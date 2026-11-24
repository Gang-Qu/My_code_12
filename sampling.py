"""Fixed complementary masks on a zero-based full image grid."""
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile
from zlib import error as ZlibError

import numpy as np
import torch

LEGACY_RULE = 'fixed-2x4-pcg64-518-v1'
RULE = 'fixed-xy-pcg64-v2'


MASK_DIR = Path(__file__).resolve().parent / 'masks'


def _generate_offsets(seed, a, b, nh, nw):
    if (a, b) == (4, 2):
        ranks = np.random.PCG64(seed).random_raw((8, nh, nw))
        return np.argsort(ranks, axis=0, kind='stable').astype(np.uint8)
    offsets = np.empty((a*b, nh, nw), dtype=np.int32)
    for row in range(nh):
        rng = np.random.PCG64([seed, a, b, row])
        offsets[:, row] = np.argsort(rng.random_raw((nw, a*b)), axis=1, kind='stable').T
    return offsets


@lru_cache(maxsize=2)
def cached_offsets(seed, a, b, nh, nw, directory):
    """Persist the existing seed/grid algorithm; concurrent writers produce identical masks."""
    name = (f'fixed_seed{seed}_v1.npz' if (a,b) == (4,2) else
            f'fixed_{a}x{b}_seed{seed}_{nh}x{nw}_v2.npz')
    path = Path(directory) / name
    try:
        with np.load(path, allow_pickle=False) as archive:
            offsets = archive['offsets']
        if (not isinstance(offsets, np.ndarray) or offsets.shape != (a*b, nh, nw)
                or offsets.dtype.kind not in 'iu'):
            raise ValueError('wrong mask shape or dtype')
        if not np.all(np.sort(offsets, axis=0) == np.arange(a*b)[:, None, None]):
            raise ValueError('each grid cell must be a permutation')
        return offsets
    except (OSError, ValueError, EOFError, TypeError, BadZipFile, KeyError, ZlibError, RuntimeError) as exc:
        print(f'Fixed mask unavailable ({type(exc).__name__}); regenerating {path}', flush=True)

    offsets = _generate_offsets(seed, a, b, nh, nw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tmp_path.open('wb') as stream:
            np.savez_compressed(stream, offsets=offsets)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return offsets


def phase_offsets(seed, mask_dir=None):
    return cached_offsets(seed, 4, 2, 518, 518, mask_dir or MASK_DIR)


def fixed_coords(hw, seed, grid_xy=(4, 2), *, mask_dir=None, persist=True):
    h, w = hw
    if h < 1 or w < 1:
        raise ValueError('Expected a nonempty frame')
    if len(grid_xy) != 2 or any(type(n) is not int or n < 1 for n in grid_xy):
        raise ValueError('phase_grid_xy requires two positive integers [horizontal, vertical]')
    a, b = grid_xy
    nh, nw = (h+b-1)//b, (w+a-1)//a
    rows, cols = np.ogrid[:nh, :nw]
    if (a, b) == (4, 2):
        offsets = phase_offsets(seed, mask_dir)[:, rows % 518, cols % 518]
    elif persist:
        offsets = cached_offsets(seed, a, b, nh, nw, mask_dir or MASK_DIR)
    else:
        offsets = _generate_offsets(seed, a, b, nh, nw)
    ys = b * rows + offsets // a
    xs = a * cols + offsets % a
    coords = np.stack((ys, xs), axis=-1).astype(np.int32)
    coords[(ys >= h) | (xs >= w)] = -1
    return coords


def gather(frame, coords):
    y, x = coords[..., 0], coords[..., 1]
    return np.where(y >= 0, frame[y.clip(0), x.clip(0)], frame.dtype.type(0))


def restore(parts, coords, hw):
    h, w = hw
    y, x = coords.unbind(-1)
    good = y >= 0
    index = y[good].long() * w + x[good].long()
    return parts.new_zeros(h*w).scatter(0, index, parts[good]).reshape(h, w)


LEARNED_RULE = 'fixed-grid-learned-pair-swaps-v1'


class _PairSwap(torch.nn.Module):
    """One binary decision per disjoint pair at each position of a repeating cell tile."""
    def __init__(self, count, stride, tile_hw, generator):
        super().__init__()
        partners = list(range(count))
        pair_ids = [0] * count
        pairs = 0
        for i in range(count):
            j = i ^ stride
            if i < j < count:
                partners[i], partners[j] = j, i
                pair_ids[i] = pair_ids[j] = pairs
                pairs += 1
        self.register_buffer('partners', torch.tensor(partners), persistent=False)
        self.register_buffer('pair_ids', torch.tensor(pair_ids), persistent=False)
        scores = torch.empty(pairs, *tile_hw).uniform_(-.01, .01, generator=generator)
        self.logits = torch.nn.Parameter(scores)

    def forward(self, values):
        _, h, w = values.shape[:3]
        rows = torch.arange(h, device=values.device) % self.logits.shape[1]
        cols = torch.arange(w, device=values.device) % self.logits.shape[2]
        logits = self.logits[self.pair_ids[:, None, None], rows[None, :, None], cols]
        if values.ndim == 4:  # (row,column) must travel with its measurement
            logits = logits[..., None]
        partner = values.index_select(0, self.partners)
        result = torch.where(logits > 0, partner, values)
        if self.training and torch.is_grad_enabled() and values.is_floating_point():
            soft = logits.sigmoid()
            # Hard, exact swap in forward; sigmoid derivative for the discrete decision.
            result = result + (soft - soft.detach()) * (partner - values).detach()
        return result


class LearnedMask(torch.nn.Module):
    def __init__(self, grid_xy, tile_hw, seed):
        super().__init__()
        for name, shape in [('phase_grid_xy', grid_xy), ('mask_tile_hw', tile_hw)]:
            if len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError(f'{name} requires two positive integers')
        count = grid_xy[0] * grid_xy[1]
        strides = [1 << i for i in range((count - 1).bit_length())]
        generator = torch.Generator().manual_seed(seed % (1 << 63))
        self.stages = torch.nn.ModuleList(
            _PairSwap(count, stride, tile_hw, generator)
            for stride in strides + strides[:-1][::-1]
        )

    def route(self, values, inverse=False):
        # Disjoint paired swaps are self-inverse; undo the stages in reverse order.
        stages = reversed(self.stages) if inverse else self.stages
        for stage in stages:
            values = stage(values)
        return values
