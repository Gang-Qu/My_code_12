"""Fixed complementary masks on a zero-based full image grid."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

LEGACY_RULE = 'fixed-2x4-pcg64-518-v1'
RULE = 'fixed-xy-pcg64-v2'


@lru_cache(maxsize=2)
def phase_offsets(seed):
    # Same random permutation as the original fixed layout on a 1036 x 2072 frame.
    if seed == 41:
        path = Path(__file__).parent / 'masks' / 'fixed_seed41_v1.npz'
        with np.load(path) as masks:
            return masks['offsets']
    ranks = np.random.PCG64(seed).random_raw((8, 518, 518))
    return np.argsort(ranks, axis=0, kind='stable').astype(np.uint8)


def fixed_coords(hw, seed, grid_xy=(4, 2)):
    h, w = hw
    if h < 1 or w < 1:
        raise ValueError('Expected a nonempty frame')
    if len(grid_xy) != 2 or any(type(n) is not int or n < 1 for n in grid_xy):
        raise ValueError('phase_grid_xy requires two positive integers [horizontal, vertical]')
    a, b = grid_xy
    nh, nw = (h+b-1)//b, (w+a-1)//a
    rows, cols = np.ogrid[:nh, :nw]
    if (a, b) == (4, 2):
        offsets = phase_offsets(seed)[:, rows % 518, cols % 518]
    else:
        offsets = np.empty((a*b, nh, nw), dtype=np.int32)
        for row in range(nh):
            rng = np.random.PCG64([seed, a, b, row])
            ranks = rng.random_raw((nw, a*b))
            offsets[:, row] = np.argsort(ranks, axis=1, kind='stable').T
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
