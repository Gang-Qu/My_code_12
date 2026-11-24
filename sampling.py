"""Fixed complementary masks on a zero-based full image grid."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

RULE = 'fixed-2x4-pcg64-518-v1'


@lru_cache(maxsize=2)
def phase_offsets(seed):
    # Same random permutation as the original fixed layout on a 1036 x 2072 frame.
    if seed == 41:
        path = Path(__file__).parent / 'masks' / 'fixed_seed41_v1.npz'
        with np.load(path) as masks:
            return masks['offsets']
    ranks = np.random.PCG64(seed).random_raw((8, 518, 518))
    return np.argsort(ranks, axis=0, kind='stable').astype(np.uint8)


def fixed_coords(hw, seed):
    h, w = hw
    if h < 1 or w < 1:
        raise ValueError('Expected a nonempty frame')
    rows, cols = np.ogrid[:(h + 1)//2, :(w + 3)//4]
    offsets = phase_offsets(seed)[:, rows % 518, cols % 518]
    ys = 2 * rows + offsets // 4
    xs = 4 * cols + offsets % 4
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
