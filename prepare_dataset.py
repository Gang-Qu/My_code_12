"""Prepare only supervised training frames. Validation and real frames stay raw."""
import hashlib
import json
from pathlib import Path
from tempfile import mkdtemp

import numpy as np

from data import paired_files, read_pair
from sampling import RULE, fixed_coords, gather
from settings import SETTINGS

PREP_KEYS = ('phase_mask_mode', 'phase_mask_seed', 'expected_hw', 'depth_scale',
             'depth_min', 'depth_max', 'black_level', 'white_level', 'gain', 'input_encoding')


def signature(c):
    if c['phase_mask_mode'] != 'fixed':
        raise ValueError("My_code_12 requires phase_mask_mode='fixed'")
    if type(c['phase_mask_seed']) is not int or c['phase_mask_seed'] < 0:
        raise ValueError('phase_mask_seed must be a nonnegative integer')
    return {key: c[key] for key in PREP_KEYS}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def prepare(c):
    manifest = dict(rule=RULE, preprocessing=signature(c), train=[])
    pairs = paired_files(c['train_y'], c['train_depth'])
    root = Path(c['prepared_dir'])
    root.mkdir(parents=True, exist_ok=True)
    generation = Path(mkdtemp(prefix='frames-', dir=root))
    for i, (image, depth) in enumerate(pairs):
        y, z, valid, native = read_pair(image, depth, c, training=True)
        if not valid.any():
            raise ValueError(f'{depth}: frame has no supervised pixels')
        coords = fixed_coords(y.shape, c['phase_mask_seed'])
        path = generation / f'frame_{i:06d}.npz'
        np.savez(path, y=gather(y, coords), depth=gather(z, coords),
                 valid=gather(valid, coords).astype(bool), coords_yx=coords)
        manifest['train'].append(dict(file=str(path.relative_to(root)), source=str(image.resolve()),
                                      depth_source=str(depth.resolve()), source_hw=native, hw=y.shape,
                                      sha256=digest(image)))
        print(f'{image.name} -> {path.name} (8 fixed masks)', flush=True)
    staged = generation / 'manifest.json'
    staged.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    staged.replace(root / 'manifest.json')
    return manifest


if __name__ == '__main__':
    prepare(SETTINGS)
