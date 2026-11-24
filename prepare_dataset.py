"""Prepare only supervised training frames. Validation and real frames stay raw."""
import hashlib
import json
from pathlib import Path
from tempfile import mkdtemp

import numpy as np

from data import paired_files, read_pair
from sampling import RULE, fixed_coords, gather
from settings import SETTINGS
from depth_audit import write_audit
from prep_preview import previews

PREP_KEYS = ('phase_mask_mode', 'phase_mask_seed', 'phase_grid_xy', 'center_crop_hw', 'expected_hw', 'depth_scale',
             'depth_min', 'depth_max', 'black_level', 'white_level', 'gain', 'input_encoding')


def signature(c):
    if c['phase_mask_mode'] not in ('fixed', 'learned'):
        raise ValueError("Unknown phase_mask_mode")
    if type(c['phase_mask_seed']) is not int or c['phase_mask_seed'] < 0:
        raise ValueError('phase_mask_seed must be a nonnegative integer')
    # Optional NPZ always stores the fixed, lossless base partition.
    return {**{key: c[key] for key in PREP_KEYS}, 'phase_mask_mode': 'fixed'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def prepare(c):
    manifest = dict(rule=RULE, preprocessing=signature(c), train=[], skipped_train=[])
    pairs = paired_files(c['train_y'], c['train_depth'])
    root = Path(c['prepared_dir']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    generation = Path(mkdtemp(prefix='frames-', dir=root))
    for i, (image, depth) in enumerate(pairs):
        audit = {} if c.get('dataset_audit', True) else None
        y, z, valid, meta = read_pair(image, depth, c, training=True, audit=audit)
        if not valid.any():
            manifest['skipped_train'].append(audit or dict(source=str(depth), supervised_pixels=0))
            continue
        coords = fixed_coords(y.shape, c['phase_mask_seed'], c['phase_grid_xy'])
        path = generation / f'frame_{i:06d}.npz'
        parts, depths, masks = (gather(a, coords) for a in (y, z, valid))
        np.savez(path, y=parts, depth=depths, valid=masks, coords_yx=coords)
        folder = path.with_suffix('')
        previews(folder, i, y, z, valid, parts, depths, masks, c)
        manifest['train'].append(dict(file=str(path.relative_to(root)), source=str(image.resolve()),
                                      depth_source=str(depth.resolve()), **meta, preview=str((folder / "index.png").relative_to(root)),
                                      sha256=digest(image)))
        if audit is not None:
            manifest['train'][-1]['depth_audit'] = audit
        print(f'{image.name} -> {path.name} ({len(coords)} fixed masks)', flush=True)
    if c.get('dataset_audit', True):
        audits = [row['depth_audit'] for row in manifest['train']] + manifest['skipped_train']
        write_audit(generation, c, {'train': audits})
    if not manifest['train']:
        raise ValueError(f"整个训练集筛选后没有监督点。请核对深度单位、范围和裁剪区域，或运行 python depth_audit.py")
    staged = generation / 'manifest.json'
    staged.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    staged.replace(root / 'manifest.json')
    return manifest


if __name__ == '__main__':
    prepare(SETTINGS)
