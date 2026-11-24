"""Depth-range audit in metres, before range filtering."""
import json
from pathlib import Path

import numpy as np

from settings import SETTINGS


def depth_counts(z, lo, hi):
    positive = z[np.isfinite(z) & (z > 0)]
    nonfinite = int((~np.isfinite(z)).sum())
    below = int((positive < lo).sum())
    above = int((positive > hi).sum())
    return dict(pixels=z.size, nonfinite=nonfinite, nonpositive=z.size-nonfinite-positive.size,
                below=below, in_range=positive.size-below-above, above=above,
                min_positive_m=float(positive.min()) if positive.size else None,
                max_positive_m=float(positive.max()) if positive.size else None)


def audit_pair(image, label, config, training):
    from data import read_pair
    info = {}
    read_pair(image, label, config, training=training, audit=info)
    return info


def split_report(rows):
    summary = {}
    for stage in ('original', 'crop'):
        known = [r[stage] for r in rows if stage in r]
        if not known:
            continue
        counts = {k: sum(r[k] for r in known) for k in
                  ('pixels', 'nonfinite', 'nonpositive', 'below', 'in_range', 'above')}
        for key, reduce in [('min_positive_m', min), ('max_positive_m', max)]:
            counts[key] = reduce((r[key] for r in known if r[key] is not None), default=None)
        summary[stage] = dict(frames=len(known), **counts)
    return dict(frames=len(rows), retained_frames=sum(r['supervised_pixels'] > 0 for r in rows),
                supervised_pixels=sum(r['supervised_pixels'] for r in rows),
                zero_supervision_frames=[r['source'] for r in rows if not r['supervised_pixels']],
                **summary, per_frame=rows)


def write_audit(folder, c, splits):
    report = dict(depth_scale=c['depth_scale'], lower_m=c['depth_min'], upper_m=c['depth_max'],
                  bounds_inclusive=True, splits={s: split_report(rows) for s, rows in splits.items()})
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'depth_range_audit.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    for name, summary in report['splits'].items():
        crop = summary.get('crop')
        detail = (f"positive={crop['min_positive_m']}..{crop['max_positive_m']} m, "
                  f"below={crop['below']} above={crop['above']}" if crop else 'pre-filter range unavailable')
        print(f"Depth audit {name}: {detail}, supervised={summary['supervised_pixels']}, kept_frames={summary['retained_frames']}/{summary['frames']}", flush=True)
    print(f'Depth audit saved to {path}', flush=True)
    return report


if __name__ == '__main__':
    from data import paired_files
    splits = {s: [audit_pair(image, label, SETTINGS, s == 'train') for image, label in
                  paired_files(SETTINGS[s+'_y'], SETTINGS[s+'_depth'])] for s in ('train', 'val')}
    write_audit(SETTINGS['output'], SETTINGS, splits)
