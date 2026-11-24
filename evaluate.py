"""Each evaluation starts from paired full images in the raw test folders."""
import csv
import json
import numpy as np
from pathlib import Path

from data import paired_files, read_pair
from metrics import EpochStats
from outputs import plot_test_bins, save_result
from reconstruct_real import geometry, load_checkpoint, reconstruct
from settings import SETTINGS
from prep_preview import previews
from sampling import gather


def evaluate(net, c, pairs, folder, device, epoch, skipped_frames=()):
    net.eval()
    stats = EpochStats(c)
    rows = []
    skipped = list(skipped_frames)
    folder.mkdir(parents=True, exist_ok=True)
    for i, (image, label) in enumerate(pairs):
        y, gt, valid, meta = read_pair(image, label, c)
        if not valid.any():
            skipped.append(str(label))
            continue
        show_mask = not rows and c.get('save_sampling_preview', False)
        if show_mask:
            pred, coords = reconstruct(net, y, c, device, return_coords=True)
            folder.mkdir(parents=True, exist_ok=True)
            sample_dir = folder / f'sampling_{coords.shape[0]}_parts'
            previews(sample_dir, i, y, gt, valid, *(gather(a, coords) for a in (y, gt, valid)), c)
            np.save(sample_dir / 'coords_yx.npy', coords)
        else:
            pred = reconstruct(net, y, c, device)
        rows.append(dict(source=image.name, **stats.update(pred[None], gt[None], valid[None])))
        save_result(folder, f'{i:04d}_{image.stem}', y, pred, geometry(y, c, meta), c, gt)
    result = dict(**stats.result(), epoch=epoch, per_image=rows, skipped_frames=skipped,
                  evaluation=dict(bin_width_m=.1, relative_error_limit=.03,
                                  rule='abs(pred-gt)/gt <= 0.03', config=c))
    (folder / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    fields = ['lower_m', 'upper_m', 'upper_inclusive', 'valid_pixels', 'pixel_fraction',
              'within_3pct_pixels', 'accuracy_3pct']
    with (folder / 'depth_bins.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['depth_bins'])
    plot_test_bins(folder, result)
    return result


if __name__ == '__main__':
    from reconstruct_real import CHECKPOINT
    net, config = load_checkpoint(CHECKPOINT, SETTINGS['device'])
    pairs = paired_files(SETTINGS['val_y'], SETTINGS['val_depth'])
    evaluate(net, config, pairs, Path(SETTINGS['output']) / 'evaluation', SETTINGS['device'],
             Path(CHECKPOINT).stem)
