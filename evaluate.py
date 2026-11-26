"""Each evaluation starts from paired full images in the raw test folders."""
from time import perf_counter
import csv
import json
import numpy as np
from pathlib import Path

from data import keep_training_frame, paired_files, read_pair, train_depth_threshold
from metrics import EpochStats
from outputs import plot_test_bins, save_result
from reconstruct_real import inference_options, geometry, load_checkpoint, reconstruct
from settings import SETTINGS
from prep_preview import previews
from sampling import gather


def evaluate(net, c, pairs, folder, device, epoch, skipped_frames=(), export_images=True):
    threshold = train_depth_threshold(c)
    net.eval()
    stats = EpochStats(c)
    rows = []
    skipped = list(skipped_frames)
    folder.mkdir(parents=True, exist_ok=True)
    for i, (image, label) in enumerate(pairs):
        start = perf_counter()
        y, gt, valid, meta = read_pair(image, label, c)
        if not keep_training_frame(meta, threshold):
            skipped.append(str(label))
            continue
        ready = perf_counter()
        show_mask = export_images and not rows and c.get('save_sampling_preview', False)
        if show_mask:
            pred, coords = reconstruct(net, y, c, device, return_coords=True)
        else:
            pred = reconstruct(net, y, c, device)
        inferred = perf_counter()
        rows.append(dict(source=image.name, **stats.update(pred[None], gt[None], valid[None])))
        scored = perf_counter()
        if show_mask:
            folder.mkdir(parents=True, exist_ok=True)
            sample_dir = folder / f'sampling_{coords.shape[0]}_parts'
            previews(sample_dir, i, y, gt, valid, *(gather(a, coords) for a in (y, gt, valid)), c)
            np.save(sample_dir / 'coords_yx.npy', coords)
        if export_images:
            save_result(folder, f'{i:04d}_{image.stem}', y, pred, geometry(y, c, meta), c, gt)
        if c.get('profile_reconstruction'):
            times = dict(read_crop=ready-start, reconstruct=inferred-ready,
                         metrics=scored-inferred, export=perf_counter()-scored)
            rows[-1]['timing_s'] = times
            print(f'{image.name} {y.shape}: ' + ' '.join(f'{k}={v:.3f}s' for k, v in times.items()), flush=True)
    result = dict(**stats.result(), epoch=epoch, per_image=rows, skipped_frames=skipped,
                  evaluation=dict(min_depth_in_range_percent=threshold, retained_frames=len(rows),
                                  total_frames=len(rows) + len(skipped), bin_width_m=.1, relative_error_limit=.03,
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
    folder = Path(SETTINGS['output']) / 'evaluation'
    net, config = load_checkpoint(CHECKPOINT, SETTINGS['device'], strict=SETTINGS['strict_inference_weights'],
                                  report_path=folder / 'weight_load.json')
    config['strict_inference_weights'] = SETTINGS['strict_inference_weights']
    config['min_train_depth_percent'] = train_depth_threshold(SETTINGS)
    config.update(val_image_extensions=SETTINGS['val_image_extensions'], depth_extensions=SETTINGS['depth_extensions'],
                  **inference_options(SETTINGS))
    pairs = paired_files(SETTINGS['val_y'], SETTINGS['val_depth'], config=config)
    evaluate(net, config, pairs, folder, SETTINGS['device'],
             Path(CHECKPOINT).stem)
