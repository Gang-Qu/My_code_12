"""Each evaluation starts from paired full images in the raw test folders."""
import json
from pathlib import Path

from data import paired_files, read_pair
from metrics import EpochStats
from outputs import plot_test_bins, save_result
from reconstruct_real import geometry, load_checkpoint, reconstruct
from settings import SETTINGS


def evaluate(net, c, pairs, folder, device, epoch):
    net.eval()
    stats = EpochStats(c)
    rows = []
    for i, (image, label) in enumerate(pairs):
        y, gt, valid, meta = read_pair(image, label, c)
        pred = reconstruct(net, y, c, device)
        rows.append(dict(source=image.name, **stats.update(pred[None], gt[None], valid[None])))
        save_result(folder, f'{i:04d}_{image.stem}', y, pred, geometry(y, c, meta), c, gt)
    result = dict(**stats.result(), epoch=epoch, per_image=rows,
                  evaluation=dict(bin_width_m=.1, relative_error_limit=.03,
                                  rule='abs(pred-gt)/gt <= 0.03', config=c))
    (folder / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    plot_test_bins(folder, result)
    return result


if __name__ == '__main__':
    from reconstruct_real import CHECKPOINT
    net, config = load_checkpoint(CHECKPOINT, SETTINGS['device'])
    pairs = paired_files(SETTINGS['val_y'], SETTINGS['val_depth'])
    evaluate(net, config, pairs, Path(SETTINGS['output']) / 'evaluation', SETTINGS['device'],
             Path(CHECKPOINT).stem)
