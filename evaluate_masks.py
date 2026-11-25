"""Evaluate one registered GT/prediction pair under a folder of binary masks."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from data import image_files, read_array
from metrics import RELATIVE_ERROR_LIMIT
from settings import SETTINGS


def depth_image(path):
    raw = read_array(path)
    if raw.ndim != 2 or raw.dtype.kind not in 'uif':
        raise ValueError(f'{path}: 深度必须是单通道数值图像，实际为 {raw.shape}, {raw.dtype}')
    return raw


def align_gt(raw, shape, geometry_npz):
    if geometry_npz:
        with np.load(geometry_npz, allow_pickle=False) as f:
            g = json.loads(f['geometry_json'].item())
        if list(raw.shape) != g['source_hw'] or list(shape) != g['hw']:
            raise ValueError('几何记录中的原图/裁剪尺寸与本次 GT/重建图不符')
        h, w = g['resized_hw']
        if raw.shape != (h, w):
            # Nearest sampling by original row/column keeps the raw depth dtype.
            row_index = Image.fromarray(np.arange(raw.shape[0], dtype=np.int32)[:, None])
            col_index = Image.fromarray(np.arange(raw.shape[1], dtype=np.int32)[None, :])
            rows = np.asarray(row_index.resize((1, h), Image.Resampling.NEAREST)).ravel()
            cols = np.asarray(col_index.resize((w, 1), Image.Resampling.NEAREST)).ravel()
            raw = raw[np.ix_(rows, cols)]
        y, x = g['crop_origin_yx']
        raw = raw[y:y+shape[0], x:x+shape[1]]
    if raw.shape != shape:
        raise ValueError(f'GT {raw.shape} 与重建网格 {shape} 不一致。请提供已配准 GT 或使用 --geometry-npz。')
    return raw


def resized_mask(path, shape):
    raw = read_array(path)
    if raw.ndim == 2:
        selected = raw > 0
    elif raw.ndim == 3 and raw.shape[-1] in (3, 4):
        selected = (raw[..., :3] > 0).any(axis=-1)
        if raw.shape[-1] == 4:
            selected &= raw[..., 3] > 0
    else:
        raise ValueError(f'{path}: 无法解析 mask 形状 {raw.shape}')
    mask = Image.fromarray(selected).resize(shape[::-1], Image.Resampling.NEAREST)
    return np.asarray(mask)


def error_totals(pred, gt, selected):
    p, g = pred[selected], gt[selected]
    if not np.isfinite(p).all():
        raise ValueError('mask 有效真值区域内存在非有限预测，请核查深度文件')
    error = np.abs(p - g)
    # Allow only floating-point roundoff at the inclusive 3% boundary.
    limit = RELATIVE_ERROR_LIMIT * g + 4 * np.spacing(g)
    return np.array([g.size, error.sum(), np.dot(error, error), (error <= limit).sum()], np.float64)


def summarize(totals):
    n, absolute, squared, hits = totals
    return dict(valid_pixels=int(n), within_3pct_pixels=int(hits),
                mae_m=absolute/n if n else None, rmse_m=np.sqrt(squared/n) if n else None,
                accuracy_3pct=hits/n if n else None)


def evaluate_masks(mask_dir, gt_file, pred_file, output, *, gt_scale=1., pred_scale=None,
                   pred_offset=0., depth_min=.5, depth_max=5., geometry_npz=None):
    pred_raw = depth_image(pred_file)
    if pred_scale is None:
        if pred_raw.dtype.kind != 'f':
            raise ValueError('整数预测需要指定 --pred-scale；若编码为毫米，请填 0.001')
        pred_scale = 1.
    if not np.isfinite([pred_scale, gt_scale, pred_offset, depth_min, depth_max]).all() or not (
        min(pred_scale, gt_scale) > 0 and 0 <= depth_min < depth_max
    ):
        raise ValueError('请检查深度换算比例、偏移和有效范围')
    masks = image_files(mask_dir)
    gt_raw = align_gt(depth_image(gt_file), pred_raw.shape, geometry_npz)
    gt = gt_raw.astype(np.float64) * gt_scale
    pred = pred_raw.astype(np.float64) * pred_scale + pred_offset
    valid = np.isfinite(gt) & (gt > 0) & (gt >= depth_min) & (gt <= depth_max)
    union = np.zeros_like(valid)
    pooled = np.zeros(4)
    rows = []
    output = Path(output)
    for path in masks:
        mask = resized_mask(path, pred.shape)
        union |= mask
        selected = mask & valid
        totals = error_totals(pred, gt, selected)
        pooled += totals
        rows.append(dict(mask=path.name, covered_pixels=int(mask.sum()), **summarize(totals)))
        folder = output / path.name
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(folder / 'mask.png')
        for name, raw in [('gt', gt_raw), ('prediction', pred_raw)]:
            tifffile.imwrite(folder / f'{name}_masked.tiff', np.where(mask, raw, 0))
        errors = np.full(pred.shape, np.nan, dtype=np.float32)
        errors[selected] = np.abs(pred[selected] - gt[selected])
        tifffile.imwrite(folder / 'abs_error_m.tiff', errors)

    used = [row for row in rows if row['valid_pixels']]
    mean = {key: float(np.mean([row[key] for row in used])) if used else None
            for key in ('mae_m', 'rmse_m', 'accuracy_3pct')}
    result = dict(gt_file=str(gt_file), pred_file=str(pred_file), mask_dir=str(mask_dir),
                  geometry_npz=str(geometry_npz) if geometry_npz else None,
                  gt_scale=gt_scale, pred_scale=pred_scale, pred_offset_m=pred_offset,
                  depth_range_m=[depth_min, depth_max], shape_hw=list(pred.shape),
                  gt_dtype=str(gt_raw.dtype), pred_dtype=str(pred_raw.dtype),
                  relative_error_limit=RELATIVE_ERROR_LIMIT, per_mask=rows,
                  mean_over_masks=dict(nonempty_masks=len(used), **mean),
                  pooled_with_repeats=summarize(pooled),
                  union_unique_pixels=summarize(error_totals(pred, gt, union & valid)))
    (output / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('mask-dir', 'gt-file', 'pred-file', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--gt-scale', type=float, default=SETTINGS['depth_scale'], help='GT metres per stored unit')
    parser.add_argument('--pred-scale', type=float, help='prediction metres per stored unit (required for integer files)')
    parser.add_argument('--pred-offset', type=float, default=0., help='prediction offset in metres after scaling')
    parser.add_argument('--depth-min', type=float, default=SETTINGS['depth_min'])
    parser.add_argument('--depth-max', type=float, default=SETTINGS['depth_max'])
    parser.add_argument('--geometry-npz', type=Path, help='matching reconstruction NPZ when GT is on the original grid')
    args = parser.parse_args()
    result = evaluate_masks(**vars(args))
    for name in ('mean_over_masks', 'union_unique_pixels'):
        print(name, json.dumps(result[name], ensure_ascii=False))
    print(args.output / 'metrics.json')
