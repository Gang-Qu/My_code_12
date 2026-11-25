"""Exact interleaved/tiled packing, common radiometry and paired datasets."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import tifffile

EXT={'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}
def read_array(path):
    path=Path(path)
    if path.suffix.lower() in ('.tif','.tiff'):
        with tifffile.TiffFile(path) as f:
            if len(f.pages)!=1:raise ValueError(f'Expected one image, not multipage TIFF: {path}')
            return f.pages[0].asarray()
    with Image.open(path) as im:
        if im.mode == "P":
            im = im.convert("RGB")
        return np.asarray(im)

def measurement(path,c):
    return convert_measurement(read_array(path), c, path)

def convert_measurement(a, c, path):
    dtype=a.dtype
    if a.ndim==3 and a.shape[-1] in (3,4):a=a[...,:3]
    elif a.ndim!=2:raise ValueError(f'Expected gray/RGB image: {path}')
    valid=np.isfinite(a).all(-1) if a.ndim==3 else np.isfinite(a)
    a=np.nan_to_num(a.astype(np.float32),nan=0.,posinf=0.,neginf=0.)
    white=c['white_level']
    if white is None:white=float(np.iinfo(dtype).max) if np.issubdtype(dtype,np.integer) else 1.
    black=float(c['black_level']);gain=float(c['gain'])
    if not np.isfinite([black,white,gain]).all() or white<=black or gain<=0:raise ValueError('Invalid radiometry')
    a=(a-black)/(white-black)
    if c['input_encoding']=='srgb':a=np.where(a<=.04045,a/12.92,(np.maximum(a+.055,0)/1.055)**2.4)
    elif c['input_encoding']!='linear':raise ValueError('Invalid input_encoding')
    if a.ndim==3:a=(a*np.array([.2126,.7152,.0722],np.float32)).sum(-1)
    a=a*gain
    if not np.isfinite(a).all():raise ValueError('Nonfinite converted measurement')
    return a.astype(np.float32),valid

def model_input(y):
    if y.ndim!=3:raise ValueError('Expected B,H,W')
    rgb=y[:,None].expand(-1,3,-1,-1)
    mean=y.new_tensor([.485,.456,.406])[None,:,None,None]
    std=y.new_tensor([.229,.224,.225])[None,:,None,None]
    return (rgb-mean)/std

def resize_pair(y, measurement_valid, depth, expected_hw):
    """Map registered, full-field measurement and depth onto the preparation grid."""
    if depth.ndim != 2:
        raise ValueError(f'Expected single-channel metric depth TIFF, got {depth.shape}')
    if expected_hw is None:
        if y.shape != depth.shape:
            raise ValueError(f'Measurement and depth shapes differ: {y.shape}, {depth.shape}')
        return y, measurement_valid, depth

    y, measurement_valid = resize_measurement(y, measurement_valid, expected_hw)
    height, width = expected_hw
    size = (width, height)
    if depth.shape != (height, width):
        depth = np.asarray(Image.fromarray(depth).resize(size, Image.Resampling.NEAREST))
    return y, measurement_valid, depth


def resize_measurement(y, valid, expected_hw):
    if expected_hw is None:
        return y, valid
    height, width = expected_hw
    size = (width, height)
    if y.shape != (height, width):
        bilinear = Image.Resampling.BILINEAR
        y = np.array(Image.fromarray(y).resize(size, bilinear))
        # Conservatively exclude any measurement influenced by invalid source pixels.
        invalid = (~valid).astype(np.float32)
        valid = np.asarray(Image.fromarray(invalid).resize(size, bilinear)) == 0
    return y, valid


def center_region(source_hw, resized_hw, c):
    h, w = resized_hw
    target = c.get('center_crop_hw') or resized_hw
    ch, cw = target
    if not (0 < ch <= h and 0 < cw <= w):
        raise ValueError(f'center_crop_hw={target} exceeds resized frame {resized_hw}')
    r, col = (h-ch)//2, (w-cw)//2
    region = np.s_[r:r+ch, col:col+cw]
    return region, dict(source_hw=list(source_hw), resized_hw=list(resized_hw),
                        crop_origin_yx=[r, col], hw=[ch,cw])


def paired_files(image_dir, depth_dir):
    files = image_files(image_dir)
    pairs = []
    for image in files:
        matches = [Path(depth_dir) / (image.stem + ext) for ext in ('.tif', '.tiff')]
        matches = [path for path in matches if path.is_file()]
        if len(matches) != 1:
            raise ValueError(f'{image}: expected exactly one matching depth TIFF in {depth_dir}')
        pairs.append((image, matches[0]))
    return pairs


def image_files(folder):
    files = sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXT)
    if not files:
        raise ValueError(f'No measurement images in {folder}')
    return files


def train_depth_threshold(c):
    pct = c.get('min_train_depth_percent', 80.)
    if type(pct) not in (int, float) or not 0 <= pct <= 100:
        raise ValueError('min_train_depth_percent must be a percentage between 0 and 100')
    return pct


def keep_training_frame(info, threshold):
    return (info['supervised_pixels'] > 0 and
            (threshold == 0 or info.get('depth_in_range_percent', info.get('depth_in_range_lower_bound_percent', 0)) >= threshold))


def read_pair(image, depth, c, training=False, audit=None):
    y, valid = measurement(image, c)
    source_hw = y.shape
    z = read_array(depth).astype(np.float32) * c['depth_scale']
    if audit is not None:
        from depth_audit import depth_counts
        audit.update(source=str(depth), original=depth_counts(z, c['depth_min'], c['depth_max']))
    target = c['expected_hw'] if training or c.get('resize_before_inference_crop', False) else source_hw
    y, valid, z = resize_pair(y, valid, z, target)
    in_range = np.isfinite(z) & (z >= c['depth_min']) & (z <= c['depth_max']) & (z > 0)
    valid = valid & in_range
    region, meta = center_region(source_hw, y.shape, c)
    if training:
        meta.update(depth_in_range_percent=100 * int(in_range[region].sum()) / in_range[region].size,
                    supervised_pixels=int(valid[region].sum()))
    if audit is not None:
        audit.update(meta)
        audit.update(crop=depth_counts(z[region], c['depth_min'], c['depth_max']),
                     supervised_pixels=int(valid[region].sum()))
    return y[region], np.where(valid, z, 0)[region], valid[region], meta
