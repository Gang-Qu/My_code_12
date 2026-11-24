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

    height, width = expected_hw
    size = (width, height)
    if y.shape != (height, width):
        bilinear = Image.Resampling.BILINEAR
        y = np.asarray(Image.fromarray(y).resize(size, bilinear))
        # Conservatively exclude any measurement influenced by invalid source pixels.
        invalid = (~measurement_valid).astype(np.float32)
        measurement_valid = np.asarray(Image.fromarray(invalid).resize(size, bilinear)) == 0
    if depth.shape != (height, width):
        depth = np.asarray(Image.fromarray(depth).resize(size, Image.Resampling.NEAREST))
    return y, measurement_valid, depth


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


def read_pair(image, depth, c, training=False):
    y, valid = measurement(image, c)
    source_hw = y.shape
    z = read_array(depth).astype(np.float32) * c['depth_scale']
    target = c['expected_hw'] if training else source_hw
    y, valid, z = resize_pair(y, valid, z, target)
    valid = valid & np.isfinite(z) & (z >= c['depth_min']) & (z <= c['depth_max']) & (z > 0)
    return y, np.where(valid, z, 0), valid, source_hw
