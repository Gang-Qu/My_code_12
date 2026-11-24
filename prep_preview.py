"""Numbered measurement and metric depth views from the exact arrays written to NPZ."""
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def panel(y, z, valid, c):
    gray = Image.fromarray((np.clip(y, 0, 1)*255).round().astype('uint8')).convert('RGB')
    t = np.clip((z-c['depth_min'])/(c['depth_max']-c['depth_min']), 0, 1)
    rgb = np.stack((t, np.sqrt(t), 1-t), axis=-1)
    rgb[~valid] = 0
    depth = Image.fromarray((rgb*255).round().astype('uint8'))
    image = Image.new('RGB', (2*y.shape[1], y.shape[0]))
    image.paste(gray, (0, 0))
    image.paste(depth, (y.shape[1], 0))
    return image


def previews(folder, number, y, z, valid, parts, depths, masks, c):
    folder.mkdir(exist_ok=True)
    panel(y, z, valid, c).save(folder/'crop.png')
    count = len(parts)
    cols = min(c['phase_grid_xy'][0], 6)
    sheet = Image.new('RGB', (cols*260, ((count+cols-1)//cols)*162), '#20252a')
    for k, (part, depth, mask) in enumerate(zip(parts, depths, masks)):
        label = f'Frame {number:06d}   {k+1:03d}/{count:03d}'
        pair = panel(part, depth, mask, c)
        labeled = Image.new('RGB', (max(pair.width, 210), pair.height+24), '#20252a')
        labeled.paste(pair, (0, 24))
        ImageDraw.Draw(labeled).text((5, 5), label, fill='white')
        labeled.save(folder/f'part_{k+1:03d}.png')
        pair = ImageOps.contain(pair, (256, 134), method=Image.Resampling.NEAREST)
        x, yy = k % cols*260, k//cols*162
        sheet.paste(pair, (x+2, yy+24))
        ImageDraw.Draw(sheet).text((x+5, yy+4), label, fill='white')
    sheet.save(folder/'index.png')
