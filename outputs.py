from pathlib import Path
import json
import numpy as np
import tifffile
from PIL import Image, ImageDraw

PREVIEW_GAP = 92
PREVIEW_TOP = 26


def preview(path,y,z,c,gt=None,full_resolution=False):
    # Only PNG is display-scaled; TIFF/NPZ retain exact metric values.
    def depth(v):
        a=np.clip((v-c['depth_min'])/(c['depth_max']-c['depth_min']),0,1)
        a=np.nan_to_num(a);return np.stack((a,np.sqrt(a),1-a),-1)
    panels = [y, z] if gt is None else [y, z, gt]
    labels = ['Measurement', 'Depth (m)', 'GT (m)'][:len(panels)]
    h, w = y.shape
    width, height = w * len(panels), h

    def paste_panel(canvas, index, x, top):
        # Bound scratch space to a strip even for 15 MP measurements.
        for row in range(0, h, 128):
            strip = panels[index][row:row+128]
            values = np.clip(strip, 0, 1) if index == 0 else depth(strip)
            piece = Image.fromarray((values * 255).round().astype(np.uint8))
            canvas.paste(piece, (x, top + row))

    if not full_resolution:
        im = Image.new('RGB', (width, height))
        for i in range(len(panels)):
            paste_panel(im, i, i*w, 0)
        im.thumbnail((1800,900))
        width, height = im.size
    lo,hi=float(c['depth_min']),float(c['depth_max'])
    ticks=np.arange(np.ceil(lo/.25),np.floor(hi/.25)+1)*.25
    # One identical continuous scale beside each depth panel. Leave sufficient
    # vertical spacing for 0.25 m labels, even for small preview images.
    bar_h=max(height, int(np.ceil((hi-lo)/.25))*16+1)
    gap=PREVIEW_GAP
    canvas=Image.new('RGB',(width+gap*(len(panels)-1),bar_h+40),'white')
    draw=ImageDraw.Draw(canvas)
    offset=0
    for i,label in enumerate(labels):
        left = i * width // len(panels)
        right = (i+1) * width // len(panels)
        if full_resolution:
            paste_panel(canvas, i, offset, PREVIEW_TOP)
        else:
            canvas.paste(im.crop((left, 0, right, height)), (offset, PREVIEW_TOP))
        draw.text((offset+5,5), label, fill='black')
        offset += right - left
        if i:
            x=offset+8;top=PREVIEW_TOP
            ramp=np.linspace(hi,lo,bar_h,dtype=np.float32)[:,None]
            colors=(depth(ramp)*255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(np.repeat(colors,16,axis=1)),(x,top))
            draw.rectangle((x,top,x+16,top+bar_h-1),outline='black')
            for value in ticks:
                yy=top+int(round((hi-value)/(hi-lo)*(bar_h-1)))
                draw.line((x+16,yy,x+21,yy),fill='black')
                draw.text((x+24,yy-5),f'{value:.2f}',fill='black')
            draw.text((x,8),'m',fill='black')
            offset+=gap
    canvas.save(path, compress_level=1 if c.get('fast_reconstruction_export', False) else 6)

def read_test_panels(path):
    """Extract full-size RGB depth panels from a save_result test PNG."""
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != 'RGB':
            raise ValueError(f'{path}: 需要 test 导出的 RGB 拼图 PNG')
        width, remainder = divmod(image.width - 2 * PREVIEW_GAP, 3)
        if remainder or width <= 0 or image.height <= PREVIEW_TOP:
            raise ValueError(f'{path}: 不符合三面板 test PNG 布局')
        panels = []
        for x in (width, 2 * width + PREVIEW_GAP):
            # The depth palette has complementary red/blue, so white is
            # unambiguously the footer even when the colorbar is taller than the image.
            stripe = np.asarray(image.crop((x, PREVIEW_TOP, x+1, image.height)))[:, 0]
            footer = np.flatnonzero(np.all(stripe == 255, axis=1))
            if not footer.size or footer[0] == 0:
                raise ValueError(f'{path}: 找不到深度面板底边')
            bottom = PREVIEW_TOP + int(footer[0])
            panel = np.asarray(image.crop((x, PREVIEW_TOP, x+width, bottom)))
            rb = panel[..., 0].astype(np.int16) + panel[..., 2]
            if rb.min() < 254 or rb.max() > 256:
                raise ValueError(f'{path}: 面板颜色与 test 深度示意图不符')
            panels.append(panel)
        pred, gt = panels
        if pred.shape != gt.shape:
            raise ValueError(f'{path}: 重建和真值面板尺寸不一致')
        return gt, pred


def save_result(folder,name,y,z,g,c,gt=None,save_arrays=True):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if save_arrays:
        tifffile.imwrite(folder/(name+'_depth_m.tiff'),z.astype(np.float32, copy=False),photometric='minisblack')
        metadata = dict(g, phase_mask_mode=c.get('phase_mask_mode', 'regular'),
                        phase_mask_seed=c.get('phase_mask_seed'))
        arrays = dict(measurement=y, depth_m=z, geometry_json=np.asarray(json.dumps(metadata)))
        if gt is not None:arrays['gt_m']=gt
        save_npz = np.savez if c.get('fast_reconstruction_export', False) else np.savez_compressed
        save_npz(folder/(name+'.npz'), **arrays)
    preview(folder/(name+'.png'),y,z,c,gt,full_resolution=True)

def plot_test_bins(folder, result):
    """Archive the current epoch's pooled final predictions on the GT depth axis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bins = result['depth_bins']
    centers = [(b['lower_m'] + b['upper_m']) / 2 for b in bins]
    fig, (population, accuracy) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                              constrained_layout=True)
    try:
        for ax, key, label, color in (
            (population, 'pixel_fraction', 'Share of all valid test pixels (%)', '#28647d'),
            (accuracy, 'accuracy_3pct', 'Within 3% of GT in bin (%)', '#b75530'),
        ):
            values = [float('nan') if b[key] is None else 100*b[key] for b in bins]
            ax.plot(centers, values, 'o-', markersize=3, color=color)
            ax.set_ylabel(label)
            ax.grid(alpha=.25)
        population.set_ylim(bottom=0)
        accuracy.set_ylim(-2, 102)
        accuracy.set_xlabel('Ground-truth depth bin center (m)')
        width = result['evaluation']['bin_width_m']
        fig.suptitle(f"Epoch {result['epoch']} — test reconstruction in {width:g} m bins")
        fig.savefig(Path(folder)/'depth_bins.png', dpi=160)
    finally:
        plt.close(fig)
