from pathlib import Path
import json
import numpy as np
import tifffile
from PIL import Image, ImageDraw

def preview(path,y,z,c,gt=None,full_resolution=False):
    # Only PNG is display-scaled; TIFF/NPZ retain exact metric values.
    def depth(v):
        a=np.clip((v-c['depth_min'])/(c['depth_max']-c['depth_min']),0,1)
        a=np.nan_to_num(a);return np.stack((a,np.sqrt(a),1-a),-1)
    panels=[np.repeat(np.clip(y,0,1)[...,None],3,axis=-1),depth(z)]
    labels=['Measurement','Depth (m)']
    if gt is not None:panels.append(depth(gt));labels.append('GT (m)')
    arr=np.concatenate(panels,axis=1);im=Image.fromarray((arr*255).round().astype(np.uint8))
    # Compact PNG preview; full arrays are in TIFF/NPZ.
    if not full_resolution:
        im.thumbnail((1800,900))
    lo,hi=float(c['depth_min']),float(c['depth_max'])
    ticks=np.arange(np.ceil(lo/.25),np.floor(hi/.25)+1)*.25
    # One identical continuous scale beside each depth panel. Leave sufficient
    # vertical spacing for 0.25 m labels, even for small preview images.
    bar_h=max(im.height, int(np.ceil((hi-lo)/.25))*16+1)
    gap=92
    canvas=Image.new('RGB',(im.width+gap*(len(panels)-1),bar_h+40),'white')
    draw=ImageDraw.Draw(canvas)
    offset=0
    for i,label in enumerate(labels):
        left=i*im.width//len(panels);right=(i+1)*im.width//len(panels)
        piece=im.crop((left,0,right,im.height))
        canvas.paste(piece,(offset,26));draw.text((offset+5,5),label,fill='black')
        offset+=piece.width
        if i:
            x=offset+8;top=26
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
    canvas.save(path)

def save_result(folder,name,y,z,g,c,gt=None,save_arrays=True):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if save_arrays:
        tifffile.imwrite(folder/(name+'_depth_m.tiff'),z.astype(np.float32),photometric='minisblack')
        metadata = dict(g, phase_mask_mode=c.get('phase_mask_mode', 'regular'),
                        phase_mask_seed=c.get('phase_mask_seed'))
        arrays = dict(measurement=y, depth_m=z, geometry_json=np.asarray(json.dumps(metadata)))
        if gt is not None:arrays['gt_m']=gt
        np.savez_compressed(folder/(name+'.npz'),**arrays)
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
