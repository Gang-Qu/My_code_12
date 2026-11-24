"""Run fixed-mask reconstruction on native-sized real measurements."""
from datetime import datetime
from pathlib import Path

import torch

from data import image_files, measurement, resize_measurement, center_region
from model import DepthModel
from outputs import save_result
from sampling import RULE, LEGACY_RULE, LEARNED_RULE
from settings import SETTINGS

CHECKPOINT = 'output/run/checkpoints/epoch_001.pt'


@torch.no_grad()
def reconstruct(net, y, c, device, return_coords=False):
    image = torch.from_numpy(y).to(device)
    result = net.forward_frame(image, c['inference_batch_size'], return_coords=return_coords)
    z = result[0] if return_coords else result
    if not z.isfinite().all():
        raise FloatingPointError('Nonfinite full-frame depth')
    pred = z.cpu().numpy()
    return (pred, result[1].cpu().numpy()) if return_coords else pred


def sampling_rule(c):
    if c['phase_mask_mode'] == 'learned':
        return LEARNED_RULE
    return RULE if 'phase_grid_xy' in c else LEGACY_RULE


def geometry(y, c, meta=None):
    return {'hw': list(y.shape), **(meta or {}),
            'rule': sampling_rule(c),
            'phase_grid_xy': c.get('phase_grid_xy', [4, 2]),
            'coordinate_origin': 'crop top-left, zero-based y,x'}


def run_real(net, c, folder, device):
    net.eval()
    for i, image in enumerate(image_files(c['real_dir'])):
        y, valid = measurement(image, c)
        source_hw = y.shape
        if c.get('resize_before_inference_crop', False):
            y, valid = resize_measurement(y, valid, c['expected_hw'])
        region, meta = center_region(source_hw, y.shape, c)
        y = y[region]
        z = reconstruct(net, y, c, device)
        save_result(folder, f'{i:04d}_{image.stem}', y, z, geometry(y, c, meta), c)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck['rule'] != sampling_rule(ck['config']):
        raise ValueError('Incompatible sampling rule')
    net = DepthModel(ck['config'], load_pretrained=False)
    net.load_state_dict(ck['model'], strict=True)
    return net.to(device).eval(), dict(ck['config'])


if __name__ == '__main__':
    net, config = load_checkpoint(CHECKPOINT, SETTINGS['device'])
    config.update(real_dir=SETTINGS['real_dir'], inference_batch_size=SETTINGS['inference_batch_size'])
    folder = Path(SETTINGS['output']) / 'real' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_real(net, config, folder, SETTINGS['device'])
