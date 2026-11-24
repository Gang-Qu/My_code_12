"""Run fixed-mask reconstruction on native-sized real measurements."""
from datetime import datetime
from pathlib import Path

import torch

from data import image_files, measurement
from model import DepthModel
from outputs import save_result
from sampling import RULE, fixed_coords, gather
from settings import SETTINGS

CHECKPOINT = 'output/run/checkpoints/epoch_001.pt'


@torch.no_grad()
def reconstruct(net, y, c, device):
    coords = fixed_coords(y.shape, c['phase_mask_seed'])
    parts = torch.from_numpy(gather(y, coords)).to(device)
    positions = torch.from_numpy(coords).to(device)
    z = net(parts, positions, y.shape, c['inference_batch_size'])
    if not z.isfinite().all():
        raise FloatingPointError('Nonfinite full-frame depth')
    return z.cpu().numpy()


def geometry(y):
    return dict(hw=list(y.shape), rule=RULE, coordinate_origin='top-left, zero-based y,x')


def run_real(net, c, folder, device):
    net.eval()
    for i, image in enumerate(image_files(c['real_dir'])):
        y, _ = measurement(image, c)
        z = reconstruct(net, y, c, device)
        save_result(folder, f'{i:04d}_{image.stem}', y, z, geometry(y), c)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck['rule'] != RULE or ck['config']['phase_mask_mode'] != 'fixed':
        raise ValueError('Incompatible sampling rule')
    net = DepthModel(ck['config'], load_pretrained=False)
    net.load_state_dict(ck['model'], strict=True)
    return net.to(device).eval(), dict(ck['config'])


if __name__ == '__main__':
    net, config = load_checkpoint(CHECKPOINT, SETTINGS['device'])
    config.update(real_dir=SETTINGS['real_dir'], inference_batch_size=SETTINGS['inference_batch_size'])
    folder = Path(SETTINGS['output']) / 'real' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_real(net, config, folder, SETTINGS['device'])
