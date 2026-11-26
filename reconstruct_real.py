"""Run fixed-mask reconstruction on native-sized real measurements."""
from datetime import datetime
from time import perf_counter
import json
from pathlib import Path

import torch

from data import configured_extensions, image_files, measurement, resize_measurement, center_region
from model import DepthModel, load_matching_tensors
from outputs import save_result
from sampling import RULE, LEGACY_RULE, LEARNED_RULE
from settings import SETTINGS

CHECKPOINT = 'output/run/checkpoints/epoch_001.pt'


def inference_options(settings):
    return {key: settings[key] for key in (
        'inference_batch_size', 'inference_precision', 'fast_reconstruction_export',
        'profile_reconstruction', 'save_sampling_preview')}


@torch.no_grad()
def reconstruct(net, y, c, device, return_coords=False):
    image = torch.from_numpy(y).to(device)
    precision = c.get('inference_precision', 'float32')
    if precision not in ('float32', 'float16', 'bfloat16'):
        raise ValueError('inference_precision must be float32, float16 or bfloat16')
    amp = precision != 'float32'
    if amp and image.device.type != 'cuda':
        raise ValueError('低精度推理需要 CUDA；当前设备请设 inference_precision=float32')
    if precision == 'bfloat16':
        with torch.cuda.device(image.device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError('当前 CUDA 设备不支持 bfloat16')
    chunk = c['inference_batch_size']
    if type(chunk) is not int or chunk < 1:
        raise ValueError('inference_batch_size must be a positive integer')
    with torch.autocast(device_type=image.device.type, enabled=amp, dtype=getattr(torch, precision)):
        result = net.forward_frame(image, chunk, return_coords=return_coords)
    z = result[0] if return_coords else result
    if not z.isfinite().all():
        raise FloatingPointError('Nonfinite full-frame depth')
    pred = z.float().cpu().numpy()
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
    extensions = configured_extensions(c, 'real_image_extensions')
    for i, image in enumerate(image_files(c['real_dir'], extensions=extensions)):
        start = perf_counter()
        y, valid = measurement(image, c)
        source_hw = y.shape
        if c.get('resize_before_inference_crop', False):
            y, valid = resize_measurement(y, valid, c['expected_hw'])
        region, meta = center_region(source_hw, y.shape, c)
        y = y[region]
        ready = perf_counter()
        z = reconstruct(net, y, c, device)
        inferred = perf_counter()
        save_result(folder, f'{i:04d}_{image.stem}', y, z, geometry(y, c, meta), c)
        if c.get('profile_reconstruction'):
            print(f'{image.name} {y.shape}: read/crop={ready-start:.3f}s '
                  f'reconstruct={inferred-ready:.3f}s export={perf_counter()-inferred:.3f}s', flush=True)


def load_checkpoint(path, device, *, strict=True, report_path=None):
    if type(strict) is not bool:
        raise ValueError('strict_inference_weights must be True or False')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck['rule'] != sampling_rule(ck['config']):
        raise ValueError('Incompatible sampling rule')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(ck['config'].get('seed', 41))
        net = DepthModel(ck['config'], load_pretrained=False)
    report = load_matching_tensors(net, ck['model'], strict=strict)
    if not report['loaded']:
        raise ValueError(f'{path}: 没有匹配当前模型的权重张量')
    report.update(source=str(Path(path).resolve()), strict_inference_weights=strict)
    net.weight_load = report
    if not strict:
        print(f"推理权重：从 {path} 匹配 {len(report['loaded'])}/{len(net.state_dict())} 个张量", flush=True)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    return net.to(device).eval(), dict(ck['config'])


if __name__ == '__main__':
    folder = Path(SETTINGS['output']) / 'real' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    net, config = load_checkpoint(CHECKPOINT, SETTINGS['device'], strict=SETTINGS['strict_inference_weights'],
                                  report_path=folder / 'weight_load.json')
    config.update(real_image_extensions=SETTINGS['real_image_extensions'], real_dir=SETTINGS['real_dir'], **inference_options(SETTINGS))
    run_real(net, config, folder, SETTINGS['device'])
