"""Train depth only. A shuffled training example contains one complete fixed partition."""
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from data import paired_files
from evaluate import evaluate
from metrics import loss_terms, make_amp
from model import DepthModel
from outputs import save_result
from prepare_dataset import digest, signature
from reconstruct_real import geometry, run_real
from sampling import LEGACY_RULE, RULE, restore
from settings import SETTINGS


def optimizer_for(net, c):
    groups = []
    for module, lr in ((net.da.pretrained, c['lr_encoder']), (net.da.depth_head, c['lr_decoder']),
                       (net.coord_embed, c['lr_extra']), (net.refiner, c['lr_extra'])):
        if module is not None:
            params = [p for p in module.parameters() if p.requires_grad]
            if params:
                groups.append(dict(params=params, lr=lr))
    if not groups:
        raise ValueError('All weights frozen: unfreeze DA or enable a trainable optional module')
    return torch.optim.AdamW(groups)


def train(c):
    prep = Path(c['prepared_dir'])
    manifest = json.loads((prep / 'manifest.json').read_text())
    saved = dict(manifest['preprocessing'])
    if manifest['rule'] == LEGACY_RULE:
        saved.update(phase_grid_xy=[4, 2], center_crop_hw=None)
    if manifest['rule'] not in (LEGACY_RULE, RULE) or saved != signature(c):
        raise ValueError('Sampling or preprocessing changed. Run prepare_dataset.py again.')
    rows = manifest['train']
    pairs = paired_files(c['val_y'], c['val_depth'])
    hashes = {row['sha256'] for row in rows}
    if any(digest(image) in hashes for image, _ in pairs):
        raise ValueError('Duplicate measurement across train/test')
    torch.manual_seed(c['seed'])
    device = torch.device(c['device'])
    net = DepthModel(c).to(device)
    opt = optimizer_for(net, c)
    amp, dtype, scaler = make_amp(c, device)
    run = Path(c['output']) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    (run / 'checkpoints').mkdir(parents=True)
    for epoch in range(1, c['epochs'] + 1):
        net.train()
        if c['freeze_da']:
            net.da.eval()
        elif c['freeze_encoder']:
            net.da.pretrained.eval()
        for step, idx in enumerate(torch.randperm(len(rows)).tolist(), 1):
            row = rows[idx]
            hw = tuple(row['hw'])
            with np.load(prep / row['file']) as frame:
                y, gt, valid, coords = (torch.from_numpy(frame[k]).to(device)
                                        for k in ('y', 'depth', 'valid', 'coords_yx'))
            gt = restore(gt, coords, hw)
            valid = restore(valid, coords, hw)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                pred = net(y, coords, hw, c['batch_size'])
            terms = loss_terms(pred[None], gt[None], valid[None], c)
            scaler.scale(terms['loss']).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=not amp)
            scaler.step(opt)
            scaler.update()
            print(f'epoch={epoch} frame={step}/{len(rows)} loss={terms["loss"].item():.6f}', flush=True)
            if step == 1 or step % c['train_save_every'] == 0:
                raw = restore(y, coords, hw).cpu().numpy()
                save_result(run/'train', f'epoch{epoch:03d}_iter{step:06d}', raw,
                            pred.detach().cpu().numpy(), geometry(raw, c), c, gt.cpu().numpy(), save_arrays=False)
        path = run / 'checkpoints' / f'epoch_{epoch:03d}.pt'
        torch.save(dict(model=net.state_dict(), config=c, rule=RULE, epoch=epoch), path)
        result = evaluate(net, c, pairs, run/'test'/f'epoch{epoch:03d}', device, epoch)
        print(f'test MAE={result["mae_m"]} accuracy_3pct={result["accuracy_3pct"]}', flush=True)
        if c['real_each_epoch']:
            run_real(net, c, run/'real'/f'epoch{epoch:03d}', device)
    return run


if __name__ == '__main__':
    train(SETTINGS)
