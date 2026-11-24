"""Train depth only. A shuffled training example contains one complete fixed partition."""
from datetime import datetime
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from data import paired_files, read_pair
from evaluate import evaluate
from depth_audit import audit_pair, write_audit
from metrics import loss_terms, make_amp
from model import DepthModel
from outputs import save_result
from prepare_dataset import digest, signature
from reconstruct_real import geometry, run_real, sampling_rule
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
    if net.calibration is not None:
        groups.append(dict(params=net.calibration.parameters(), lr=c['lr_calibration'], weight_decay=0.))
    if net.mask is not None:
        params = list(net.mask.parameters())
        if params:
            groups.append(dict(params=params, lr=c['lr_mask'], weight_decay=0.))
    if not groups:
        raise ValueError('All weights frozen: unfreeze DA or enable a trainable optional module')
    return torch.optim.AdamW(groups)


def training_records(c):
    source = c.get('training_input', 'prepared')
    if source == 'raw':
        return [dict(image=image, depth=depth, sha256=digest(image))
                for image, depth in paired_files(c['train_y'], c['train_depth'])]
    if source != 'prepared':
        raise ValueError("training_input must be 'raw' or 'prepared'")
    prep = Path(c['prepared_dir'])
    manifest = json.loads((prep / 'manifest.json').read_text())
    saved = dict(manifest['preprocessing'])
    if manifest['rule'] == LEGACY_RULE:
        saved.update(phase_grid_xy=[4, 2], center_crop_hw=None)
    if manifest['rule'] not in (LEGACY_RULE, RULE) or saved != signature(c):
        raise ValueError('Sampling or preprocessing changed. Run prepare_dataset.py again.')
    return manifest['train']


def audit_training_rows(rows, c, raw):
    result = []
    for row in rows:
        if raw:
            audit = audit_pair(row['image'], row['depth'], c, training=True)
        elif 'depth_audit' in row:
            audit = row['depth_audit']
        else:
            # Old caches have already replaced excluded labels by zero.
            with np.load(Path(c['prepared_dir']) / row['file']) as frame:
                audit = dict(source=row['file'], supervised_pixels=int(frame['valid'].sum()),
                             range_status='pre-filter labels unavailable in legacy cache')
        result.append(audit)
    return result


def show_progress(epoch, epochs, step, steps, loss, metrics=None):
    tty = sys.stdout.isatty()
    if not tty and metrics is None:
        return
    width = max(1, shutil.get_terminal_size((100, 24)).columns - 1)
    left = f'loss={loss:.6f}'
    if metrics is not None:
        mae, acc = metrics['mae_m'], metrics['accuracy_3pct']
        detail = (f' test_MAE={mae:.4f} acc3%={acc:.1%}' if mae is not None else ' test=n/a')
    else:
        detail = ''
    right = f'epoch={epoch}/{epochs} frame={step}/{steps} {100*step/steps:.0f}%'
    if len(left + detail + right) + 2 <= width or not tty:
        left += detail
    gap = max(1, width-len(left)-len(right)) if tty else 2
    line = left + ' '*gap + right
    last = metrics is not None and epoch == epochs
    print(('\r' if tty else '') + line, end='\n' if last or not tty else '', flush=True)


def train(c):
    rows = training_records(c)
    raw_input = c.get('training_input', 'prepared') == 'raw'
    sample_online = raw_input or c['phase_mask_mode'] == 'learned'
    pairs = paired_files(c['val_y'], c['val_depth'])
    hashes = {row['sha256'] for row in rows}
    if any(digest(image) in hashes for image, _ in pairs):
        raise ValueError('Duplicate measurement across train/test')
    run = Path(c['output']) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    (run / 'checkpoints').mkdir(parents=True)
    report = write_audit(run, c, dict(train=audit_training_rows(rows, c, raw_input),
                                      val=[audit_pair(y, z, c, training=False) for y, z in pairs]))
    empty = report['splits']['train']['zero_supervision_frames']
    if empty:
        raise ValueError(f'No supervised pixels in {empty[0]}; see {run / "depth_range_audit.json"}')
    torch.manual_seed(c['seed'])
    device = torch.device(c['device'])
    net = DepthModel(c).to(device)
    opt = optimizer_for(net, c)
    amp, dtype, scaler = make_amp(c, device)
    for epoch in range(1, c['epochs'] + 1):
        net.train()
        if c['freeze_da']:
            net.da.eval()
        elif c['freeze_encoder']:
            net.da.pretrained.eval()
        for step, idx in enumerate(torch.randperm(len(rows)).tolist(), 1):
            row = rows[idx]
            if raw_input:
                y, gt, valid, meta = read_pair(row['image'], row['depth'], c, training=True)
                y, gt, valid = (torch.from_numpy(a).to(device) for a in (y, gt, valid))
            else:
                hw = tuple(row['hw'])
                with np.load(Path(c['prepared_dir']) / row['file']) as frame:
                    y, gt, valid, coords = (torch.from_numpy(frame[k]).to(device)
                                            for k in ('y', 'depth', 'valid', 'coords_yx'))
                if sample_online:
                    y = restore(y, coords, hw)
                gt = restore(gt, coords, hw)
                valid = restore(valid, coords, hw)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                if sample_online:
                    pred = net.forward_frame(y, c['batch_size'])
                else:
                    pred = net(y, coords, hw, c['batch_size'])
            terms = loss_terms(pred[None], gt[None], valid[None], c)
            scaler.scale(terms['loss']).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=not amp)
            scaler.step(opt)
            scaler.update()
            if sys.stdout.isatty() or step == len(rows):
                last_loss = terms['loss'].item()
                show_progress(epoch, c['epochs'], step, len(rows), last_loss)
            if step == 1 or step % c['train_save_every'] == 0:
                raw = (y if sample_online else restore(y, coords, hw)).cpu().numpy()
                save_result(run/'train', f'epoch{epoch:03d}_iter{step:06d}', raw,
                            pred.detach().cpu().numpy(), geometry(raw, c, meta if raw_input else None), c, gt.cpu().numpy(), save_arrays=False)
        path = run / 'checkpoints' / f'epoch_{epoch:03d}.pt'
        torch.save(dict(model=net.state_dict(), config=c, rule=sampling_rule(c), epoch=epoch), path)
        result = evaluate(net, c, pairs, run/'test'/f'epoch{epoch:03d}', device, epoch)
        show_progress(epoch, c['epochs'], len(rows), len(rows), last_loss, result)
        if c['real_each_epoch']:
            run_real(net, c, run/'real'/f'epoch{epoch:03d}', device)
    return run


if __name__ == '__main__':
    train(SETTINGS)
