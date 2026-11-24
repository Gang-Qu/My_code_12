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
        return [dict(image=image, depth=depth)
                for image, depth in paired_files(c['train_y'], c['train_depth'])], []
    if source != 'prepared':
        raise ValueError("training_input must be 'raw' or 'prepared'")
    prep = Path(c['prepared_dir'])
    manifest = json.loads((prep / 'manifest.json').read_text())
    saved = dict(manifest['preprocessing'])
    if manifest['rule'] == LEGACY_RULE:
        saved.update(phase_grid_xy=[4, 2], center_crop_hw=None)
    if manifest['rule'] not in (LEGACY_RULE, RULE) or saved != signature(c):
        raise ValueError('Sampling or preprocessing changed. Run prepare_dataset.py again.')
    return manifest['train'], manifest.get('skipped_train', [])


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


def check_split_overlap(rows, pairs, c, raw, prefiltered):
    # Identical file contents must have equal lengths. Decode labels only when
    # a byte-for-byte match could involve a frame excluded from supervision.
    sizes = {image.stat().st_size for image, _ in pairs}
    matches = {}
    candidate_sizes = set() if raw else sizes
    for row in rows:
        if raw:
            size = row['image'].stat().st_size
            if size not in sizes:
                continue
            candidate_sizes.add(size)
            key = digest(row['image'])
        else:
            key = row['sha256']
        matches.setdefault(key, []).append(row)
    if not matches:
        return
    for image, depth in pairs:
        if image.stat().st_size not in candidate_sizes:
            continue
        group = matches.get(digest(image))
        if not group:
            continue
        if prefiltered:
            raise ValueError('Duplicate measurement across train/test')
        if not read_pair(image, depth, c)[2].any():
            continue
        for row in group:
            if raw:
                valid = read_pair(row['image'], row['depth'], c, training=True)[2]
            else:
                with np.load(Path(c['prepared_dir']) / row['file']) as frame:
                    valid = frame['valid']
            if valid.any():
                raise ValueError('Duplicate measurement across train/test')


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
    export_every = c.get('test_export_every', 1)
    if type(export_every) is not int or export_every < 1:
        raise ValueError('test_export_every must be a positive integer')
    rows, skipped_at_prepare = training_records(c)
    raw_input = c.get('training_input', 'prepared') == 'raw'
    sample_online = raw_input or c['phase_mask_mode'] == 'learned'
    pairs = paired_files(c['val_y'], c['val_depth'])
    run = Path(c['output']) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    (run / 'checkpoints').mkdir(parents=True)
    audit_enabled = c.get('dataset_audit', True)
    skipped_val = []
    if audit_enabled:
        train_audit = audit_training_rows(rows, c, raw_input)
        val_audit = [audit_pair(y, z, c, training=False) for y, z in pairs]
        report = write_audit(run, c, dict(train=train_audit + skipped_at_prepare, val=val_audit))
        rows = [row for row, info in zip(rows, train_audit) if info['supervised_pixels'] > 0]
        pairs = [pair for pair, info in zip(pairs, val_audit) if info['supervised_pixels'] > 0]
        skipped_val = report['splits']['val']['zero_supervision_frames']
    no_supervision = '整个训练集在裁剪和有效深度筛选后没有监督点。请核对单位、深度范围及裁剪，或运行 python depth_audit.py'
    if not rows:
        raise ValueError(no_supervision)
    check_split_overlap(rows, pairs, c, raw_input, audit_enabled)
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
        empty_indices = set()
        step = 0
        for idx in torch.randperm(len(rows)).tolist():
            row = rows[idx]
            if raw_input:
                y, gt, valid, meta = read_pair(row['image'], row['depth'], c, training=True)
                if not audit_enabled and epoch == 1 and not valid.any():
                    empty_indices.add(idx)
                    continue
                y, gt, valid = (torch.from_numpy(a).to(device) for a in (y, gt, valid))
            else:
                hw = tuple(row['hw'])
                with np.load(Path(c['prepared_dir']) / row['file']) as frame:
                    mask = frame['valid']
                    if not audit_enabled and epoch == 1 and not mask.any():
                        empty_indices.add(idx)
                        continue
                    valid = torch.from_numpy(mask).to(device)
                    y, gt, coords = (torch.from_numpy(frame[k]).to(device)
                                     for k in ('y', 'depth', 'coords_yx'))
                if sample_online:
                    y = restore(y, coords, hw)
                gt = restore(gt, coords, hw)
                valid = restore(valid, coords, hw)
            step += 1
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
            last_loss = terms['loss'].detach()
            if sys.stdout.isatty():
                show_progress(epoch, c['epochs'], step, len(rows)-len(empty_indices), last_loss.item())
            if step == 1 or step % c['train_save_every'] == 0:
                raw = (y if sample_online else restore(y, coords, hw)).cpu().numpy()
                save_result(run/'train', f'epoch{epoch:03d}_iter{step:06d}', raw,
                            pred.detach().cpu().numpy(), geometry(raw, c, meta if raw_input else None), c, gt.cpu().numpy(), save_arrays=False)
        if not audit_enabled and epoch == 1 and (empty_indices or skipped_at_prepare):
            skipped = [r['source'] for r in skipped_at_prepare]
            for i in sorted(empty_indices):
                row = rows[i]
                skipped.append(str(row['depth'] if raw_input else row.get('depth_source', row['file'])))
            (run / 'skipped_train.json').write_text(json.dumps(skipped, indent=2), encoding='utf8')
        if empty_indices:
            rows = [row for i, row in enumerate(rows) if i not in empty_indices]
        if not rows:
            raise ValueError(no_supervision)
        path = run / 'checkpoints' / f'epoch_{epoch:03d}.pt'
        torch.save(dict(model=net.state_dict(), config=c, rule=sampling_rule(c), epoch=epoch), path)
        result = evaluate(net, c, pairs, run/'test'/f'epoch{epoch:03d}', device, epoch,
                          skipped_frames=skipped_val, export_images=epoch == c['epochs'] or epoch % export_every == 0)
        if not audit_enabled and epoch == 1:
            skipped_val = result['skipped_frames']
            excluded = set(skipped_val)
            pairs = [pair for pair in pairs if str(pair[1]) not in excluded]
        show_progress(epoch, c['epochs'], len(rows), len(rows), last_loss.item(), result)
        if c['real_each_epoch']:
            run_real(net, c, run/'real'/f'epoch{epoch:03d}', device)
    return run


if __name__ == '__main__':
    train(SETTINGS)
