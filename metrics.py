from decimal import Decimal
import numpy as np
import torch

DEPTH_BIN_WIDTH_M = Decimal('0.1')
RELATIVE_ERROR_LIMIT = .03
DELTA_BASE = 1.25


def depth_bin_edges(minimum, maximum):
    lo, hi = Decimal(str(minimum)), Decimal(str(maximum))
    bins = int(((hi - lo) / DEPTH_BIN_WIDTH_M).to_integral_value(rounding='ROUND_CEILING'))
    return np.array([float(lo + i * DEPTH_BIN_WIDTH_M) for i in range(bins)] + [float(hi)])


def _depth_statistics(pred, gt, valid, edges=None):
    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)
    if not np.isfinite(p).all():
        raise FloatingPointError('Nonfinite prediction at valid depth pixels')
    if not (np.isfinite(g) & (g > 0)).all():
        raise ValueError('Depth metrics require finite positive GT at valid=True pixels')

    e = np.abs(p - g)
    ratio = np.full_like(g, np.inf)
    positive = p > 0
    ratio[positive] = np.maximum(p[positive] / g[positive], g[positive] / p[positive])
    # Each column is reduced before the next pixel-sized temporary is created.
    def columns():
        yield e
        yield e / g
        yield e * e
        for k in (1, 2, 3):
            yield ratio < DELTA_BASE**k
        yield e / g <= RELATIVE_ERROR_LIMIT

    if edges is None:
        return np.array([len(g), *(v.sum() for v in columns())], dtype=np.float64), None

    # Compare edges at the precision in which the input depth was stored, so
    # e.g. float32(0.7) is assigned to the bin starting at 0.7.
    boundaries = edges.astype(gt.dtype) if np.issubdtype(gt.dtype, np.floating) else edges
    n_bins = len(edges) - 1
    ids = np.searchsorted(boundaries[1:-1], g, side='right')
    ids[(g < boundaries[0]) | (g > boundaries[-1])] = n_bins
    grouped = np.empty((n_bins + 1, 8), dtype=np.float64)
    grouped[:, 0] = np.bincount(ids, minlength=n_bins + 1)
    for j, values in enumerate(columns(), 1):
        grouped[:, j] = np.bincount(ids, weights=values, minlength=n_bins + 1)
    return grouped.sum(axis=0), grouped[:n_bins]


def scores(s):
    n = s[0]
    names = ['mae_m', 'abs_rel', 'rmse_m', 'delta1', 'delta2', 'delta3', 'accuracy_3pct']
    values = dict.fromkeys(names, None) if not n else dict(zip(
        names, [s[1]/n, s[2]/n, (s[3]/n)**.5, *list(s[4:]/n)]))
    return dict(valid_pixels=int(n), within_3pct_pixels=int(s[7]), **values)


def loss_terms(pred, gt, valid, c):
    """L1 metres plus supervised finite differences on the supplied pixel grid.

    Mask before subtraction so NaN/Inf in invalid GT cannot enter autograd.
    Edge loss compares gradients; it is not total-variation smoothing.
    """
    if pred.shape != gt.shape or gt.shape != valid.shape or pred.ndim != 3:
        raise ValueError(f'loss要求相同[B,H,W]：pred={pred.shape}, gt={gt.shape}, valid={valid.shape}')
    if valid.dtype != torch.bool:
        raise ValueError(f'loss valid.dtype={valid.dtype}，要求torch.bool')
    if not torch.isfinite(pred).all():
        raise FloatingPointError('Nonfinite depth prediction')
    count = valid.sum()
    if count.item() == 0:
        raise ValueError('当前batch没有有效深度像素；检查预处理有效掩码与深度范围')
    if not (torch.isfinite(gt[valid]) & (gt[valid] > 0)).all():
        raise ValueError('loss: valid=True的真值必须是有限正米制深度')
    residual = torch.where(valid, pred, 0.) - torch.where(valid, gt, 0.)
    l1 = residual.abs().sum() / count
    horizontal = valid[..., :, 1:] & valid[..., :, :-1]
    vertical = valid[..., 1:, :] & valid[..., :-1, :]
    edge_sum = ((residual[..., :, 1:] - residual[..., :, :-1])[horizontal].abs().sum()
                + (residual[..., 1:, :] - residual[..., :-1, :])[vertical].abs().sum())
    pairs = horizontal.sum() + vertical.sum()
    edge = edge_sum / pairs.clamp_min(1)  # zero valid pairs -> zero loss, no epsilon
    result = dict(loss=c['depth_l1_weight'] * l1 + c['depth_edge_weight'] * edge,
                  depth_l1=l1, depth_edge=edge)
    return result


def make_amp(c, device):
    """Configure training AMP. Reconstruction precision is set separately."""
    enabled = bool(c['amp'] and device.type == 'cuda')
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[c['amp_dtype']]
    if enabled and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError('amp_dtype=bfloat16不受当前GPU支持，请使用float16或设置amp=False')
    use_scaler = enabled and dtype == torch.float16
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    return enabled, dtype, scaler


class EpochStats:
    """Pool pixels/edges, never average per-batch percentages or per-image MAEs."""
    def __init__(self, c):
        self.c = c
        self.total = np.zeros(8)
        self.edges = depth_bin_edges(c['depth_min'], c['depth_max'])
        self.bins = np.zeros((len(self.edges) - 1, 8))
        self.edge_sum = 0.
        self.edge_pairs = 0

    def update(self, pred, gt, valid):
        if pred.shape != gt.shape or gt.shape != valid.shape or pred.ndim != 3:
            raise ValueError('EpochStats requires matching [B,H,W] arrays')
        ss, bins = _depth_statistics(pred, gt, valid, self.edges)
        self.total += ss
        self.bins += bins
        # The caller supplies the training grid (full frame or legacy subimage).
        r = np.where(valid, pred, 0.).astype(np.float64) - np.where(valid, gt, 0.).astype(np.float64)
        mx = valid[..., :, 1:] & valid[..., :, :-1]
        my = valid[..., 1:, :] & valid[..., :-1, :]
        self.edge_sum += float(np.abs(np.diff(r, axis=-1)[mx]).sum()
                               + np.abs(np.diff(r, axis=-2)[my]).sum())
        self.edge_pairs += int(mx.sum() + my.sum())
        return scores(ss)

    def result(self):
        result = scores(self.total)
        l1 = result['mae_m']
        edge = self.edge_sum / self.edge_pairs if self.edge_pairs else 0.
        result.update(depth_l1=l1, depth_edge=edge if l1 is not None else None,
                      edge_pairs=self.edge_pairs,
                      loss=None if l1 is None else self.c['depth_l1_weight']*l1 + self.c['depth_edge_weight']*edge)
        result['depth_bins'] = [dict(lower_m=lo, upper_m=hi,
                                    upper_inclusive=(i == len(self.bins)-1),
                                    pixel_fraction=float(s[0]/self.total[0]) if self.total[0] else None,
                                    **scores(s))
                                for i, (lo, hi, s) in enumerate(zip(self.edges[:-1], self.edges[1:], self.bins))]
        return result
