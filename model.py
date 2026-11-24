from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).parent/'da_vendor'))
from depth_anything_v2.dpt import DepthAnythingV2
from refiner import FrameRefiner

def configure_attention(net,backend):
    """Select implementation only; state_dict names/shapes remain unchanged."""
    from depth_anything_v2.dinov2_layers.attention import Attention
    if backend not in ('sdpa','original'):
        raise ValueError(f'attention_backend={backend!r}，要求sdpa或original')
    if backend=='sdpa' and not hasattr(torch.nn.functional,'scaled_dot_product_attention'):
        raise RuntimeError('attention_backend=sdpa需要PyTorch>=2.0；可设置original恢复原实现')
    for layer in net.modules():
        if isinstance(layer,Attention):
            layer.use_sdpa=(backend=='sdpa')

from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F

from data import model_input
from sampling import LearnedMask, fixed_coords, restore

SPECS = {'vits': (64, [48, 96, 192, 384]), 'vitb': (128, [96, 192, 384, 768]),
         'vitl': (256, [256, 512, 1024, 1024])}


class InverseDepthCalibration(nn.Module):
    """Shared positive inverse-depth mapping, fitted only on training labels."""
    def __init__(self):
        super().__init__()
        self.raw_gain = nn.Parameter(torch.tensor(-3.))
        self.bias = nn.Parameter(torch.tensor(-1.5))

    def forward(self, relative):
        inverse_m = F.softplus(F.softplus(self.raw_gain) * relative.float() + self.bias)
        return (inverse_m + 1e-6).reciprocal()


class DepthModel(nn.Module):
    def __init__(self, c, load_pretrained=True):
        super().__init__()
        features, channels = SPECS[c['encoder']]
        model_type = c.get('da_model_type', 'metric')
        if model_type not in ('metric', 'relative'):
            raise ValueError("da_model_type must be 'metric' or 'relative'")
        maximum = c['metric_max_depth'] if model_type == 'metric' else None
        if not 0 < c['depth_min'] < c['depth_max'] or (maximum is not None and c['depth_max'] > maximum):
            raise ValueError('Check metric depth range and depth unit')
        self.da = DepthAnythingV2(encoder=c['encoder'], features=features,
                                  out_channels=channels, max_depth=maximum)
        self.calibration = InverseDepthCalibration() if model_type == 'relative' else None
        configure_attention(self.da, c['attention_backend'])
        if load_pretrained:
            weights = c['relative_da_weights'] if model_type == 'relative' else c['da_weights']
            state = torch.load(weights, map_location='cpu', weights_only=True)
            self.da.load_state_dict(state.get('state_dict', state), strict=True)
        self.checkpoint_activations = c.get('checkpoint_da_activations', False)
        self.grid_xy = tuple(c.get('phase_grid_xy', (4, 2)))
        self.register_buffer('_base_grid', None, persistent=False)
        self._grid_key = None
        self.mask_seed = c['phase_mask_seed']
        if type(self.mask_seed) is not int or self.mask_seed < 0:
            raise ValueError('phase_mask_seed must be a nonnegative integer')
        self.mask_mode = c['phase_mask_mode']
        if self.mask_mode not in ('fixed', 'learned'):
            raise ValueError("phase_mask_mode must be 'fixed' or 'learned'")
        self.mask = (LearnedMask(self.grid_xy, c['mask_tile_hw'], self.mask_seed)
                     if self.mask_mode == 'learned' else None)
        self.coord_embed = None
        if c['use_coordinate_prior']:
            self.coord_embed = nn.Conv2d(2, self.da.pretrained.embed_dim, 14, stride=14, bias=False)
            nn.init.zeros_(self.coord_embed.weight)
        self.refiner = FrameRefiner('measurement') if c['use_refiner'] else None
        self.da.requires_grad_(not c['freeze_da'])
        self.da.pretrained.requires_grad_(not (c['freeze_encoder'] or c['freeze_da']))

    def predict_parts(self, parts, coords, hw):
        h, w = parts.shape[-2:]
        pad = (0, -w % 14, 0, -h % 14)
        x = model_input(F.pad(parts, pad))
        tokens = None
        if self.coord_embed is not None:
            # Channels are normalized row, column in the whole measurement.
            scale = parts.new_tensor([max(hw[0]-1, 1), max(hw[1]-1, 1)])
            xy = (2 * coords.float() / scale - 1).masked_fill(coords < 0, 0)
            xy = F.pad(xy.permute(0, 3, 1, 2), pad)
            tokens = self.coord_embed(xy).flatten(2).transpose(1, 2)
        depth = self.da(x, coord_tokens=tokens)[:, :h, :w].float()
        return self.calibration(depth) if self.calibration is not None else depth

    def _predict_all_parts(self, parts, coords, hw, chunk):
        preds = []
        for y, pos in zip(parts.split(chunk), coords.split(chunk)):
            if self.checkpoint_activations and self.training and torch.is_grad_enabled():
                depth = checkpoint(self.predict_parts, y, pos, hw, use_reentrant=False)
            else:
                depth = self.predict_parts(y, pos, hw)
            preds.append(depth)
        return torch.cat(preds)

    def _base_coords(self, hw, device):
        key = (tuple(hw), self.mask_seed, self.grid_xy)
        device = torch.device(device)
        if self._grid_key != key or self._base_grid.device != device:
            self._base_grid = torch.from_numpy(fixed_coords(hw, self.mask_seed, self.grid_xy, persist=self.mask_mode == 'fixed')).to(device)
            self._grid_key = key
        return self._base_grid

    def sampling_coords(self, hw, device):
        base = self._base_coords(hw, device)
        return self.mask.route(base) if self.mask is not None else base

    def forward_frame(self, frame, chunk=1, return_coords=False):
        hw = frame.shape
        base = self._base_coords(hw, frame.device)
        coords = self.mask.route(base) if self.mask is not None else base
        # Training differentiates both the split and its inverse. Deployment needs
        # only the final integer coordinates: a single gather and inverse scatter.
        train_mask = self.mask is not None and self.training and torch.is_grad_enabled()
        scatter_coords = base if train_mask else coords
        rows, cols = scatter_coords.unbind(-1)
        parts = frame[rows.clamp_min(0).long(), cols.clamp_min(0).long()].masked_fill(rows < 0, 0)
        if train_mask:
            parts = self.mask.route(parts)
        preds = self._predict_all_parts(parts, coords, hw, chunk)
        if train_mask:
            preds = self.mask.route(preds, inverse=True)
        final = self._refine(restore(preds, scatter_coords, hw), frame)
        return (final, coords) if return_coords else final

    def forward(self, parts, coords, hw, chunk=1):
        """Compatibility with prepared, fixed partitions."""
        if self.mask is not None:
            raise ValueError('Learnable masks require forward_frame on the raw image')
        pred = self._predict_all_parts(parts, coords, hw, chunk)
        coarse = restore(pred, coords, hw)
        if self.refiner is None:
            return coarse
        return self._refine(coarse, restore(parts, coords, hw))

    def _refine(self, coarse, y):
        if self.refiner is None:
            return coarse
        residual = self.refiner(torch.stack((coarse, y))[None])
        return coarse + residual[0, 0].float()
