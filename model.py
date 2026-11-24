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
from sampling import restore

SPECS = {'vits': (64, [48, 96, 192, 384]), 'vitb': (128, [96, 192, 384, 768]),
         'vitl': (256, [256, 512, 1024, 1024])}


class DepthModel(nn.Module):
    def __init__(self, c, load_pretrained=True):
        super().__init__()
        features, channels = SPECS[c['encoder']]
        if not 0 < c['depth_min'] < c['depth_max'] <= c['metric_max_depth']:
            raise ValueError('Check metric depth range and depth unit')
        self.da = DepthAnythingV2(encoder=c['encoder'], features=features,
                                  out_channels=channels, max_depth=c['metric_max_depth'])
        configure_attention(self.da, c['attention_backend'])
        if load_pretrained:
            state = torch.load(c['da_weights'], map_location='cpu', weights_only=True)
            self.da.load_state_dict(state.get('state_dict', state), strict=True)
        self.checkpoint_activations = c.get('checkpoint_da_activations', False)
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
        return self.da(x, coord_tokens=tokens)[:, :h, :w].float()

    def forward(self, parts, coords, hw, chunk=1):
        preds = []
        for y, pos in zip(parts.split(chunk), coords.split(chunk)):
            if self.checkpoint_activations and self.training and torch.is_grad_enabled():
                depth = checkpoint(self.predict_parts, y, pos, hw, use_reentrant=False)
            else:
                depth = self.predict_parts(y, pos, hw)
            preds.append(depth)
        coarse = restore(torch.cat(preds), coords, hw)
        if self.refiner is None:
            return coarse
        y = restore(parts, coords, hw)
        residual = self.refiner(torch.stack((coarse, y))[None])
        return coarse + residual[0, 0].float()
