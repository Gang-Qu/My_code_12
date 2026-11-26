import unittest

import torch

from metrics import loss_terms
from model import DepthModel
from sampling import gather, restore
from settings import SETTINGS


class TrainingSpeedTest(unittest.TestCase):
    def test_padded_fixed_grid_reuses_linear_indices_and_matches_gradient(self):
        torch.set_num_threads(2)
        c = dict(SETTINGS, phase_grid_xy=[5, 3], freeze_da=True)
        net = DepthModel(c, load_pretrained=False)
        hw = (29, 61)
        x = torch.randn(hw, requires_grad=True)
        coords = net.sampling_coords(hw, 'cpu')
        self.assertIsNotNone(net._base_linear)
        # A pointwise stand-in isolates split/restore and padding from DA arithmetic.
        net._predict_all_parts = lambda parts, positions, shape, chunk: 2*parts + 1
        out = net.forward_frame(x, 6)
        expected = restore(torch.from_numpy(gather(x.detach().numpy(), coords.numpy()))*2+1, coords, hw)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(out.sum(), x)[0], torch.full_like(x, 2))

    def test_dense_loss_matches_masked_reference_and_gradient(self):
        torch.manual_seed(8)
        p = torch.randn(1, 7, 11, requires_grad=True)
        g = torch.rand_like(p) + .5
        valid = g > .9
        g[~valid] = float('nan')
        got = loss_terms(p, g, valid, SETTINGS)
        r = p - g.nan_to_num()
        pairs = [torch.diff(r, dim=dim)[mask] for dim, mask in (
            (2, valid[:, :, 1:] & valid[:, :, :-1]), (1, valid[:, 1:, :] & valid[:, :-1, :]))]
        ref = r[valid].abs().mean() + SETTINGS['depth_edge_weight'] * torch.cat(pairs).abs().mean()
        torch.testing.assert_close(got['loss'], ref)
        torch.testing.assert_close(torch.autograd.grad(got['loss'], p)[0], torch.autograd.grad(ref, p)[0])
