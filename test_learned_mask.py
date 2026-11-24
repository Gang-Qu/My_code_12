import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import tifffile

from sampling import LearnedMask, fixed_coords, gather, restore
from settings import SETTINGS
from model import DepthModel
from prepare_dataset import prepare
from train import train
from reconstruct_real import load_checkpoint, reconstruct, run_real


class LearnableRoutingTest(unittest.TestCase):
    def test_exact_partition_and_surrogate_gradients(self):
        torch.set_num_threads(2)
        for grid in ([1, 1], [3, 1], [3, 2], [4, 2]):
            with self.subTest(grid=grid):
                module = LearnedMask(grid, [3, 5], seed=91)
                hw = (35, 67)
                base = torch.from_numpy(fixed_coords(hw, 91, grid))
                coords = module.route(base)
                parts = torch.arange(base[..., 0].numel()).reshape(base.shape[:-1]).float()
                routed = module.route(parts)
                self.assertTrue(torch.equal(module.route(routed, inverse=True), parts))
                self.assertTrue(torch.equal(restore(routed, coords, hw), restore(parts, base, hw)))
                if grid == [1, 1]:
                    continue
                before = coords.clone()
                # The hard forward is discrete; the task gradient reaches its continuous parameters.
                loss = (routed * torch.linspace(-1, 1, routed.numel()).reshape_as(routed)).mean()
                loss.backward()
                self.assertTrue(any(p.grad.abs().sum().item() > 0 for p in module.parameters()))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters()))
                module.eval()
                self.assertTrue(torch.equal(module.route(base), before))
                with torch.no_grad():
                    next(module.parameters()).neg_()
                self.assertFalse(torch.equal(module.route(base), before))

    def test_fixed_online_reproduces_prepared_forward(self):
        torch.set_num_threads(2)
        hw = (45, 63)
        c = dict(SETTINGS, phase_grid_xy=[3, 2], use_coordinate_prior=True, use_refiner=True)
        net = DepthModel(c, load_pretrained=False).eval()
        array = np.random.default_rng(5).random(hw, dtype=np.float32)
        coords = fixed_coords(hw, c['phase_mask_seed'], c['phase_grid_xy'])
        with torch.no_grad():
            cached = net(torch.from_numpy(gather(array, coords)), torch.from_numpy(coords), hw)
            online = net.forward_frame(torch.from_numpy(array))
        torch.testing.assert_close(online, cached, rtol=0, atol=0)

    def test_training_without_prepared_files_and_checkpoint_replay(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = dict(SETTINGS, training_input='raw', phase_mask_mode='learned', mask_tile_hw=[3, 5],
                     phase_grid_xy=[3, 2], device='cpu', expected_hw=[57, 88], center_crop_hw=[51, 73],
                     epochs=1, freeze_da=True, lr_mask=.02, output=tmp+'/output',
                     prepared_dir=tmp+'/does_not_exist', da_weights=tmp+'/da.pth')
            for split, k in [('train', 1), ('val', 3)]:
                for suffix in ['y', 'depth']:
                    path = root / (split+suffix)
                    path.mkdir()
                    c[split+'_'+suffix] = str(path)
                y = (np.arange(65*85).reshape(65,85)*k % 255).astype('uint8')
                Image.fromarray(y).save(Path(c[split+'_y'])/'image.png')
                tifffile.imwrite(Path(c[split+'_depth'])/'image.tif', (1+y/127.).astype('float32'))
            baseline = DepthModel(c, load_pretrained=False)
            torch.save(baseline.da.state_dict(), c['da_weights'])
            run = train(c)
            self.assertFalse(Path(c['prepared_dir']).exists())
            net, config = load_checkpoint(run/'checkpoints/epoch_001.pt', 'cpu')
            hw = (51, 73)
            self.assertFalse(torch.equal(baseline.sampling_coords(hw, 'cpu'),
                                         net.sampling_coords(hw, 'cpu')))
            saved_coords = np.load(run/'test/epoch001/sampling_6_parts/coords_yx.npy')
            np.testing.assert_array_equal(saved_coords, net.sampling_coords(hw, 'cpu').numpy())
            self.assertTrue((run/'test/epoch001/sampling_6_parts/index.png').is_file())
            with np.load(run/'test/epoch001/0000_image.npz') as f:
                image = f['measurement']
                np.testing.assert_array_equal(reconstruct(net, image, config, 'cpu'), f['depth_m'])
            net.train()
            # Surrogate training and hard deployment produce identical sampled predictions.
            pred = net.forward_frame(torch.from_numpy(image), chunk=1).detach().numpy()
            net.eval()
            np.testing.assert_allclose(pred, reconstruct(net, image, config, 'cpu'), rtol=1e-5, atol=1e-5)
            config['real_dir'] = c['val_y']
            run_real(net, config, root/'real', 'cpu')
            np.testing.assert_array_equal(tifffile.imread(root/'real/0000_image_depth_m.tiff'),
                                          tifffile.imread(run/'test/epoch001/0000_image_depth_m.tiff'))
            # Optional disk cache contains the lossless base permutation. Training from
            # it must make the same mask update as reading the raw files directly.
            cached_config = dict(c, training_input='prepared', prepared_dir=tmp+'/cache')
            prepare(cached_config)
            cached_run = train(cached_config)
            cached_net, _ = load_checkpoint(cached_run/'checkpoints/epoch_001.pt', 'cpu')
            for learned, cached in zip(net.mask.parameters(), cached_net.mask.parameters()):
                torch.testing.assert_close(learned, cached, rtol=0, atol=0)
