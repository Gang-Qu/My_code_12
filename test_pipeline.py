import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import tifffile
import torch

from data import read_pair
from model import DepthModel
from prepare_dataset import prepare
from reconstruct_real import load_checkpoint, reconstruct, run_real
from sampling import fixed_coords, gather
from settings import SETTINGS
from train import train


class PipelineTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_optional_modules_identity_and_gradients(self):
        c = dict(SETTINGS, freeze_da=True)
        hw = (55, 81)
        pos = torch.from_numpy(fixed_coords(hw, 41))
        parts = torch.rand(pos.shape[:-1])
        baseline = DepthModel(c, load_pretrained=False).eval()
        with torch.no_grad():
            reference = baseline(parts, pos, hw, 1)
        for prior, refiner in ((True, False), (False, True), (True, True)):
            candidate = DepthModel(dict(c, use_coordinate_prior=prior, use_refiner=refiner),
                                   load_pretrained=False).eval()
            candidate.da.load_state_dict(baseline.da.state_dict(), strict=True)
            pred = candidate(parts, pos, hw, 3)
            torch.testing.assert_close(pred, reference, atol=2e-5, rtol=1e-5)
            pred.mean().backward()
            if prior:
                self.assertGreater(candidate.coord_embed.weight.grad.abs().max().item(), 0)
            if refiner:
                self.assertGreater(candidate.refiner[-1].weight.grad.abs().max().item(), 0)

    def test_train_test_and_real(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = dict(SETTINGS, prepared_dir=tmp+'/prep', output=tmp+'/out', device='cpu',
                     expected_hw=[56, 84], center_crop_hw=[51, 73], phase_grid_xy=[3, 2], checkpoint_da_activations=True, da_weights=tmp+'/da.pth', epochs=1, real_dir=tmp+'/real',
                     real_each_epoch=True, freeze_da=True, use_coordinate_prior=True, use_refiner=True)
            for split, hw in [('train', (65, 85)), ('val', (61, 97))]:
                for kind in ('y', 'depth'):
                    folder = root / (split+kind)
                    folder.mkdir()
                    c[split+'_'+kind] = str(folder)
                pixels = np.arange(np.prod(hw), dtype=np.uint8).reshape(hw)
                Image.fromarray(pixels).save(Path(c[split+'_y'])/'scene.png')
                tifffile.imwrite(Path(c[split+'_depth'])/'scene.tif', np.ones(hw, np.float32))
            Path(c['real_dir']).mkdir()
            Image.fromarray(np.ones((57, 91), np.uint8)).save(Path(c['real_dir'])/'real.png')
            row = prepare(c)['train'][0]
            self.assertEqual(row['crop_origin_yx'], [2, 5])
            self.assertEqual(row['hw'], [51, 73])
            index = Path(c['prepared_dir']) / row['preview']
            self.assertTrue(index.is_file())
            self.assertEqual(len(list(index.parent.glob('part_*.png'))), 6)
            with np.load(Path(c['prepared_dir']) / row['file']) as frame:
                y, gt, valid, _ = read_pair(Path(row['source']), Path(row['depth_source']), c, True)
                coords = fixed_coords(y.shape, c['phase_mask_seed'], c['phase_grid_xy'])
                np.testing.assert_equal(frame['coords_yx'], coords)
                for key, full in [('y', y), ('depth', gt), ('valid', valid)]:
                    np.testing.assert_equal(frame[key], gather(full, coords))
            # Synthetic compatible state exercises strict loading without downloading test weights.
            torch.save(DepthModel(c, load_pretrained=False).da.state_dict(), c['da_weights'])
            run = train(c)
            net, config = load_checkpoint(run/'checkpoints/epoch_001.pt', 'cpu')
            test_dir = run/'test/epoch001'
            with np.load(test_dir/'0000_scene.npz') as result:
                geom = json.loads(str(result['geometry_json']))
                self.assertEqual(geom['resized_hw'], [61, 97])
                self.assertEqual(geom['crop_origin_yx'], [5, 12])
                self.assertEqual(result['depth_m'].shape, (51, 73))
                np.testing.assert_allclose(reconstruct(net, result['measurement'], config, 'cpu'),
                                           result['depth_m'], rtol=1e-5)
            self.assertEqual(json.loads((test_dir/'metrics.json').read_text())['valid_pixels'], 51*73)
            self.assertTrue((test_dir/'depth_bins.png').is_file())
            run_real(net, config, root/'real_again', 'cpu')
            actual = tifffile.imread(root/'real_again/0000_real_depth_m.tiff')
            self.assertEqual(actual.shape, (51, 73))
            np.testing.assert_equal(actual, tifffile.imread(run/'real/epoch001/0000_real_depth_m.tiff'))
