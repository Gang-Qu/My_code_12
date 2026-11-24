import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import tifffile
import torch

from evaluate import evaluate
from model import DepthModel
from prepare_dataset import prepare
from settings import SETTINGS
from train import train


class DepthFilterTests(unittest.TestCase):
    def fixture(self, root):
        c = dict(SETTINGS, prepared_dir=str(root/'prep'), output=str(root/'out'),
                 da_weights=str(root/'da.pt'), expected_hw=None, center_crop_hw=None,
                 device='cpu', epochs=1, freeze_da=True, use_refiner=True,
                 save_sampling_preview=False)
        for split, level in [('train', 64), ('val', 96)]:
            for suffix in ('y', 'depth'):
                folder = root / (split+suffix)
                folder.mkdir()
                c[split+'_'+suffix] = str(folder)
            for name in ('empty', 'mixed'):
                image = np.full((28, 56), 0 if name == 'empty' else level, np.uint8)
                Image.fromarray(image).save(Path(c[split+'_y'])/(name+'.png'))
                z = np.full(image.shape, 10., np.float32)
                if name == 'mixed':
                    z[0, :5] = [.1, .5, 2, 5, np.nan]
                tifffile.imwrite(Path(c[split+'_depth'])/(name+'.tif'), z)
        return c

    def test_mixed_frames_complete_raw_and_prepared_training(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self.fixture(root)
            torch.save(DepthModel(c, load_pretrained=False).da.state_dict(), c['da_weights'])
            for mode in ('raw', 'prepared'):
                c['training_input'] = mode
                if mode == 'prepared':
                    manifest = prepare(c)
                    self.assertEqual(len(manifest['train']), 1)
                run = train(c)
                self.assertTrue((run/'checkpoints/epoch_001.pt').is_file())
                report = json.loads((run/'depth_range_audit.json').read_text())
                for split in ('train', 'val'):
                    part = report['splits'][split]
                    self.assertEqual((part['frames'], part['retained_frames'], part['supervised_pixels']), (2, 1, 3))
                    self.assertEqual(len(part['zero_supervision_frames']), 1)
                metrics = json.loads((run/'test/epoch001/metrics.json').read_text())
                self.assertEqual(metrics['valid_pixels'], 3)
                self.assertEqual([r['source'] for r in metrics['per_image']], ['mixed.png'])
                self.assertEqual(len(metrics['skipped_frames']), 1)

    def test_standalone_empty_validation_exports_unavailable_metrics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self.fixture(root)
            pairs = [(Path(c['val_y'])/'empty.png', Path(c['val_depth'])/'empty.tif')]
            # Identity accepts eval(), but cannot perform a depth-model forward.
            result = evaluate(torch.nn.Identity(), c, pairs, root/'eval', 'cpu', 1)
            self.assertIsNone(result['accuracy_3pct'])
            self.assertEqual(result['valid_pixels'], 0)
            self.assertEqual(result['skipped_frames'], [str(pairs[0][1])])
            self.assertTrue((root/'eval'/'metrics.json').is_file())

    def test_all_empty_train_reports_before_loading_weights(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self.fixture(root)
            tifffile.imwrite(Path(c['train_depth'])/'mixed.tif', np.zeros((28, 56), np.float32))
            with self.assertRaisesRegex(ValueError, '整个训练集'):
                train(c)
            audit, = (root/'out').glob('*/depth_range_audit.json')
            self.assertEqual(json.loads(audit.read_text())['splits']['train']['retained_frames'], 0)
