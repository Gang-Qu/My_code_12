import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
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
        c = dict(SETTINGS, dataset_audit=True, prepared_dir=str(root/'prep'), output=str(root/'out'),
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

    def test_audit_off_still_filters_raw_and_prepared(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = dict(self.fixture(root), dataset_audit=False, epochs=2, test_export_every=2)
            torch.save(DepthModel(c, load_pretrained=False).da.state_dict(), c['da_weights'])
            with patch('depth_audit.depth_counts', side_effect=AssertionError('audit was enabled')):
                for mode in ('raw', 'prepared'):
                    c['training_input'] = mode
                    if mode == 'prepared':
                        manifest = prepare(c)
                        self.assertEqual(len(manifest['train']), 1)
                        self.assertFalse(list((root/'prep').rglob('depth_range_audit.json')))
                    run = train(c)
                    self.assertFalse((run/'depth_range_audit.json').exists())
                    for epoch in (1, 2):
                        stats = json.loads((run/f'test/epoch{epoch:03d}/metrics.json').read_text())
                        directory = run/f'test/epoch{epoch:03d}'
                        self.assertTrue((directory/'depth_bins.png').is_file())
                        self.assertEqual(bool(list(directory.glob('*_depth_m.tiff'))), epoch == 2)
                        self.assertEqual(stats['valid_pixels'], 3)
                        self.assertEqual(len(stats['skipped_frames']), 1)

    def test_duplicate_valid_measurement_detected_with_audit_off(self):
        with TemporaryDirectory() as tmp:
            c = dict(self.fixture(Path(tmp)), dataset_audit=False)
            original = Path(c['train_y'])/'mixed.png'
            (Path(c['val_y'])/'mixed.png').write_bytes(original.read_bytes())
            for mode in ('raw', 'prepared'):
                c['training_input'] = mode
                if mode == 'prepared':
                    prepare(c)
                with self.assertRaisesRegex(ValueError, 'Duplicate measurement'):
                    train(c)

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
