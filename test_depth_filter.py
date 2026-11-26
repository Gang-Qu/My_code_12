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
from data import keep_training_frame, read_pair, train_depth_threshold
from model import DepthModel
from prepare_dataset import prepare
from settings import SETTINGS
from train import train, training_records


class DepthFilterTests(unittest.TestCase):
    def fixture(self, root):
        c = dict(SETTINGS, dataset_audit=True, min_train_depth_percent=0, prepared_dir=str(root/'prep'), output=str(root/'out'),
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
                suffix = '.TIF' if split == 'train' else '.png'
                Image.fromarray(image).save(Path(c[split+'_y'])/(name+suffix))
                z = np.full(image.shape, 10., np.float32)
                if name == 'mixed':
                    z[0, :5] = [.1, .5, 2, 5, np.nan]
                tifffile.imwrite(Path(c[split+'_depth'])/(name+'.tif'), z)
        return c

    def test_train_only_pairs_tiffs_before_label_lookup_and_filters_cached_sources(self):
        with TemporaryDirectory() as tmp:
            c = self.fixture(Path(tmp))
            folder = Path(c['train_y'])
            (folder / 'mixed.png').write_bytes(b'ignored same-stem PNG')
            (folder / 'preview.png').write_bytes(b'ignored unpaired PNG')
            d = Path(c['train_depth'])
            (d/'mixed.tif').rename(d/'mixed.TIFF')
            raw, _ = training_records(c)
            self.assertEqual({r['image'].name for r in raw}, {'mixed.TIF', 'empty.TIF'})
            manifest = prepare(c)
            self.assertEqual([Path(r['source']).name for r in manifest['train']], ['mixed.TIF'])
            duplicate = dict(manifest['train'][0], source=str(folder/'mixed.png'), file='absent.npz')
            manifest['train'].insert(0, duplicate)
            (Path(c['prepared_dir'])/'manifest.json').write_text(json.dumps(manifest))
            cached, _ = training_records(dict(c, training_input='prepared'))
            self.assertEqual([Path(r['source']).name for r in cached], ['mixed.TIF'])
            (folder/'mixed.TIF').unlink()
            (folder/'empty.TIF').unlink()
            with self.assertRaisesRegex(ValueError, 'tif'):
                training_records(c)

    def test_configurable_training_and_label_formats(self):
        with TemporaryDirectory() as tmp:
            c = self.fixture(Path(tmp))
            c.update(train_image_extensions='PNG', depth_extensions=['.PNG'])
            folder = Path(c['train_y'])
            Image.fromarray(np.ones((28, 56), np.uint8)).save(folder/'mixed.png')
            Image.fromarray(np.ones((28, 56), np.uint16)).save(Path(c['train_depth'])/'mixed.png')
            rows, _ = training_records(c)
            self.assertEqual([(r['image'].name, r['depth'].name) for r in rows], [('mixed.png', 'mixed.png')])
            prepare(c)
            cached, _ = training_records(dict(c, training_input='prepared'))
            self.assertEqual(Path(cached[0]['source']).name, 'mixed.png')
            with self.assertRaisesRegex(ValueError, '未缓存'):
                training_records(dict(c, training_input='prepared',
                                      train_image_extensions=['png', 'tif'], depth_extensions=['tif']))
            with self.assertRaisesRegex(ValueError, 'train_image_extensions'):
                training_records(dict(c, train_image_extensions=['.invalid']))

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

    def test_depth_coverage_is_independent_of_measurement_validity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = dict(SETTINGS, expected_hw=None, center_crop_hw=None)
            tifffile.imwrite(root/'image.tif', np.array([[np.nan, 1, 1, 1, 1]], np.float32))
            tifffile.imwrite(root/'depth.tif', np.array([[.5, 5., 1., 1., 0.]], np.float32))
            _, _, valid, meta = read_pair(root/'image.tif', root/'depth.tif', c, training=True)
            self.assertEqual(meta['depth_in_range_percent'], 80)
            self.assertEqual(valid.sum(), 3)
            self.assertTrue(keep_training_frame(meta, train_depth_threshold(c)))
            self.assertFalse(keep_training_frame(meta, 81))
            for value in (-1, 101, float('nan'), True):
                with self.assertRaises(ValueError):
                    train_depth_threshold(dict(c, min_train_depth_percent=value))

    def test_percent_boundary_applies_to_train_and_test(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self.fixture(root)
            c.update(min_train_depth_percent=80, center_crop_hw=[20, 25])
            # The full frame is mostly outside range. In the crop one image has
            # exactly 400/500 in range and its sibling only 399/500.
            for name, n in [('mixed', 400), ('below', 399)]:
                z = np.full((28, 56), 10., np.float32)
                region = np.full(500, np.nan, np.float32)
                region[:n:2] = .5
                region[1:n:2] = 5.
                z[4:24, 15:40] = region.reshape(20, 25)
                tifffile.imwrite(Path(c['train_depth'])/(name+'.tif'), z)
                if name == 'below':
                    Image.fromarray(np.full(z.shape, 173, np.uint8)).save(Path(c['train_y'])/'below.tiff')
            # A retained and a rejected test frame straddle the same threshold.
            # The rejected test measurement duplicates a retained training image.
            for name in ('mixed', 'below'):
                (Path(c['val_depth'])/(name+'.tif')).write_bytes(
                    (Path(c['train_depth'])/(name+'.tif')).read_bytes())
            (Path(c['val_y'])/'below.tiff').write_bytes((Path(c['train_y'])/'mixed.TIF').read_bytes())
            torch.save(DepthModel(c, load_pretrained=False).da.state_dict(), c['da_weights'])
            for audit in (True, False):
                c['dataset_audit'] = audit
                for mode in ('raw', 'prepared'):
                    c['training_input'] = mode
                    if mode == 'prepared':
                        self.assertEqual(len(prepare(c)['train']), 1)
                    run = train(c)
                    if audit:
                        summary = json.loads((run/'depth_range_audit.json').read_text())['splits']['train']
                        self.assertEqual(summary['retained_frames'], 1)
                        self.assertEqual(summary['supervised_pixels'], 400)
                    else:
                        self.assertEqual(len(json.loads((run/'skipped_train.json').read_text())), 2)
                    stats = json.loads((run/'test/epoch001/metrics.json').read_text())
                    self.assertEqual(stats['valid_pixels'], 400)
                    self.assertEqual([r['source'] for r in stats['per_image']], ['mixed.png'])
                    self.assertEqual({Path(p).stem for p in stats['skipped_frames']}, {'empty', 'below'})
                    self.assertEqual(sum(b['valid_pixels'] for b in stats['depth_bins']), 400)
                    self.assertEqual(stats['evaluation']['min_depth_in_range_percent'], 80)
                    if audit:
                        val = json.loads((run/'depth_range_audit.json').read_text())['splits']['val']
                        self.assertEqual(val['retained_frames'], 1)
                        self.assertEqual(set(val['skipped_frames']), set(stats['skipped_frames']))

    def test_audit_off_still_filters_raw_and_prepared(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = dict(self.fixture(root), dataset_audit=False, epochs=2, test_export_every=2,
                     train_prefetch=True, profile_train_steps=1, batch_size=6)
            torch.save(DepthModel(c, load_pretrained=False).da.state_dict(), c['da_weights'])
            with patch('depth_audit.depth_counts', side_effect=AssertionError('audit was enabled')):
                for mode in ('raw', 'prepared'):
                    c['training_input'] = mode
                    if mode == 'prepared':
                        manifest = prepare(c)
                        self.assertEqual(len(manifest['train']), 1)
                        self.assertFalse(list((root/'prep').rglob('depth_range_audit.json')))
                    run = train(c)
                    timing, = (run/'train_timing.jsonl').read_text().splitlines()
                    profile = json.loads(timing)
                    self.assertEqual((profile['hw'], profile['parts'], profile['da_chunks']), ([28, 56], 8, 2))
                    self.assertTrue(profile['prefetch'])
                    self.assertGreater(profile['seconds']['backward'], 0)
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
            original = Path(c['train_y'])/'mixed.TIF'
            (Path(c['val_y'])/'mixed.png').unlink()
            (Path(c['val_y'])/'mixed.tiff').write_bytes(original.read_bytes())
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
            c['min_train_depth_percent'] = 80
            pairs = [(Path(c['val_y'])/'mixed.png', Path(c['val_depth'])/'mixed.tif')]
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
