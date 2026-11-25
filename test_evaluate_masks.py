import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import tifffile

from evaluate_masks import align_gt, evaluate_masks, evaluate_test_png
from outputs import preview, read_test_panels


class MaskEvaluationTests(unittest.TestCase):
    def test_16bit_depth_resize_overlap_and_threshold(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            masks = root / 'masks'
            masks.mkdir()
            for name, row in [('left', [1, 0]), ('all', [255, 255]), ('empty', [0, 0])]:
                Image.fromarray(np.array([row], np.uint8)).save(masks / (name+'.png'))
            gt = np.array([[1, 1, 2, 9]], np.float32)
            codes = np.array([[1030, 1031, 2000, 4000]], np.uint16)
            tifffile.imwrite(root / 'gt.tif', gt)
            Image.fromarray(codes).save(root / 'pred.png')
            args = (masks, root / 'gt.tif', root / 'pred.png', root / 'out')
            with self.assertRaisesRegex(ValueError, 'pred_scale'):
                evaluate_masks(*args)
            evaluate_masks(*args, pred_scale=.001)
            report = json.loads((root/'out/metrics.json').read_text())
            per = {r['mask']: r for r in report['per_mask']}
            self.assertIsNone(per['empty.png']['accuracy_3pct'])
            self.assertEqual(per['left.png']['within_3pct_pixels'], 1)
            self.assertAlmostEqual(per['left.png']['mae_m'], .0305)
            self.assertAlmostEqual(report['mean_over_masks']['accuracy_3pct'], (.5 + 2/3)/2)
            self.assertAlmostEqual(report['pooled_with_repeats']['accuracy_3pct'], 3/5)
            self.assertAlmostEqual(report['union_unique_pixels']['accuracy_3pct'], 2/3)
            folder = root/'out/left.png'
            for name in ('gt_masked.png', 'prediction_masked.png', 'abs_error.png'):
                with Image.open(folder/name) as preview:
                    self.assertEqual(preview.mode, 'L')
                    self.assertEqual(preview.size, (4, 1))
                    values = np.asarray(preview)
                    self.assertTrue((values[0, :2] > 0).all())
                    self.assertTrue((values[0, 2:] == 0).all())
            with Image.open(root/'out/all.png/gt_masked.png') as preview:
                values = np.asarray(preview)
                self.assertGreater(values[0, 2], values[0, 0])
            with Image.open(root/'out/all.png/abs_error.png') as preview:
                values = np.asarray(preview)
                self.assertGreater(values[0, 1], values[0, 2])  # 31 mm versus zero error
                self.assertEqual(values[0, 3], 0)  # invalid GT
            saved = tifffile.imread(folder/'prediction_masked.tiff')
            self.assertEqual(saved.dtype, codes.dtype)
            np.testing.assert_array_equal(saved, [[1030, 1031, 0, 0]])
            np.testing.assert_array_equal(tifffile.imread(folder/'gt_masked.tiff'), [[1, 1, 0, 0]])
            np.testing.assert_allclose(tifffile.imread(folder/'abs_error_m.tiff'),
                                       [[.03, .031, np.nan, np.nan]], rtol=1e-6)

    def test_saved_test_png_dimensions_colors_and_mask_metrics(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            masks = root/'masks'
            masks.mkdir()
            Image.fromarray(np.array([[255, 0]], np.uint8)).save(masks/'left.png')
            c = dict(depth_min=1., depth_max=5.)
            for h, w in [(7, 11), (285, 17)]:
                gt = np.ones((h, w), np.float32)
                pred = np.full_like(gt, 5.)
                gt[-1, 0], pred[-1, 0] = 5., 1.
                path = root/f'0000_scene{h}.png'
                preview(path, np.zeros_like(gt), pred, c, gt, full_resolution=True)
                gt_rgb, pred_rgb = read_test_panels(path)
                self.assertEqual(gt_rgb.shape, (h, w, 3))
                np.testing.assert_array_equal(gt_rgb[0, 0], [0, 0, 255])
                np.testing.assert_array_equal(pred_rgb[0, 0], [255, 255, 0])
                np.testing.assert_array_equal(pred_rgb[-1, 0], gt_rgb[0, 0])
                out = root/'out'
                for numeric in (False, True):
                    if numeric:
                        np.savez(path.with_suffix('.npz'), gt_m=gt, depth_m=pred)
                    result = evaluate_test_png(masks, path, out, depth_min=1., depth_max=5.)
                    folder = out/path.stem
                    if numeric:
                        self.assertAlmostEqual(result['union_unique_pixels']['mae_m'], 4.)
                    else:
                        self.assertIsNone(result['metric_source'])
                        folder = folder/'png_only'
                    for name, rgb in [('gt', gt_rgb), ('prediction', pred_rgb)]:
                        with Image.open(folder/'left.png'/f'{name}_masked.png') as im:
                            self.assertEqual(im.size, (w, h))
                            masked = np.asarray(im)
                            np.testing.assert_array_equal(masked[:, 0], rgb[:, 0])
                            self.assertFalse(masked[:, -1].any())
                np.savez(path.with_suffix('.npz'), gt_m=gt, depth_m=pred.T)
                with self.assertRaisesRegex(ValueError, '逐像素'):
                    evaluate_test_png(masks, path, out)

    def test_align_raw_gt_from_recorded_geometry(self):
        raw = np.arange(24, dtype=np.uint16).reshape(4, 6)
        with TemporaryDirectory() as temp:
            path = Path(temp)/'frame.npz'
            geo = dict(source_hw=[4, 6], resized_hw=[8, 12], crop_origin_yx=[2, 4], hw=[2, 4])
            np.savez(path, geometry_json=json.dumps(geo))
            result = align_gt(raw, (2, 4), path)
            np.testing.assert_array_equal(result, [[8, 8, 9, 9], [8, 8, 9, 9]])
            direct = align_gt(raw, (2, 4), None)
            np.testing.assert_array_equal(direct, [[6, 8, 9, 11], [18, 20, 21, 23]])
            self.assertEqual(direct.dtype, raw.dtype)
