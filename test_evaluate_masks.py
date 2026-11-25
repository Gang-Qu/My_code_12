import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import tifffile

from evaluate_masks import align_gt, evaluate_masks


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
            with self.assertRaisesRegex(ValueError, 'pred-scale'):
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
            saved = tifffile.imread(folder/'prediction_masked.tiff')
            self.assertEqual(saved.dtype, codes.dtype)
            np.testing.assert_array_equal(saved, [[1030, 1031, 0, 0]])
            np.testing.assert_array_equal(tifffile.imread(folder/'gt_masked.tiff'), [[1, 1, 0, 0]])
            np.testing.assert_allclose(tifffile.imread(folder/'abs_error_m.tiff'),
                                       [[.03, .031, np.nan, np.nan]], rtol=1e-6)

    def test_align_raw_gt_from_recorded_geometry(self):
        raw = np.arange(24, dtype=np.uint16).reshape(4, 6)
        with TemporaryDirectory() as temp:
            path = Path(temp)/'frame.npz'
            geo = dict(source_hw=[4, 6], resized_hw=[8, 12], crop_origin_yx=[2, 4], hw=[2, 4])
            np.savez(path, geometry_json=json.dumps(geo))
            result = align_gt(raw, (2, 4), path)
            np.testing.assert_array_equal(result, [[8, 8, 9, 9], [8, 8, 9, 9]])
            with self.assertRaisesRegex(ValueError, 'geometry-npz'):
                align_gt(raw, (2, 4), None)
