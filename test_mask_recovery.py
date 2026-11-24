from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import sampling
from metrics import EpochStats
from settings import SETTINGS


class MaskRecoveryTests(unittest.TestCase):
    def test_missing_corrupt_invalid_and_reused(self):
        for grid in ((4, 2), (3, 2)):
            with self.subTest(grid=grid), tempfile.TemporaryDirectory() as folder:
                hw = (15, 23)
                def read():
                    sampling.cached_offsets.cache_clear()  # simulate a fresh process
                    return sampling.fixed_coords(hw, 41, grid, mask_dir=folder)
                ref = read()
                path, = Path(folder).glob('*.npz')
                with patch.object(sampling, '_generate_offsets', side_effect=AssertionError('regenerated valid NPZ')):
                    np.testing.assert_array_equal(read(), ref)
                path.write_bytes(b'truncated')
                np.testing.assert_array_equal(read(), ref)
                with np.load(path) as data:
                    invalid = np.zeros_like(data['offsets'])
                np.savez(path, offsets=invalid)
                np.testing.assert_array_equal(read(), ref)
                with np.load(path) as data:
                    self.assertGreater(data['offsets'].max(), 0)
                good = ref[..., 0] >= 0
                np.testing.assert_array_equal(np.sort(ref[..., 0][good]*hw[1]+ref[..., 1][good]), np.arange(np.prod(hw)))

    def test_test_bins_pool_pixels_across_frames(self):
        stats = EpochStats(dict(SETTINGS, depth_min=.5, depth_max=.8))
        for g, p in [([.5, .5, .5, .8], [.51, .52, .52, .78]), ([.6], [.615])]:
            gt = np.array(g, dtype=np.float32)[None,None]
            stats.update(np.array(p)[None,None], gt, np.ones_like(gt, dtype=bool))
        bins = stats.result()['depth_bins']
        self.assertEqual([r['valid_pixels'] for r in bins], [3, 1, 1])
        self.assertEqual([r['within_3pct_pixels'] for r in bins], [1, 1, 1])
        np.testing.assert_allclose([r['pixel_fraction'] for r in bins], [.6,.2,.2])
        np.testing.assert_allclose([r['accuracy_3pct'] for r in bins], [1/3,1,1])
