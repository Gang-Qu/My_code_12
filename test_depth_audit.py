import io
import json
import tempfile
from pathlib import Path
import tifffile
import unittest
from unittest.mock import patch
import numpy as np

from depth_audit import audit_pair, depth_counts, write_audit
from settings import SETTINGS
from train import show_progress


class DepthAuditTests(unittest.TestCase):
    def test_range_boundaries_and_invalid_values(self):
        z = np.array([np.nan, np.inf, -1, 0, .25, .5, 1, 5, 5.1])
        result = depth_counts(z, .5, 5.)
        self.assertEqual([result[k] for k in ('nonfinite', 'nonpositive', 'below', 'in_range', 'above')],
                         [2, 2, 1, 3, 1])
        self.assertEqual(result['min_positive_m'], .25)
        self.assertEqual(result['max_positive_m'], 5.1)
        self.assertIsNone(depth_counts(z[:4], .5, 5.)['min_positive_m'])

    def test_report_uses_unfiltered_source_and_actual_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = dict(SETTINGS, min_train_depth_percent=0, expected_hw=None, center_crop_hw=[2, 2], depth_scale=.001)
            y = np.ones((4, 4), np.float32)
            y[1, 1] = np.nan
            z = np.full((4, 4), 10000., np.float32)
            z[1:3, 1:3] = [[500, 1000], [0, 6000]]
            tifffile.imwrite(root/'y.tif', y)
            tifffile.imwrite(root/'z.tif', z)
            audit = audit_pair(root/'y.tif', root/'z.tif', c, True)
            write_audit(root, c, {'train': [audit]})
            result = json.loads((root/'depth_range_audit.json').read_text())['splits']['train']
            self.assertEqual(result['original']['above'], 13)
            self.assertEqual(result['crop']['above'], 1)
            self.assertEqual(result['crop']['in_range'], 2)
            self.assertEqual(result['supervised_pixels'], 1)

    def test_one_live_line_and_redirected_summary(self):
        for tty in [True, False]:
            stream = io.StringIO()
            stream.isatty = lambda: tty
            with patch('sys.stdout', stream):
                show_progress(1, 1, 1, 2, .4)
                show_progress(1, 1, 2, 2, .25, {'mae_m': .2, 'accuracy_3pct': .4})
            output = stream.getvalue()
            self.assertEqual(output.count('\n'), 1)
            self.assertEqual('\r' in output, tty)
            self.assertLess(output.index('loss='), output.index('epoch='))
            self.assertIn('100%', output)
