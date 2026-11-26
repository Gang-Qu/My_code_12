import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from outputs import read_test_panels, save_result
from reconstruct_real import reconstruct
from settings import SETTINGS
from sampling import LearnedMask


class ReconstructionSpeedTests(unittest.TestCase):
    def test_fast_export_preserves_full_resolution_arrays_and_png(self):
        y = np.random.default_rng(41).random((301, 117), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, fast in [('baseline', False), ('fast', True)]:
                save_result(root, name, y, 2+y, {}, dict(SETTINGS, fast_reconstruction_export=fast), gt=1+y)
            with zipfile.ZipFile(root/'fast.npz') as archive:
                self.assertTrue(all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()))
            with np.load(root/'baseline.npz') as a, np.load(root/'fast.npz') as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
            with Image.open(root/'baseline.png') as a, Image.open(root/'fast.png') as b:
                np.testing.assert_array_equal(np.array(a), np.array(b))
            gt, pred = read_test_panels(root/'fast.png')
            self.assertEqual(gt.shape, (*y.shape, 3))
            self.assertEqual(pred.shape, gt.shape)

    def test_tiled_inference_routing_matches_all_stages(self):
        mask = LearnedMask([5, 3], [4, 7], 41).eval()
        for h, w in [(17, 23), (2, 17), (12, 3)]:
            values = torch.arange(15*h*w*2).reshape(15, h, w, 2)
            with torch.no_grad():
                expected = values
                for stage in mask.stages:
                    expected = stage(expected)
                actual = mask.route(values)
                self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(torch.equal(mask.route(actual, inverse=True), values))
                mask.stages[0].logits.neg_()

    def test_optional_cuda_precision_is_explicit(self):
        class Echo(torch.nn.Module):
            def forward_frame(self, image, chunk, return_coords):
                return image
        y = np.array([[1.5]], dtype=np.float32)
        np.testing.assert_equal(reconstruct(Echo(), y, SETTINGS, 'cpu'), y)
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            reconstruct(Echo(), y, dict(SETTINGS, inference_precision='float16'), 'cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_cuda_autocast_returns_float32_depth(self):
        class Projection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Conv2d(1, 1, 1)

            def forward_frame(self, image, chunk, return_coords):
                depth = self.layer(image[None, None])[0, 0]
                self.compute_dtype = depth.dtype
                return depth
        net = Projection().cuda().eval()
        y = np.ones((16, 16), dtype=np.float32)
        fp32 = reconstruct(net, y, SETTINGS, 'cuda')
        fp16 = reconstruct(net, y, dict(SETTINGS, inference_precision='float16'), 'cuda')
        self.assertEqual(net.compute_dtype, torch.float16)
        self.assertEqual(fp16.dtype, np.float32)
        np.testing.assert_allclose(fp16, fp32, atol=.002)
