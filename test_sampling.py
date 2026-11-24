import unittest
import numpy as np
import torch
from sampling import fixed_coords, gather, restore, phase_offsets

class FixedSampling(unittest.TestCase):
    def test_persisted_mask_matches_original_generation(self):
        ranks = np.random.PCG64(41).random_raw((8, 518, 518))
        np.testing.assert_equal(phase_offsets(41), np.argsort(ranks, axis=0, kind='stable'))
        small = fixed_coords((27, 57), 42)
        large = fixed_coords((60, 120), 42)[:, :14, :15]
        good = small[..., 0] >= 0
        np.testing.assert_equal(small[good], large[good])

    def test_coverage_and_inverse(self):
        for hw in [(1, 1), (33, 71), (56, 112)]:
            coords = fixed_coords(hw, 41)
            self.assertEqual(coords.shape[0], 8)
            y = np.arange(np.prod(hw)).reshape(hw)
            patches = gather(y, coords)
            tensor = torch.tensor(patches, dtype=torch.float32, requires_grad=True)
            frame = restore(tensor, torch.from_numpy(coords), hw)
            np.testing.assert_array_equal(frame.detach(), y)
            frame.sum().backward()
            np.testing.assert_equal(tensor.grad.numpy(), coords[..., 0] >= 0)
            good = coords[..., 0] >= 0
            ids = coords[good]
            np.testing.assert_equal(np.sort(ids[:, 0] * hw[1] + ids[:, 1]), y.ravel())
