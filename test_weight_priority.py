import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from model import load_training_weights
from settings import SETTINGS


class WeightPriorityTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        folder = Path(temp.name)
        self.net = nn.Module()
        self.net.da = nn.Linear(1, 1, bias=False)
        self.net.refiner = nn.Linear(1, 1, bias=False)
        self.c = dict(SETTINGS, da_weights=str(folder / 'metric.pth'),
                      checkpoint=str(folder / 'epoch.pt'), use_refiner=True)
        torch.save({'state_dict': {'weight': torch.tensor([[1.]])}}, self.c['da_weights'])
        state = {'da.weight': torch.tensor([[2.]]), 'refiner.weight': torch.tensor([[3.]])}
        torch.save({'model': state, 'config': self.c}, self.c['checkpoint'])

    def test_both_priorities_preserve_refiner(self):
        for priority, expected, source in [(True, 1., 'da_weights'), (False, 2., 'checkpoint')]:
            with self.subTest(priority=priority):
                report = load_training_weights(self.net, dict(self.c, da_weight_priority=priority))
                self.assertEqual(self.net.da.weight.item(), expected)
                self.assertEqual(self.net.refiner.weight.item(), 3.)
                self.assertEqual(report['final_da_source'], source)

    def test_checkpoint_alone(self):
        report = load_training_weights(self.net, dict(self.c, da_weights=None))
        self.assertEqual(list(report['sources']), ['checkpoint'])
        self.assertEqual(self.net.da.weight.item(), 2.)

    def test_wrong_model_type_and_no_source(self):
        with self.assertRaisesRegex(ValueError, 'da_model_type'):
            load_training_weights(self.net, dict(self.c, da_model_type='relative', relative_da_weights=None))
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            load_training_weights(self.net, dict(self.c, da_weights=None, checkpoint=None))

    def test_file_not_found_identifies_setting(self):
        with self.assertRaisesRegex(FileNotFoundError, 'da_weights'):
            load_training_weights(self.net, dict(self.c, da_weights=self.c['da_weights']+'.missing'))
