import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from model import load_training_weights
from settings import SETTINGS
from reconstruct_real import load_checkpoint, sampling_rule


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
            load_training_weights(self.net, dict(self.c, da_model_type='relative', relative_da_weights=None, strict_weight_load=True))
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            load_training_weights(self.net, dict(self.c, da_weights=None, checkpoint=None))

    def test_partial_priority_is_per_tensor(self):
        self.net.da = nn.Linear(1, 1)
        self.net.extra = nn.Linear(2, 1, bias=False)
        initial = self.net.extra.weight.detach().clone()
        torch.save({'weight': torch.tensor([[1.]]), 'bias': torch.ones(3),
                    'obsolete': torch.ones(2)}, self.c['da_weights'])
        torch.save({'model': {'da.bias': torch.tensor([4.]), 'da.weight': torch.tensor([[2.]]),
                              'refiner.weight': torch.ones(2, 1)}, 'config': self.c}, self.c['checkpoint'])
        for priority, weight, source in [(True, 1., 'mixed'), (False, 2., 'checkpoint')]:
            with self.subTest(priority=priority):
                report = load_training_weights(self.net, dict(self.c, da_weight_priority=priority))
                self.assertEqual(self.net.da.weight.item(), weight)
                self.assertEqual(self.net.da.bias.item(), 4.)
                torch.testing.assert_close(self.net.extra.weight, initial)
                self.assertEqual(report['final_da_source'], source)
                self.assertIn('bias', report['details']['da_weights']['shape_mismatch'])
                self.assertIn('obsolete', report['details']['da_weights']['unexpected'])
                self.assertIn('extra.weight', report['details']['checkpoint']['missing'])
        with self.assertRaises(RuntimeError):
            load_training_weights(self.net, dict(self.c, strict_weight_load=True))

    def test_source_without_matches_can_fall_back(self):
        torch.save({'unknown': torch.ones(3)}, self.c['da_weights'])
        report = load_training_weights(self.net, self.c)
        self.assertEqual(report['final_da_source'], 'checkpoint')
        with self.assertRaisesRegex(ValueError, '匹配'):
            load_training_weights(self.net, dict(self.c, checkpoint=None))

    def test_checkpoint_with_changed_sampling_transfers_matching_network_weights(self):
        self.net.mask = nn.Linear(1, 1, bias=False)
        initial = self.net.mask.weight.clone()
        state = {'mask.weight': torch.tensor([[9.]]), 'da.weight': torch.tensor([[6.]])}
        torch.save({'config': self.c, 'model': state, 'rule': 'old'}, self.c['checkpoint'])
        c = dict(self.c, phase_mask_mode='learned', phase_grid_xy=[3, 2], da_weights=None)
        report = load_training_weights(self.net, c)
        self.assertEqual(self.net.da.weight.item(), 6.)
        torch.testing.assert_close(self.net.mask.weight, initial)
        self.assertIn('mask.weight', report['details']['checkpoint']['incompatible'])

    def test_replay_can_explicitly_use_partial_weights(self):
        # Preserve the default exact-replay contract while exercising the optional path.
        state = {'da.weight': torch.tensor([[8.]]), 'refiner.weight': torch.ones(4, 1)}
        torch.save(dict(model=state, config=self.c, rule=sampling_rule(self.c)), self.c['checkpoint'])
        initial = self.net.refiner.weight.detach().clone()
        with patch('reconstruct_real.DepthModel', return_value=self.net):
            with self.assertRaises(RuntimeError):
                load_checkpoint(self.c['checkpoint'], 'cpu')
            path = Path(self.c['checkpoint']).parent / 'eval' / 'weight_load.json'
            net, _ = load_checkpoint(self.c['checkpoint'], 'cpu', strict=False, report_path=path)
            self.assertEqual(json.loads(path.read_text()), net.weight_load)
            self.assertEqual(net.da.weight.item(), 8.)
            torch.testing.assert_close(net.refiner.weight, initial)
            self.assertIn('refiner.weight', net.weight_load['shape_mismatch'])

    def test_file_not_found_identifies_setting(self):
        with self.assertRaisesRegex(FileNotFoundError, 'da_weights'):
            load_training_weights(self.net, dict(self.c, da_weights=self.c['da_weights']+'.missing'))
