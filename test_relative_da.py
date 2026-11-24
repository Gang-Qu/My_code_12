from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch
from model import DepthModel
from reconstruct_real import load_checkpoint, sampling_rule
from settings import SETTINGS
from train import optimizer_for


class RelativeDATest(unittest.TestCase):
    def test_relative_weights_calibration_and_replay(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as tmp:
            c = dict(SETTINGS, da_model_type='relative', freeze_da=True,
                     metric_max_depth=1., relative_da_weights=tmp+'/relative.pth',
                     da_weights=tmp+'/absent.pth')
            source = DepthModel(c, load_pretrained=False)
            torch.save(source.da.state_dict(), c['relative_da_weights'])
            model = DepthModel(c).eval()
            self.assertIsInstance(model.da.depth_head.scratch.output_conv2[-1], torch.nn.ReLU)
            torch.testing.assert_close(model.da.pretrained.cls_token, source.da.pretrained.cls_token)
            relative = torch.tensor([0., 1., 8.])
            metric = model.calibration(relative)
            self.assertTrue((metric[:-1] > metric[1:]).all() and metric[-1] > 0)
            frame = torch.rand(52, 76)
            pred = model.forward_frame(frame)
            optimizer = optimizer_for(model, c)
            pred.mean().backward()
            initial = model.calibration.bias.item()
            optimizer.step()
            self.assertNotEqual(initial, model.calibration.bias.item())
            path = Path(tmp) / 'checkpoint.pt'
            torch.save(dict(config=c, rule=sampling_rule(c), model=model.state_dict()), path)
            Path(c['relative_da_weights']).unlink()
            restored, _ = load_checkpoint(path, 'cpu')
            torch.testing.assert_close(restored.forward_frame(frame), model.forward_frame(frame))

    def test_missing_type_uses_legacy_metric(self):
        c = dict(SETTINGS)
        c.pop('da_model_type', None)
        model = DepthModel(c, load_pretrained=False)
        self.assertIsNone(model.calibration)
        self.assertIsInstance(model.da.depth_head.scratch.output_conv2[-1], torch.nn.Sigmoid)
