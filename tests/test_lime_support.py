"""Lightweight LIME checks; no dataset or trained checkpoint required."""
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src import dr_pipeline as pipeline
from src.lime_support import explain_lime_image, run_lime_targets


class ToyModel(torch.nn.Module):
    def forward(self, images):
        signal = images[:, 0, :8, :8].mean((1, 2))
        return torch.stack((signal, -signal), dim=1)


class LimeTests(unittest.TestCase):
    def setUp(self):
        pixels = np.full((24, 24, 3), 80, dtype=np.uint8)
        pixels[:8, :8, 0] = 240
        self.image = Image.fromarray(pixels)
        self.model = ToyModel().eval()
        self.transform = pipeline._build_transform(24, False, {}, {'enabled': False})

    def test_repeatable_signed_map_and_input_probabilities(self):
        args = (self.model, self.image, self.transform, torch.device('cpu'), 0)
        first = explain_lime_image(*args, seed=17, num_samples=40, n_segments=8)
        second = explain_lime_image(*args, seed=17, num_samples=40, n_segments=8)
        self.assertEqual(first[0].shape, (24, 24))
        self.assertTrue(np.isfinite(first[0]).all())
        np.testing.assert_allclose(first[0], second[0])
        self.assertTrue(np.isfinite(first[1]))

    def test_audit_outputs_and_failure_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.image.save(root / 'test.png')
            conf = {'paths': {key: directory for key in ['figures_dir', 'tables_dir', 'logs_dir']},
                    'data': {'image_size': 24, 'num_classes': 2, 'label_order': ['A', 'B']},
                    'training': {'device': 'cpu', 'backbone': 'resnet50'},
                    'preprocessing': {'enabled': False},
                    'xai': {'device': 'cpu', 'lime_num_samples': 40, 'lime_n_segments': 8}}
            targets = pd.DataFrame([
                dict(sample_id='valid', true_class=0, pred_class=0, confidence=.8, image_path=str(root/'test.png')),
                dict(sample_id='missing', true_class=1, pred_class=0, confidence=.6, image_path=str(root/'absent.png'))])
            signed = np.zeros((24, 24), dtype=np.float32)
            signed[:8, :8] = 1
            with patch.object(pipeline, '_load_model', return_value=self.model), patch(
                'src.lime_support.explain_lime_image', return_value=(signed, .95, np.zeros((24,24), dtype=int))
            ):
                frame, table, status_path, status = run_lime_targets(
                    pipeline, conf, targets, 17, 'test', None, 'unit_test')
            self.assertEqual(status, 'partial')
            self.assertEqual(frame.sample_id.tolist(), ['valid'])
            self.assertTrue(Path(frame.iloc[0].artifact_path).is_file())
            self.assertTrue(Path(table).is_file())
            self.assertIn('missing', pipeline._load_json(status_path)['errors'])
            self.assertIn('faith_delta_k20', frame.columns)
            self.assertEqual(int(frame.iloc[0].aopc_pass), int(frame.iloc[0].aopc_delta > 0))


if __name__ == '__main__':
    unittest.main()
