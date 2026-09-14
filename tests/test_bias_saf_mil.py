"""Tests for BiasSAFMILDetector (SAF-MIL: Saliency-Aware Fusion + Learned Gate)."""
import os
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training'))

from detectors import DETECTOR
from detectors.bias_saf_mil_detector import BiasSAFMILDetector, SAFMILDetector, topk_mil_logits


class DummyConfig:
    hidden_size = 32
    patch_size = 2


class DummyVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()
        self.patch_embed = nn.Conv2d(3, 32, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(32)
        self.head_bias = nn.Parameter(torch.zeros(32))

    def forward(self, x):
        b = x.shape[0]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_token = tokens.mean(dim=1)
        all_tokens = torch.cat([cls_token.unsqueeze(1), tokens], dim=1)
        all_tokens = self.norm(all_tokens) + self.head_bias
        return type('CLIPOut', (), {
            'pooler_output': all_tokens[:, 0],
            'last_hidden_state': all_tokens,
        })()


@pytest.fixture
def saf_detector(monkeypatch):
    """Instantiate BiasSAFMILDetector with a lightweight mock backbone."""
    monkeypatch.setattr(
        BiasSAFMILDetector, 'build_backbone',
        lambda self, config: DummyVisionModel()
    )
    cfg = {
        'model_name': 'bias_saf_mil',
        'use_patch': True,
        'use_sspanet': True,
        'lambda_mil': 0.3,
        'mil_topk': 4,
        'tau_saliency': 1.0,
        'gate_hidden_dim': 16,
        'lambda_gate': 1.0,
        'fusion_alpha_init': 0.1,
    }
    return BiasSAFMILDetector(cfg)


def test_registry_and_aliases(saf_detector):
    assert DETECTOR.get('bias_saf_mil') is BiasSAFMILDetector
    assert DETECTOR.get('saf_mil') is BiasSAFMILDetector
    assert isinstance(saf_detector, SAFMILDetector)


def test_bitfit_parameter_freezing(saf_detector):
    """Backbone weights should be frozen; only backbone biases, heads, and gate should train."""
    for name, param in saf_detector.backbone.named_parameters():
        if 'bias' in name:
            assert param.requires_grad, f"Backbone bias {name} must be trainable"
        else:
            assert not param.requires_grad, f"Backbone weight {name} must be frozen"

    assert saf_detector.fusion_alpha.requires_grad
    assert saf_detector.head.weight.requires_grad
    assert saf_detector.patch_head.weight.requires_grad
    for p in saf_detector.gate.parameters():
        assert p.requires_grad


def test_forward_output_keys_and_shapes(saf_detector):
    data = dict(
        image=torch.randn(4, 3, 8, 8),
        label=torch.tensor([0, 0, 1, 1])
    )
    out = saf_detector(data)

    expected_keys = [
        'cls', 'prob', 'feat', 'feat_norm', 'cls_only_prob',
        'feature_fusion_prob', 'patch_logits', 'mil_logits',
        'mil_prob', 'gating_w', 'gating_w_f', 'saliency_attn', 'diagnostics'
    ]
    for k in expected_keys:
        assert k in out, f"Missing key {k} in forward output"

    assert out['cls'].shape == (4, 2)
    assert out['prob'].shape == (4,)
    assert out['feature_fusion_prob'].shape == (4,)
    assert out['cls_only_prob'].shape == (4,)
    assert out['patch_logits'].shape == (4, 4, 4)
    assert out['mil_logits'].shape == (4,)
    assert out['mil_prob'].shape == (4,)
    assert out['gating_w'].shape == (4,)
    # Saliency attention weights should sum to 1 across spatial patches
    assert out['saliency_attn'].shape == (4, 1, 16)
    torch.testing.assert_close(
        out['saliency_attn'].sum(dim=-1),
        torch.ones(4, 1),
        atol=1e-5, rtol=1e-5
    )


def test_gate_initialization_balanced(saf_detector):
    """At step 0, final gate layer is initialized to zero -> w = sigmoid(0) = 0.5."""
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor([0, 1]))
    out = saf_detector(data)
    torch.testing.assert_close(out['gating_w'], torch.full((2,), 0.5))
    expected_prob = 0.5 * out['feature_fusion_prob'] + 0.5 * out['mil_prob']
    torch.testing.assert_close(out['prob'], expected_prob)


def test_saliency_guided_pooling_focuses_on_anomalous_patches(saf_detector):
    """A patch with a very high logit must receive significantly higher attention weight."""
    data = dict(image=torch.randn(1, 3, 8, 8), label=torch.tensor([1]))
    # Mock patch logits: patch (0, 0) is anomalous (+10.0), others are -5.0
    patch_logits = torch.full((1, 4, 4), -5.0)
    patch_logits[0, 0, 0] = 10.0

    attn = F.softmax(patch_logits.flatten(1) / saf_detector.tau_saliency, dim=-1)
    # The first patch must dominate the attention weight (> 0.99)
    assert attn[0, 0].item() > 0.99


def test_triple_supervision_gradient_flow(saf_detector):
    """Backpropagating overall loss updates CLS head, patch head, gate, and alpha."""
    data = dict(image=torch.randn(4, 3, 8, 8), label=torch.tensor([0, 0, 1, 1]))
    out = saf_detector(data)
    losses = saf_detector.get_losses(data, out)

    assert 'overall' in losses
    assert 'loss_ce' in losses
    assert 'loss_mil' in losses
    assert 'loss_gate' in losses

    losses['overall'].backward()

    assert saf_detector.head.weight.grad is not None
    assert saf_detector.patch_head.weight.grad is not None
    assert saf_detector.fusion_alpha.grad is not None
    # Gate parameters must receive gradients from loss_gate
    for p in saf_detector.gate.parameters():
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()


def test_evaluate_saf_mil(saf_detector, tmp_path):
    """Verify test.py evaluation and metrics reporting for bias_saf_mil."""
    from training.metrics.utils import format_compact_test_report
    from training.metrics.reporting import write_metrics_csv, write_summary_csv
    import importlib.util
    spec = importlib.util.spec_from_file_location('test_eval_module', ROOT / 'training/test.py')
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)

    class MockDataset:
        data_dict = {'image': ['video_1/0.png', 'video_1/1.png', 'video_2/0.png', 'video_2/1.png']}

    class MockLoader:
        dataset = MockDataset()
        def __iter__(self):
            yield dict(image=torch.randn(4, 3, 8, 8), label=torch.tensor([0, 0, 1, 1]))

    result, arrays = test_module.evaluate(
        saf_detector.eval(), MockLoader(), torch.device('cpu'), ensemble_weight=0.5
    )

    assert 'auc' in result
    assert 'video_auc' in result
    assert result.get('learned_gate') is True
    assert result.get('is_feat_model') is True

    # Check that predictions arrays contain prob, mil_prob, feature_fusion_prob
    for k in ('prob', 'feature_fusion_prob', 'mil_prob', 'cls_only_prob'):
        assert k in arrays
        assert len(arrays[k]) == 4

    report_text = format_compact_test_report('Celeb-DF-v2', result)
    assert 'Fusion:' in report_text or 'Learned Fusion' in report_text

    # Verify CSV export
    csv_path = tmp_path / 'metrics_saf_mil.csv'
    write_metrics_csv(csv_path, result, 'bias_saf_mil', 1024, 'Celeb-DF-v2', 'ckpt.pth', 0.5)
    assert csv_path.exists()
