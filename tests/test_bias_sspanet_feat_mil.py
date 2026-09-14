from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from detectors import DETECTOR
from detectors.bias_sspanet_feat_mil_detector import (
    BiasSSPANetFeatMILDetector,
    BiasSSPANetFFMILDetector,
)

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]


class SmallBackbone(nn.Module):
    """Cheap differentiable backbone for detector tests."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, patch_size=2)
        self.patch = nn.Conv2d(3, 8, 2, 2, bias=True)
        self.layer_norm = nn.LayerNorm(8)
        self.mix = nn.Linear(8, 8, bias=True)
        self.post_layernorm = nn.LayerNorm(8)

    def forward(self, image):
        tokens = self.mix(self.layer_norm(self.patch(image).flatten(2).transpose(1, 2)))
        cls = self.post_layernorm(tokens.mean(1))
        return SimpleNamespace(pooler_output=cls, last_hidden_state=torch.cat([cls[:, None], tokens], 1))


@pytest.fixture
def feat_detector(monkeypatch):
    monkeypatch.setattr(BiasSSPANetFeatMILDetector, 'build_backbone', lambda self, config: SmallBackbone())
    torch.manual_seed(2)
    return BiasSSPANetFeatMILDetector(dict(mil_topk=3, lambda_mil=0.3, fusion_alpha_init=0.1))


def test_registry_and_aliases():
    assert DETECTOR['bias_sspanet_feat_mil'] is BiasSSPANetFeatMILDetector
    assert DETECTOR['bias_sspanet_ff_mil'] is BiasSSPANetFeatMILDetector
    assert BiasSSPANetFFMILDetector is BiasSSPANetFeatMILDetector


def test_fusion_alpha_exists_and_trainable(feat_detector):
    assert hasattr(feat_detector, 'fusion_alpha')
    assert isinstance(feat_detector.fusion_alpha, nn.Parameter)
    assert feat_detector.fusion_alpha.requires_grad
    assert torch.isclose(feat_detector.fusion_alpha.data, torch.tensor(0.1))


def test_bitfit_parameter_freezing(feat_detector):
    # Backbone weight matrices must be frozen; only biases are trained
    for name, param in feat_detector.backbone.named_parameters():
        if 'bias' in name:
            assert param.requires_grad, f"Expected {name} to be trainable"
        else:
            assert not param.requires_grad, f"Expected {name} to be frozen"

    # Detector head, patch_head, sspanet, and fusion_alpha must be trainable
    assert feat_detector.head.weight.requires_grad
    assert feat_detector.patch_head.weight.requires_grad
    for p in feat_detector.sspanet.parameters():
        assert p.requires_grad


def test_forward_output_keys_and_shapes(feat_detector):
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor([0, 1]))
    out = feat_detector(data)

    expected_keys = {'cls', 'prob', 'feat', 'feat_norm', 'cls_only_prob', 'patch_logits', 'mil_logits', 'mil_prob'}
    assert expected_keys.issubset(out.keys())

    assert out['cls'].shape == (2, 2)
    assert out['prob'].shape == (2,)
    assert out['cls_only_prob'].shape == (2,)
    assert out['mil_prob'].shape == (2,)
    assert out['patch_logits'].shape == (2, 4, 4)
    assert out['feat'].shape == (2, 8)
    assert 'fusion_alpha' in out['diagnostics']


def test_dual_supervision_gradient_flow(feat_detector):
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor([0, 1]))
    out = feat_detector(data)
    losses = feat_detector.get_losses(data, out)

    losses['overall'].backward()

    # fusion_alpha must receive gradients
    assert feat_detector.fusion_alpha.grad is not None
    assert feat_detector.fusion_alpha.grad.abs() > 0

    # Backbone biases must receive gradients; backbone weights must have no gradient
    for name, param in feat_detector.backbone.named_parameters():
        if 'bias' in name:
            assert param.grad is not None
        else:
            assert param.grad is None

    # SSPANet must receive gradients (from both CE through Mean(SSPANet) and Top-k MIL)
    for p in feat_detector.sspanet.parameters():
        assert p.grad is not None
        assert p.grad.abs().sum() > 0


def test_train_metrics(feat_detector):
    data = dict(image=torch.randn(4, 3, 8, 8), label=torch.tensor([0, 1, 0, 1]))
    out = feat_detector(data)
    metrics = feat_detector.get_train_metrics(data, out)
    for k in ('acc', 'auc', 'eer', 'ap'):
        assert k in metrics
        assert 0.0 <= metrics[k] <= 1.0
