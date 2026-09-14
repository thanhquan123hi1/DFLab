"""Unit and contract tests for CAMILDetector (Cross-Attention Multiple Instance Learning)."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
import yaml

from detectors.camil_detector import CAMILDetector, BiasCAMILDetector, topk_mil_logits
from detectors.modules.sspanet import ATTN_Block
from metrics.registry import DETECTOR

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]


class SmallBackbone(nn.Module):
    """Cheap differentiable backbone for fast detector unit tests."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, patch_size=2)
        self.patch = nn.Conv2d(3, 8, 2, 2)
        self.layer_norm = nn.LayerNorm(8)
        self.mix = nn.Linear(8, 8)
        self.post_layernorm = nn.LayerNorm(8)

    def forward(self, image):
        tokens = self.mix(self.layer_norm(self.patch(image).flatten(2).transpose(1, 2)))
        cls = self.post_layernorm(tokens.mean(1))
        return SimpleNamespace(pooler_output=cls, last_hidden_state=torch.cat([cls[:, None], tokens], 1))


@pytest.fixture
def camil_detector(monkeypatch):
    monkeypatch.setattr(CAMILDetector, 'build_backbone', lambda self, config: SmallBackbone())
    torch.manual_seed(42)
    return CAMILDetector(dict(
        mil_topk=4,
        lambda_mil=0.3,
        cross_attn_heads=2,
        fusion_gamma_init=0.0
    ))


def test_registry_lookup():
    """Verify that camil and bias_camil are properly registered and accessible."""
    import detectors
    import detectors.camil_detector as camil_mod

    assert DETECTOR['camil'] is CAMILDetector
    assert DETECTOR['bias_camil'] is CAMILDetector
    assert detectors.CAMILDetector is CAMILDetector
    assert detectors.BiasCAMILDetector is CAMILDetector
    assert camil_mod.CAMILDetector is CAMILDetector
    assert camil_mod.BiasCAMILDetector is CAMILDetector
    assert BiasCAMILDetector is CAMILDetector


def test_camil_initialization_and_trainable_counts(camil_detector):
    """Verify trainable parameter groups, freezing contract, and count consistency."""
    counts = camil_detector.trainable_counts
    expected_groups = {
        'backbone_bias', 'head', 'sspanet', 'patch_head',
        'norm_cls', 'norm_patch', 'cross_attn', 'fusion_gamma'
    }
    assert expected_groups.issubset(counts.keys())
    assert counts['backbone_bias'] == 32  # 4 layers * 8 bias params each in SmallBackbone
    assert counts['fusion_gamma'] == 1

    # Check backbone parameter freezing: only biases are trainable
    for name, param in camil_detector.backbone.named_parameters():
        if 'bias' in name:
            assert param.requires_grad is True, f"Expected backbone bias {name} to be trainable"
        else:
            assert param.requires_grad is False, f"Expected backbone weight {name} to be frozen"

    # Check total trainable count consistency
    actual_trainable = sum(p.numel() for p in camil_detector.parameters() if p.requires_grad)
    assert actual_trainable == sum(counts.values())


def test_camil_forward_shapes(camil_detector):
    """Verify forward pass output keys and tensor shapes."""
    # SmallBackbone: image 8x8, patch_size 2 -> 4x4 grid = 16 patches
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor([0, 1]))
    out = camil_detector(data)

    assert out['cls'].shape == (2, 2)
    assert out['prob'].shape == (2,)
    assert out['mil_prob'].shape == (2,)
    assert out['cls_only_prob'].shape == (2,)
    assert out['feat'].shape == (2, 8)
    assert out['feat_norm'].shape == (2, 8)
    assert out['patch_logits'].shape == (2, 4, 4)
    assert out['mil_logits'].shape == (2,)
    assert 'attn_weights' in out
    assert out['attn_weights'].shape == (2, 1, 16)

    # Attention weights over key patches must sum to 1.0 per query token
    torch.testing.assert_close(
        out['attn_weights'].sum(dim=-1),
        torch.ones(2, 1),
        atol=1e-5, rtol=1e-5
    )


def test_camil_epoch_zero_residual_contract(camil_detector):
    """Verify that when fusion_gamma == 0.0, fused feat equals cls and prob equals cls_only_prob."""
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor([0, 1]))
    out = camil_detector(data)

    # When fusion_gamma == 0, fused feature is exactly cls
    outputs = camil_detector.backbone(data['image'])
    cls_expected = outputs.pooler_output
    torch.testing.assert_close(out['feat'], cls_expected)

    # Classification probabilities must match cls_only_prob
    torch.testing.assert_close(out['prob'], out['cls_only_prob'])
    assert out['diagnostics']['fusion_gamma'].item() == 0.0


def test_camil_diagnostics_dictionary(camil_detector):
    """Verify diagnostic dictionary keys, shapes (0-dim scalars), and finite values."""
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor([0, 1]))
    out = camil_detector(data)
    diag = out['diagnostics']

    required_keys = [
        'fusion_gamma',
        'patch_probability_mean',
        'patch_probability_std',
        'patch_entropy',
        'sspa_relative_change',
        'branch_disagreement',
    ]
    for k in required_keys:
        assert k in diag, f"Missing diagnostic key: {k}"
        val = diag[k]
        assert val.ndim == 0, f"Diagnostic {k} must be scalar (0-dim), got shape {val.shape}"
        assert torch.isfinite(val), f"Diagnostic {k} must be finite, got {val}"


def test_camil_losses_calculation(camil_detector):
    """Verify CE + MIL loss calculation and diagnostic metrics propagation."""
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor([0, 1]))
    out = camil_detector(data)
    losses = camil_detector.get_losses(data, out)

    expected_overall = losses['loss_ce'] + 0.3 * losses['loss_mil']
    torch.testing.assert_close(losses['overall'], expected_overall)
    assert 'real_loss' in losses
    assert 'fake_loss' in losses
    assert 'real_prob' in losses
    assert 'fake_prob' in losses
    assert 'real_mil_prob' in losses
    assert 'fake_mil_prob' in losses
    assert 'fusion_gamma' in losses


@pytest.mark.parametrize('labels', [[0, 1], [0, 0], [1, 1], [0, 4]])
def test_camil_backward_gradients_zero_init(camil_detector, labels):
    """Verify gradient flow at step 0 (fusion_gamma == 0.0)."""
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor(labels))
    out = camil_detector(data)
    losses = camil_detector.get_losses(data, out)
    losses['overall'].backward()

    # fusion_gamma receives non-zero gradient from dL/d(fused) * attn_out
    assert camil_detector.fusion_gamma.grad is not None
    assert camil_detector.fusion_gamma.grad.abs().sum() > 0

    # Modules receiving direct gradients
    for name, group in [('head', camil_detector.head),
                        ('patch_head', camil_detector.patch_head),
                        ('sspanet', camil_detector.sspanet)]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in group.parameters()), \
            f"Expected non-zero gradient in {name}"

    # Attention branch parameters exist in computation graph with grad tensor initialized
    assert camil_detector.cross_attn.in_proj_weight.grad is not None
    assert camil_detector.norm_cls.weight.grad is not None
    assert camil_detector.norm_patch.weight.grad is not None

    # Backbone: biases receive gradient, weights are frozen with no grad
    for name, param in camil_detector.backbone.named_parameters():
        if 'bias' in name:
            assert param.grad is not None and param.grad.abs().sum() > 0, \
                f"Expected gradient for backbone bias {name}"
        else:
            assert param.grad is None, f"Expected no gradient for backbone weight {name}"


def test_camil_backward_gradients_active_gamma(monkeypatch):
    """Verify full gradient flow across all modules when gamma > 0."""
    monkeypatch.setattr(CAMILDetector, 'build_backbone', lambda self, config: SmallBackbone())
    torch.manual_seed(42)
    detector = CAMILDetector(dict(
        mil_topk=4,
        lambda_mil=0.3,
        cross_attn_heads=2,
        fusion_gamma_init=0.1
    ))
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor([0, 1]))
    out = detector(data)
    losses = detector.get_losses(data, out)
    losses['overall'].backward()

    # cross_attn parameters must receive non-zero gradient
    assert detector.cross_attn.in_proj_weight.grad is not None
    assert detector.cross_attn.in_proj_weight.grad.abs().sum() > 0
    assert detector.cross_attn.out_proj.weight.grad is not None
    assert detector.cross_attn.out_proj.weight.grad.abs().sum() > 0

    # LayerNorms in fusion branch must receive non-zero gradient
    assert detector.norm_cls.weight.grad is not None
    assert detector.norm_cls.weight.grad.abs().sum() > 0
    assert detector.norm_patch.weight.grad is not None
    assert detector.norm_patch.weight.grad.abs().sum() > 0

    # fusion_gamma receives non-zero gradient
    assert detector.fusion_gamma.grad is not None
    assert detector.fusion_gamma.grad.abs().sum() > 0

    # Head and patch_head receive non-zero gradient
    assert detector.head.weight.grad is not None and detector.head.weight.grad.abs().sum() > 0
    assert detector.patch_head.weight.grad is not None and detector.patch_head.weight.grad.abs().sum() > 0

    # SSPANet receives non-zero gradient
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in detector.sspanet.parameters())

    # Backbone biases receive non-zero gradient, weights are frozen
    for name, param in detector.backbone.named_parameters():
        if 'bias' in name:
            assert param.grad is not None and param.grad.abs().sum() > 0
        else:
            assert param.grad is None


def test_camil_ablation_without_patch(monkeypatch):
    """Verify detector behaves properly when use_patch=False."""
    monkeypatch.setattr(CAMILDetector, 'build_backbone', lambda self, config: SmallBackbone())
    model = CAMILDetector(dict(use_patch=False, use_sspanet=False, lambda_mil=0))
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor([0, 1]))
    out = model(data)
    assert 'attn_weights' not in out
    assert 'patch_logits' not in out
    losses = model.get_losses(data, out)
    assert losses['loss_mil'] == 0

    with pytest.raises(ValueError, match='lambda_mil must be nonnegative and zero when use_patch=false'):
        CAMILDetector(dict(use_patch=False, lambda_mil=0.3))


def test_camil_mil_weight_zero_freezes_patch_head(monkeypatch):
    """Verify that setting lambda_mil=0 freezes patch_head while keeping cross_attn trainable."""
    monkeypatch.setattr(CAMILDetector, 'build_backbone', lambda self, config: SmallBackbone())
    model = CAMILDetector(dict(mil_topk=4, lambda_mil=0.0, use_patch=True, cross_attn_heads=2))
    assert model.patch_head.weight.requires_grad is False
    assert model.patch_head.bias.requires_grad is False
    assert 'patch_head' not in model.trainable_counts
    assert 'cross_attn' in model.trainable_counts
    assert 'backbone_bias' in model.trainable_counts


def test_camil_eval_mode_consistency(camil_detector):
    """Verify that eval mode produces identical outputs with or without inference flag."""
    camil_detector.eval()
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor([0, 1]))
    with torch.no_grad():
        out1 = camil_detector(data)
        out2 = camil_detector(data, inference=True)
        torch.testing.assert_close(out1['prob'], out2['prob'])


def test_camil_yaml_config_contract(monkeypatch):
    """Verify that training/config/detector/camil.yaml loads and instantiates CAMILDetector."""
    monkeypatch.setattr(CAMILDetector, 'build_backbone', lambda self, cfg: SmallBackbone())
    config_path = ROOT / 'training/config/detector/camil.yaml'
    assert config_path.exists()
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    assert cfg['model_name'] == 'camil'
    assert cfg['use_patch'] is True
    assert cfg['use_sspanet'] is True
    assert cfg['cross_attn_heads'] == 4
    assert cfg['cross_attn_dropout'] == 0.0
    assert cfg['fusion_gamma_init'] == 0.0
    assert cfg['lambda_mil'] == 0.3
    assert cfg['mil_topk'] == 16
    assert cfg['optimizer']['adam']['lr'] == 0.0003
    assert cfg['lr_T_max'] == 15
    assert cfg['label_smoothing'] == 0.1
    assert cfg['weight_real'] == 1.0
    assert cfg['weight_fake'] == 1.0
    assert cfg['grad_clip_norm'] == 5.0
    assert cfg['save_ckpt'] is True
    assert cfg['save_feat'] is True
    assert cfg['train_dataset'] == ['FaceForensics++']
    assert cfg['test_dataset'] == ['Celeb-DF-v2', 'FaceShifter', 'DeeperForensics-1.0']

    # Verify instantiation with the actual YAML config (with SmallBackbone)
    cfg_copy = dict(cfg, cross_attn_heads=2)  # SmallBackbone hidden_size=8, so 2 heads
    model = CAMILDetector(cfg_copy)
    assert model.mil_weight == 0.3
    assert model.mil_topk == 16
    assert model.fusion_gamma.item() == 0.0


def test_camil_huggingface_clip_contract(monkeypatch):
    """Verify contract with real transformers CLIPVisionModel architecture."""
    transformers = pytest.importorskip('transformers')
    config = transformers.CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=28,
        patch_size=7,
    )
    backbone = transformers.CLIPVisionModel(config).vision_model
    monkeypatch.setattr(CAMILDetector, 'build_backbone', lambda self, cfg: backbone)

    model = CAMILDetector(dict(
        mil_topk=4,
        cross_attn_heads=2,
        fusion_gamma_init=0.1
    ))
    data = dict(image=torch.randn(2, 3, 28, 28), label=torch.tensor([0, 1]))
    out = model(data)
    # 28 / 7 = 4x4 = 16 patches
    assert out['feat'].shape == (2, 16)
    assert out['patch_logits'].shape == (2, 4, 4)
    assert out['attn_weights'].shape == (2, 1, 16)

    model.get_losses(data, out)['overall'].backward()

    # BitFit contract: LayerNorm biases trainable, weights frozen
    assert backbone.encoder.layers[0].layer_norm1.weight.requires_grad is False
    assert backbone.encoder.layers[0].layer_norm1.weight.grad is None
    assert backbone.encoder.layers[0].layer_norm1.bias.requires_grad is True
    assert backbone.encoder.layers[0].layer_norm1.bias.grad.abs().sum() > 0

    # Self-attention biases trainable, weights frozen
    assert backbone.encoder.layers[0].self_attn.q_proj.weight.requires_grad is False
    assert backbone.encoder.layers[0].self_attn.q_proj.weight.grad is None
    assert backbone.encoder.layers[0].self_attn.q_proj.bias.requires_grad is True
    assert backbone.encoder.layers[0].self_attn.q_proj.bias.grad.abs().sum() > 0

    # MLP biases trainable, weights frozen
    assert backbone.encoder.layers[0].mlp.fc1.weight.requires_grad is False
    assert backbone.encoder.layers[0].mlp.fc1.weight.grad is None
    assert backbone.encoder.layers[0].mlp.fc1.bias.requires_grad is True
    assert backbone.encoder.layers[0].mlp.fc1.bias.grad.abs().sum() > 0


def test_camil_trainer_smoke(camil_detector, tmp_path):
    """Verify Trainer train_step records gradient norms for all CAMIL parameter groups, loads checkpoint, and evaluates."""
    from trainer.trainer import Trainer
    from training.test import evaluate
    import json

    class DummyDataset(torch.utils.data.Dataset):
        data_dict = {'image': ['real/a/0.png', 'fake/a/0.png']}
        def __len__(self):
            return 2
        def __getitem__(self, i):
            return dict(image=torch.ones(3, 8, 8) * (i + 1), label=torch.tensor(i % 2))

    loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
    config = dict(
        cuda=False,
        ddp=False,
        model_name='camil',
        log_dir=str(tmp_path),
        optimizer={'type': 'adam'},
        selection_dataset='source',
        validation_split='val',
        log_interval=1,
        save_ckpt=True,
        grad_clip_norm=5.0
    )
    optimizer = torch.optim.Adam([p for p in camil_detector.parameters() if p.requires_grad], lr=0.001)
    import logging
    trainer = Trainer(config, camil_detector, optimizer, None, logging.getLogger('test_trainer'), time_now='smoke')

    trainer.train_epoch(0, loader, {'source': loader})
    checkpoint = tmp_path / 'camil_smoke/validation/source/ckpt_best.pth'
    assert checkpoint.exists()
    state = torch.load(checkpoint, map_location='cpu')
    assert state.get('architecture') == 'camil_v1'

    # Strict state_dict reload into fresh CAMILDetector instance
    camil_detector.load_state_dict(state['state_dict'], strict=True)

    # End-to-end evaluation smoke test via test.py evaluate()
    result, arrays = evaluate(camil_detector.eval(), loader, torch.device('cpu'), max_samples=2, patch_limit=2)
    assert result['n'] == 2 and len(arrays['image_names']) == 2
    assert arrays['patch_prob'].shape == (2, 4, 4)
    assert arrays['attn_weights'].shape == (2, 1, 16)

    # Verify all parameter groups logged in history.jsonl
    train_log = tmp_path / 'camil_smoke/history.jsonl'
    assert train_log.exists()
    lines = [json.loads(line) for line in train_log.read_text().strip().split('\n')]
    assert len(lines) > 0
    first_step = lines[0]
    for key in ('grad_backbone_bias', 'grad_fusion_gamma', 'grad_sspanet', 'grad_head',
                'grad_patch_head', 'grad_cross_attn', 'grad_norm_cls', 'grad_norm_patch'):
        assert key in first_step, f"Missing {key} in history.jsonl"

    # Verify run.json
    trainable_json = tmp_path / 'camil_smoke/run.json'
    assert trainable_json.exists()
    t_data = json.loads(trainable_json.read_text())['trainable_parameters']
    assert 'backbone_bias' in t_data
    assert 'cross_attn' in t_data
    assert 'norm_cls' in t_data
    assert 'norm_patch' in t_data
    assert 'fusion_gamma' in t_data

    for writer in trainer.writers.values():
        writer.close()


def test_camil_batch_size_one(camil_detector):
    """Verify forward and loss calculation on batch size 1 (edge case)."""
    data = dict(image=torch.randn(1, 3, 8, 8), label=torch.tensor([1]))
    out = camil_detector(data)
    assert out['cls'].shape == (1, 2)
    assert out['prob'].shape == (1,)
    assert out['mil_prob'].shape == (1,)
    assert out['feat'].shape == (1, 8)
    assert out['patch_logits'].shape == (1, 4, 4)
    assert out['attn_weights'].shape == (1, 1, 16)
    losses = camil_detector.get_losses(data, out)
    assert torch.isfinite(losses['overall'])


def test_camil_batch_size_one_backward(monkeypatch):
    """Verify backward autograd pass succeeds on batch size 1."""
    monkeypatch.setattr(CAMILDetector, 'build_backbone', lambda self, config: SmallBackbone())
    model = CAMILDetector(dict(mil_topk=4, lambda_mil=0.3, cross_attn_heads=2, fusion_gamma_init=0.1))
    data = dict(image=torch.randn(1, 3, 8, 8), label=torch.tensor([1]))
    out = model(data)
    losses = model.get_losses(data, out)
    losses['overall'].backward()
    assert model.fusion_gamma.grad is not None
    assert torch.isfinite(model.fusion_gamma.grad)
    assert model.cross_attn.in_proj_weight.grad is not None
    assert torch.isfinite(model.cross_attn.in_proj_weight.grad).all()


def test_camil_fusion_gamma_learnable_step(camil_detector):
    """Verify that an optimizer step actually updates fusion_gamma away from 0.0."""
    optimizer = torch.optim.Adam([p for p in camil_detector.parameters() if p.requires_grad], lr=0.01)
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor([0, 1]))
    gamma_before = camil_detector.fusion_gamma.item()
    assert gamma_before == 0.0

    out = camil_detector(data)
    losses = camil_detector.get_losses(data, out)
    losses['overall'].backward()
    optimizer.step()

    gamma_after = camil_detector.fusion_gamma.item()
    assert gamma_after != 0.0, f"Expected fusion_gamma to update from 0.0, got {gamma_after}"


def test_camil_invalid_input_shape(camil_detector):
    """Verify that non-4D inputs raise ValueError."""
    with pytest.raises(ValueError, match='CAMILDetector expects'):
        camil_detector(dict(image=torch.randn(2, 5, 3, 8, 8)))


def test_camil_grid_mismatch(monkeypatch):
    """Verify that images with dimensions not matching patch grid raise ValueError."""
    class MismatchBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8, patch_size=2)
        def forward(self, image):
            # Returns 16 tokens regardless of input size
            cls = torch.randn(image.shape[0], 8)
            tokens = torch.randn(image.shape[0], 16, 8)
            return SimpleNamespace(pooler_output=cls, last_hidden_state=torch.cat([cls[:, None], tokens], 1))

    monkeypatch.setattr(CAMILDetector, 'build_backbone', lambda self, cfg: MismatchBackbone())
    model = CAMILDetector(dict(mil_topk=4, cross_attn_heads=2))
    # 6x6 image with patch_size 2 -> h*w = 3*3 = 9 != 16 tokens
    with pytest.raises(ValueError, match='Patch token count does not match image grid'):
        model(dict(image=torch.randn(2, 3, 6, 6)))


def test_camil_invalid_cross_attn_heads(monkeypatch):
    """Verify that an embed_dim not divisible by num_heads raises an exception."""
    monkeypatch.setattr(CAMILDetector, 'build_backbone', lambda self, cfg: SmallBackbone())
    with pytest.raises(ValueError):
        CAMILDetector(dict(mil_topk=4, cross_attn_heads=3))  # 8 is not divisible by 3


def test_bias_camil_alias_equivalence(monkeypatch):
    """Verify BiasCAMILDetector alias behaves identically to CAMILDetector."""
    monkeypatch.setattr(BiasCAMILDetector, 'build_backbone', lambda self, cfg: SmallBackbone())
    torch.manual_seed(42)
    model = BiasCAMILDetector(dict(mil_topk=4, lambda_mil=0.3, cross_attn_heads=2, fusion_gamma_init=0.0))
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor([0, 1]))
    out = model(data)
    assert 'attn_weights' in out
    assert 'mil_prob' in out
    losses = model.get_losses(data, out)
    losses['overall'].backward()
    assert model.fusion_gamma.grad is not None

