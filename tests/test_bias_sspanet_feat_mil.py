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


def test_bdg_computation_and_bounds(feat_detector):
    from detectors.bias_sspanet_feat_mil_detector import compute_dynamic_gating
    # 1. Check default attributes on detector
    assert hasattr(feat_detector, 'gating_w_min')
    assert hasattr(feat_detector, 'gating_w_max')
    assert hasattr(feat_detector, 'gating_tau')
    assert feat_detector.gating_w_min == 0.05
    assert feat_detector.gating_w_max == 0.85
    assert feat_detector.gating_tau == 0.15

    # 2. Check forward outputs for BDG
    data = dict(image=torch.randn(4, 3, 8, 8), label=torch.tensor([0, 0, 1, 1]))
    out = feat_detector(data)
    for k in ('gating_w_cls', 'gating_w_f', 'bdg_cls_mil_prob', 'bdg_f_mil_prob'):
        assert k in out, f"Missing key {k} in forward output"
        assert out[k].shape == (4,)
        if 'gating_w' in k:
            assert (out[k] >= feat_detector.gating_w_min).all()
            assert (out[k] <= feat_detector.gating_w_max).all()

    # Check BDG formulas
    exp_cls = (1.0 - out['gating_w_cls']) * out['cls_only_prob'] + out['gating_w_cls'] * out['mil_prob']
    exp_f = (1.0 - out['gating_w_f']) * out['prob'] + out['gating_w_f'] * out['mil_prob']
    torch.testing.assert_close(out['bdg_cls_mil_prob'], exp_cls)
    torch.testing.assert_close(out['bdg_f_mil_prob'], exp_f)

    # 3. Flat noise behavior (mil_conf -> 0 => w drops near w_min)
    flat_patch_logits = torch.zeros(2, 4, 4)
    confident_base = torch.tensor([0.05, 0.95])
    w_noisy, base_conf, mil_conf = compute_dynamic_gating(confident_base, flat_patch_logits)
    assert (mil_conf < 0.05).all()
    assert (w_noisy < 0.15).all()

    # 4. Sharp localized anomaly with confused base => w leaps near w_max
    sharp_patch_logits = torch.full((1, 4, 4), -8.0)
    sharp_patch_logits[0, 1, 1] = 8.0
    confused_base = torch.tensor([0.50])
    w_sharp, base_conf2, mil_conf2 = compute_dynamic_gating(confused_base, sharp_patch_logits)
    assert mil_conf2.item() > 0.4
    assert w_sharp.item() > 0.70

    # 5. Test numpy array compatibility and extreme numerical stability
    import numpy as np
    w_np, bc_np, mc_np = compute_dynamic_gating(confident_base.numpy(), flat_patch_logits.numpy())
    assert isinstance(w_np, np.ndarray)
    assert (w_np < 0.15).all()

    # Extreme logits in numpy should not trigger overflow warnings or NaNs
    extreme_logits = np.array([[-500.0, 500.0, -1000.0, 1000.0]])
    w_ext, bc_ext, mc_ext = compute_dynamic_gating([0.5], extreme_logits)
    assert np.isfinite(w_ext).all()
    assert 0.05 <= w_ext[0] <= 0.85


def test_evaluate_seven_ablation_branches(feat_detector, tmp_path):
    import csv
    import numpy as np
    from metrics.utils import format_compact_test_report
    from metrics.reporting import write_metrics_csv, write_summary_csv
    import importlib.util
    spec = importlib.util.spec_from_file_location('test_eval_module', ROOT / 'training/test.py')
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)

    class MockDataset:
        data_dict = {'image': ['video_1/0.png', 'video_1/1.png', 'video_2/0.png', 'video_2/1.png']}

    class MockLoader:
        dataset = MockDataset()
        def __iter__(self):
            # Batch of 4 samples: 2 real, 2 fake across 2 videos
            yield dict(
                image=torch.randn(4, 3, 8, 8),
                label=torch.tensor([0, 0, 1, 1])
            )

    result, arrays = test_module.evaluate(
        feat_detector.eval(), MockLoader(), torch.device('cpu'), ensemble_weight=0.5
    )

    # Verify all 7 ablation variants are present in result
    expected_branches = [
        'bdg_cls_mil',
        'bdg_f_mil',
        'ens_cls_mil',
        'ens_f_mil',
        'feature_fusion',
        'mil',
        'cls_only',
    ]
    for branch in expected_branches:
        assert f'{branch}_auc' in result, f"Missing frame auc for {branch}"
        assert f'{branch}_video_auc' in result, f"Missing video auc for {branch}"
        assert f'{branch}_eer' in result, f"Missing frame eer for {branch}"
        assert f'{branch}_video_eer' in result, f"Missing video eer for {branch}"
        assert f'{branch}_acc' in result, f"Missing frame acc for {branch}"
        assert f'{branch}_video_acc' in result, f"Missing video acc for {branch}"

    # Verify arrays contain all probability predictions
    for prob_key in ('bdg_cls_mil_prob', 'bdg_f_mil_prob', 'ens_cls_mil_prob', 'ens_f_mil_prob',
                     'feature_fusion_prob', 'mil_prob', 'cls_only_prob', 'prob'):
        assert prob_key in arrays
        assert len(arrays[prob_key]) == 4

    # Verify CSV export contains rows for all 7 branches at frame & video levels (14 rows total)
    csv_path = tmp_path / 'metrics_7_branches.csv'
    write_metrics_csv(csv_path, result, 'bias_sspanet_feat_mil', 1024, 'Celeb-DF-v2', 'ckpt.pth', 0.5)
    with csv_path.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 14
    branch_levels = [(r['branch'], r['level']) for r in rows]
    for b in ('bdg_cls_mil', 'bdg_f_mil', 'ens_cls_mil', 'ens_f_mil', 'feature_fusion', 'mil', 'cls'):
        assert (b, 'frame') in branch_levels
        assert (b, 'video') in branch_levels

    # Verify ensemble_weight only present for ensemble branches
    for r in rows:
        if r['branch'] in ('ens_cls_mil', 'ens_f_mil'):
            assert r['ensemble_weight'] == '0.5'
        else:
            assert r['ensemble_weight'] == ''

    # Verify write_summary_csv includes dataset rows + AVERAGE rows for all 7 branches
    summary_csv = tmp_path / 'summary_7_branches.csv'
    dataset_results = [('Celeb-DF-v2', result)]
    write_summary_csv(summary_csv, dataset_results, 'bias_sspanet_feat_mil', 1024, 'ckpt.pth', 0.5)
    with summary_csv.open(encoding='utf-8-sig') as f:
        sum_rows = list(csv.DictReader(f))
    assert len(sum_rows) == 28  # 14 for Celeb-DF-v2 + 14 for AVERAGE
    avg_branches = [r['branch'] for r in sum_rows if r['dataset'] == 'AVERAGE']
    assert set(avg_branches) == {'bdg_cls_mil', 'bdg_f_mil', 'ens_cls_mil', 'ens_f_mil', 'feature_fusion', 'mil', 'cls'}

    # Verify format_compact_test_report displays Leaderboard with all 7 branches
    report_text = format_compact_test_report('Celeb-DF-v2', result)
    assert '[Ablation Leaderboard - Ranked by Video AUC]:' in report_text
    assert 'Adaptive BDG (CLS + MIL)' in report_text
    assert 'Adaptive BDG (F + MIL)' in report_text
    assert 'Ensemble 50/50 (CLS + MIL)' in report_text
    assert 'Ensemble 50/50 (F + MIL)' in report_text
    assert 'Feature Fusion (F)' in report_text
    assert 'MIL (Patch Head only)' in report_text
    assert 'CLS only' in report_text
    assert 'Rank' in report_text and 'Video AUC' in report_text
    assert 'Gating Behavior' in report_text
