import io
import torch
import pytest
from detectors.bias_gmil_detector import BiasGMILDetector
from test_ln_sspanet_mil import SmallBackbone


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(BiasGMILDetector, 'build_backbone', lambda self, config: SmallBackbone())
    torch.manual_seed(17)
    return BiasGMILDetector(dict(model_name='bias_gmil', mil_topk=3, lambda_mil=.3,
                                 gate_hidden_dim=7, lambda_fusion=1.0))


def test_initial_fusion_gradients_and_learning(model):
    batch = dict(image=torch.randn(6, 3, 8, 8), label=torch.tensor([0, 1, 0, 1, 0, 1]))
    out = model(batch)
    torch.testing.assert_close(out['gating_w'], torch.full((6,), .5))
    torch.testing.assert_close(out['prob'], .5 * (out['cls_only_prob'] + out['mil_prob']))
    model.get_losses(batch, out)['loss_fusion'].backward()
    for parameter in (model.gate[-1].weight, model.head.weight, model.patch_head.weight,
                      model.backbone.mix.bias):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert model.backbone.mix.weight.grad is None
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    optimizer.step()
    optimizer.zero_grad()
    updated = model(batch)
    assert not torch.allclose(updated['gating_w'], out['gating_w'])
    model.get_losses(batch, updated)['loss_fusion'].backward()
    assert model.gate[0].weight.grad.abs().sum() > 0


def test_checkpoint_roundtrip(model):
    model.eval()
    with torch.no_grad():
        model.gate[-1].weight.fill_(.1)
    batch = dict(image=torch.randn(3, 3, 8, 8))
    stream = io.BytesIO()
    torch.save(dict(config=model.config, state_dict=model.state_dict()), stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True)
    restored = BiasGMILDetector(checkpoint['config']).eval()
    restored.load_state_dict(checkpoint['state_dict'], strict=True)
    for key in ('prob', 'cls_only_prob', 'mil_prob', 'gating_w'):
        torch.testing.assert_close(model(batch)[key], restored(batch)[key])


def test_all_patches_and_detached_features(model):
    model.mil_topk = 16
    cls_logits = torch.randn(2, 2, requires_grad=True)
    patches = torch.randn(2, 4, 4, requires_grad=True)
    mil = patches.flatten(1).mean(1)
    w, _, _ = model._fusion_gate(cls_logits, cls_logits.softmax(1)[:, 1], patches, mil)
    w.sum().backward()
    assert cls_logits.grad is None and patches.grad is None
    assert torch.isfinite(w).all()


def test_fusion_loss_routes_toward_correct_branch(model):
    # For a fake example, MIL=.9 is better than CLS=.3: gradient descent
    # must increase MIL weight. For a real example it must decrease it.
    for label, expected_sign in ((1, -1), (0, 1)):
        gate_logit = torch.tensor(0., requires_grad=True)
        weight = gate_logit.sigmoid()
        predictions = dict(cls=torch.tensor([[0., 0.]]),
                           mil_logits=torch.tensor([0.]),
                           prob=(.3 + .6 * weight).reshape(1))
        loss = model.get_losses({'label': torch.tensor([label])}, predictions)['loss_fusion']
        loss.backward()
        assert gate_logit.grad.item() * expected_sign > 0


def test_config_scheduler_matches_training_length():
    import yaml
    from pathlib import Path
    path = Path(__file__).parents[1] / 'training/config/detector/bias_gmil.yaml'
    config = yaml.safe_load(path.read_text())
    assert config['metric_scoring'] == 'video_auc'
    assert config['lr_T_max'] == config['nEpochs'] - config['start_epoch'] == 10
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=config['optimizer']['adam']['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['lr_T_max'], eta_min=config['lr_eta_min'])
    for _ in range(config['nEpochs']):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(config['lr_eta_min'])


def test_evaluation_four_outputs(model):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    from metrics.utils import format_compact_test_report
    spec = importlib.util.spec_from_file_location('gmil_evaluation', Path(__file__).parents[1] / 'training/test.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Loader:
        dataset = SimpleNamespace(data_dict={'image': ['real/a.png', 'real/b.png', 'fake/a.png', 'fake/b.png']})
        def __iter__(self):
            yield dict(image=torch.randn(4, 3, 8, 8), label=torch.tensor([0, 0, 1, 1]))
    result, arrays = module.evaluate(model.eval(), Loader(), torch.device('cpu'))
    assert result['video_n'] == 2
    for prefix in ('', 'cls_only_', 'mil_', 'ensemble_'):
        assert prefix + 'video_auc' in result
    assert arrays['gating_w'].shape == (4,)
    text = format_compact_test_report('tiny', result)
    assert all(label in text for label in ('Fusion:', 'CLS:', 'MIL:', 'Ensemble:'))
    assert 'Adaptive BDG' not in text
    assert 'Fusion' in text
    assert '[Ablation Leaderboard - Ranked by Video AUC]:' in text

    from metrics.reporting import _get_metric_rows
    rows = _get_metric_rows(result, 'bias_gmil', 1024, 'tiny', 'ckpt.pth', 0.5)
    branches = set(r['branch'] for r in rows)
    assert 'learned_fusion' in branches
    assert 'ensemble' in branches
    assert 'mil' in branches
    assert 'cls' in branches
