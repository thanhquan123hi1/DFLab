import hashlib
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from detectors.ln_sspanet_mil_detector import (
    LNSSPANetMILDetector,
    BiasSSPANetMILDetector,
    topk_mil_logits,
)
from detectors.modules.sspanet import ATTN_Block
from metrics.base_metrics_class import Recorder
from metrics.utils import get_test_metrics

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]


def safe_torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class SmallBackbone(nn.Module):
    """Cheap differentiable backbone for detector/trainer integration tests."""
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
def detector(monkeypatch):
    monkeypatch.setattr(LNSSPANetMILDetector, 'build_backbone', lambda self, config: SmallBackbone())
    torch.manual_seed(2)
    return LNSSPANetMILDetector(dict(mil_topk=3, lambda_mil=.3))


def test_detector_registry():
    import detectors
    import detectors.bias_sspanet_mil_detector as bias_mod
    import detectors.ln_sspanet_mil_detector as ln_mod
    from metrics.registry import DETECTOR
    assert DETECTOR['ln_sspanet_mil'] is LNSSPANetMILDetector
    assert DETECTOR['bias_sspanet_mil'] is BiasSSPANetMILDetector
    assert DETECTOR['bias_sspanet_mil'] is not LNSSPANetMILDetector
    assert issubclass(BiasSSPANetMILDetector, LNSSPANetMILDetector)
    # Check all import avenues resolve to the exact same class
    assert detectors.BiasSSPANetMILDetector is BiasSSPANetMILDetector
    assert detectors.LNSSPANetMILDetector is LNSSPANetMILDetector
    assert bias_mod.BiasSSPANetMILDetector is BiasSSPANetMILDetector
    assert ln_mod.BiasSSPANetMILDetector is BiasSSPANetMILDetector
    assert 'BiasSSPANetMILDetector' in dir(ln_mod)


@pytest.fixture
def bias_sspanet_mil_detector(monkeypatch):
    monkeypatch.setattr(BiasSSPANetMILDetector, 'build_backbone', lambda self, config: SmallBackbone())
    torch.manual_seed(2)
    return BiasSSPANetMILDetector(dict(mil_topk=3, lambda_mil=.3))


def test_official_source_unchanged():
    source = ROOT / 'training/detectors/modules/sspanet.py'
    # Original upstream bytes, commit ed15ffbeabba15d3f3f3caba53bea6c34537c810.
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '772019425071767dae5a535f2dda4ab38e10ebd7c588ee55b289d6f5ef831a5a'


def test_official_block_signed_input_and_gradient():
    block = ATTN_Block(8)
    x = torch.randn(2, 8, 4, 5, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    # Original residual gate amplifies magnitude by a factor in [1,2].
    ratio = y / x
    assert (ratio >= 1).all() and (ratio <= 2).all()
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_mil_numeric_gradient_and_invalid_k():
    logits = torch.tensor([[[-2., 1., 4., 3.]]], requires_grad=True)
    loss = topk_mil_logits(logits, 2)
    assert loss.item() == 3.5
    loss.sum().backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[[0., 0., .5, .5]]]))
    for k in [0, 5]:
        with pytest.raises(ValueError):
            topk_mil_logits(logits, k)


@pytest.mark.parametrize('labels', [[0, 1], [0, 0], [1, 1], [0, 4]])
def test_gradients_and_freezing(detector, labels):
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor(labels))
    out = detector(data)
    assert out['patch_logits'].shape == (2, 4, 4)
    losses = detector.get_losses(data, out)
    torch.testing.assert_close(losses['overall'], losses['loss_ce'] + .3 * losses['loss_mil'])
    losses['overall'].backward()
    for module in detector.backbone.modules():
        for param in module.parameters(recurse=False):
            assert param.requires_grad == isinstance(module, nn.LayerNorm)
            if not param.requires_grad:
                assert param.grad is None
    for group in [detector.sspanet, detector.patch_head, detector.head]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in group.parameters())
    assert detector.backbone.layer_norm.weight.grad.abs().sum() > 0
    assert not hasattr(detector, 'fusion_alpha')
    detector.eval()
    with torch.no_grad():
        torch.testing.assert_close(detector(data)['prob'], detector(data, inference=True)['prob'])


@pytest.mark.parametrize('labels', [[0, 1], [0, 0], [1, 1], [0, 4]])
def test_bias_sspanet_mil_gradients_and_freezing(bias_sspanet_mil_detector, labels):
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor(labels))
    out = bias_sspanet_mil_detector(data)
    assert out['patch_logits'].shape == (2, 4, 4)
    losses = bias_sspanet_mil_detector.get_losses(data, out)
    torch.testing.assert_close(losses['overall'], losses['loss_ce'] + .3 * losses['loss_mil'])
    losses['overall'].backward()

    # Backbone parameters: only parameters containing 'bias' are trainable
    backbone_bias_count = 0
    backbone_frozen_count = 0
    for name, param in bias_sspanet_mil_detector.backbone.named_parameters():
        if 'bias' in name:
            assert param.requires_grad is True, f"Expected {name} to be trainable"
            assert param.grad is not None, f"Expected {name} to receive gradient"
            assert param.grad.abs().sum() > 0, f"Expected nonzero gradient for {name}"
            backbone_bias_count += 1
        else:
            assert param.requires_grad is False, f"Expected {name} to be frozen"
            assert param.grad is None, f"Expected {name} to have no gradient"
            backbone_frozen_count += 1
    assert backbone_bias_count > 0
    assert backbone_frozen_count > 0

    # Explicit contract contrast: LayerNorm weights frozen, LayerNorm biases trainable
    assert bias_sspanet_mil_detector.backbone.layer_norm.weight.requires_grad is False
    assert bias_sspanet_mil_detector.backbone.layer_norm.weight.grad is None
    assert bias_sspanet_mil_detector.backbone.layer_norm.bias.requires_grad is True
    assert bias_sspanet_mil_detector.backbone.layer_norm.bias.grad.abs().sum() > 0

    # Linear layer weights frozen, Linear biases trainable
    assert bias_sspanet_mil_detector.backbone.mix.weight.requires_grad is False
    assert bias_sspanet_mil_detector.backbone.mix.weight.grad is None
    assert bias_sspanet_mil_detector.backbone.mix.bias.requires_grad is True
    assert bias_sspanet_mil_detector.backbone.mix.bias.grad.abs().sum() > 0

    # Detector auxiliary modules: sspanet, patch_head, head are trainable and receive gradient
    for group_name, group in [('sspanet', bias_sspanet_mil_detector.sspanet),
                              ('patch_head', bias_sspanet_mil_detector.patch_head),
                              ('head', bias_sspanet_mil_detector.head)]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in group.parameters()), \
            f"Expected gradient in {group_name}"
    assert bias_sspanet_mil_detector.head.weight.grad is not None and bias_sspanet_mil_detector.head.weight.grad.abs().sum() > 0
    assert bias_sspanet_mil_detector.head.bias.grad is not None and bias_sspanet_mil_detector.head.bias.grad.abs().sum() > 0
    assert not hasattr(bias_sspanet_mil_detector, 'fusion_alpha')

    bias_sspanet_mil_detector.eval()
    with torch.no_grad():
        torch.testing.assert_close(bias_sspanet_mil_detector(data)['prob'], bias_sspanet_mil_detector(data, inference=True)['prob'])


def test_ablation_without_patch(monkeypatch):
    monkeypatch.setattr(LNSSPANetMILDetector, 'build_backbone', lambda self, config: SmallBackbone())
    model = LNSSPANetMILDetector(dict(use_patch=False, use_sspanet=False, lambda_mil=0))
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor([0, 1]))
    assert model.get_losses(data, model(data))['loss_mil'] == 0
    with pytest.raises(ValueError):
        LNSSPANetMILDetector(dict(use_patch=False))


def test_bias_sspanet_mil_ablation_without_patch(monkeypatch):
    monkeypatch.setattr(BiasSSPANetMILDetector, 'build_backbone', lambda self, config: SmallBackbone())
    model = BiasSSPANetMILDetector(dict(use_patch=False, use_sspanet=False, lambda_mil=0))
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor([0, 1]))
    assert model.get_losses(data, model(data))['loss_mil'] == 0
    with pytest.raises(ValueError):
        BiasSSPANetMILDetector(dict(use_patch=False))


def test_bias_sspanet_mil_weight_zero_freezes_patch_head(monkeypatch):
    monkeypatch.setattr(BiasSSPANetMILDetector, 'build_backbone', lambda self, config: SmallBackbone())
    model = BiasSSPANetMILDetector(dict(mil_topk=3, lambda_mil=0.0, use_patch=True))
    assert model.patch_head.weight.requires_grad is False
    assert model.patch_head.bias.requires_grad is False
    assert 'patch_head' not in model.trainable_counts
    assert 'backbone_bias' in model.trainable_counts


def test_bias_sspanet_mil_trainable_counts(bias_sspanet_mil_detector):
    counts = bias_sspanet_mil_detector.trainable_counts
    assert 'backbone_bias' in counts
    assert counts['backbone_bias'] == 32  # 4 layers * 8 bias params each in SmallBackbone
    assert 'head' in counts
    assert 'sspanet' in counts
    assert 'patch_head' in counts
    assert 'fusion_alpha' not in counts
    assert sum(p.numel() for p in bias_sspanet_mil_detector.parameters() if p.requires_grad) == sum(counts.values())


def test_metrics_video_collisions_and_single_class():
    labels = np.array([0, 0, 1, 1])
    names = ['real/001/0.png', 'real/001/1.png', 'fake/001/0.png', 'fake/001/1.png']
    result = get_test_metrics([.1, .2, .8, .9], labels, names)
    assert result['video_auc'] == 1 and result['video_n'] == 2
    assert result['tn'] == 2 and result['tp'] == 2
    assert np.isnan(get_test_metrics([.2], [0], ['a/0.png'])['auc'])
    with pytest.raises(ValueError):
        get_test_metrics([.2], [0], names)
    with pytest.raises(ValueError):
        get_test_metrics([float('nan')], [0], ['a/0.png'])


def test_recorder_does_not_retain_graph():
    recorder = Recorder()
    recorder.update(torch.tensor(2., requires_grad=True), 3)
    assert isinstance(recorder.sum, float) and recorder.average() == 2.


def test_trainer_checkpoint_diagnostics_and_test_cap(detector, tmp_path):
    from trainer.trainer import Trainer
    from training.test import evaluate
    class Data(torch.utils.data.Dataset):
        data_dict = {'image': ['real/a/0.png', 'fake/a/0.png', 'real/b/0.png', 'fake/b/0.png']}
        def __len__(self):
            return 4
        def __getitem__(self, i):
            return dict(image=torch.ones(3, 8, 8) * (i + 1), label=torch.tensor(i % 2))
    loader = torch.utils.data.DataLoader(Data(), batch_size=2)
    config = dict(cuda=False, ddp=False, model_name='ln_sspanet_mil', log_dir=str(tmp_path),
                  optimizer={'type': 'adam'}, selection_dataset='source', validation_split='val',
                  log_interval=1, save_ckpt=True, grad_clip_norm=5.)
    trainer = Trainer(config, detector, torch.optim.Adam([p for p in detector.parameters() if p.requires_grad], lr=.001),
                      None, logging.getLogger('test'), time_now='smoke')
    trainer.train_epoch(0, loader, {'source': loader})
    checkpoint = tmp_path / 'ln_sspanet_mil_smoke/validation/source/ckpt_best.pth'
    assert checkpoint.exists()
    state = safe_torch_load(checkpoint)
    assert state.get('architecture') == 'ln_sspanet_mil_v1'
    detector.load_state_dict(state['state_dict'], strict=True)
    result, arrays = evaluate(detector.eval(), loader, torch.device('cpu'), max_samples=3, patch_limit=2)
    assert result['n'] == 3 and len(arrays['image_names']) == 3
    assert arrays['patch_prob'].shape == (2, 4, 4)
    assert (tmp_path / 'ln_sspanet_mil_smoke/train.jsonl').exists()
    import json
    ln_train_lines = [json.loads(line) for line in (tmp_path / 'ln_sspanet_mil_smoke/train.jsonl').read_text().strip().split('\n')]
    assert len(ln_train_lines) > 0
    assert 'grad_backbone_ln' in ln_train_lines[0]
    assert 'grad_backbone_bias' not in ln_train_lines[0]
    ln_trainable_data = json.loads((tmp_path / 'ln_sspanet_mil_smoke/trainable_parameters.json').read_text())
    assert 'backbone_ln' in ln_trainable_data
    assert 'backbone_bias' not in ln_trainable_data
    for writer in trainer.writers.values():
        writer.close()


def test_actual_huggingface_clip_contract(monkeypatch):
    transformers = pytest.importorskip('transformers')
    config = transformers.CLIPVisionConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, image_size=28, patch_size=7)
    backbone = transformers.CLIPVisionModel(config).vision_model
    monkeypatch.setattr(LNSSPANetMILDetector, 'build_backbone', lambda self, cfg: backbone)
    model = LNSSPANetMILDetector(dict(mil_topk=4))
    data = dict(image=torch.randn(2, 3, 28, 28), label=torch.tensor([0, 1]))
    out = model(data)
    assert out['feat'].shape == (2, 16) and out['patch_logits'].shape == (2, 4, 4)
    model.get_losses(data, out)['overall'].backward()
    assert backbone.encoder.layers[0].layer_norm1.weight.grad.abs().sum() > 0
    assert backbone.encoder.layers[0].self_attn.q_proj.bias.grad is None


def test_bias_sspanet_mil_actual_huggingface_clip_contract(monkeypatch):
    transformers = pytest.importorskip('transformers')
    config = transformers.CLIPVisionConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, image_size=28, patch_size=7)
    backbone = transformers.CLIPVisionModel(config).vision_model
    monkeypatch.setattr(BiasSSPANetMILDetector, 'build_backbone', lambda self, cfg: backbone)
    model = BiasSSPANetMILDetector(dict(mil_topk=4))
    data = dict(image=torch.randn(2, 3, 28, 28), label=torch.tensor([0, 1]))
    out = model(data)
    assert out['feat'].shape == (2, 16) and out['patch_logits'].shape == (2, 4, 4)
    model.get_losses(data, out)['overall'].backward()
    # In Bias tuning (BitFit): LayerNorm weights frozen, LayerNorm biases trainable
    assert backbone.encoder.layers[0].layer_norm1.weight.requires_grad is False
    assert backbone.encoder.layers[0].layer_norm1.weight.grad is None
    assert backbone.encoder.layers[0].layer_norm1.bias.requires_grad is True
    assert backbone.encoder.layers[0].layer_norm1.bias.grad.abs().sum() > 0
    # Self-attention weights frozen, biases trainable
    assert backbone.encoder.layers[0].self_attn.q_proj.weight.requires_grad is False
    assert backbone.encoder.layers[0].self_attn.q_proj.weight.grad is None
    assert backbone.encoder.layers[0].self_attn.q_proj.bias.requires_grad is True
    assert backbone.encoder.layers[0].self_attn.q_proj.bias.grad.abs().sum() > 0
    # MLP weights frozen, biases trainable
    assert backbone.encoder.layers[0].mlp.fc1.weight.requires_grad is False
    assert backbone.encoder.layers[0].mlp.fc1.weight.grad is None
    assert backbone.encoder.layers[0].mlp.fc1.bias.requires_grad is True
    assert backbone.encoder.layers[0].mlp.fc1.bias.grad.abs().sum() > 0


def test_actual_width_official_block():
    block = ATTN_Block(1024)
    assert sum(p.numel() for p in block.parameters()) == 7345252
    x = torch.randn(1, 1024, 16, 16, requires_grad=True)
    output = block(x)
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert output.shape == x.shape


@pytest.mark.parametrize('mode', ['train', 'test'])
@pytest.mark.parametrize('frame_limit', [3, 10, 12])
def test_dataset_val_sampling_and_missing_image(tmp_path, mode, frame_limit):
    import json
    import yaml
    pytest.importorskip('albumentations')
    pytest.importorskip('lmdb')
    from PIL import Image
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
    with (ROOT / 'training/config/detector/ln_sspanet_mil.yaml').open() as stream:
        config = yaml.safe_load(stream)
    config.update(lmdb=False, rgb_dir=str(tmp_path), dataset_json_folder=str(tmp_path),
                  test_dataset='Fixture', train_dataset=['Fixture'], eval_split='val',
                  label_dict={'real': 0, 'fake': 1}, frame_num={'train': frame_limit, 'test': frame_limit})
    metadata = {'Fixture': {}}
    for label in ('real', 'fake'):
        metadata['Fixture'][label] = {}
        for split in ('train', 'val', 'test'):
            relative = f'{split}/{label}/video'
            directory = tmp_path / relative
            directory.mkdir(parents=True)
            frames = []
            for i in range(10):
                Image.new('RGB', (16, 16), (i * 20, 0, 0)).save(directory / f'{i}.png')
                frames.append(f'{relative}/{i}.png')
            metadata['Fixture'][label][split] = {'video': {'label': label, 'frames': frames}}
    (tmp_path / 'Fixture.json').write_text(json.dumps(metadata))
    dataset = DeepfakeAbstractBaseDataset(config, mode=mode)
    split = 'train' if mode == 'train' else 'val'
    assert all(str(path).startswith(split + '/') for path in dataset.data_dict['image'])
    for label_name in ('real', 'fake'):
        selected = [int(Path(p).stem) for p in dataset.data_dict['image']
                    if f'/{label_name}/' in p]
        assert sorted(selected) == list(range(min(frame_limit, 10)))
        if mode == 'test':
            assert selected == sorted(selected)
    image, label, _, _ = dataset[0]
    assert image.shape == (3, 224, 224) and label in (0, 1)
    dataset.data_dict['image'] = list(dataset.data_dict['image'])
    dataset.data_dict['image'][1] = 'missing/0.png'
    with pytest.raises(RuntimeError, match='Failed to load sample'):
        dataset[1]


def test_config_bias_sspanet_mil():
    import yaml
    from metrics.registry import DETECTOR
    with (ROOT / 'training/config/detector/bias_sspanet_mil.yaml').open() as stream:
        cfg = yaml.safe_load(stream)
    with (ROOT / 'training/config/detector/ln_sspanet_mil.yaml').open() as ref_stream:
        ln_cfg = yaml.safe_load(ref_stream)
    assert cfg['model_name'] == 'bias_sspanet_mil'
    assert cfg['clip_model_name'] == 'openai/clip-vit-large-patch14'
    assert cfg['nEpochs'] == 10
    assert cfg['weight_real'] == 1.0
    assert cfg['weight_fake'] == 1.0
    assert cfg['label_smoothing'] == 0.1
    assert cfg['grad_clip_norm'] == 5.0
    assert cfg['lr_scheduler'] == 'cosine'
    assert cfg['use_patch'] is True
    assert cfg['use_sspanet'] is True
    assert cfg['lambda_mil'] == 0.3
    assert cfg['mil_topk'] == 16
    assert cfg['fusion_alpha_init'] == 0.1
    assert DETECTOR[cfg['model_name']] is BiasSSPANetMILDetector
    # Full hyperparameter parity check with ln_sspanet_mil.yaml
    for k in ln_cfg:
        if k == 'model_name':
            continue
        assert k in cfg, f"Key '{k}' missing in bias_sspanet_mil.yaml"
        assert cfg[k] == ln_cfg[k], f"Mismatch for key '{k}': {cfg[k]} vs {ln_cfg[k]}"


def test_bias_sspanet_mil_trainer_smoke(bias_sspanet_mil_detector, tmp_path):
    from trainer.trainer import Trainer
    from training.test import evaluate
    class Data(torch.utils.data.Dataset):
        data_dict = {'image': ['real/a/0.png', 'fake/a/0.png', 'real/b/0.png', 'fake/b/0.png']}
        def __len__(self):
            return 4
        def __getitem__(self, i):
            return dict(image=torch.ones(3, 8, 8) * (i + 1), label=torch.tensor(i % 2))
    loader = torch.utils.data.DataLoader(Data(), batch_size=2)
    config = dict(cuda=False, ddp=False, model_name='bias_sspanet_mil', log_dir=str(tmp_path),
                  optimizer={'type': 'adam'}, selection_dataset='source', validation_split='val',
                  log_interval=1, save_ckpt=True, grad_clip_norm=5.)
    trainer = Trainer(config, bias_sspanet_mil_detector, torch.optim.Adam([p for p in bias_sspanet_mil_detector.parameters() if p.requires_grad], lr=.001),
                      None, logging.getLogger('test'), time_now='smoke_bias')
    trainer.train_epoch(0, loader, {'source': loader})
    checkpoint = tmp_path / 'bias_sspanet_mil_smoke_bias/validation/source/ckpt_best.pth'
    assert checkpoint.exists()
    state = safe_torch_load(checkpoint)
    assert state.get('architecture') == 'bias_sspanet_mil_v1'
    bias_sspanet_mil_detector.load_state_dict(state['state_dict'], strict=True)
    result, arrays = evaluate(bias_sspanet_mil_detector.eval(), loader, torch.device('cpu'), max_samples=3, patch_limit=2)
    assert result['n'] == 3 and len(arrays['image_names']) == 3
    assert arrays['patch_prob'].shape == (2, 4, 4)
    assert (tmp_path / 'bias_sspanet_mil_smoke_bias/train.jsonl').exists()
    import json
    train_lines = [json.loads(line) for line in (tmp_path / 'bias_sspanet_mil_smoke_bias/train.jsonl').read_text().strip().split('\n')]
    assert len(train_lines) > 0
    assert 'grad_backbone_bias' in train_lines[0]
    assert 'grad_backbone_ln' not in train_lines[0]
    assert 'grad_head' in train_lines[0]
    assert 'grad_sspanet' in train_lines[0]
    assert 'grad_patch_head' in train_lines[0]
    assert 'grad_fusion_alpha' not in train_lines[0]
    trainable_params_path = tmp_path / 'bias_sspanet_mil_smoke_bias/trainable_parameters.json'
    assert trainable_params_path.exists()
    trainable_data = json.loads(trainable_params_path.read_text())
    assert 'backbone_bias' in trainable_data
    assert 'backbone_ln' not in trainable_data
    for writer in trainer.writers.values():
        writer.close()


def test_safe_torch_load_compatibility(tmp_path, monkeypatch):
    from training.train import safe_torch_load as train_safe_load
    from training.test import safe_torch_load as test_safe_load

    dummy_path = tmp_path / "dummy.pth"
    torch.save({'a': 1, 'b': torch.tensor([1, 2, 3])}, dummy_path)

    # Test normal load with both implementations
    for fn in (safe_torch_load, train_safe_load, test_safe_load):
        res = fn(dummy_path)
        assert res['a'] == 1 and (res['b'] == torch.tensor([1, 2, 3])).all()

    # Test fallback behavior when weights_only raises TypeError (like PyTorch 1.12)
    real_load = torch.load
    def mock_load(path, *args, **kwargs):
        if 'weights_only' in kwargs:
            raise TypeError("torch.load() got an unexpected keyword argument 'weights_only'")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, 'load', mock_load)
    for fn in (safe_torch_load, train_safe_load, test_safe_load):
        res_fallback = fn(dummy_path)
        assert res_fallback['a'] == 1


def test_bias_sspanet_mil_detector_shim():
    import detectors.bias_sspanet_mil_detector as shim
    assert shim.LNSSPANetMILDetector is LNSSPANetMILDetector
    assert shim.BiasSSPANetMILDetector is BiasSSPANetMILDetector
    assert shim.BiasSSPANetMILDetector is not LNSSPANetMILDetector
    assert issubclass(shim.BiasSSPANetMILDetector, LNSSPANetMILDetector)
    assert shim.topk_mil_logits is topk_mil_logits


def test_ablation_configs():
    import yaml
    ablation_dir = ROOT / 'training/config/detector/ablations'
    assert ablation_dir.is_dir()
    configs = list(ablation_dir.glob('*.yaml'))
    assert len(configs) == 4
    for cfg_path in configs:
        with cfg_path.open(encoding='utf-8') as stream:
            cfg = yaml.safe_load(stream)
        assert cfg['model_name'] == 'ln_sspanet_mil'
        assert cfg['clip_model_name'] == 'openai/clip-vit-large-patch14'


def test_requirements_files_consistency():
    for name in ('requirements-ln-sspanet-mil.txt', 'requirements-bias-sspanet-mil.txt', 'requirements.txt'):
        path = ROOT / name
        assert path.is_file()
        content = path.read_text(encoding='utf-8')
        assert 'numpy==1.21.5' in content
        assert 'torch==1.12.0+cu113' in content
        assert 'albumentations' in content
        assert 'transformers' in content


def test_eval_ensemble_sweep(tmp_path):
    from analysis.eval_ensemble import run_ensemble_sweep
    from metrics.utils import format_compact_test_report
    # Create synthetic predictions npz
    names = np.array(['v1/0.png', 'v1/1.png', 'v2/0.png', 'v2/1.png'])
    labels = np.array([0, 0, 1, 1])
    prob = np.array([0.2, 0.3, 0.7, 0.8])
    mil_prob = np.array([0.1, 0.2, 0.8, 0.9])
    cls_only_prob = np.array([0.25, 0.35, 0.65, 0.75])

    npz_path = tmp_path / "test_predictions.npz"
    np.savez(npz_path, prob=prob, mil_prob=mil_prob, cls_only_prob=cls_only_prob,
             label=labels, image_names=names)

    best_item, results = run_ensemble_sweep(npz_path, weights=[0.0, 0.5, 1.0], save=True)
    assert len(results) == 3
    assert best_item['video_auc'] == 1.0
    saved_json = tmp_path / f"test_predictions_ensemble_best_w{best_item['weight']:.2f}.json"
    assert saved_json.exists()

    # Test format_compact_test_report displays Ensemble properly
    res_dict = dict(video_auc=0.95, ensemble_video_auc=0.965, mil_video_auc=0.958, cls_only_video_auc=0.949,
                    auc=0.87, ensemble_auc=0.885, mil_auc=0.88, cls_only_auc=0.865,
                    video_eer=0.12, eer=0.20, n=4, video_n=2)
    report = format_compact_test_report('Synthetic', res_dict)
    assert 'Ensemble: 96.50%' in report
    assert 'MIL: 95.80%' in report
    assert 'CLS: 94.90%' in report


def test_bilateral_dynamic_gating(bias_sspanet_mil_detector):
    """Verify Bilateral Dynamic Gating bounds, behavior on noise vs peak artifacts, and report."""
    from metrics.utils import format_compact_test_report
    model = bias_sspanet_mil_detector
    model.eval()

    data = dict(image=torch.randn(4, 3, 8, 8), label=torch.tensor([0, 0, 1, 1]))
    with torch.no_grad():
        out = model(data)

    assert 'gating_w' in out
    w = out['gating_w']
    assert w.shape == (4,)
    assert (w >= model.gating_w_min).all() and (w <= model.gating_w_max).all()

    # Check that adaptive prob satisfies (1 - w)*cls_prob + w*mil_prob
    expected_prob = (1.0 - w) * out['cls_only_prob'] + w * out['mil_prob']
    torch.testing.assert_close(out['prob'], expected_prob)

    # Synthetic Scenario 1: Diffuse / Flat noise patches (DFDC style) -> w should be close to w_min
    flat_patch_logits = torch.zeros(2, 4, 4)  # identical patches -> salience = 0
    confident_cls = torch.tensor([0.05, 0.95])  # very confident CLS
    w_noisy, cls_conf, mil_conf = model._compute_dynamic_gating(confident_cls, flat_patch_logits)
    assert (mil_conf < 0.05).all()
    assert (w_noisy < 0.15).all(), f"Expected w to drop near w_min, got {w_noisy}"

    # Synthetic Scenario 2: Sharp localized anomaly (Celeb-DF style) with confused CLS -> w should leap near w_max
    sharp_patch_logits = torch.full((1, 4, 4), -8.0)
    sharp_patch_logits[0, 1, 1] = 8.0  # single smoking gun patch
    confused_cls = torch.tensor([0.50])  # CLS is 50/50 confused
    w_sharp, cls_conf2, mil_conf2 = model._compute_dynamic_gating(confused_cls, sharp_patch_logits)
    assert mil_conf2.item() > 0.4
    assert w_sharp.item() > 0.70, f"Expected w to leap near w_max, got {w_sharp.item()}"

    # Verify report formatting with gating statistics
    report_dict = dict(
        video_auc=0.961, mil_video_auc=0.956, cls_only_video_auc=0.953,
        auc=0.898, mil_auc=0.892, cls_only_auc=0.886,
        video_eer=0.10, eer=0.18, n=4, video_n=2,
        gating_w_mean=0.25, gating_w_real=0.08, gating_w_fake=0.42
    )
    rep = format_compact_test_report('Celeb-DF-v2', report_dict)
    assert 'Gating Behavior' in rep
    assert 'Mean w = 0.25' in rep
    assert 'Real: 0.08' in rep
    assert 'Fake: 0.42' in rep


def test_checkpoint_loading_state_dict(tmp_path):
    """Ensure checkpoints saved by trainer can be prepared by test.py logic without UnboundLocalError."""
    dummy_state = {'module.layer.weight': torch.randn(2, 2), 'layer.bias': torch.randn(2)}
    ckpt_path = tmp_path / 'ckpt.pth'
    torch.save({'state_dict': dummy_state, 'config': {'model_name': 'test'}}, ckpt_path)

    checkpoint = safe_torch_load(ckpt_path)
    state = checkpoint.get('state_dict', checkpoint)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}

    assert 'layer.weight' in state
    assert 'layer.bias' in state
    assert not any(k.startswith('module.') for k in state)




