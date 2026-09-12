import hashlib
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from detectors.biasln_detector import BiasLNDetector, topk_mil_logits
from detectors.modules.sspanet import ATTN_Block
from metrics.base_metrics_class import Recorder
from metrics.utils import get_test_metrics

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]


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
    monkeypatch.setattr(BiasLNDetector, 'build_backbone', lambda self, config: SmallBackbone())
    torch.manual_seed(2)
    return BiasLNDetector(dict(mil_topk=3, lambda_mil=.3))


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
    assert detector.fusion_alpha.grad is not None
    detector.eval()
    with torch.no_grad():
        torch.testing.assert_close(detector(data)['prob'], detector(data, inference=True)['prob'])


def test_ablation_without_patch(monkeypatch):
    monkeypatch.setattr(BiasLNDetector, 'build_backbone', lambda self, config: SmallBackbone())
    model = BiasLNDetector(dict(use_patch=False, use_sspanet=False, lambda_mil=0))
    data = dict(image=torch.randn(2, 3, 8, 8), label=torch.tensor([0, 1]))
    assert model.get_losses(data, model(data))['loss_mil'] == 0
    with pytest.raises(ValueError):
        BiasLNDetector(dict(use_patch=False))


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
    config = dict(cuda=False, ddp=False, model_name='biasln', log_dir=str(tmp_path),
                  optimizer={'type': 'adam'}, selection_dataset='source', validation_split='val',
                  log_interval=1, save_ckpt=True, grad_clip_norm=5.)
    trainer = Trainer(config, detector, torch.optim.Adam([p for p in detector.parameters() if p.requires_grad], lr=.001),
                      None, logging.getLogger('test'), time_now='smoke')
    trainer.train_epoch(0, loader, {'source': loader})
    checkpoint = tmp_path / 'biasln_smoke/validation/source/ckpt_best.pth'
    assert checkpoint.exists()
    state = torch.load(checkpoint, weights_only=False)
    detector.load_state_dict(state['state_dict'], strict=True)
    result, arrays = evaluate(detector.eval(), loader, torch.device('cpu'), max_samples=3, patch_limit=2)
    assert result['n'] == 3 and len(arrays['image_names']) == 3
    assert arrays['patch_prob'].shape == (2, 4, 4)
    assert (tmp_path / 'biasln_smoke/train.jsonl').exists()
    for writer in trainer.writers.values():
        writer.close()


def test_actual_huggingface_clip_contract(monkeypatch):
    transformers = pytest.importorskip('transformers')
    config = transformers.CLIPVisionConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, image_size=28, patch_size=7)
    backbone = transformers.CLIPVisionModel(config).vision_model
    monkeypatch.setattr(BiasLNDetector, 'build_backbone', lambda self, cfg: backbone)
    model = BiasLNDetector(dict(mil_topk=4))
    data = dict(image=torch.randn(2, 3, 28, 28), label=torch.tensor([0, 1]))
    out = model(data)
    assert out['feat'].shape == (2, 16) and out['patch_logits'].shape == (2, 4, 4)
    model.get_losses(data, out)['overall'].backward()
    assert backbone.encoder.layers[0].layer_norm1.weight.grad.abs().sum() > 0
    assert backbone.encoder.layers[0].self_attn.q_proj.bias.grad is None


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
    with (ROOT / 'training/config/detector/biasln.yaml').open() as stream:
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
