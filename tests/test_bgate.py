"""BGATE invariants, branch isolation and CPU integration without downloading CLIP."""
import json
import logging
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from detectors.ln_sspanet_mil_detector import LNSSPANetMILDetector
from detectors.bias_sspanet_mil_detector import BiasSSPANetMILDetector
from detectors.bgate_mil_detector import MODELS, BRANCHES
from bgate_common import training_config,ROOT,seed_everything,check_overlap
from trainer.bgate_trainer import BGateTrainer
from bgate_logging import new_run,load_checkpoint,unique_dir
from metrics.bgate_evaluation import evaluate


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8,patch_size=2)
        self.patch = nn.Conv2d(3,8,2,2)
        self.norm = nn.LayerNorm(8)
        self.mix = nn.Linear(8,8)

    def forward(self,x):
        tokens = self.mix(self.norm(self.patch(x).flatten(2).transpose(1,2)))
        cls = tokens.mean(1)
        return SimpleNamespace(pooler_output=cls,last_hidden_state=torch.cat((cls[:,None],tokens),1))


@pytest.fixture(autouse=True)
def tiny(monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(LNSSPANetMILDetector,'build_backbone',lambda self,cfg:Backbone())


def config(kind='bias',epochs=2):
    cfg = training_config(ROOT/f'config/detector/{kind}_sspanet_bgate_mil.yaml')
    cfg.update(mil_topk=3,manualSeed=1024,nEpochs=epochs,workers=0,log_interval=1,
               train_batchSize=2,test_batchSize=2)
    return cfg


def batch():
    return dict(image=torch.randn(2,3,8,8),label=torch.tensor([0,1]))


@pytest.mark.parametrize('kind,baseline',[('bias',BiasSSPANetMILDetector),('ln',LNSSPANetMILDetector)])
def test_base_identity_rng_trainables_and_gate(kind,baseline):
    cfg = config(kind)
    torch.manual_seed(17)
    old = baseline(cfg).eval()
    state_after = torch.get_rng_state().clone()
    torch.manual_seed(17)
    new = MODELS[cfg['model_name']](cfg).eval()
    assert torch.equal(torch.get_rng_state(),state_after)
    for name,value in old.state_dict().items():
        assert torch.equal(value,new.state_dict()[name])
    assert sum(p.numel() for p in new.gate.parameters())==97
    assert {n for n,p in old.backbone.named_parameters() if p.requires_grad} == {
        n for n,p in new.backbone.named_parameters() if p.requires_grad}
    data = batch()
    a,b = old(data),new(data)
    assert torch.equal(a['prob'],b['feature_fusion_prob'])
    assert torch.equal(a['mil_prob'],b['mil_prob'])
    new.set_epoch(1)
    out = new(data)
    assert torch.equal(out['gating_w'],torch.full((2,),.5))
    assert torch.allclose(out['bounded_gate_prob'],out['fixed_ensemble_prob'])
    with torch.no_grad():
        new.gate[-1].bias.fill_(100)
    assert new(data)['gating_w'].max() <= .75
    with torch.no_grad():
        new.gate[-1].bias.fill_(-100)
    assert new(data)['gating_w'].min() >= .25


@pytest.mark.parametrize('kind',['bias','ln'])
def test_gradient_separation(kind):
    cfg = config(kind)
    model = MODELS[cfg['model_name']](cfg)
    model.set_epoch(1)
    data = batch()
    model.get_losses(data,model(data))['loss_gate'].backward()
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('gate.'))
    assert model.gate[-1].weight.grad is not None
    model.zero_grad(set_to_none=True)
    model.get_losses(data,model(data))['loss_base'].backward()
    assert all(p.grad is None for p in model.gate.parameters())
    assert model.patch_head.weight.grad is not None
    assert any(p.grad is not None for p in model.backbone.parameters() if p.requires_grad)


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.images = torch.randn(4,3,8,8)
        self.data_dict = dict(image=['real/a/0.png','real/a/1.png','fake/b/0.png','fake/b/1.png'],label=[0,0,1,1])
    def __len__(self):
        return 4
    def __getitem__(self,index):
        return dict(image=self.images[index],label=torch.tensor(self.data_dict['label'][index]))


def trainer(cfg,path):
    return BGateTrainer(MODELS[cfg['model_name']](cfg),cfg,path,torch.device('cpu'),logging.getLogger('bgate-test'))


@pytest.mark.parametrize('kind',['bias','ln'])
def test_warmup_checkpoint_resume_and_evaluation(tmp_path,kind):
    seed_everything(2)
    cfg = config(kind)
    run = new_run(tmp_path,cfg['model_name'],cfg['manualSeed'])
    tr = trainer(cfg,run)
    data = batch()
    original = {k:v.clone() for k,v in tr.model.gate.state_dict().items()}
    tr.model.train()
    tr.model.set_epoch(0)
    tr.train_step(data)
    assert not tr.gate_optimizer.state
    assert all(torch.equal(v,tr.model.gate.state_dict()[k]) for k,v in original.items())
    tr.model.set_epoch(1)
    tr.train_step(data)
    assert tr.gate_optimizer.state
    assert any(not torch.equal(v,tr.model.gate.state_dict()[k]) for k,v in original.items())
    loader = torch.utils.data.DataLoader(Dataset(),batch_size=2)
    metrics,arrays = evaluate(tr.model,loader,torch.device('cpu'))
    assert set(metrics['branches'])==set(BRANCHES)
    assert metrics['branches']['bounded_gate']['video_n']==2
    assert np.allclose(arrays['fixed_ensemble_prob'],.5*(arrays['feature_fusion_prob']+arrays['mil_prob']))
    assert np.allclose(arrays['bounded_gate_prob'],(1-arrays['gating_w'])*arrays['feature_fusion_prob']+arrays['gating_w']*arrays['mil_prob'])
    tr.close()
    # Full two-epoch integration with all on-disk artifacts.
    tr = trainer(cfg,run)
    tr.fit(loader,loader)
    checkpoint = load_checkpoint(run/'checkpoints/last.pt')
    a = evaluate(tr.model,loader,torch.device('cpu'))[1]
    resumed = trainer(cfg,run)
    resumed.resume(checkpoint)
    b = evaluate(resumed.model,loader,torch.device('cpu'))[1]
    assert np.array_equal(a['bounded_gate_prob'],b['bounded_gate_prob'])
    assert resumed.start_epoch==2
    assert resumed.step==tr.step
    assert resumed.scheduler.state_dict()==tr.scheduler.state_dict()
    assert (run/'checkpoints/best.pt').exists()
    assert len((run/'validation/metrics.jsonl').read_text().splitlines())==2
    tr.model.train()
    resumed.model.train()
    # Identical next update verifies optimizer states, not just prediction reload.
    tr.train_step(data)
    resumed.train_step(data)
    assert all(torch.equal(v,resumed.model.state_dict()[k]) for k,v in tr.model.state_dict().items())
    tr.close()
    resumed.close()


def test_unique_paths_and_overlap(tmp_path):
    a = new_run(tmp_path,'bias_sspanet_bgate_mil',1024)
    b = new_run(tmp_path,'bias_sspanet_bgate_mil',1024)
    assert a!=b and 'bgate_v1' in a.parts
    assert unique_dir(a/'tests')!=unique_dir(a/'tests')
    loader = torch.utils.data.DataLoader(Dataset(),batch_size=2)
    with pytest.raises(ValueError,match='overlap'):
        check_overlap(loader,loader)


def test_config_and_registry_isolation():
    from detectors import DETECTOR
    assert DETECTOR['bias_sspanet_mil'] is BiasSSPANetMILDetector
    assert DETECTOR['ln_sspanet_mil'] is LNSSPANetMILDetector
    cfg = config()
    assert cfg['gate_radius']==.25 and cfg['lr_T_max']==15
    assert cfg['validation_split']=='val' and cfg['selection_dataset']=='FaceForensics++'


def test_cli_train_resume_test(tmp_path,monkeypatch):
    import sys
    import bgate_common as common
    import training.train_bgate as train_cli
    import training.test_bgate as test_cli
    def make_loader(cfg,mode,dataset=None,split=None):
        data = Dataset()
        data.data_dict['image'] = [mode+'/'+p for p in data.data_dict['image']]
        return torch.utils.data.DataLoader(data,batch_size=2)
    monkeypatch.setattr(common,'make_loader',make_loader)
    monkeypatch.setattr(sys,'argv',['train','--config',str(ROOT/'config/detector/bias_sspanet_bgate_mil.yaml'),
        '--epochs','2','--workers','0','--output-root',str(tmp_path),'--device','cpu'])
    train_cli.main()
    runs = list((tmp_path/'bgate_v1/bias_sspanet_bgate_mil/seed_1024').iterdir())
    assert len(runs)==1
    run = runs[0]
    assert not (run/'.training.lock').exists()
    monkeypatch.setattr(sys,'argv',['train','--resume',str(run/'checkpoints/last.pt'),'--device','cpu'])
    train_cli.main()
    assert (run/'train/resume_events.jsonl').exists()
    for _ in range(2):
        monkeypatch.setattr(sys,'argv',['test','--weights',str(run/'checkpoints/best.pt'),
            '--datasets','Celeb-DF-v2','--device','cpu'])
        test_cli.main()
    outputs = list((run/'tests').iterdir())
    assert len(outputs)==2
    for output in outputs:
        info = json.loads((output/'evaluation.json').read_text())
        assert info['train_seed']==1024
        assert info['fixed_ensemble_base']=='feature_fusion'
        assert (output/'summary.csv').exists()
        assert (output/'Celeb-DF-v2/predictions.npz').exists()
        metrics = json.loads((output/'Celeb-DF-v2/metrics.json').read_text())
        assert set(metrics['branches'])==set(BRANCHES)


def test_albumentations_private_rng_reproducible():
    import random
    import albumentations as A
    from dataset.bgate_dataset import seed_transform
    image = np.arange(192,dtype=np.uint8).reshape(8,8,3)
    def sequence():
        random.seed(123)
        transform = A.Compose([A.RandomBrightnessContrast(p=1)])
        frames = []
        for _ in range(4):
            seed_transform(transform)
            frames.append(transform(image=image)['image'])
        return frames
    a,b = sequence(),sequence()
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    assert any(not np.array_equal(a[0],x) for x in a[1:])


def test_evaluation_rejects_shuffled_ids():
    cfg = config()
    model = MODELS[cfg['model_name']](cfg)
    loader = torch.utils.data.DataLoader(Dataset(),batch_size=2,shuffle=True)
    with pytest.raises(ValueError,match='sequential'):
        evaluate(model,loader,torch.device('cpu'))


def test_resume_rejects_changed_source():
    from bgate_logging import verify_resume
    original = dict(config={'seed':1},manifests={'frames':4},source_hashes={'models.py':'abc'})
    verify_resume(original,original['config'],original['manifests'],original['source_hashes'])
    with pytest.raises(ValueError,match='source changed'):
        verify_resume(original,original['config'],original['manifests'],{'models.py':'xyz'})


class StochasticDataset(Dataset):
    def __getitem__(self,index):
        import random
        sample = super().__getitem__(index)
        sample['image'] = sample['image'] + random.random() + float(np.random.random()) + torch.rand(())
        return sample


def test_interrupted_epoch_resume_matches_uninterrupted(tmp_path,monkeypatch):
    import trainer.bgate_trainer as trainer_module
    cfg = config()
    def build(path):
        seed_everything(77)
        train = torch.utils.data.DataLoader(StochasticDataset(),batch_size=2,shuffle=True)
        val = torch.utils.data.DataLoader(Dataset(),batch_size=2)
        return trainer(cfg,path),train,val
    full,train,val = build(tmp_path/'full')
    full.fit(train,val)
    expected = {k:v.clone() for k,v in full.model.state_dict().items()}
    full.close()
    interrupted,train,val = build(tmp_path/'interrupted')
    save = trainer_module.atomic_save
    def stop_after_commit(payload,path):
        save(payload,path)
        if path.name=='last.pt' and payload['epoch']==0:
            raise RuntimeError('simulated interruption')
    with monkeypatch.context() as patch:
        patch.setattr(trainer_module,'atomic_save',stop_after_commit)
        with pytest.raises(RuntimeError,match='simulated'):
            interrupted.fit(train,val)
    interrupted.close()
    checkpoint = load_checkpoint(tmp_path/'interrupted/checkpoints/last.pt')
    resumed,train,val = build(tmp_path/'interrupted')
    resumed.resume(checkpoint)
    resumed.fit(train,val)
    assert all(torch.equal(v,resumed.model.state_dict()[k]) for k,v in expected.items())
    resumed.close()
