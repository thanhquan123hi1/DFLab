"""Configuration and baseline dataset reuse without changing baseline entry points."""
import copy
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent


def read_config(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    base = config.pop('base_config', None)
    if base:
        parent = read_config(path.parent / base)
        parent.update(config)
        config = parent
    return config


def training_config(path):
    config = read_config(ROOT / 'config/train_config.yaml')
    config.update(read_config(path))
    config.update(mode='train', ddp=False, local_rank=0)
    if config.get('video_mode') or config.get('SWA') or config.get('dataset_type'):
        raise ValueError('BGATE v1 supports the baseline frame dataset, without SWA')
    if config.get('pretrained'):
        raise ValueError('Use CLIP initialization or --resume; detector warm-start is not implemented')
    if config['validation_dataset'] != ['FaceForensics++'] or config.get('validation_split') != 'val':
        raise ValueError('BGATE v1 selection is fixed to FF++ val')
    if config['optimizer']['type'] != 'adam' or config.get('lr_scheduler') != 'cosine':
        raise ValueError('BGATE v1 supports the documented Adam/cosine baseline')
    return config


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(config, mode, dataset=None, split=None):
    from dataset.bgate_dataset import BGateDataset
    cfg = copy.deepcopy(config)
    if mode == 'test':
        cfg.update(test_dataset=dataset, eval_split=split)
    data = BGateDataset(cfg, mode=mode)
    loader = torch.utils.data.DataLoader(
        data, batch_size=cfg['train_batchSize' if mode == 'train' else 'test_batchSize'],
        shuffle=mode == 'train', num_workers=int(cfg['workers']),
        collate_fn=data.collate_fn, drop_last=False, worker_init_fn=seed_worker,
        persistent_workers=False)
    return loader


def manifest(loader):
    paths = [str(p).replace('\\', '/') for p in loader.dataset.data_dict['image']]
    labels = [int(y) for y in loader.dataset.data_dict['label']]
    digest = hashlib.sha256(json.dumps(list(zip(paths, labels)), ensure_ascii=False).encode()).hexdigest()
    return dict(frames=len(paths), videos=len({p.rsplit('/',1)[0] for p in paths}),
                ordered_manifest_sha256=digest)


def check_overlap(train, validation):
    def videos(loader):
        return {str(p).replace('\\','/').rsplit('/',1)[0] for p in loader.dataset.data_dict['image']}
    if videos(train) & videos(validation):
        raise ValueError('Train/validation video overlap')


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
