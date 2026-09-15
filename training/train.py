# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from detectors import DETECTOR
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter
from run_naming import get_run_name


def safe_torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='training/config/detector/ln_sspanet_mil.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank','--local-rank', type=int, default=0)
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task')

# [NEW] Thêm tham số weights_path giống test.py
parser.add_argument('--weights_path', type=str, default=None, help='Path to pretrained weights (overrides config)')
parser.add_argument('--seed', '--manualSeed', dest='seed', type=int, default=None,
                    help='Random seed (overrides manualSeed in YAML)')

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    np.random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_training_data(config):
    from dataset import (
        DeepfakeAbstractBaseDataset,
        FFBlendDataset,
        FWABlendDataset,
        I2GDataset,
        IIDDataset,
        LRLDataset,
        LSDADataset,
        SBIDataset,
        pairDataset,
    )

    # Only use the blending dataset class in training
    if 'dataset_type' in config and config['dataset_type'] == 'blend':
        if config['model_name'] == 'facexray':
            train_set = FFBlendDataset(config)
        elif config['model_name'] == 'fwa':
            train_set = FWABlendDataset(config)
        elif config['model_name'] == 'sbi':
            train_set = SBIDataset(config, mode='train')
        elif config['model_name'] == 'lsda':
            train_set = LSDADataset(config, mode='train')
        else:
            raise NotImplementedError(
                'Only facexray, fwa, sbi, and lsda are currently supported for blending dataset'
            )
    elif 'dataset_type' in config and config['dataset_type'] == 'pair':
        train_set = pairDataset(config, mode='train')  # Only use the pair dataset class in training
    elif 'dataset_type' in config and config['dataset_type'] == 'iid':
        train_set = IIDDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'I2G':
        train_set = I2GDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'lrl':
        train_set = LRLDataset(config, mode='train')
    else:
        train_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='train',
                )
    if config['model_name'] == 'lsda':
        from dataset.lsda_dataset import CustomSampler
        custom_sampler = CustomSampler(num_groups=2*360, n_frame_per_vid=config['frame_num']['train'], batch_size=config['train_batchSize'], videos_per_group=5)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                sampler=custom_sampler, 
                collate_fn=train_set.collate_fn,
            )
    elif config['ddp']:
        sampler = DistributedSampler(train_set)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sampler
            )
    else:
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                )
    return train_data_loader


def prepare_testing_data(config):
    from dataset import DeepfakeAbstractBaseDataset, LRLDataset

    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        config['eval_split'] = config.get('validation_split', 'val')
        if not config.get('dataset_type', None) == 'lrl':
            test_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='test',
            )
        else:
            test_set = LRLDataset(
                config=config,
                mode='test',
            )

        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False,
            )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config.get('validation_dataset', config['train_dataset']):
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=[p for p in model.parameters() if p.requires_grad],
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=[p for p in model.parameters() if p.requires_grad],
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs']/4),
        )
        return scheduler
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap', 'video_auc']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def main():
    global args
    args = parser.parse_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    config.update(config2)
    config['local_rank']=args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False
    
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    
    # [NEW] Logic ưu tiên: CLI Argument > YAML Config
    if args.weights_path:
        config['pretrained'] = args.weights_path
    if args.seed is not None:
        config['manualSeed'] = args.seed
        
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    
    # Resolve the seed before naming the run; initialize RNGs below as before.
    if config.get('manualSeed') is None:
        config['manualSeed'] = random.randint(1, 10000)
    timenow = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).strftime('%Hh%M')
    logger_path =  os.path.join(
                config['log_dir'],
                get_run_name(config, time_now=timenow)
            )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    config['ddp']= args.ddp
    if config.get('video_mode', False):
        raise ValueError(f"{config['model_name']} is frame-level; disable video_mode")
    if config.get('SWA') or config['optimizer']['type'] == 'sam':
        raise ValueError('LN+SSPANet currently supports Adam/SGD, without SAM/SWA')
    if args.ddp and not torch.cuda.is_available():
        raise ValueError('DDP entry point requires CUDA/NCCL')
    
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        # dist.init_process_group(backend='gloo')
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )
        logger.addFilter(RankFilter(0))
        
    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    train_paths = {str(p).replace('\\', '/') for p in train_data_loader.dataset.data_dict['image']}
    train_videos = {p.rsplit('/', 1)[0] for p in train_paths}
    for key, loader in test_data_loaders.items():
        val_videos = {str(p).replace('\\', '/').rsplit('/', 1)[0] for p in loader.dataset.data_dict['image']}
        if train_videos & val_videos:
            raise ValueError(f'Train/validation video overlap in {key}')
    logger.info('Checkpoint selection uses validation_dataset=%s split=%s; final test is training/test.py',
                list(test_data_loaders), config.get('validation_split', 'val'))

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    if config.get('pretrained') and not os.path.isfile(config['pretrained']):
        raise FileNotFoundError(config['pretrained'])

    # --- [NEW] LOAD PRETRAINED WEIGHTS (Xử lý thông minh) ---
    if config.get('pretrained') is not None and os.path.exists(config['pretrained']):
        logger.info(f"🔄 Loading pretrained weights from: {config['pretrained']}")
        try:
            # Load checkpoint
            checkpoint = safe_torch_load(config['pretrained'], map_location='cpu')
            
            # Xử lý trường hợp checkpoint lưu cả epoch/optimizer (dict lồng nhau)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Xử lý prefix 'module.' (do DataParallel/DDP)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            # Fail if the requested checkpoint is not compatible with this architecture.
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=True)
            
            if len(missing_keys) > 0:
                logger.warning(f"⚠️ Missing keys (initialized randomly): {missing_keys[:5]} ... total {len(missing_keys)}")
            if len(unexpected_keys) > 0:
                logger.warning(f"⚠️ Unexpected keys (ignored): {unexpected_keys[:5]} ... total {len(unexpected_keys)}")
                
            logger.info("✅ Successfully loaded pretrained weights!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load pretrained weights: {e}")
            raise
    else:
        logger.info("ℹ️ No pretrained weights provided via CLI or YAML. Training from scratch.")
    # --------------------------------------------------------

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    from trainer.trainer import Trainer

    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow)

    # start training
    best_metric = None
    for epoch in range(config['start_epoch'], config['nEpochs']):
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader,
                    test_data_loaders=test_data_loaders,
                )
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!")
        if scheduler is not None:
            scheduler.step()
    
    if best_metric is not None:
        logger.info("Stop Training on best Testing metric {}".format(parse_metric_for_print(best_metric))) 
    
    # update
    if 'svdd' in config['model_name']:
        model.update_R(epoch)

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()


if __name__ == '__main__':
    main()
