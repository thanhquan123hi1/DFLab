"""LN-SSPANet-MIL training with source-validation selection and auditable diagnostics."""
import datetime
import os
import time
from collections import defaultdict
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from metrics.utils import binary_metrics, get_test_metrics, write_json


def get_vietnam_time_str():
    vn_tz = datetime.timezone(datetime.timedelta(hours=7))
    return datetime.datetime.now(vn_tz).strftime('%Hh%M')


def get_run_name(config, time_now=None):
    model_name = config.get('model_name', 'model')
    task = f"_{config['task_target']}" if config.get('task_target') else ''
    model_part = f"{model_name}{task}"

    if time_now in ('smoke', 'smoke_bias'):
        return f"{model_part}_{time_now}"

    stamp = time_now or get_vietnam_time_str()
    seed = config.get('manualSeed', config.get('seed'))
    if seed is not None:
        return f"{model_part}_{seed}_{stamp}"
    return f"{model_part}_{stamp}"


class Trainer:
    def __init__(self, config, model, optimizer, scheduler, logger, metric_scoring='auc',
                 time_now=None, swa_model=None, log_dir=None):
        if config.get('SWA') or config['optimizer']['type'] == 'sam':
            raise ValueError('This audited LN+SSPANet trainer supports Adam/SGD; SAM/SWA need separate BN validation')
        self.config, self.optimizer, self.scheduler = config, optimizer, scheduler
        self.logger, self.metric_scoring = logger, metric_scoring
        self.device = torch.device('cuda', config.get('local_rank', 0)) if config.get('cuda', True) and torch.cuda.is_available() else torch.device('cpu')
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.model = model.to(self.device)
        if config.get('ddp'):
            self.model = DDP(model, device_ids=[self.device.index], find_unused_parameters=True)
        self.writers = {}
        self.best_metrics_all_time = {}
        self.best_score = float('inf') if metric_scoring == 'eer' else -float('inf')
        if log_dir is not None:
            self.log_dir = log_dir
        else:
            self.log_dir = os.path.join(config['log_dir'], get_run_name(config, time_now=time_now))
        if self.rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            write_json(os.path.join(self.log_dir, 'config.json'), config)
            write_json(os.path.join(self.log_dir, 'trainable_parameters.json'), model.trainable_counts)

    @property
    def module(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def get_writer(self, phase, dataset_key, metric_key='all'):
        key = (phase, dataset_key)
        if key not in self.writers:
            self.writers[key] = SummaryWriter(os.path.join(self.log_dir, phase, dataset_key))
        return self.writers[key]

    def move(self, data):
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in data.items()}

    def train_step(self, data):
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(data)
        losses = self.module.get_losses(data, prediction)
        if not torch.isfinite(losses['overall']):
            raise FloatingPointError('Nonfinite overall loss; inspect CE/MIL and input data')
        losses['overall'].backward()
        norms = defaultdict(list)
        for name, param in self.module.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name.startswith('backbone.'):
                    is_bias = ('backbone_bias' in getattr(self.module, 'trainable_counts', {}) or
                               'bias' in self.config.get('model_name', ''))
                    group = 'backbone_bias' if is_bias else 'backbone_ln'
                else:
                    group = name.split('.')[0]
                norms[group].append(param.grad.detach().float().norm())
        for group, values in norms.items():
            losses['grad_' + group] = torch.stack(values).norm()
        total = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.config.get('grad_clip_norm', 5.0), error_if_nonfinite=True)
        losses['grad_total_before_clip'] = total.detach()
        self.optimizer.step()
        return losses, prediction

    def train_epoch(self, epoch, train_data_loader, test_data_loaders=None):
        if len(train_data_loader) == 0:
            raise ValueError('Training loader is empty')
        if hasattr(train_data_loader.sampler, 'set_epoch'):
            train_data_loader.sampler.set_epoch(epoch)
        self.model.train()
        records = defaultdict(Recorder)
        labels, predictions = [], []
        start = time.monotonic()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        interval = max(1, int(self.config.get('log_interval', 100)))
        for iteration, batch in enumerate(train_data_loader):
            data = self.move(batch)
            losses, pred = self.train_step(data)
            size = len(data['label'])
            for key, value in losses.items():
                records[key].update(value, size)
            labels.extend((data['label'] != 0).long().detach().cpu().tolist())
            predictions.extend(pred['prob'].detach().cpu().tolist())
            if (iteration + 1) % interval == 0 or iteration + 1 == len(train_data_loader):
                step = epoch * len(train_data_loader) + iteration
                if self.rank == 0:
                    values = {k: r.average() for k, r in records.items()}
                    values.update(binary_metrics(labels, predictions))
                    values['lr'] = self.optimizer.param_groups[0]['lr']
                    values['images_per_second'] = len(labels) / max(time.monotonic() - start, 1e-6)
                    if self.device.type == 'cuda':
                        values['peak_vram_mb'] = torch.cuda.max_memory_allocated(self.device) / 2**20
                    self.logger.info('train epoch=%d step=%d %s', epoch, step, values)
                    write_json(os.path.join(self.log_dir, 'train.jsonl'), dict(epoch=epoch, step=step, rank=0, **values), append=True)
                    for key, value in values.items():
                        self.get_writer('train', 'all').add_scalar(key, value, step)
                records.clear()
                labels, predictions = [], []
                start = time.monotonic()
        # All ranks reach the same boundary. Validation uses the unwrapped module
        # to avoid DDP forward collectives while only rank zero evaluates.
        if dist.is_initialized():
            dist.barrier()
        if self.rank == 0 and test_data_loaders:
            self.test_epoch(epoch, iteration, test_data_loaders, epoch)
        if dist.is_initialized():
            dist.barrier()
        return self.best_metrics_all_time

    def save_checkpoint(self, path, epoch, score=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arch = f"{self.config.get('model_name', 'ln_sspanet_mil')}_v1"
        torch.save({'state_dict': self.module.state_dict(), 'config': self.config,
                    'epoch': epoch, 'selection_score': score,
                    'architecture': arch}, path)
        self.logger.info('Saved checkpoint: %s', path)

    @torch.no_grad()
    def test_one_dataset(self, loader):
        records = defaultdict(Recorder)
        pred, labels, local, cls = [], [], [], []
        gating = []
        patch_examples = []
        example_count = 0
        for batch in loader:
            data = self.move(batch)
            output = self.module(data, inference=True)
            for key, value in self.module.get_losses(data, output).items():
                records[key].update(value, len(data['label']))
            pred.extend(output['prob'].cpu().tolist())
            labels.extend((data['label'] != 0).long().cpu().tolist())
            cls.extend(output['cls_only_prob'].cpu().tolist())
            if output.get('gating_w') is not None:
                gating.extend(output['gating_w'].cpu().tolist())
            if 'mil_prob' in output:
                local.extend(output['mil_prob'].cpu().tolist())
            if 'patch_logits' in output and example_count < 16:
                examples = output['patch_logits'][:16-example_count].sigmoid().cpu().numpy()
                patch_examples.append(examples)
                example_count += len(examples)
        names = loader.dataset.data_dict['image']
        result = get_test_metrics(pred, labels, names)
        result.update({'cls_only_' + k: v for k, v in binary_metrics(labels, cls).items()})
        if local:
            result.update({'mil_' + k: v for k, v in binary_metrics(labels, local).items()})
        return result, {k: r.average() for k, r in records.items()}, dict(
            prob=np.asarray(pred), label=np.asarray(labels), cls_only_prob=np.asarray(cls),
            mil_prob=np.asarray(local), image_names=np.asarray(names), gating_w=np.asarray(gating),
            patch_prob=np.concatenate(patch_examples) if patch_examples else np.empty((0,)),
            patch_image_names=np.asarray(names[:example_count]))

    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        self.module.eval()
        selection = self.config.get('selection_dataset', next(iter(test_data_loaders)))
        if selection not in test_data_loaders:
            raise ValueError('selection_dataset must be one of validation_dataset')
        for key, loader in test_data_loaders.items():
            result, losses, arrays = self.test_one_dataset(loader)
            self.logger.info('validation epoch=%d dataset=%s metrics=%s losses=%s',
                             epoch, key, {k: v for k, v in result.items() if k not in ('pred', 'label')}, losses)
            write_json(os.path.join(self.log_dir, 'validation.jsonl'),
                       dict(epoch=epoch, dataset=key, split=self.config.get('validation_split', 'val'), metrics=result, losses=losses), append=True)
            writer = self.get_writer('validation', key)
            for name, value in result.items():
                if name not in ('pred', 'label'):
                    writer.add_scalar('metrics/' + name, value, step)
            for name, value in losses.items():
                writer.add_scalar('losses_and_diagnostics/' + name, value, step)
            out_dir = os.path.join(self.log_dir, 'validation', key)
            np.savez_compressed(os.path.join(out_dir, 'predictions_last.npz'), **arrays)
            score = result[self.metric_scoring]
            better = score < self.best_score if self.metric_scoring == 'eer' else score > self.best_score
            if key == selection and np.isfinite(score) and better:
                self.best_score = score
                self.best_metrics_all_time[key] = {self.metric_scoring: score, 'epoch': epoch}
                if self.config.get('save_ckpt', True):
                    self.save_checkpoint(os.path.join(out_dir, 'ckpt_best.pth'), epoch, score)
                np.savez_compressed(os.path.join(out_dir, 'predictions_best.npz'), **arrays)
        if self.config.get('save_ckpt', True):
            self.save_checkpoint(os.path.join(self.log_dir, 'ckpt_last.pth'), epoch)
