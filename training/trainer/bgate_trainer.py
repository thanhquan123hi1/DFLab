"""Single-device trainer with independent base/gate optimizers and epoch-boundary resume."""
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from metrics.utils import write_json, binary_metrics
from detectors.bgate_mil_detector import ARCHITECTURE, BRANCHES
from bgate_common import move
from metrics.bgate_evaluation import evaluate
from bgate_logging import atomic_save, rng_state, restore_rng, gate_stats


class BGateTrainer:
    def __init__(self, model, config, run_dir, device, logger):
        self.model = model.to(device)
        self.config, self.run_dir, self.device, self.logger = config, Path(run_dir), device, logger
        self.base_params = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('gate.')]
        self.gate_params = list(model.gate.parameters())
        opt = config['optimizer']['adam']
        self.base_optimizer = torch.optim.Adam(self.base_params, lr=opt['lr'],
            betas=(opt['beta1'],opt['beta2']), eps=opt['eps'],
            weight_decay=opt['weight_decay'], amsgrad=opt.get('amsgrad',False))
        self.gate_optimizer = torch.optim.Adam(self.gate_params, lr=config.get('gate_lr',1e-4))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.base_optimizer,
            T_max=config['lr_T_max'], eta_min=config['lr_eta_min'])
        self.writer = SummaryWriter(str(self.run_dir/'train/tensorboard'))
        self.step, self.start_epoch, self.best_score = 0, 0, -float('inf')

    def train_step(self, data):
        self.base_optimizer.zero_grad(set_to_none=True)
        self.gate_optimizer.zero_grad(set_to_none=True)
        pred = self.model(data)
        losses = self.model.get_losses(data,pred)
        if not torch.isfinite(losses['overall']):
            raise FloatingPointError('Nonfinite BGATE loss')
        losses['overall'].backward()
        groups = defaultdict(list)
        for name,p in self.model.named_parameters():
            if p.grad is not None:
                group = name.split('.')[0]
                groups[group].append(p.grad.detach().norm())
        values = {k:float(v.detach()) for k,v in losses.items()}
        values.update({'grad_'+k:float(torch.stack(v).norm()) for k,v in groups.items()})
        values['grad_base_before_clip'] = float(torch.nn.utils.clip_grad_norm_(
            self.base_params,self.config.get('grad_clip_norm',5.),error_if_nonfinite=True))
        if self.model.gate_active:
            values['grad_gate_before_clip'] = float(torch.nn.utils.clip_grad_norm_(
                self.gate_params,self.config.get('gate_grad_clip_norm',5.),error_if_nonfinite=True))
        self.base_optimizer.step()
        if self.model.gate_active:
            self.gate_optimizer.step()
        self.step += 1
        return values,pred

    def checkpoint(self, epoch):
        return dict(architecture=ARCHITECTURE, state_dict=self.model.state_dict(), config=self.config,
            epoch=epoch, next_epoch=epoch+1, global_step=self.step, best_score=self.best_score,
            gate_active=self.model.gate_active, train_seed=self.config['manualSeed'],
            base_optimizer=self.base_optimizer.state_dict(), gate_optimizer=self.gate_optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(), rng=rng_state(), run_dir=str(self.run_dir.resolve()),
            selection='FaceForensics++/val/bounded_gate/video_auc',
            base_model_id=self.config['model_name'])

    def resume(self, checkpoint):
        if checkpoint['architecture'] != ARCHITECTURE or checkpoint['config'] != self.config:
            raise ValueError('Resume requires the original BGATE configuration')
        self.model.load_state_dict(checkpoint['state_dict'],strict=True)
        self.base_optimizer.load_state_dict(checkpoint['base_optimizer'])
        self.gate_optimizer.load_state_dict(checkpoint['gate_optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.step, self.start_epoch = checkpoint['global_step'],checkpoint['next_epoch']
        self.best_score = checkpoint['best_score']
        self.model.set_epoch(checkpoint['epoch'])
        restore_rng(checkpoint['rng'])

    def fit(self, train_loader, val_loader):
        interval = max(1,int(self.config.get('log_interval',100)))
        for epoch in range(self.start_epoch,self.config['nEpochs']):
            self.model.set_epoch(epoch)
            self.model.train()
            totals, count = defaultdict(float),0
            weights, labels = [],[]
            predictions = {branch: [] for branch in BRANCHES}
            started = time.monotonic()
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            for index,batch in enumerate(train_loader):
                data = move(batch,self.device)
                vals,pred = self.train_step(data)
                n = len(data['label'])
                count += n
                for k,v in vals.items():
                    totals[k] += v*n
                weights.extend(pred['gating_w'].detach().cpu().tolist())
                labels.extend((data['label']!=0).long().cpu().tolist())
                for branch in BRANCHES:
                    predictions[branch].extend(pred[branch+'_prob'].detach().cpu().tolist())
                if (index+1)%interval==0 or index+1==len(train_loader):
                    row = dict(epoch=epoch,step=self.step,gate_active=self.model.gate_active,
                        lr_base=self.base_optimizer.param_groups[0]['lr'],
                        lr_gate=self.gate_optimizer.param_groups[0]['lr'],**vals)
                    write_json(self.run_dir/'train/steps.jsonl',row,append=True)
                    self.logger.info('epoch=%d step=%d base=%.5f gate=%.5f active=%s',
                        epoch,self.step,vals['loss_base'],vals['loss_gate'],self.model.gate_active)
                    for k,v in vals.items():
                        self.writer.add_scalar('steps/'+k,v,self.step)
            if not count:
                raise ValueError('Empty training loader')
            stats = gate_stats(weights)
            for label,name in ((0,'real'),(1,'fake')):
                subset = np.asarray(weights)[np.asarray(labels)==label]
                if len(subset):
                    stats[name+'_mean'] = float(subset.mean())
            row = dict(epoch=epoch,step=self.step,losses={k:v/count for k,v in totals.items()},
                gate=stats,branches={k:binary_metrics(labels,p) for k,p in predictions.items()},
                images_per_second=count/max(time.monotonic()-started,1e-6),
                peak_vram_mb=torch.cuda.max_memory_allocated(self.device)/2**20 if self.device.type=='cuda' else 0)
            write_json(self.run_dir/'train/epochs.jsonl',row,append=True)
            metrics,arrays = evaluate(self.model,val_loader,self.device)
            write_json(self.run_dir/'validation/metrics.jsonl',dict(epoch=epoch,**metrics),append=True)
            for branch,items in metrics['branches'].items():
                for k,v in items.items():
                    if isinstance(v,(int,float)):
                        self.writer.add_scalar(f'validation/{branch}/{k}',v,epoch)
            score = metrics['branches']['bounded_gate']['video_auc']
            if not np.isfinite(score):
                raise ValueError('Validation video AUC is undefined; check real/fake labels and video grouping')
            improved = score>self.best_score
            if improved:
                self.best_score = score
            self.logger.info('validation epoch=%d video AUC: %s',epoch,
                {k:round(v['video_auc'],6) for k,v in metrics['branches'].items()})
            self.scheduler.step()
            payload = self.checkpoint(epoch)
            if improved:
                atomic_save(payload,self.run_dir/'checkpoints/best.pt')
                np.savez_compressed(self.run_dir/'validation/predictions_best.npz',**arrays)
            # Commit the resume boundary only after the new best artifact is saved.
            atomic_save(payload,self.run_dir/'checkpoints/last.pt')
            self.writer.flush()

    def close(self):
        self.writer.close()
