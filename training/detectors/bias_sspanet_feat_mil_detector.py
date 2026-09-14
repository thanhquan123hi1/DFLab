"""Bias-tuned (BitFit) CLIP + official SSPANet + Feature Fusion (CLS + alpha * Mean(SSPANet)) + CE / patch Top-k MIL detector."""
import logging
import math
import torch
from torch import nn
from torch.nn import functional as F
from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from .modules.sspanet import ATTN_Block
try:
    from detectors import DETECTOR
except (ImportError, ModuleNotFoundError):
    from metrics.registry import DETECTOR

logger = logging.getLogger(__name__)


def topk_mil_logits(patch_logits, k):
    """Pool logits, not probabilities; gradients reach selected patches."""
    flat = patch_logits.flatten(1)
    if not 1 <= k <= flat.shape[1]:
        raise ValueError(f'mil_topk={k} must be in [1, {flat.shape[1]}]')
    return flat.topk(k, dim=1).values.mean(dim=1)


def compute_dynamic_gating(base_prob, patch_logits, w_min=0.05, w_max=0.85, tau=0.15):
    """
    Bilateral Dynamic Gating (BDG):
    Dynamically calculates sample-level MIL decision weight w(x) in [w_min, w_max].
    - Base confidence reflects distance from decision boundary: 2 * |base_prob - 0.5|
    - MIL confidence reflects patch peak salience and concentration: salience * (1 - entropy)
    """
    if torch.is_tensor(base_prob) and torch.is_tensor(patch_logits):
        base_conf = 2.0 * torch.abs(base_prob.detach() - 0.5)
        probs = patch_logits.detach().sigmoid().flatten(1)
        if base_conf.device != probs.device:
            base_conf = base_conf.to(probs.device)
        salience = probs.max(dim=1).values - probs.mean(dim=1)
        mass = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(mass * mass.clamp_min(1e-8).log()).sum(dim=1) / math.log(max(2, probs.shape[1]))
        mil_conf = salience * (1.0 - entropy).clamp(min=0.0, max=1.0)
        ratio = (mil_conf - 0.5 * base_conf) / max(1e-6, tau)
        w = w_min + (w_max - w_min) * torch.sigmoid(ratio)
        return w, base_conf, mil_conf
    else:
        import numpy as np
        base_prob = np.asarray(base_prob, dtype=float)
        patch_logits = np.asarray(patch_logits, dtype=float)
        base_conf = 2.0 * np.abs(base_prob - 0.5)
        safe_logits = np.clip(patch_logits, -88.0, 88.0)
        probs = 1.0 / (1.0 + np.exp(-safe_logits))
        probs = probs.reshape(probs.shape[0], -1)
        salience = probs.max(axis=1) - probs.mean(axis=1)
        sum_mass = np.maximum(probs.sum(axis=1, keepdims=True), 1e-6)
        mass = probs / sum_mass
        safe_mass = np.maximum(mass, 1e-8)
        entropy = -np.sum(mass * np.log(safe_mass), axis=1) / math.log(max(2, probs.shape[1]))
        mil_conf = np.clip(salience * np.clip(1.0 - entropy, 0.0, 1.0), 0.0, 1.0)
        ratio = np.clip((mil_conf - 0.5 * base_conf) / max(1e-6, tau), -88.0, 88.0)
        w = w_min + (w_max - w_min) * (1.0 / (1.0 + np.exp(-ratio)))
        return w, base_conf, mil_conf


@DETECTOR.register_module(module_name='bias_sspanet_feat_mil')
@DETECTOR.register_module(module_name='bias_sspanet_ff_mil')
class BiasSSPANetFeatMILDetector(AbstractDetector):
    """
    BitFit + official SSPANet + Feature Fusion (CLS + alpha * Mean(SSPANet)) + Top-k MIL.

    Restores the original feature-level fusion formulation where the classifier operates
    on the combined CLS and global average pooled SSPANet patch representations:
        fused = Normalize(cls) + alpha * Normalize(Mean(refined_patches))
    This provides dual supervision back to SSPANet from both CrossEntropy and MIL losses.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.backbone = self.build_backbone(self.config)
        dim = self.backbone.config.hidden_size
        self.use_patch = bool(self.config.get('use_patch', True))
        self.use_sspanet = bool(self.config.get('use_sspanet', True))
        self.mil_weight = float(self.config.get('lambda_mil', 0.3))
        self.mil_topk = int(self.config.get('mil_topk', 16))
        if self.mil_weight < 0 or (not self.use_patch and self.mil_weight != 0):
            raise ValueError('lambda_mil must be nonnegative and zero when use_patch=false')
        self.head = nn.Linear(dim, 2)
        self.gating_w_min = float(self.config.get('gating_w_min', 0.05))
        self.gating_w_max = float(self.config.get('gating_w_max', 0.85))
        self.gating_tau = float(self.config.get('gating_tau', 0.15))
        if self.use_patch:
            self.sspanet = ATTN_Block(dim) if self.use_sspanet else nn.Identity()
            self.patch_head = nn.Conv2d(dim, 1, 1)
            self.fusion_alpha = nn.Parameter(torch.tensor(float(self.config.get('fusion_alpha_init', 0.1))))
            if self.mil_weight == 0:
                self.patch_head.requires_grad_(False)
        self.build_loss(self.config)
        self._setup_trainable_params()

    def build_backbone(self, config):
        from transformers import CLIPVisionModel
        return CLIPVisionModel.from_pretrained(
            config.get('clip_model_name', 'openai/clip-vit-large-patch14'),
            local_files_only=config.get('local_files_only', False)).vision_model

    def build_loss(self, config):
        weights = torch.tensor([float(config.get('weight_real', 1.0)), float(config.get('weight_fake', 1.0))])
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('Class weights must be finite and positive')
        self.loss_ce = nn.CrossEntropyLoss(weight=weights,
            label_smoothing=float(config.get('label_smoothing', 0.0)))

    def _setup_trainable_params(self):
        self.backbone.requires_grad_(False)
        for name, param in self.backbone.named_parameters():
            if 'bias' in name:
                param.requires_grad_(True)
        counts = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                group = 'backbone_bias' if name.startswith('backbone.') else name.split('.')[0]
                counts[group] = counts.get(group, 0) + param.numel()
        self.trainable_counts = counts
        logger.info('Trainable parameter groups: %s; total=%d', counts, sum(counts.values()))

    def _extract(self, data_dict):
        image = data_dict['image']
        if image.ndim != 4:
            raise ValueError('BiasSSPANetFeatMIL expects [B,3,H,W] frames, not video tensors')
        outputs = self.backbone(image)
        cls = outputs.pooler_output
        if not self.use_patch:
            return cls, cls, None, None
        size = self.backbone.config.patch_size
        h, w = image.shape[-2] // size, image.shape[-1] // size
        tokens = outputs.last_hidden_state[:, 1:]
        if tokens.shape[1] != h * w:
            raise ValueError('Patch token count does not match image grid')
        patch_map = tokens.transpose(1, 2).reshape(image.shape[0], -1, h, w)
        refined = self.sspanet(patch_map)
        local = refined.mean(dim=(2, 3))
        fused = F.normalize(cls, dim=1, eps=1e-6) + self.fusion_alpha * F.normalize(local, dim=1, eps=1e-6)
        return fused, cls, refined, patch_map

    def features(self, data_dict):
        return self._extract(data_dict)[0]

    def classifier(self, features):
        return self.head(features)

    def compute_dynamic_gating(self, base_prob, patch_logits, w_min=None, w_max=None, tau=None):
        w_min = self.gating_w_min if w_min is None else w_min
        w_max = self.gating_w_max if w_max is None else w_max
        tau = self.gating_tau if tau is None else tau
        return compute_dynamic_gating(base_prob, patch_logits, w_min=w_min, w_max=w_max, tau=tau)

    def _compute_dynamic_gating(self, cls_prob, patch_logits):
        return self.compute_dynamic_gating(cls_prob, patch_logits)

    def forward(self, data_dict, inference=False):
        fused, cls, refined, patch_map = self._extract(data_dict)
        normalized = F.normalize(fused, dim=1, eps=1e-6)
        logits = self.classifier(normalized)
        cls_only_prob = self.head(F.normalize(cls, dim=1, eps=1e-6)).softmax(1)[:, 1]
        result = {
            'cls': logits,
            'prob': logits.softmax(1)[:, 1],
            'feat': fused,
            'feat_norm': normalized,
            'cls_only_prob': cls_only_prob,
        }
        if refined is not None:
            patch_logits = self.patch_head(refined).squeeze(1)
            mil_logits = topk_mil_logits(patch_logits, self.mil_topk)
            mil_prob = mil_logits.sigmoid()
            w_cls, _, _ = self.compute_dynamic_gating(cls_only_prob, patch_logits)
            w_f, _, _ = self.compute_dynamic_gating(result['prob'], patch_logits)
            bdg_cls_mil_prob = (1.0 - w_cls) * cls_only_prob + w_cls * mil_prob
            bdg_f_mil_prob = (1.0 - w_f) * result['prob'] + w_f * mil_prob
            result.update(
                patch_logits=patch_logits,
                mil_logits=mil_logits,
                mil_prob=mil_prob,
                gating_w=w_cls,
                gating_w_cls=w_cls,
                gating_w_f=w_f,
                bdg_cls_mil_prob=bdg_cls_mil_prob,
                bdg_f_mil_prob=bdg_f_mil_prob,
            )
            with torch.no_grad():
                probs = patch_logits.detach().sigmoid().flatten(1)
                mass = probs / probs.sum(1, keepdim=True).clamp_min(1e-6)
                result['diagnostics'] = {
                    'fusion_alpha': self.fusion_alpha.detach(),
                    'gating_w_cls_mean': w_cls.detach().mean(),
                    'gating_w_f_mean': w_f.detach().mean(),
                    'gating_weight_mean': w_cls.detach().mean(),
                    'patch_probability_mean': probs.mean(),
                    'patch_probability_std': probs.std(dim=1, unbiased=False).mean(),
                    'patch_entropy': (-(mass * mass.clamp_min(1e-8).log()).sum(1) / math.log(max(2, probs.shape[1]))).mean(),
                    'sspa_relative_change': ((refined.detach() - patch_map.detach()).flatten(1).norm(dim=1) /
                        patch_map.detach().flatten(1).norm(dim=1).clamp_min(1e-6)).mean(),
                    'branch_disagreement': ((result['prob'].detach() >= .5) != (result['mil_prob'].detach() >= .5)).float().mean(),
                }
        return result

    def get_losses(self, data_dict, pred_dict):
        label = (data_dict['label'] != 0).long()
        ce = self.loss_ce(pred_dict['cls'], label)
        mil = ce.new_zeros(())
        if self.use_patch and self.mil_weight > 0:
            weight = self.loss_ce.weight[label]
            per_sample = F.binary_cross_entropy_with_logits(pred_dict['mil_logits'], label.float(), reduction='none')
            mil = (per_sample * weight).sum() / weight.sum()
        result = {
            'overall': ce + self.mil_weight * mil,
            'loss_ce': ce,
            'loss_mil': mil,
            'loss_mil_weighted': self.mil_weight * mil,
        }
        with torch.no_grad():
            for value, name in [(0, 'real'), (1, 'fake')]:
                mask = label == value
                if mask.any():
                    result[f'{name}_loss'] = self.loss_ce(pred_dict['cls'][mask], label[mask])
                    result[f'{name}_prob'] = pred_dict['prob'][mask].mean()
                    if 'mil_prob' in pred_dict:
                        result[f'{name}_mil_prob'] = pred_dict['mil_prob'][mask].mean()
            result.update(pred_dict.get('diagnostics', {}))
        return result

    def get_train_metrics(self, data_dict, pred_dict):
        label = (data_dict['label'] != 0).long()
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred_dict['cls'].detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}


# Backward compatibility aliases
BiasSSPANetFFMILDetector = BiasSSPANetFeatMILDetector

__all__ = ['BiasSSPANetFeatMILDetector', 'BiasSSPANetFFMILDetector', 'topk_mil_logits', 'compute_dynamic_gating']
