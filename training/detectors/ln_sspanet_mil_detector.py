"""LN-SSPANet-MIL: LayerNorm CLIP + official SSPANet + CE / patch Top-k MIL detector."""
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


@DETECTOR.register_module(module_name='ln_sspanet_mil')
class LNSSPANetMILDetector(AbstractDetector):
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
        if self.use_patch:
            self.sspanet = ATTN_Block(dim) if self.use_sspanet else nn.Identity()
            self.patch_head = nn.Conv2d(dim, 1, 1)
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
        for module in self.backbone.modules():
            if isinstance(module, nn.LayerNorm):
                module.requires_grad_(True)
        counts = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                group = 'backbone_ln' if name.startswith('backbone.') else name.split('.')[0]
                counts[group] = counts.get(group, 0) + param.numel()
        self.trainable_counts = counts
        logger.info('Trainable parameter groups: %s; total=%d', counts, sum(counts.values()))

    def _extract(self, data_dict):
        image = data_dict['image']
        if image.ndim != 4:
            raise ValueError('LNSSPANetMIL expects [B,3,H,W] frames, not video tensors')
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
        return cls, cls, refined, patch_map

    def features(self, data_dict):
        return self._extract(data_dict)[0]

    def classifier(self, features):
        return self.head(features)

    def forward(self, data_dict, inference=False):
        _, cls, refined, patch_map = self._extract(data_dict)
        normalized = F.normalize(cls, dim=1, eps=1e-6)
        logits = self.classifier(normalized)
        prob = logits.softmax(1)[:, 1]
        result = {'cls': logits, 'prob': prob, 'feat': cls, 'feat_norm': normalized,
                  'cls_only_prob': prob}
        if refined is not None:
            patch_logits = self.patch_head(refined).squeeze(1)
            mil_logits = topk_mil_logits(patch_logits, self.mil_topk)
            result.update(patch_logits=patch_logits, mil_logits=mil_logits, mil_prob=mil_logits.sigmoid())
            with torch.no_grad():
                probs = patch_logits.detach().sigmoid().flatten(1)
                mass = probs / probs.sum(1, keepdim=True).clamp_min(1e-6)
                result['diagnostics'] = {
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
        result = {'overall': ce + self.mil_weight * mil, 'loss_ce': ce,
                  'loss_mil': mil, 'loss_mil_weighted': self.mil_weight * mil}
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


# Backward compatibility alias
def __getattr__(name):
    if name in ('BiasSSPANetMILDetector', 'BiasLNDetector'):
        from .bias_sspanet_mil_detector import BiasSSPANetMILDetector, BiasLNDetector
        globals()['BiasSSPANetMILDetector'] = BiasSSPANetMILDetector
        globals()['BiasLNDetector'] = BiasLNDetector
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + ['BiasSSPANetMILDetector', 'BiasLNDetector'])


__all__ = ['LNSSPANetMILDetector', 'BiasSSPANetMILDetector', 'BiasLNDetector', 'topk_mil_logits']
