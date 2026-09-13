"""CAMIL: Cross-Attention Multiple Instance Learning detector with BitFit CLIP backbone."""
import logging
import math
import torch
from torch import nn
from torch.nn import functional as F
from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from .modules.sspanet import ATTN_Block
from .ln_sspanet_mil_detector import topk_mil_logits
try:
    from detectors import DETECTOR
except (ImportError, ModuleNotFoundError):
    from metrics.registry import DETECTOR

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='camil')
@DETECTOR.register_module(module_name='bias_camil')
class CAMILDetector(AbstractDetector):
    """
    Cross-Attention Multiple Instance Learning (CAMIL) Detector.

    Combines:
      - CLIP ViT backbone with bias-tuning (BitFit: only backbone biases are trainable).
      - SSPANet (ATTN_Block) for spatial-frequency feature enhancement.
      - Cross-Attention Fusion: Query = LayerNorm(CLS), Key/Value = LayerNorm(SSPANet patches).
      - Zero-initialized residual connection: fused = cls + fusion_gamma * attn_out.
      - Multiple Instance Learning: patch_head Conv2d + top-k MIL logits.
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
        if self.use_patch:
            self.sspanet = ATTN_Block(dim) if self.use_sspanet else nn.Identity()
            self.patch_head = nn.Conv2d(dim, 1, 1)
            num_heads = int(self.config.get('cross_attn_heads', 4))
            dropout = float(self.config.get('cross_attn_dropout', 0.0))
            if num_heads <= 0 or dim % num_heads != 0:
                raise ValueError(f'cross_attn_heads={num_heads} must be positive and divide dim={dim}')
            if not 0.0 <= dropout < 1.0:
                raise ValueError(f'cross_attn_dropout={dropout} must be in [0, 1)')
            self.norm_cls = nn.LayerNorm(dim)
            self.norm_patch = nn.LayerNorm(dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.fusion_gamma = nn.Parameter(
                torch.tensor(float(self.config.get('fusion_gamma_init', 0.0)))
            )
            if self.mil_weight == 0:
                self.patch_head.requires_grad_(False)

        self.build_loss(self.config)
        self._setup_trainable_params()

    def build_backbone(self, config):
        from transformers import CLIPVisionModel
        return CLIPVisionModel.from_pretrained(
            config.get('clip_model_name', 'openai/clip-vit-large-patch14'),
            local_files_only=config.get('local_files_only', False)
        ).vision_model

    def build_loss(self, config):
        weights = torch.tensor([
            float(config.get('weight_real', 1.0)),
            float(config.get('weight_fake', 1.0))
        ])
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('Class weights must be finite and positive')
        self.loss_ce = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=float(config.get('label_smoothing', 0.0))
        )

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
            raise ValueError('CAMILDetector expects [B,3,H,W] frames, not video tensors')
        outputs = self.backbone(image)
        cls = outputs.pooler_output
        if not self.use_patch:
            return cls, cls, None, None, None

        size = self.backbone.config.patch_size
        h, w = image.shape[-2] // size, image.shape[-1] // size
        tokens = outputs.last_hidden_state[:, 1:]
        if tokens.shape[1] != h * w:
            raise ValueError('Patch token count does not match image grid')
        patch_map = tokens.transpose(1, 2).reshape(image.shape[0], -1, h, w)
        refined = self.sspanet(patch_map)

        # Flatten refined patches back to sequence format [B, N, D]
        refined_patches = refined.flatten(2).transpose(1, 2)

        # Cross-Attention Fusion
        # Query: Norm(CLS) [B, 1, D]
        # Key & Value: Norm(SSPANet patches) [B, N, D]
        q = self.norm_cls(cls).unsqueeze(1)
        k = self.norm_patch(refined_patches)
        v = k
        attn_out, attn_weights = self.cross_attn(q, k, v, need_weights=True)
        attn_out = attn_out.squeeze(1)

        fused = cls + self.fusion_gamma * attn_out
        return fused, cls, refined, patch_map, attn_weights

    def features(self, data_dict):
        return self._extract(data_dict)[0]

    def classifier(self, features):
        return self.head(features)

    def forward(self, data_dict, inference=False):
        fused, cls, refined, patch_map, attn_weights = self._extract(data_dict)
        normalized = F.normalize(fused, dim=1, eps=1e-6)
        logits = self.classifier(normalized)
        result = {
            'cls': logits,
            'prob': logits.softmax(1)[:, 1],
            'feat': fused,
            'feat_norm': normalized,
            'cls_only_prob': self.head(F.normalize(cls, dim=1, eps=1e-6)).softmax(1)[:, 1],
        }
        if attn_weights is not None:
            result['attn_weights'] = attn_weights

        if refined is not None:
            patch_logits = self.patch_head(refined).squeeze(1)
            mil_logits = topk_mil_logits(patch_logits, self.mil_topk)
            result.update(
                patch_logits=patch_logits,
                mil_logits=mil_logits,
                mil_prob=mil_logits.sigmoid()
            )
            with torch.no_grad():
                probs = patch_logits.detach().sigmoid().flatten(1)
                mass = probs / probs.sum(1, keepdim=True).clamp_min(1e-6)
                result['diagnostics'] = {
                    'fusion_gamma': self.fusion_gamma.detach(),
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
            'loss_mil_weighted': self.mil_weight * mil
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


# Alias for backward compatibility / alternative naming
BiasCAMILDetector = CAMILDetector

__all__ = ['CAMILDetector', 'BiasCAMILDetector', 'topk_mil_logits']
