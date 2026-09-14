"""Bias-tuned (BitFit) CLIP + official SSPANet + Saliency-Guided Feature Fusion + Learned Gating (SAF-MIL)."""
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


@DETECTOR.register_module(module_name='bias_saf_mil')
@DETECTOR.register_module(module_name='saf_mil')
class BiasSAFMILDetector(AbstractDetector):
    """
    Saliency-Aware Fusion with Multiple Instance Learning (SAF-MIL).

    Key Innovations:
    1. Saliency-Guided Feature Fusion:
       Uses spatial saliency from Patch Head to weight patch tokens dynamically
       instead of Global Average Pooling (which dilutes localized fake artifacts).
    2. Lightweight Learned Gating with Joint Supervision:
       Uses a 2-layer MLP to learn the optimal sample-level combination weight w(x)
       between Feature Fusion (Global) and MIL (Local), trained end-to-end with
       joint supervision loss L_gate.
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
        self.tau_saliency = float(self.config.get('tau_saliency', 1.0))
        self.lambda_gate = float(self.config.get('lambda_gate', 1.0))
        self.learned_gate = True

        if self.mil_weight < 0 or (not self.use_patch and self.mil_weight != 0):
            raise ValueError('lambda_mil must be nonnegative and zero when use_patch=false')
        if not math.isfinite(self.lambda_gate) or self.lambda_gate < 0:
            raise ValueError('lambda_gate must be finite and nonnegative')
        if self.tau_saliency <= 0:
            raise ValueError('tau_saliency must be positive')

        self.head = nn.Linear(dim, 2)
        if self.use_patch:
            self.sspanet = ATTN_Block(dim) if self.use_sspanet else nn.Identity()
            self.patch_head = nn.Conv2d(dim, 1, 1)
            self.fusion_alpha = nn.Parameter(
                torch.tensor(float(self.config.get('fusion_alpha_init', 0.1)))
            )
            gate_hidden = int(self.config.get('gate_hidden_dim', 16))
            if gate_hidden < 1:
                raise ValueError('gate_hidden_dim must be positive')
            self.gate = nn.Sequential(
                nn.Linear(5, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, 1)
            )
            # Initialize final gate layer to zero -> w = sigmoid(0) = 0.5 (balanced ensemble) at step 0
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

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
            raise ValueError('BiasSAFMILDetector expects [B,3,H,W] frames, not video tensors')
        outputs = self.backbone(image)
        cls = outputs.pooler_output
        if not self.use_patch:
            return cls, cls, None, None, None, None

        size = self.backbone.config.patch_size
        h, w = image.shape[-2] // size, image.shape[-1] // size
        tokens = outputs.last_hidden_state[:, 1:]
        if tokens.shape[1] != h * w:
            raise ValueError('Patch token count does not match image grid')

        patch_map = tokens.transpose(1, 2).reshape(image.shape[0], -1, h, w)
        refined = self.sspanet(patch_map)
        patch_logits = self.patch_head(refined).squeeze(1)

        # Saliency-Guided Feature Fusion:
        # Compute spatial attention weights from patch logits
        # Shape: [B, 1, N] where N = H * W
        saliency_attn = F.softmax(patch_logits.flatten(1) / max(1e-6, self.tau_saliency), dim=-1).unsqueeze(1)
        refined_flat = refined.flatten(2)  # [B, D, N]
        # Weighted sum: [B, D, N] x [B, N, 1] -> [B, D, 1] -> [B, D]
        local = torch.bmm(refined_flat, saliency_attn.transpose(1, 2)).squeeze(-1)

        fused = F.normalize(cls, dim=1, eps=1e-6) + self.fusion_alpha * F.normalize(local, dim=1, eps=1e-6)
        return fused, cls, refined, patch_map, patch_logits, saliency_attn

    def features(self, data_dict):
        return self._extract(data_dict)[0]

    def classifier(self, features):
        return self.head(features)

    def forward(self, data_dict, inference=False):
        fused, cls, refined, patch_map, patch_logits, saliency_attn = self._extract(data_dict)
        normalized = F.normalize(fused, dim=1, eps=1e-6)
        feat_logits = self.classifier(normalized)
        feat_prob = feat_logits.softmax(1)[:, 1]
        cls_only_prob = self.head(F.normalize(cls, dim=1, eps=1e-6)).softmax(1)[:, 1]

        result = {
            'cls': feat_logits,
            'prob': feat_prob,
            'feat': fused,
            'feat_norm': normalized,
            'cls_only_prob': cls_only_prob,
            'feature_fusion_prob': feat_prob,
        }

        if refined is not None:
            mil_logits = topk_mil_logits(patch_logits, self.mil_topk)
            mil_prob = mil_logits.sigmoid()

            # Routing features for the learned gate (detached to preserve branch specialization)
            flat = patch_logits.detach().flatten(1)
            top = flat.topk(self.mil_topk, dim=1).values
            remaining = flat.shape[1] - self.mil_topk
            rest_mean = (flat.sum(1) - top.sum(1)) / remaining if remaining else top.mean(1)
            gate_features = torch.stack((
                (feat_logits[:, 1] - feat_logits[:, 0]).detach(),
                mil_logits.detach(),
                feat_prob.detach() - mil_prob.detach(),
                top.std(1, unbiased=False),
                top.mean(1) - rest_mean,
            ), dim=1)

            w = self.gate(gate_features).squeeze(1).sigmoid()
            # Joint prediction mixture
            final_prob = (1.0 - w) * feat_prob + w * mil_prob

            result.update(
                prob=final_prob,
                patch_logits=patch_logits,
                mil_logits=mil_logits,
                mil_prob=mil_prob,
                gating_w=w,
                gating_w_f=w,
                saliency_attn=saliency_attn,
            )

            with torch.no_grad():
                probs = patch_logits.detach().sigmoid().flatten(1)
                mass = probs / probs.sum(1, keepdim=True).clamp_min(1e-6)
                result['diagnostics'] = {
                    'fusion_alpha': self.fusion_alpha.detach(),
                    'gating_weight_mean': w.detach().mean(),
                    'patch_probability_mean': probs.mean(),
                    'patch_probability_std': probs.std(dim=1, unbiased=False).mean(),
                    'patch_entropy': (-(mass * mass.clamp_min(1e-8).log()).sum(1) / math.log(max(2, probs.shape[1]))).mean(),
                    'sspa_relative_change': ((refined.detach() - patch_map.detach()).flatten(1).norm(dim=1) /
                        patch_map.detach().flatten(1).norm(dim=1).clamp_min(1e-6)).mean(),
                    'branch_disagreement': ((feat_prob.detach() >= .5) != (mil_prob.detach() >= .5)).float().mean(),
                }
        return result

    def get_losses(self, data_dict, pred_dict):
        label = (data_dict['label'] != 0).long()
        ce = self.loss_ce(pred_dict['cls'], label)
        mil = ce.new_zeros(())
        loss_gate = ce.new_zeros(())

        if self.use_patch and self.mil_weight > 0:
            weight = self.loss_ce.weight[label]
            per_sample_mil = F.binary_cross_entropy_with_logits(pred_dict['mil_logits'], label.float(), reduction='none')
            mil = (per_sample_mil * weight).sum() / weight.sum()

            if self.lambda_gate > 0 and 'prob' in pred_dict and pred_dict.get('gating_w') is not None:
                # Float32 BCE avoids half-precision arithmetic instabilities
                per_sample_gate = F.binary_cross_entropy(pred_dict['prob'].float(), label.float(), reduction='none')
                loss_gate = (per_sample_gate * weight).sum() / weight.sum()

        overall = ce + self.mil_weight * mil + self.lambda_gate * loss_gate
        result = {
            'overall': overall,
            'loss_ce': ce,
            'loss_mil': mil,
            'loss_mil_weighted': self.mil_weight * mil,
            'loss_gate': loss_gate,
            'loss_gate_weighted': self.lambda_gate * loss_gate,
        }

        with torch.no_grad():
            for value, name in [(0, 'real'), (1, 'fake')]:
                mask = label == value
                if mask.any():
                    result[f'{name}_loss'] = self.loss_ce(pred_dict['cls'][mask], label[mask])
                    result[f'{name}_prob'] = pred_dict['prob'][mask].mean()
                    if 'mil_prob' in pred_dict:
                        result[f'{name}_mil_prob'] = pred_dict['mil_prob'][mask].mean()
                    if pred_dict.get('gating_w') is not None:
                        result[f'{name}_gating_w'] = pred_dict['gating_w'][mask].mean()
            result.update(pred_dict.get('diagnostics', {}))
        return result

    def get_train_metrics(self, data_dict, pred_dict):
        label = (data_dict['label'] != 0).long()
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred_dict['cls'].detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}


# Backward compatibility aliases
SAFMILDetector = BiasSAFMILDetector

__all__ = ['BiasSAFMILDetector', 'SAFMILDetector', 'topk_mil_logits']
