"""BitFit SSPANet-MIL with a jointly trained decision gate."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .bias_sspanet_mil_detector import BiasSSPANetMILDetector
from metrics.registry import DETECTOR


@DETECTOR.register_module(module_name='bias_gmil')
class BiasGMILDetector(BiasSSPANetMILDetector):
    def __init__(self, config=None):
        super().__init__(config)
        if not self.use_patch or self.mil_weight <= 0:
            raise ValueError('bias_gmil requires use_patch=true and lambda_mil>0')
        hidden = int(self.config.get('gate_hidden_dim', 16))
        self.lambda_fusion = float(self.config.get('lambda_fusion', 1.0))
        if hidden < 1 or not math.isfinite(self.lambda_fusion) or self.lambda_fusion <= 0:
            raise ValueError('gate_hidden_dim and lambda_fusion must be positive')
        self.gate = nn.Sequential(nn.Linear(5, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self._setup_trainable_params()

    def _fusion_gate(self, logits, cls_prob, patch_logits, mil_logits):
        # Detach routing features only; the probability mixture retains branch gradients.
        flat = patch_logits.detach().flatten(1)
        top = flat.topk(self.mil_topk, dim=1).values
        remaining = flat.shape[1] - self.mil_topk
        rest_mean = ((flat.sum(1) - top.sum(1)) / remaining
                     if remaining else top.mean(1))
        features = torch.stack((
            (logits[:, 1] - logits[:, 0]).detach(), mil_logits.detach(),
            cls_prob.detach() - mil_logits.detach().sigmoid(),
            top.std(1, unbiased=False), top.mean(1) - rest_mean,
        ), dim=1)
        self_gate_logits = self.gate(features).squeeze(1)
        w = self_gate_logits.sigmoid()
        return w, 2 * (cls_prob.detach() - .5).abs(), 2 * (mil_logits.detach().sigmoid() - .5).abs()

    def get_losses(self, data_dict, pred_dict):
        losses = super().get_losses(data_dict, pred_dict)
        labels = (data_dict['label'] != 0).long()
        weights = self.loss_ce.weight[labels]
        # Float32 BCE avoids half-precision probability arithmetic.
        per_sample = F.binary_cross_entropy(pred_dict['prob'].float(), labels.float(), reduction='none')
        fusion = (per_sample * weights).sum() / weights.sum()
        losses.update(loss_fusion=fusion, loss_fusion_weighted=self.lambda_fusion * fusion)
        losses['overall'] = losses['overall'] + self.lambda_fusion * fusion
        return losses
