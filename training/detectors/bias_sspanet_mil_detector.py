"""Bias-tuned (BitFit) CLIP + official SSPANet + CE / patch Top-k MIL detector."""
import logging
try:
    from detectors import DETECTOR
except (ImportError, ModuleNotFoundError):
    from metrics.registry import DETECTOR
from .ln_sspanet_mil_detector import LNSSPANetMILDetector, topk_mil_logits

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='bias_sspanet_mil')
class BiasSSPANetMILDetector(LNSSPANetMILDetector):
    """
    Bias-tuning (BitFit) variant of LNSSPANetMILDetector.

    Freezes all weights in the backbone CLIP ViT and only optimizes bias parameters,
    along with the auxiliary detector modules (SSPANet, classifier head, and patch head).
    """

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


# Backward compatibility alias
BiasLNDetector = BiasSSPANetMILDetector

__all__ = ['BiasSSPANetMILDetector', 'BiasLNDetector', 'LNSSPANetMILDetector', 'topk_mil_logits']
