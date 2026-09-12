"""Backward compatibility shim for biasln_detector."""
from .ln_sspanet_mil_detector import (
    LNSSPANetMILDetector,
    BiasLNDetector,
    topk_mil_logits,
)

__all__ = ['LNSSPANetMILDetector', 'BiasLNDetector', 'topk_mil_logits']
