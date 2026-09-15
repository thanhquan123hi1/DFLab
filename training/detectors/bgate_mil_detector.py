"""Feature-fusion baselines with an independently optimized bounded decision gate."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from detectors.ln_sspanet_mil_detector import LNSSPANetMILDetector
from detectors.bias_sspanet_mil_detector import BiasSSPANetMILDetector
from metrics.registry import DETECTOR

ARCHITECTURE = 'sspanet_bgate_mil_v1'
BRANCHES = ('cls_only', 'feature_fusion', 'mil', 'fixed_ensemble', 'bounded_gate')


class BoundedGateMixin:
    def __init__(self, config=None):
        super().__init__(config)
        if not self.use_patch or self.mil_weight <= 0:
            raise ValueError('BGATE requires patches and positive lambda_mil')
        self.gate_radius = float(self.config.get('gate_radius', .25))
        self.gate_regularization = float(self.config.get('gate_regularization', .1))
        self.gate_warmup_epochs = int(self.config.get('gate_warmup_epochs', 1))
        hidden = int(self.config.get('gate_hidden_dim', 16))
        if not 0 < self.gate_radius < .5 or hidden < 1 or self.gate_warmup_epochs < 0:
            raise ValueError('Invalid gate radius, hidden size or warm-up')
        if not math.isfinite(self.gate_regularization) or self.gate_regularization < 0:
            raise ValueError('Invalid gate regularization')
        # Do not perturb the baseline RNG stream (including subsequent data sampling).
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(self.config.get('manualSeed', 1024)) + 17011)
            self.gate = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        self.trainable_counts['gate'] = sum(p.numel() for p in self.gate.parameters())
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.gate_active = self.epoch >= self.gate_warmup_epochs

    def forward(self, data_dict, inference=False):
        out = super().forward(data_dict, inference=inference)
        pf, pl = out['prob'], out['mil_prob']
        f, l = pf.detach().float(), pl.detach().float()
        u = torch.stack((f, l, f-l, (f-l).abs()), dim=1)
        w = .5 + self.gate_radius * self.gate(u).squeeze(1).tanh() if self.gate_active else torch.full_like(f, .5)
        out.update(feature_fusion_prob=pf, fixed_ensemble_prob=.5*(pf+pl),
                   bounded_gate_prob=(1-w)*f+w*l, gating_w=w)
        # Keep prob/cls semantics of the baseline for its CE and diagnostics.
        return out

    def get_losses(self, data_dict, pred_dict):
        out = super().get_losses(data_dict, pred_dict)
        base = out['overall']
        p = pred_dict['bounded_gate_prob'].float().clamp(1e-7, 1-1e-7)
        y = (data_dict['label'] != 0).float()
        bce = F.binary_cross_entropy(p, y)
        reg = (pred_dict['gating_w']-.5).square().mean()
        gate = bce + self.gate_regularization * reg
        out.update(loss_base=base, loss_gate_bce=bce, loss_gate_reg=reg,
                   loss_gate=gate, overall=base+gate if self.gate_active else base)
        return out


@DETECTOR.register_module(module_name='bias_sspanet_bgate_mil')
class BiasSSPANetBGateMILDetector(BoundedGateMixin, BiasSSPANetMILDetector):
    pass


@DETECTOR.register_module(module_name='ln_sspanet_bgate_mil')
class LNSSPANetBGateMILDetector(BoundedGateMixin, LNSSPANetMILDetector):
    pass


MODELS = {'bias_sspanet_bgate_mil': BiasSSPANetBGateMILDetector,
          'ln_sspanet_bgate_mil': LNSSPANetBGateMILDetector}
