"""
HF Prediction Module (Eq.7-8)
Fuses gated expert predictions with base class features via skip connection,
then passes through deterministic layers + BayesianLinear output.

  - num_det_layers=2 (default): Linear→Act → Linear→Act → BayesianLinear
  - num_det_layers=1:           Linear→Act → BayesianLinear
  - num_det_layers=0:           BayesianLinear only
  - residual=True:              mu = h_fused + correction (HF learns correction)
  - activation: 'leaky_relu' (default) or 'tanh'
"""

import torch
import torch.nn as nn
from models.bayesian_linear import BayesianLinear


def _get_activation(name: str) -> nn.Module:
    if name == 'tanh':
        return nn.Tanh()
    return nn.LeakyReLU(0.01)


class HFPredictionModule(nn.Module):

    def __init__(self, fused_dim: int, base_dim: int, hidden_dim: int = 64,
                 output_dim: int = 1, num_det_layers: int = 2,
                 activation: str = 'leaky_relu', residual: bool = False,
                 bayesian_bias: bool = True):
        super().__init__()
        self.residual = residual
        in_dim = fused_dim + base_dim

        if num_det_layers >= 2:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), _get_activation(activation),
                nn.Linear(hidden_dim, hidden_dim), _get_activation(activation),
            )
            bay_in = hidden_dim
        elif num_det_layers == 1:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), _get_activation(activation),
            )
            bay_in = hidden_dim
        else:
            self.layers = nn.Identity()
            bay_in = in_dim

        self.bayesian_output = BayesianLinear(bay_in, output_dim, bias=bayesian_bias)

    def forward(self, h_fused: torch.Tensor, h_base: torch.Tensor):
        h = torch.cat([h_fused, h_base], dim=-1)
        h_s = self.layers(h)
        correction, var = self.bayesian_output(h_s)
        if self.residual:
            mu = h_fused + correction
        else:
            mu = correction
        return mu, var

    def kl_divergence(self) -> torch.Tensor:
        return self.bayesian_output.kl_divergence()
