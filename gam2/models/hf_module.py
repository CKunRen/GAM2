"""HF prediction module (Eqs. (6)-(11) of the paper).

The module receives the concatenation of the quality-gated expert predictions
and the base feature (Eq. (6)):

    [q_1 h_1, ..., q_{s-1} h_{s-1} ; h_base]

and maps it to the HF output through ``hidden_layers`` deterministic layers
followed by the Bayesian output layer.  With scalar expert outputs the input
width is h + s - 1, as in Table 1.
"""

import torch
import torch.nn as nn

from .base_module import make_activation
from .bayesian_linear import BayesianLinear


class HFPredictionModule(nn.Module):
    """h_s = f_HF([q_1 h_1, ..., q_{s-1} h_{s-1} ; h_base]; theta_HF)."""

    def __init__(self, hidden_size: int, num_experts: int,
                 output_dim: int = 1, hidden_layers: int = 2,
                 activation: str = 'relu', bias: bool = True,
                 residual: bool = False):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError('hidden_layers must be >= 1')
        in_dim = hidden_size + num_experts * output_dim
        self.in_dim = in_dim
        self.hidden_layers = hidden_layers
        self.residual = residual

        layers = [nn.Linear(in_dim, hidden_size), make_activation(activation)]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size),
                       make_activation(activation)]
        self.layers = nn.Sequential(*layers)
        self.output = BayesianLinear(hidden_size, output_dim, bias=bias)

    def forward(self, fused: torch.Tensor, skip: torch.Tensor = None):
        h_s = self.layers(fused)
        mu, var = self.output(h_s)
        if self.residual and skip is not None:
            mu = skip + mu
        return mu, var

    def kl_divergence(self) -> torch.Tensor:
        return self.output.kl_divergence()
