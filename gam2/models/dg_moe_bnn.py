"""DG-MoEs-BNN: dual-gated mixture-of-experts Bayesian neural network.

The module assembles the four components of the paper:

1. base class module            Eq. (1)
2. expert modules               Eq. (2)
3. dual-level gating            Eqs. (3)-(5)
4. HF prediction module         Eqs. (6)-(11)

Routing is the sum over all other experts, Eq. (4).  Because the expert features
themselves depend on the routed inputs, (4) is a coupled system; it is solved by
``routing_exchanges`` Jacobi exchanges (default: one), i.e. the features are
first computed with r_t = 0 and the routed inputs are then formed from those
features.  One exchange is what the paper reports: it exchanges information in
both directions for every ordered pair (i, t).
"""

import torch
import torch.nn as nn

from .base_module import BaseClassModule
from .expert_module import ExpertModule
from .gating import QualityGate, RoutingGates
from .hf_module import HFPredictionModule


def table1_hidden_size(input_dim: int) -> int:
    """Table 1: h = 16 for d <= 10, 64 for d > 10."""
    return 16 if input_dim <= 10 else 64


class DGMoEBNN(nn.Module):
    """Dual-gated mixture-of-experts Bayesian neural network."""

    def __init__(self, input_dim: int, num_lf_models: int,
                 hidden_size: int = None, output_dim: int = 1,
                 kernel_size: int = 1, conv_padding: int = 0,
                 use_conv: bool = True, base_layers: int = 1,
                 activation: str = 'leaky_relu', expert_layers: int = 2,
                 hf_layers: int = 2, gate_hidden: int = 16,
                 routing_exchanges: int = 1, routing_init: float = -2.0,
                 bayesian_bias: bool = True, residual: bool = True):
        super().__init__()
        if num_lf_models < 1:
            raise ValueError('at least one low-fidelity model is required')
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_lf_models
        self.routing_exchanges = int(routing_exchanges)
        self.hidden_size = (hidden_size if hidden_size is not None
                            else table1_hidden_size(input_dim))

        self.base_module = BaseClassModule(
            input_dim, self.hidden_size, kernel_size=kernel_size,
            padding=conv_padding, activation=activation,
            use_conv=use_conv, base_layers=base_layers)
        self.experts = nn.ModuleList([
            ExpertModule(self.hidden_size, output_dim,
                         hidden_layers=expert_layers, activation=activation)
            for _ in range(num_lf_models)
        ])
        self.routing_gates = (RoutingGates(num_lf_models, init_value=routing_init)
                              if num_lf_models > 1 else None)
        self.quality_gate = QualityGate(self.hidden_size, gate_hidden,
                                        activation=activation)
        self.hf_module = HFPredictionModule(
            self.hidden_size, num_lf_models, output_dim=output_dim,
            hidden_layers=hf_layers, activation=activation,
            bias=bayesian_bias, residual=residual)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def _expert_pass(self, h_base: torch.Tensor, features):
        """One sweep of experts over the given (routed) features."""
        preds, penult = [], []
        zero = torch.zeros_like(h_base)
        for t, expert in enumerate(self.experts):
            if self.routing_gates is None:
                r = zero
            else:
                r = self.routing_gates.routed_feature(features, t)
            y_t, h_t_minus = expert(torch.cat([h_base, r], dim=-1))
            preds.append(y_t)
            penult.append(h_t_minus)
        return preds, penult

    def forward(self, x: torch.Tensor, stage: int = 2):
        """Returns (mu, var, predictions, quality_gates, routing_gates)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h_base = self.base_module(x)

        # sweep 0: experts without routing (r_t = 0)
        preds, features = self._expert_pass(h_base, None if self.routing_gates
                                            is None else
                                            [torch.zeros_like(h_base)
                                             for _ in range(self.num_experts)])
        # Jacobi exchanges of Eq. (4)
        if stage != 1 and self.routing_gates is not None:
            for _ in range(self.routing_exchanges):
                preds, features = self._expert_pass(h_base, features)

        # quality gates, Eq. (5)
        q = self.quality_gate(features)
        q_list = [q[:, t:t + 1] for t in range(self.num_experts)]

        # fusion, Eq. (6)
        fused = torch.cat([q_list[t] * preds[t]
                           for t in range(self.num_experts)] + [h_base], dim=-1)
        skip = sum(q_list[t] * preds[t] for t in range(self.num_experts))
        mu, var = self.hf_module(fused, skip)

        alphas = (self.routing_gates.all_alphas()
                  if self.routing_gates is not None else {})
        return mu, var, preds, q_list, alphas

    def kl_divergence(self) -> torch.Tensor:
        """KL divergence of the Bayesian output layer (Eq. (14))."""
        return self.hf_module.kl_divergence()

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def architecture_summary(self) -> dict:
        """Layer widths, for checks against the configuration table."""
        n = self.num_experts
        return dict(
            input_dim=self.input_dim,
            use_conv=self.base_module.use_conv,
            hidden_size=self.hidden_size,
            num_lf_models=n,
            base_length=self.base_module.length,
            expert_in=2 * self.hidden_size,
            hf_in=self.hf_module.in_dim,
            hf_in_expected=self.hidden_size + n * self.output_dim,
            routing_gates=n * (n - 1),
        )
