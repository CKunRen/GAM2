"""Dual-level gating: inter-expert routing gates (Eqs. (3)-(4)) and expert
quality gates (Eq. (5)).

Routing gates
-------------
One scalar w_ij per *ordered* pair (i, j), i != j, with

    alpha_ij = 1[sigmoid(w_ij) > 0.5]                      (Eq. (3))

and a straight-through estimator for the gradient.  The routed feature of expert
t is the sum over all other experts (Eq. (4)):

    r_t = sum_{i != t} alpha_it * h_i^-

Quality gates
-------------
A shared MLP applied to each expert's penultimate feature, normalised across
experts (Eq. (5)):

    q_t = softmax_t( f_quality(h_t^-; theta_q) )
"""

import torch
import torch.nn as nn

from .base_module import make_activation


class RoutingGates(nn.Module):
    """alpha_ij in {0, 1}, one parameter per ordered pair."""

    def __init__(self, num_experts: int, init_value: float = -2.0):
        super().__init__()
        self.num_experts = num_experts
        n = max(num_experts * (num_experts - 1), 1)
        self.weight = nn.Parameter(torch.full((n,), float(init_value)))

    def _index(self, i: int, j: int) -> int:
        n = self.num_experts
        return j * (n - 1) + i if i < j else j * (n - 1) + (i - 1)

    def alpha(self, i: int, j: int) -> torch.Tensor:
        s = torch.sigmoid(self.weight[self._index(i, j)])
        hard = (s > 0.5).to(s.dtype)
        return hard - s.detach() + s          # straight-through estimator

    def all_alphas(self) -> dict:
        return {(i, j): self.alpha(i, j)
                for i in range(self.num_experts)
                for j in range(self.num_experts) if i != j}

    def get_sparsity_values(self) -> torch.Tensor:
        """Sigmoid activations, used by the sparsity penalty (Eq. (16))."""
        return torch.sigmoid(self.weight)

    def routed_feature(self, features, j: int) -> torch.Tensor:
        r = torch.zeros_like(features[0])
        for i in range(self.num_experts):
            if i != j:
                r = r + self.alpha(i, j) * features[i]
        return r


class QualityGate(nn.Module):
    """q_t = softmax_t(f_quality(h_t^-)), shared MLP across experts."""

    def __init__(self, hidden_size: int, gate_hidden: int = 16,
                 activation: str = 'relu'):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, gate_hidden),
                                 make_activation(activation),
                                 nn.Linear(gate_hidden, 1))

    def forward(self, penultimate: list) -> torch.Tensor:
        logits = torch.cat([self.net(h) for h in penultimate], dim=-1)
        return torch.softmax(logits, dim=-1)
