"""
Dual-Level Gating Mechanism (Eq.3-6)
- Routing gates alpha_ij: input-independent binary scalars (STE)
- Quality gates q_t: input-dependent via expert penultimate features h_{t-}
"""

import math
import torch
import torch.nn as nn


class RoutingGates(nn.Module):
    """
    Inter-expert routing gates (Eq.3-4).
    alpha_ij in {0, 1} -- hard binary gate with STE.
    """

    def __init__(self, num_experts: int, init_value: float = -2.0):
        super().__init__()
        self.num_experts = num_experts
        num_gates = num_experts * (num_experts - 1)
        self.w = nn.Parameter(torch.full((num_gates,), init_value))

    def _get_index(self, i: int, j: int) -> int:
        assert i != j
        if i < j:
            return j * (self.num_experts - 1) + i
        else:
            return j * (self.num_experts - 1) + (i - 1)

    def _hard_sigmoid(self, w: torch.Tensor) -> torch.Tensor:
        soft = torch.sigmoid(w)
        hard = (soft > 0.5).float()
        return hard - soft.detach() + soft

    def get_alpha(self, i: int, j: int) -> torch.Tensor:
        idx = self._get_index(i, j)
        return self._hard_sigmoid(self.w[idx])

    def get_all_alphas(self) -> torch.Tensor:
        return self._hard_sigmoid(self.w)

    def compute_routing(self, h_minus_list: list, j: int) -> torch.Tensor:
        r_j = torch.zeros_like(h_minus_list[0])
        for i in range(self.num_experts):
            if i != j:
                alpha_ij = self.get_alpha(i, j)
                r_j = r_j + alpha_ij * h_minus_list[i]
        return r_j


class QualityGate(nn.Module):
    """
    Expert quality gate (Eq.5-6).

    Input: h_{t-} (penultimate-layer feature of expert t, h-dimensional)
    Output: q_t (batch, 1) per expert, softmax-normalized across experts.

    Shared MLP applied to each expert's h_{t-} independently,
    then softmax across all experts for normalization.
    This makes q_t input-dependent (varies with x) and expert-specific.
    """

    def __init__(self, expert_hidden: int, num_experts: int,
                 temperature: float = 1.0, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        # Shared MLP: h_{t-} (h-dim) -> 1 scalar logit per expert
        self.gate_net = nn.Sequential(
            nn.Linear(expert_hidden, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        with torch.no_grad():
            self.gate_net[-1].weight.fill_(0.0)
            self.gate_net[-1].bias.fill_(0.0)

    def forward_all(self, h_minus_list: list) -> list:
        """
        Compute quality gates for ALL experts at once (with softmax normalization).

        Args:
            h_minus_list: list of (batch, h) tensors -- penultimate features per expert

        Returns:
            list of (batch, 1) quality gate values, softmax-normalized
        """
        logits = []
        for h_m in h_minus_list:
            logit = self.gate_net(h_m)  # (batch, 1)
            logits.append(logit)
        logits = torch.cat(logits, dim=-1)  # (batch, num_experts)
        q = torch.softmax(logits / self.temperature, dim=-1)
        return [q[:, t:t+1] for t in range(len(h_minus_list))]

    def forward(self, h_t: torch.Tensor, expert_idx: int = 0,
                h_base: torch.Tensor = None) -> torch.Tensor:
        """Legacy per-expert call (for compatibility). NOT recommended."""
        logit = self.gate_net(h_t)
        return torch.sigmoid(logit)
