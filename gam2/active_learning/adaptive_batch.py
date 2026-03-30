"""
Adaptive Batch Size (Eq.19-21)
Determines the number of new samples per active learning iteration.
"""

import math
import torch
import numpy as np


class AdaptiveBatchSize:
    """
    Adaptive batch size strategy for active learning.

    N_U^base from d (input dimension)
    V^(i) = mean_var_i / mean_var_0            (Eq.19)
    Δ_gate^(i) from q_t and α_ij changes       (Eq.20)
    G^(i) = Δ_gate^(i) / Δ_gate^(1)            (Eq.19)
    N_U^(i) = clamp(round(N_base*(β*V+((1-β)*G)), N_min, N_max)  (Eq.21)
    """

    def __init__(self, input_dim: int, beta: float = 0.5,
                 n_min: int = 2, n_max_factor: int = 2,
                 n_base_override: int = None):
        """
        Args:
            input_dim: d, input dimensionality
            beta: balance between V and G (default 0.5)
            n_min: minimum batch size
            n_max_factor: N_max = n_max_factor * N_base
            n_base_override: if set, directly use this as n_base
        """
        self.input_dim = input_dim
        self.beta = beta
        self.n_min = n_min

        if n_base_override is not None:
            self.n_base = n_base_override
        elif input_dim < 10:
            self.n_base = 5
        elif input_dim <= 20:
            self.n_base = 10
        elif input_dim <= 50:
            self.n_base = 15
        else:
            self.n_base = 20

        self.n_max = n_max_factor * self.n_base

        # State tracking
        self.initial_mean_var = None
        self.prev_q_values = None
        self.prev_alpha_values = None
        self.delta_gate_1 = None  # Δ_gate^(1) for normalization

    def initialize(self, mean_var: float, q_values: list, alpha_values: dict):
        """
        Record initial state after first training.

        Args:
            mean_var: mean HF prediction variance over candidates
            q_values: list of mean q_t values
            alpha_values: dict of α_ij values
        """
        self.initial_mean_var = mean_var
        self.prev_q_values = [float(q) for q in q_values]
        self.prev_alpha_values = {k: float(v) for k, v in alpha_values.items()}

    def compute_batch_size(self, iteration: int, mean_var: float,
                            q_values: list, alpha_values: dict) -> int:
        """
        Compute adaptive batch size for iteration i.

        Args:
            iteration: current iteration (1-indexed)
            mean_var: current mean HF prediction variance
            q_values: list of current mean q_t values
            alpha_values: dict of current α_ij values

        Returns:
            N_U: batch size for this iteration
        """
        if iteration == 0 or self.initial_mean_var is None:
            return self.n_base

        # V^(i) = mean_var_i / mean_var_0  (Eq.19)
        V = mean_var / max(self.initial_mean_var, 1e-10)
        V = min(V, 1.0)  # Clamp to [0, 1]

        # Δ_gate^(i) (Eq.20)
        delta_gate = self._compute_delta_gate(q_values, alpha_values)

        # G^(i) = Δ_gate^(i) / Δ_gate^(1)  (Eq.19)
        if iteration == 1:
            self.delta_gate_1 = delta_gate
            G = 1.0
        elif delta_gate < 1e-8 and self.delta_gate_1 < 1e-8:
            # Gates have not started learning at all — do not penalize batch size.
            G = 1.0
        else:
            G = delta_gate / max(self.delta_gate_1, 1e-10)
            G = min(G, 1.0)

        # Update previous gate values
        self.prev_q_values = [float(q) for q in q_values]
        self.prev_alpha_values = {k: float(v) for k, v in alpha_values.items()}

        # N_U^(i) (Eq.21)
        raw = self.n_base * (self.beta * V + (1 - self.beta) * G) + 0.5
        n_u = min(max(int(raw), self.n_min), self.n_max)

        return n_u

    def _compute_delta_gate(self, q_values: list, alpha_values: dict) -> float:
        """
        Compute Δ_gate^(i) (Eq.20).

        Δ_gate = (1/(s-1)) Σ|q_t^(i) - q_t^(i-1)| +
                 (1/N_α) Σ|α_ij^(i) - α_ij^(i-1)|
        """
        if self.prev_q_values is None:
            return 1.0

        # Quality gate changes
        num_experts = len(q_values)
        q_change = sum(
            abs(float(q_values[t]) - self.prev_q_values[t])
            for t in range(num_experts)
        ) / max(num_experts, 1)

        # Routing gate changes
        alpha_change = 0.0
        n_alpha = len(alpha_values)
        if n_alpha > 0 and self.prev_alpha_values:
            for key, val in alpha_values.items():
                prev_val = self.prev_alpha_values.get(key, 0.0)
                alpha_change += abs(float(val) - prev_val)
            alpha_change /= n_alpha

        return q_change + alpha_change
