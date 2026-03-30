"""
Loss Functions (Eq.12-14, 16-18)
"""

import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    """L_base = MSE(f_base(x), y) over all data (Eq.12)"""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - target) ** 2)


class ExpertLoss(nn.Module):
    """L_t = MSE(ŷ_t, y_t) per expert (Eq.13)"""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - target) ** 2)


class HFLoss(nn.Module):
    """
    L_s = MSE_s + D_KL(q(W_s)||p(W_s))  (Eq.14)
    The KL term is scaled by 1/N_s for proper scaling.
    """

    def __init__(self, kl_weight: float = 1.0):
        super().__init__()
        self.kl_weight = kl_weight

    def forward(self, mu: torch.Tensor, target: torch.Tensor,
                kl_div: torch.Tensor, n_samples: int) -> torch.Tensor:
        mse = torch.mean((mu - target) ** 2)
        # Scale KL by 1/N for proper ELBO
        kl_scaled = self.kl_weight * kl_div / max(n_samples, 1)
        return mse + kl_scaled


class GateSparsityLoss(nn.Module):
    """
    L_gate = λ_gate · Σ|α_ij|  (Eq.16)
    Sparsity regularization for routing gates.
    """

    def __init__(self, lambda_gate: float = 0.01):
        super().__init__()
        self.lambda_gate = lambda_gate

    def forward(self, alpha_values: dict) -> torch.Tensor:
        if not alpha_values:
            return torch.tensor(0.0)
        total = torch.stack([v.abs() for v in alpha_values.values()]).sum()
        return self.lambda_gate * total


class TotalLoss(nn.Module):
    """
    L_total = Σ ω_t · L_t + ω_s · L_s + L_gate  (Eq.17)
    Fixed-weight version.
    """

    def __init__(self, num_experts: int, lambda_gate: float = 0.01,
                 kl_weight: float = 1.0):
        super().__init__()
        self.expert_loss = ExpertLoss()
        self.hf_loss = HFLoss(kl_weight)
        self.gate_loss = GateSparsityLoss(lambda_gate)
        # Default: equal weights
        self.omega_t = 1.0  # weight for each expert loss
        self.omega_s = 1.0  # weight for HF loss

    def forward(self, lf_preds, lf_targets, mu, hf_target,
                kl_div, n_hf, alpha_values):
        """
        Args:
            lf_preds: list of (batch, 1) LF predictions
            lf_targets: list of (batch, 1) LF targets (can be None for missing)
            mu: (batch, 1) HF prediction mean
            hf_target: (batch, 1) HF target
            kl_div: scalar KL divergence
            n_hf: number of HF samples
            alpha_values: dict of routing gate values
        """
        loss = torch.tensor(0.0, device=mu.device)

        # Expert losses
        for t, (pred, target) in enumerate(zip(lf_preds, lf_targets)):
            if target is not None:
                loss = loss + self.omega_t * self.expert_loss(pred, target)

        # HF loss (includes KL)
        loss = loss + self.omega_s * self.hf_loss(mu, hf_target, kl_div, n_hf)

        # Gate sparsity
        loss = loss + self.gate_loss(alpha_values)

        return loss


class AutoWeightedTotalLoss(nn.Module):
    """
    L_total_auto = Σ 1/(2s_t²) · L_t + ln(s_t) + L_gate  (Eq.18)
    Automatic uncertainty weighting [Kendall et al., 2018].
    s_t are learnable task-specific noise parameters.
    """

    def __init__(self, num_tasks: int, lambda_gate: float = 0.01,
                 kl_weight: float = 1.0):
        """
        Args:
            num_tasks: s (total fidelity levels, including HF)
            lambda_gate: gate sparsity weight
            kl_weight: KL divergence weight
        """
        super().__init__()
        # Learnable log(s_t) for each task (initialized to 0 → s_t = 1)
        self.log_s = nn.Parameter(torch.zeros(num_tasks))
        self.expert_loss = ExpertLoss()
        self.hf_loss = HFLoss(kl_weight)
        self.gate_loss = GateSparsityLoss(lambda_gate)

    def forward(self, lf_preds, lf_targets, mu, hf_target,
                kl_div, n_hf, alpha_values):
        loss = torch.tensor(0.0, device=mu.device)
        task_idx = 0

        # Expert losses with automatic weighting
        for t, (pred, target) in enumerate(zip(lf_preds, lf_targets)):
            if target is not None:
                s_t = torch.exp(self.log_s[task_idx])
                l_t = self.expert_loss(pred, target)
                loss = loss + 0.5 / (s_t ** 2) * l_t + torch.log(s_t)
            task_idx += 1

        # HF loss with automatic weighting
        s_hf = torch.exp(self.log_s[task_idx])
        l_hf = self.hf_loss(mu, hf_target, kl_div, n_hf)
        loss = loss + 0.5 / (s_hf ** 2) * l_hf + torch.log(s_hf)

        # Gate sparsity (not weighted)
        loss = loss + self.gate_loss(alpha_values)

        return loss
