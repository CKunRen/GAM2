"""
Bayesian Linear Layer (Eq.9-11)
Only the HF output layer is Bayesian.
Weights: w_{s,ij} ~ N(μ_{s,ij}, σ²_{s,ij})
Stores log(σ) to ensure positivity.
Optional deterministic bias for better expressiveness.
"""

import torch
import torch.nn as nn
import math


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with optional deterministic bias.

    w_{s,ij} ~ N(μ_{s,ij}, σ²_{s,ij})  (Eq.9)
    μ̂_s = [Σ_j μ_{s,ij} · h_{s,j}] + b    (Eq.10, extended)
    σ̂²_s = [Σ_j σ²_{s,ij} · h²_{s,j}]      (Eq.11)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable mean and log-std of weights
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.empty(out_features, in_features))

        # Optional deterministic bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_log_sigma, -2.3)

    @property
    def weight_sigma(self) -> torch.Tensor:
        return torch.exp(self.weight_log_sigma)

    def forward(self, x: torch.Tensor):
        mu = torch.mm(x, self.weight_mu.t())
        if self.bias is not None:
            mu = mu + self.bias

        sigma_sq = self.weight_sigma ** 2
        var = torch.mm(x ** 2, sigma_sq.t())

        return mu, var

    def kl_divergence(self) -> torch.Tensor:
        mu = self.weight_mu
        sigma_sq = self.weight_sigma ** 2
        kl = 0.5 * torch.sum(mu ** 2 + sigma_sq - 1.0 - torch.log(sigma_sq))
        return kl

    def sample_forward(self, x: torch.Tensor) -> torch.Tensor:
        epsilon = torch.randn_like(self.weight_mu)
        w = self.weight_mu + self.weight_sigma * epsilon
        out = torch.mm(x, w.t())
        if self.bias is not None:
            out = out + self.bias
        return out
