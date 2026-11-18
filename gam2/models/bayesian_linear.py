"""Bayesian output layer (Eqs. (9)-(11) of the paper).

Only the output layer of the HF module is Bayesian.  The weights follow

    w_ij ~ N(mu_ij, sigma_ij^2)                            (Eq. (9))

and the predictive mean and variance are obtained analytically:

    mu_hat  = sum_j mu_ij h_j + b                          (Eq. (10))
    var_hat = sum_j sigma_ij^2 h_j^2                       (Eq. (11))

The variance is the output-layer *weight* variance; no observation-noise term is
added.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    """Variational Bayesian linear layer with a standard-normal prior."""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(
            torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_log_sigma, -2.3)

    @property
    def weight_sigma(self) -> torch.Tensor:
        return torch.exp(self.weight_log_sigma)

    def forward(self, x: torch.Tensor):
        mu = F.linear(x, self.weight_mu)
        if self.bias is not None:
            mu = mu + self.bias
        var = F.linear(x ** 2, self.weight_sigma ** 2)
        return mu, var

    def kl_divergence(self) -> torch.Tensor:
        s2 = self.weight_sigma ** 2
        return 0.5 * torch.sum(self.weight_mu ** 2 + s2 - 1.0 - torch.log(s2))
