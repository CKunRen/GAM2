"""
Base Class Module (Eq.1)
Shared feature extractor.
  - use_conv=True  (default): Conv1d(1→16→32→64) + Linear(64d→h) — for high-d
  - use_conv=False: Linear MLP for low-d problems
  - activation: 'leaky_relu' (default) or 'tanh'
  - base_layers: number of Linear layers in MLP mode (1 or 2, default 1)
"""

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    if name == 'tanh':
        return nn.Tanh()
    return nn.LeakyReLU(0.01)


class BaseClassModule(nn.Module):
    """
    h_base = f_base(x; θ_base)  (Eq.1)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 use_conv: bool = True, activation: str = 'leaky_relu',
                 base_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_conv = use_conv

        if use_conv:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=1), _get_activation(activation),
                nn.Conv1d(16, 32, kernel_size=1), _get_activation(activation),
                nn.Conv1d(32, 64, kernel_size=1), _get_activation(activation),
            )
            self.fc = nn.Sequential(
                nn.Linear(64 * input_dim, hidden_dim),
                _get_activation(activation),
            )
        else:
            # MLP for low-d; base_layers controls depth
            if base_layers >= 2:
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim), _get_activation(activation),
                    nn.Linear(hidden_dim, hidden_dim), _get_activation(activation),
                )
            else:
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    _get_activation(activation),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            h = x.unsqueeze(1)
            h = self.conv_layers(h)
            h = h.view(h.size(0), -1)
            return self.fc(h)
        else:
            return self.fc(x)
