"""
Expert Module (Eq.2)
Each LF model gets a dedicated expert.
  - num_hidden=2 (default): Linear→Act → Linear→Act → Linear(→1)
  - num_hidden=1:           Linear→Act → Linear(→1)
  - activation: 'leaky_relu' (default) or 'tanh'
Exposes h_{t-} (penultimate features) for routing.
"""

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    if name == 'tanh':
        return nn.Tanh()
    return nn.LeakyReLU(0.01)


class ExpertModule(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 output_dim: int = 1, num_hidden: int = 2,
                 activation: str = 'leaky_relu'):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden

        if num_hidden >= 2:
            self.layer1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), _get_activation(activation))
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), _get_activation(activation))
        else:
            # Single hidden layer
            self.layer1 = nn.Identity()
            self.layer2 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), _get_activation(activation))

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            h_t:       (batch, output_dim) LF prediction
            h_t_minus: (batch, hidden_dim) penultimate features for routing
        """
        h = self.layer1(x)
        h_t_minus = self.layer2(h)
        h_t = self.output_layer(h_t_minus)
        return h_t, h_t_minus
