"""Base class module (Eq. (1) of the paper).

A shared feature extractor applied to every input before the experts.  Two
stems are supported, both configurable:

    convolutional (default for d > 1)
        Conv1d(1 -> 16, k) -> act -> Conv1d(16 -> 32, k) -> act
            -> Conv1d(32 -> 64, k) -> act -> flatten -> Linear(64 L -> h) -> act

    MLP  (``use_conv=False``; used by the one-dimensional examples)
        Linear(d -> h) -> act [-> Linear(h -> h) -> act]

The number of MLP layers is controlled by ``base_layers``.
"""

import torch
import torch.nn as nn


def make_activation(name: str) -> nn.Module:
    if name == 'relu':
        return nn.ReLU()
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.01)
    if name == 'tanh':
        return nn.Tanh()
    raise ValueError(f'unknown activation: {name}')


class BaseClassModule(nn.Module):
    """h_base = f_base(x; theta_base)."""

    def __init__(self, input_dim: int, hidden_size: int,
                 kernel_size: int = 1, padding: int = 0,
                 activation: str = 'relu', use_conv: bool = True,
                 base_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size, padding=padding),
                make_activation(activation),
                nn.Conv1d(16, 32, kernel_size, padding=padding),
                make_activation(activation),
                nn.Conv1d(32, 64, kernel_size, padding=padding),
                make_activation(activation),
            )
            length = input_dim + 3 * (2 * padding - kernel_size + 1)
            if length <= 0:
                raise ValueError(
                    f'convolution stem leaves no length for d={input_dim}, '
                    f'kernel={kernel_size}, padding={padding}')
            self.length = length
            self.fc = nn.Sequential(nn.Linear(64 * length, hidden_size),
                                    make_activation(activation))
        else:
            self.length = input_dim
            layers = [nn.Linear(input_dim, hidden_size),
                      make_activation(activation)]
            for _ in range(max(base_layers - 1, 0)):
                layers += [nn.Linear(hidden_size, hidden_size),
                           make_activation(activation)]
            self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if not self.use_conv:
            return self.mlp(x)
        h = self.conv(x.unsqueeze(1)).flatten(1)
        return self.fc(h)
