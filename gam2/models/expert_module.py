"""Expert module (Eq. (2) of the paper).

One expert per low-fidelity source.  It receives the base feature and the routed
feature of the other experts, and returns both the LF prediction and the
penultimate feature used by the routing and quality gates:

    Linear(2h -> h) -> act [-> Linear(h -> h) -> act] -> Linear(h -> out)

``hidden_layers`` selects one or two hidden layers (the paper uses two for the
multi-dimensional examples and one for the one-dimensional ones).
"""

import torch
import torch.nn as nn

from .base_module import make_activation


class ExpertModule(nn.Module):
    """(y_t, h_t^-) = f_t([h_base ; r_t])."""

    def __init__(self, hidden_size: int, output_dim: int = 1,
                 hidden_layers: int = 2, activation: str = 'relu'):
        super().__init__()
        if hidden_layers not in (1, 2):
            raise ValueError('hidden_layers must be 1 or 2')
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        self.layer1 = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size),
                                    make_activation(activation))
        if hidden_layers == 2:
            self.layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                        make_activation(activation))
        else:
            self.layer2 = None
        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor):
        z = self.layer1(x)
        if self.layer2 is not None:
            z = self.layer2(z)
        return self.out(z), z
