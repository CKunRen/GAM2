"""
Full DG-MoE-BNN Assembly
Assembles Base Class, Expert Modules, Dual-Level Gating, and HF Prediction Module.

Dual-level gating:
  - Routing gates alpha_ij in {0,1}: binary, input-independent (STE)
  - Quality gates q_t in [0,1]: input-dependent via expert penultimate features h_{t-}
"""

import torch
import torch.nn as nn
from models.base_module import BaseClassModule
from models.expert_module import ExpertModule
from models.gating import RoutingGates, QualityGate
from models.hf_module import HFPredictionModule


class DGMoEBNN(nn.Module):
    """
    Dual-Gated Mixture-of-Experts Bayesian Neural Network.
    """

    def __init__(self, input_dim: int, num_lf_models: int,
                 base_hidden: int = 64, expert_hidden: int = 64,
                 hf_hidden: int = 64, output_dim: int = 1,
                 use_conv: bool = True,
                 expert_layers: int = 2,
                 hf_det_layers: int = 2,
                 activation: str = 'leaky_relu',
                 base_layers: int = 1,
                 residual: bool = False,
                 quality_temperature: float = 1.0,
                 routing_init: float = -2.0,
                 bayesian_bias: bool = True,
                 # legacy kwargs (ignored)
                 quality_mode: str = 'softmax_input', **kwargs):
        super().__init__()
        self.init_kwargs = dict(
            input_dim=input_dim,
            num_lf_models=num_lf_models,
            base_hidden=base_hidden,
            expert_hidden=expert_hidden,
            hf_hidden=hf_hidden,
            output_dim=output_dim,
            use_conv=use_conv,
            expert_layers=expert_layers,
            hf_det_layers=hf_det_layers,
            activation=activation,
            base_layers=base_layers,
            residual=residual,
            quality_temperature=quality_temperature,
            routing_init=routing_init,
            bayesian_bias=bayesian_bias,
            quality_mode=quality_mode,
            **kwargs,
        )
        self.input_dim = input_dim
        self.num_experts = num_lf_models
        self.output_dim = output_dim
        self.hf_hidden = hf_hidden
        self.use_conv = use_conv
        self.expert_layers = expert_layers
        self.hf_det_layers = hf_det_layers
        self.activation = activation
        self.base_layers = base_layers
        self.residual = residual
        self.quality_temperature = quality_temperature
        self.routing_init = routing_init
        self.bayesian_bias = bayesian_bias
        self.quality_mode = quality_mode

        # 1. Base class module
        self.base_module = BaseClassModule(
            input_dim, base_hidden,
            use_conv=use_conv, activation=activation,
            base_layers=base_layers)

        # 2. Expert modules
        expert_input_dim = base_hidden + expert_hidden
        self.experts = nn.ModuleList([
            ExpertModule(expert_input_dim, expert_hidden, output_dim,
                         num_hidden=expert_layers, activation=activation)
            for _ in range(num_lf_models)
        ])

        # 3. Dual-level gating
        if num_lf_models > 1:
            self.routing_gates = RoutingGates(num_lf_models,
                                              init_value=routing_init)
        else:
            self.routing_gates = None

        # Quality gate: shared MLP on h_{t-} (expert penultimate features)
        self.quality_gate = QualityGate(
            expert_hidden=expert_hidden,
            num_experts=num_lf_models,
            temperature=quality_temperature)

        # 4. HF prediction module
        self.hf_module = HFPredictionModule(
            fused_dim=output_dim,
            base_dim=base_hidden,
            hidden_dim=hf_hidden,
            output_dim=output_dim,
            num_det_layers=hf_det_layers,
            activation=activation,
            residual=residual,
            bayesian_bias=bayesian_bias,
        )

        self.base_hidden = base_hidden
        self.expert_hidden = expert_hidden

    def forward(self, x: torch.Tensor, stage: int = 2):
        batch_size = x.size(0)

        # 1. Base class features
        h_base = self.base_module(x)

        # 2. Expert computation — collect predictions AND penultimate features
        lf_preds = []
        h_minus_list = []

        if stage == 1 or self.routing_gates is None:
            zero_routing = torch.zeros(batch_size, self.expert_hidden,
                                       device=x.device)
            for t in range(self.num_experts):
                expert_input = torch.cat([h_base, zero_routing], dim=-1)
                h_t, h_t_minus = self.experts[t](expert_input)
                lf_preds.append(h_t)
                h_minus_list.append(h_t_minus)
        else:
            h_minus_init = [
                torch.zeros(batch_size, self.expert_hidden, device=x.device)
                for _ in range(self.num_experts)
            ]
            for j in range(self.num_experts):
                r_j = self.routing_gates.compute_routing(h_minus_init, j)
                expert_input = torch.cat([h_base, r_j], dim=-1)
                h_t, h_t_minus = self.experts[j](expert_input)
                lf_preds.append(h_t)
                h_minus_list.append(h_t_minus)
                h_minus_init[j] = h_t_minus

        # 3. Quality gates — input-dependent via h_{t-} (Eq.5-6)
        q_values = self.quality_gate.forward_all(h_minus_list)

        # 4. Gate-weighted fusion (Eq.7)
        h_fused = torch.zeros(batch_size, self.output_dim, device=x.device)
        for t in range(self.num_experts):
            h_fused = h_fused + q_values[t] * lf_preds[t]

        # 5. HF prediction (Eq.8, 10, 11)
        mu, var = self.hf_module(h_fused, h_base)

        # 6. Collect routing gate values
        alpha_values = {}
        if self.routing_gates is not None:
            for i in range(self.num_experts):
                for j in range(self.num_experts):
                    if i != j:
                        alpha_values[(i, j)] = self.routing_gates.get_alpha(i, j)

        return mu, var, lf_preds, q_values, alpha_values

    def kl_divergence(self) -> torch.Tensor:
        return self.hf_module.kl_divergence()

    def get_parameter_groups(self, base_lr: float):
        param_groups = [
            {'params': self.base_module.parameters(), 'lr': base_lr * 0.01},
            {'params': self.experts.parameters(), 'lr': base_lr * 0.1},
            {'params': self.hf_module.parameters(), 'lr': base_lr},
        ]
        if self.routing_gates is not None:
            param_groups.append(
                {'params': self.routing_gates.parameters(), 'lr': base_lr}
            )
        param_groups.append(
            {'params': self.quality_gate.parameters(), 'lr': base_lr}
        )
        return param_groups

    def rebuild_same(self):
        return type(self)(**self.init_kwargs)
