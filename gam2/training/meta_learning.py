"""
Meta-Learning for Base Class Module (Stage 1, Step 1)
Reptile-style meta-learning using all fidelity data.

Each inner-loop task uses a *temporary linear head* (discarded after the task)
so that the base module learns meaningful features — not a degenerate mean proxy.
"""

import torch
import torch.nn as nn
from copy import deepcopy


class MetaLearner:
    """
    Reptile-style meta-learning for the base class module.

    Samples tasks from [D_1,...,D_s] with replacement,
    performs k rounds of inner-loop training + outer-loop update.
    A fresh Linear(hidden_dim, 1) head is created per task.
    """

    def __init__(self, base_module: nn.Module, inner_lr: float = 0.01,
                 outer_lr: float = 0.001, inner_steps: int = 5,
                 num_tasks_per_round: int = 4):
        self.base_module = base_module
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.num_tasks_per_round = num_tasks_per_round
        self.loss_fn = nn.MSELoss()
        # hidden_dim stored by BaseClassModule
        self.hidden_dim = base_module.hidden_dim

    def create_tasks(self, datasets: list, batch_size: int = 32):
        tasks = []
        for _ in range(self.num_tasks_per_round):
            idx = torch.randint(0, len(datasets), (1,)).item()
            X, y = datasets[idx]
            n = X.size(0)
            indices = torch.randint(0, n, (min(batch_size, n),))
            tasks.append((X[indices], y[indices]))
        return tasks

    def meta_train_step(self, datasets: list, batch_size: int = 32):
        """One round of Reptile meta-learning with per-task linear head."""
        device = next(self.base_module.parameters()).device
        tasks = self.create_tasks(datasets, batch_size)

        # Save original parameters
        original_params = {
            name: param.clone()
            for name, param in self.base_module.named_parameters()
        }

        accumulated_params = {
            name: torch.zeros_like(param)
            for name, param in self.base_module.named_parameters()
        }

        total_loss = 0.0

        for X_task, y_task in tasks:
            X_task = X_task.to(device)
            y_task = y_task.to(device)

            # Reset to original params
            with torch.no_grad():
                for name, param in self.base_module.named_parameters():
                    param.copy_(original_params[name])

            # Fresh temporary linear head for this task
            temp_head = nn.Linear(self.hidden_dim, 1).to(device)

            # Inner loop: k steps of SGD on base_module + temp_head
            inner_params = list(self.base_module.parameters()) + list(temp_head.parameters())
            inner_optimizer = torch.optim.SGD(inner_params, lr=self.inner_lr)

            for _ in range(self.inner_steps):
                inner_optimizer.zero_grad()
                h = self.base_module(X_task)
                pred = temp_head(h)                   # proper learned projection
                loss = self.loss_fn(pred, y_task)
                loss.backward()
                inner_optimizer.step()
                total_loss += loss.item()

            # Accumulate only base_module params (NOT temp_head)
            with torch.no_grad():
                for name, param in self.base_module.named_parameters():
                    accumulated_params[name] += param

        # Average
        num_tasks = len(tasks)
        with torch.no_grad():
            for name in accumulated_params:
                accumulated_params[name] /= num_tasks

        # Reptile update: θ ← θ + outer_lr * (θ_avg - θ)
        with torch.no_grad():
            for name, param in self.base_module.named_parameters():
                param.copy_(
                    original_params[name]
                    + self.outer_lr * (accumulated_params[name] - original_params[name])
                )

        return total_loss / (num_tasks * self.inner_steps)

    def train(self, datasets: list, num_rounds: int = 100,
              batch_size: int = 32, verbose: bool = True):
        loss_history = []
        for round_idx in range(num_rounds):
            loss = self.meta_train_step(datasets, batch_size)
            loss_history.append(loss)
            if verbose and (round_idx + 1) % 20 == 0:
                print(f"  Meta-learning round {round_idx+1}/{num_rounds}, "
                      f"loss: {loss:.6f}")
        return loss_history
