"""
Two-Stage Trainer for DG-MoE-BNN
Stage 1: Independent module pre-training (meta-learn base -> train experts -> train HF)
Stage 2: Joint gated fine-tuning with differentiated LRs (Eq.15)

Also supports end-to-end training (all params jointly from scratch).

All training methods return loss histories for analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from models.dg_moe_bnn import DGMoEBNN
from training.meta_learning import MetaLearner
from training.losses import AutoWeightedTotalLoss, ExpertLoss, HFLoss


class GAM2Trainer:
    """
    Two-stage training for the DG-MoE-BNN.

    Stage 1: Sequential independent pre-training
      Step 1: Meta-learn base class -> freeze
      Step 2: Train experts in parallel (alpha_ij=0) -> freeze
      Step 3: Train HF module (q_t=1/(s-1)) -> freeze

    Stage 2: Joint fine-tuning 10-20 epochs
      Differentiated LRs (Eq.15), balanced mini-batch, gradient clipping
      Optimize L_total_auto (Eq.18)
    """

    def __init__(self, model: DGMoEBNN, base_lr: float = 0.001,
                 lambda_gate: float = 0.01, kl_weight: float = 1.0,
                 grad_clip: float = 1.0, weight_decay: float = 1e-4):
        self.model = model
        self.base_lr = base_lr
        self.lambda_gate = lambda_gate
        self.kl_weight = kl_weight
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def stage1_step1_meta_learn(self, datasets: list, device: torch.device,
                                 num_rounds: int = 100, verbose: bool = True):
        if verbose:
            print("Stage 1 Step 1: Meta-learning base class module...")
        self.model.to(device)

        meta_learner = MetaLearner(
            self.model.base_module,
            inner_lr=self.base_lr,
            outer_lr=self.base_lr * 0.1,
            inner_steps=5,
            num_tasks_per_round=min(len(datasets), 4),
        )
        loss_history = meta_learner.train(datasets, num_rounds=num_rounds, verbose=verbose)

        for param in self.model.base_module.parameters():
            param.requires_grad = False

        if verbose:
            print("  Base class module frozen.")
        return loss_history

    def stage1_step2_train_experts(self, lf_datasets: list, device: torch.device,
                                    epochs: int = 200, verbose: bool = True):
        if verbose:
            print("Stage 1 Step 2: Training expert modules independently...")
        self.model.to(device)
        loss_fn = ExpertLoss()
        loss_history = {}

        for t in range(self.model.num_experts):
            X_t, y_t = lf_datasets[t]
            X_t, y_t = X_t.to(device), y_t.to(device)
            expert_losses = []

            optimizer = torch.optim.Adam(
                self.model.experts[t].parameters(),
                lr=self.base_lr, weight_decay=self.weight_decay
            )

            for epoch in range(epochs):
                optimizer.zero_grad()
                with torch.no_grad():
                    h_base = self.model.base_module(X_t)
                zero_r = torch.zeros(X_t.size(0), self.model.expert_hidden,
                                     device=device)
                expert_input = torch.cat([h_base, zero_r], dim=-1)
                h_t, _ = self.model.experts[t](expert_input)
                loss = loss_fn(h_t, y_t)
                loss.backward()
                optimizer.step()

                expert_losses.append(loss.item())

                if verbose and (epoch + 1) % 50 == 0:
                    print(f"  Expert {t+1}, epoch {epoch+1}/{epochs}, "
                          f"loss: {loss.item():.6f}")

            loss_history[t] = expert_losses

        for expert in self.model.experts:
            for param in expert.parameters():
                param.requires_grad = False

        if verbose:
            print("  All expert modules frozen.")
        return loss_history

    def stage1_step3_train_hf(self, lf_datasets: list, hf_dataset: tuple,
                               device: torch.device, epochs: int = 200,
                               verbose: bool = True):
        if verbose:
            print("Stage 1 Step 3: Training HF prediction module...")
        self.model.to(device)

        X_s, y_s = hf_dataset
        X_s, y_s = X_s.to(device), y_s.to(device)

        hf_loss_fn = HFLoss(self.kl_weight)
        optimizer = torch.optim.Adam(
            self.model.hf_module.parameters(),
            lr=self.base_lr, weight_decay=self.weight_decay
        )

        loss_history = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            with torch.no_grad():
                h_base = self.model.base_module(X_s)
                zero_r = torch.zeros(X_s.size(0), self.model.expert_hidden,
                                     device=device)
                lf_preds = []
                for t in range(self.model.num_experts):
                    expert_input = torch.cat([h_base, zero_r], dim=-1)
                    h_t, _ = self.model.experts[t](expert_input)
                    lf_preds.append(h_t)

                uniform_q = 1.0 / max(self.model.num_experts, 1)
                h_fused = torch.zeros_like(lf_preds[0])
                for h_t in lf_preds:
                    h_fused = h_fused + uniform_q * h_t

            mu, var = self.model.hf_module(h_fused, h_base)
            kl_div = self.model.hf_module.kl_divergence()
            loss = hf_loss_fn(mu, y_s, kl_div, X_s.size(0))
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  HF module, epoch {epoch+1}/{epochs}, "
                      f"loss: {loss.item():.6f}")

        for param in self.model.hf_module.parameters():
            param.requires_grad = False

        if verbose:
            print("  HF prediction module frozen.")
        return loss_history

    def stage1(self, datasets: list, device: torch.device,
               meta_rounds: int = 100, expert_epochs: int = 200,
               hf_epochs: int = 200, verbose: bool = True):
        if verbose:
            print("=" * 60)
            print("STAGE 1: Independent Module Pre-training")
            print("=" * 60)

        lf_datasets = datasets[:-1]
        hf_dataset = datasets[-1]

        meta_loss = self.stage1_step1_meta_learn(datasets, device, meta_rounds, verbose)
        expert_loss = self.stage1_step2_train_experts(lf_datasets, device, expert_epochs, verbose)
        hf_loss = self.stage1_step3_train_hf(lf_datasets, hf_dataset, device, hf_epochs, verbose)

        return {'meta': meta_loss, 'expert': expert_loss, 'hf': hf_loss}

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def stage2(self, datasets: list, device: torch.device,
               epochs: int = 20, verbose: bool = True):
        if verbose:
            print("=" * 60)
            print("STAGE 2: Joint Gated Fine-tuning")
            print("=" * 60)

        self.model.to(device)
        num_tasks = len(datasets)

        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        # Loss function with auto-weighting
        auto_loss = AutoWeightedTotalLoss(
            num_tasks, self.lambda_gate, self.kl_weight
        ).to(device)

        # Differentiated learning rates (Eq.15)
        param_groups = self.model.get_parameter_groups(self.base_lr)
        param_groups.append({
            'params': auto_loss.parameters(), 'lr': self.base_lr
        })
        optimizer = torch.optim.Adam(param_groups, weight_decay=self.weight_decay)

        lf_datasets = datasets[:-1]
        hf_X, hf_y = datasets[-1]
        hf_X, hf_y = hf_X.to(device), hf_y.to(device)

        loss_history = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            # Expert losses on their own data
            for t in range(self.model.num_experts):
                X_t, y_t = lf_datasets[t]
                X_t, y_t = X_t.to(device), y_t.to(device)
                _, _, lf_p, _, _ = self.model(X_t, stage=2)
                s_t = torch.exp(auto_loss.log_s[t])
                l_t = torch.mean((lf_p[t] - y_t) ** 2)
                total_loss = total_loss + 0.5 / (s_t ** 2) * l_t + torch.log(s_t)

            # HF loss
            mu, var, _, _, alpha_values = self.model(hf_X, stage=2)
            kl_div = self.model.kl_divergence()
            s_hf = torch.exp(auto_loss.log_s[-1])
            l_hf = torch.mean((mu - hf_y) ** 2) + self.kl_weight * kl_div / max(hf_X.size(0), 1)
            total_loss = total_loss + 0.5 / (s_hf ** 2) * l_hf + torch.log(s_hf)

            # Gate sparsity (Eq.16)
            if alpha_values:
                gate_reg = self.lambda_gate * sum(
                    v.abs() for v in alpha_values.values()
                )
                total_loss = total_loss + gate_reg

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()

            loss_history.append(total_loss.item())

            if verbose and (epoch + 1) % 5 == 0:
                alpha_str = ""
                if alpha_values:
                    alpha_str = ", alpha=" + ",".join(
                        f"{v.item():.3f}" for v in alpha_values.values()
                    )
                q_strs = []
                with torch.no_grad():
                    _, _, _, q_vals, _ = self.model(hf_X, stage=2)
                    for q in q_vals:
                        q_strs.append(f"{q.mean().item():.3f}")
                q_str = ",".join(q_strs)
                print(f"  Epoch {epoch+1}/{epochs}, loss: {total_loss.item():.6f}, "
                      f"q=[{q_str}]{alpha_str}")

        return loss_history

    # ------------------------------------------------------------------
    # End-to-End Training (all params jointly, no stage separation)
    # ------------------------------------------------------------------

    def train_e2e(self, datasets: list, device: torch.device,
                  epochs: int = 5000, hf_weight: float = 5.0,
                  lr: float = None, warmup_epochs: int = 0,
                  batch_size: int = 0,
                  verbose: bool = True):
        """
        End-to-end training: all parameters trained jointly.
        Better for small datasets where sequential freezing fails.

        Args:
            datasets: [(X1,y1),...,(Xk,yk),(X_hf,y_hf)]
            epochs: total joint training epochs
            hf_weight: relative weight of HF loss vs expert losses
            lr: learning rate (defaults to self.base_lr)
            warmup_epochs: if > 0, first train base+experts on LF data only
            batch_size: if > 0, use mini-batch for LF data (for large datasets).
                        Each epoch iterates through ALL batches (DataLoader-style).
        """
        if verbose:
            print("=" * 60)
            print("END-TO-END Joint Training")
            print("=" * 60)

        self.model.to(device)
        if lr is None:
            lr = self.base_lr

        lf_datasets = datasets[:-1]
        hf_X, hf_y = datasets[-1]
        hf_X, hf_y = hf_X.to(device), hf_y.to(device)
        n_hf = hf_X.size(0)

        # Determine number of batches per epoch for large datasets
        use_batches = batch_size > 0
        n_batches = 1
        if use_batches:
            max_n = max(lf_datasets[t][0].size(0) if torch.is_tensor(lf_datasets[t][0])
                        else len(lf_datasets[t][0]) for t in range(self.model.num_experts))
            n_batches = max(1, (max_n + batch_size - 1) // batch_size)

        # Move all LF data to device once
        lf_device = []
        for t in range(self.model.num_experts):
            X_t, y_t = lf_datasets[t]
            lf_device.append((X_t.to(device), y_t.to(device)))

        # Compute total steps for cosine scheduler
        warmup_steps = warmup_epochs * n_batches if warmup_epochs > 0 else 0
        joint_steps = epochs * n_batches

        # Phase 1: Expert warmup (train base + experts on LF data only)
        if warmup_epochs > 0:
            if verbose:
                print(f"  Warmup: {warmup_epochs} epochs x {n_batches} batches on LF data...")
            base_expert_params = (list(self.model.base_module.parameters())
                                  + list(self.model.experts.parameters()))
            opt_warmup = torch.optim.Adam(base_expert_params, lr=lr,
                                          weight_decay=self.weight_decay)
            sched_warmup = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_warmup, T_max=warmup_steps, eta_min=lr * 0.01)
            for ep in range(warmup_epochs):
                perm = [torch.randperm(lf_device[t][0].size(0)) for t in range(self.model.num_experts)]
                for b in range(n_batches):
                    opt_warmup.zero_grad()
                    loss = torch.tensor(0.0, device=device)
                    for t in range(self.model.num_experts):
                        X_t, y_t = lf_device[t]
                        if use_batches and X_t.size(0) > batch_size:
                            idx = perm[t][b*batch_size:(b+1)*batch_size]
                            X_t, y_t = X_t[idx], y_t[idx]
                        _, _, lf_p, _, _ = self.model(X_t, stage=1)
                        loss = loss + torch.mean((lf_p[t] - y_t) ** 2)
                    loss.backward()
                    opt_warmup.step()
                    sched_warmup.step()

        # Phase 2: Joint training
        for param in self.model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=joint_steps, eta_min=lr * 0.01
        )

        loss_history = []
        for epoch in range(epochs):
            perm = [torch.randperm(lf_device[t][0].size(0)) for t in range(self.model.num_experts)]
            epoch_loss = 0.0

            for b in range(n_batches):
                optimizer.zero_grad()
                total_loss = torch.tensor(0.0, device=device)

                # Expert losses on LF data
                for t in range(self.model.num_experts):
                    X_t, y_t = lf_device[t]
                    if use_batches and X_t.size(0) > batch_size:
                        idx = perm[t][b*batch_size:(b+1)*batch_size]
                        X_t, y_t = X_t[idx], y_t[idx]
                    _, _, lf_p, _, _ = self.model(X_t, stage=2)
                    l_t = torch.mean((lf_p[t] - y_t) ** 2)
                    total_loss = total_loss + l_t

                # HF loss (every batch — few samples, most important)
                mu, var, _, _, alpha_values = self.model(hf_X, stage=2)
                kl_div = self.model.kl_divergence()
                l_hf = torch.mean((mu - hf_y) ** 2)
                kl_scaled = self.kl_weight * kl_div / max(n_hf, 1)
                total_loss = total_loss + hf_weight * l_hf + kl_scaled

                # Gate sparsity
                if alpha_values:
                    gate_reg = self.lambda_gate * sum(
                        v.abs() for v in alpha_values.values()
                    )
                    total_loss = total_loss + gate_reg

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                scheduler.step()

                epoch_loss += total_loss.item()

            loss_history.append(epoch_loss / n_batches)

            if verbose and (epoch + 1) % max(1, 500 // n_batches) == 0:
                alpha_str = ""
                if alpha_values:
                    alpha_str = ", alpha=" + ",".join(
                        f"{v.item():.3f}" for v in alpha_values.values()
                    )
                q_strs = []
                with torch.no_grad():
                    _, _, _, q_vals, _ = self.model(hf_X, stage=2)
                    for q in q_vals:
                        q_strs.append(f"{q.mean().item():.3f}")
                q_str = ",".join(q_strs)
                print(f"  Epoch {epoch+1}/{epochs} ({(epoch+1)*n_batches} steps), "
                      f"loss: {epoch_loss/n_batches:.6f}, q=[{q_str}]{alpha_str}")

        return loss_history

    # ------------------------------------------------------------------
    # Three-Phase Training (warmup → frozen HF → joint fine-tune)
    # ------------------------------------------------------------------

    def train_three_phase(self, datasets: list, device: torch.device,
                          warmup_epochs: int = 15000,
                          frozen_epochs: int = 8000,
                          joint_epochs: int = 5000,
                          hf_weight: float = 8.0,
                          lr_warmup: float = 0.005,
                          lr_frozen: float = 0.003,
                          lr_joint: float = 0.001,
                          verbose: bool = True):
        """
        Three-phase training: best for small HF datasets.

        Phase 1: Train base + experts on LF data (warmup)
        Phase 2: Freeze base/experts, train HF module + gates on HF data
        Phase 3: Unfreeze all, gentle joint fine-tuning

        Args:
            datasets: [(X1,y1),...,(Xk,yk),(X_hf,y_hf)]
        """
        if verbose:
            print("=" * 60)
            print("THREE-PHASE Training")
            print("=" * 60)

        self.model.to(device)
        lf_datasets = datasets[:-1]
        hf_X, hf_y = datasets[-1]
        hf_X, hf_y = hf_X.to(device), hf_y.to(device)
        n_hf = hf_X.size(0)

        all_loss = []

        # ---- Phase 1: Expert warmup ----
        if verbose:
            print(f"  Phase 1: Expert warmup ({warmup_epochs} epochs)...")
        base_expert_params = (list(self.model.base_module.parameters())
                              + list(self.model.experts.parameters()))
        opt1 = torch.optim.Adam(base_expert_params, lr=lr_warmup,
                                weight_decay=self.weight_decay)
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt1, T_max=warmup_epochs, eta_min=lr_warmup * 0.01)

        for ep in range(warmup_epochs):
            opt1.zero_grad()
            loss = torch.tensor(0.0, device=device)
            for t in range(self.model.num_experts):
                X_t, y_t = lf_datasets[t]
                X_t, y_t = X_t.to(device), y_t.to(device)
                _, _, lf_p, _, _ = self.model(X_t, stage=1)
                loss = loss + torch.mean((lf_p[t] - y_t) ** 2)
            loss.backward()
            opt1.step()
            sched1.step()
            all_loss.append(loss.item())

        # ---- Phase 2: Frozen experts, train HF + gates ----
        if verbose:
            print(f"  Phase 2: Frozen experts, train HF+gates ({frozen_epochs} epochs)...")
        for p in self.model.base_module.parameters():
            p.requires_grad = False
        for p in self.model.experts.parameters():
            p.requires_grad = False

        trainable = list(self.model.hf_module.parameters()) + \
                    list(self.model.quality_gate.parameters())
        if self.model.routing_gates is not None:
            trainable += list(self.model.routing_gates.parameters())

        opt2 = torch.optim.Adam(trainable, lr=lr_frozen,
                                weight_decay=self.weight_decay)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt2, T_max=frozen_epochs, eta_min=lr_frozen * 0.01)

        for ep in range(frozen_epochs):
            opt2.zero_grad()
            mu, var, _, _, alpha_values = self.model(hf_X, stage=2)
            kl_div = self.model.kl_divergence()
            loss = hf_weight * torch.mean((mu - hf_y) ** 2)
            loss = loss + self.kl_weight * kl_div / max(n_hf, 1)
            if alpha_values:
                loss = loss + self.lambda_gate * sum(
                    v.abs() for v in alpha_values.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            opt2.step()
            sched2.step()
            all_loss.append(loss.item())

        # ---- Phase 3: Joint fine-tuning ----
        if joint_epochs > 0:
            if verbose:
                print(f"  Phase 3: Joint fine-tuning ({joint_epochs} epochs)...")
            for p in self.model.parameters():
                p.requires_grad = True

            opt3 = torch.optim.Adam(self.model.parameters(), lr=lr_joint,
                                    weight_decay=self.weight_decay)
            sched3 = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt3, T_max=joint_epochs, eta_min=lr_joint * 0.01)

            for ep in range(joint_epochs):
                opt3.zero_grad()
                total_loss = torch.tensor(0.0, device=device)
                for t in range(self.model.num_experts):
                    X_t, y_t = lf_datasets[t]
                    X_t, y_t = X_t.to(device), y_t.to(device)
                    _, _, lf_p, _, _ = self.model(X_t, stage=2)
                    total_loss = total_loss + torch.mean((lf_p[t] - y_t) ** 2)
                mu, var, _, _, alpha_values = self.model(hf_X, stage=2)
                kl_div = self.model.kl_divergence()
                total_loss = total_loss + hf_weight * torch.mean((mu - hf_y) ** 2)
                total_loss = total_loss + self.kl_weight * kl_div / max(n_hf, 1)
                if alpha_values:
                    total_loss = total_loss + self.lambda_gate * sum(
                        v.abs() for v in alpha_values.values())
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                opt3.step()
                sched3.step()
                all_loss.append(total_loss.item())

        return all_loss

    def train(self, datasets: list, device: torch.device,
              meta_rounds: int = 100, expert_epochs: int = 200,
              hf_epochs: int = 200, stage2_epochs: int = 20,
              mode: str = 'two_stage', e2e_epochs: int = 5000,
              e2e_hf_weight: float = 5.0, e2e_lr: float = None,
              e2e_warmup: int = 0, e2e_batch_size: int = 0,
              tp_warmup: int = 15000, tp_frozen: int = 8000,
              tp_joint: int = 5000, tp_hf_weight: float = 8.0,
              tp_lr_warmup: float = 0.005, tp_lr_frozen: float = 0.003,
              tp_lr_joint: float = 0.001,
              verbose: bool = True):
        """
        Main training entry point.

        Args:
            mode: 'two_stage', 'e2e', or 'three_phase'
        """
        if mode == 'three_phase':
            tp_loss = self.train_three_phase(
                datasets, device,
                warmup_epochs=tp_warmup,
                frozen_epochs=tp_frozen,
                joint_epochs=tp_joint,
                hf_weight=tp_hf_weight,
                lr_warmup=tp_lr_warmup,
                lr_frozen=tp_lr_frozen,
                lr_joint=tp_lr_joint,
                verbose=verbose,
            )
            return {
                'stage1_meta':   [],
                'stage1_expert': {t: [] for t in range(self.model.num_experts)},
                'stage1_hf':     [],
                'stage2':        tp_loss,
            }
        elif mode == 'e2e':
            e2e_loss = self.train_e2e(
                datasets, device,
                epochs=e2e_epochs,
                hf_weight=e2e_hf_weight,
                lr=e2e_lr,
                warmup_epochs=e2e_warmup,
                batch_size=e2e_batch_size,
                verbose=verbose,
            )
            # Return compatible format for loss history
            return {
                'stage1_meta':   [],
                'stage1_expert': {t: [] for t in range(self.model.num_experts)},
                'stage1_hf':     [],
                'stage2':        e2e_loss,
            }
        else:
            s1_loss = self.stage1(datasets, device, meta_rounds,
                                  expert_epochs, hf_epochs, verbose)
            s2_loss = self.stage2(datasets, device, stage2_epochs, verbose)

            return {
                'stage1_meta':   s1_loss['meta'],
                'stage1_expert': s1_loss['expert'],
                'stage1_hf':     s1_loss['hf'],
                'stage2':        s2_loss,
            }
