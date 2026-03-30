"""
GASA Main Loop (Algorithm 1)
Gate-Guided Active Sampling Algorithm.
"""

import torch
import numpy as np
from copy import deepcopy

from models.dg_moe_bnn import DGMoEBNN
from training.trainer import GAM2Trainer
from utils.metrics import rmse, improvement_rate
from active_learning.adaptive_batch import AdaptiveBatchSize
from active_learning.sample_selection import SampleSelector
from active_learning.model_selection import GEAFModelSelector


class GASA:
    """
    Gate-Guided Active Sampling Algorithm (Algorithm 1).

    Orchestrates the full active learning loop:
    1. Adaptive batch size determination
    2. Variance-weighted sample selection
    3. GEAF-based model selection
    4. Network retraining (full or fine-tune based on G^(i))
    5. Stopping criterion based on improvement rate
    """

    def __init__(self, model: DGMoEBNN, input_dim: int, bounds: np.ndarray,
                 costs: list, eval_functions: list,
                 hf_test_X: np.ndarray = None, hf_test_y: np.ndarray = None,
                 K: int = 3, epsilon: float = 0.01, gamma: float = 0.2,
                 max_iterations: int = 50, base_lr: float = 0.001,
                 lambda_gate: float = 0.01, n_candidates: int = 1000,
                 verbose: bool = True, train_mode: str = 'two_stage',
                 initial_train_kwargs: dict = None,
                 retrain_train_kwargs: dict = None,
                 finetune_train_kwargs: dict = None,
                 rmse_target: float = None,
                 min_hf_per_iteration: int = 0,
                 rmse_eval_fn=None,
                 normalize_rg: bool = False,
                 cost_exponent: float = 1.0,
                 high_dim_threshold: int = 20,
                 n_base_override: int = None):
        """
        Args:
            model: DG-MoE-BNN model
            input_dim: d, input dimensionality
            bounds: (d, 2) bounds for each input dimension
            costs: [C_1, ..., C_s] computational costs
            eval_functions: list of callable f_t(x) → y_t for each model
            hf_test_X: (N_test, d) test inputs (for RMSE tracking)
            hf_test_y: (N_test, 1) test HF outputs
            K: number of consecutive iterations for stopping
            epsilon: improvement rate threshold
            gamma: gate stability switching threshold
            max_iterations: maximum number of AL iterations
            base_lr: base learning rate
            lambda_gate: gate sparsity weight
            n_candidates: number of candidate samples per iteration
            verbose: print progress
        """
        self.model = model
        self.input_dim = input_dim
        self.bounds = bounds
        self.costs = costs
        self.eval_functions = eval_functions
        self.hf_test_X = hf_test_X
        self.hf_test_y = hf_test_y
        self.K = K
        self.epsilon = epsilon
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.base_lr = base_lr
        self.lambda_gate = lambda_gate
        self.verbose = verbose
        self.train_mode = train_mode
        self.initial_train_kwargs = initial_train_kwargs or {}
        self.retrain_train_kwargs = retrain_train_kwargs or {}
        self.finetune_train_kwargs = finetune_train_kwargs or {}
        self.rmse_target = rmse_target
        self.min_hf_per_iteration = min_hf_per_iteration
        self.rmse_eval_fn = rmse_eval_fn

        # Components
        self.batch_sizer = AdaptiveBatchSize(input_dim, n_base_override=n_base_override)
        self.sample_selector = SampleSelector(input_dim, bounds, n_candidates)
        self.model_selector = GEAFModelSelector(
            costs, normalize_rg=normalize_rg,
            cost_exponent=cost_exponent,
            high_dim_threshold=high_dim_threshold,
        )

    def _get_gate_values(self, model, datasets, device):
        """Extract mean gate values over candidate points."""
        candidates = self.sample_selector.generate_candidates()
        X = torch.tensor(candidates, dtype=torch.float32, device=device)

        model.eval()
        with torch.no_grad():
            _, _, _, q_vals, alpha_vals = model(X, stage=2)
            q_means = [q.mean().item() for q in q_vals]
            alpha_means = {k: v.item() for k, v in alpha_vals.items()}

        return q_means, alpha_means

    def _evaluate_rmse(self, model, device):
        """Evaluate current RMSE on test data."""
        if self.rmse_eval_fn is not None:
            return float(self.rmse_eval_fn(model, device))

        if self.hf_test_X is None or self.hf_test_y is None:
            return None

        X = torch.tensor(self.hf_test_X, dtype=torch.float32, device=device)
        model.eval()
        with torch.no_grad():
            mu, _, _, _, _ = model(X, stage=2)
        return rmse(mu, self.hf_test_y)

    def run(self, datasets: list, device: torch.device):
        """
        Run the full GASA active learning loop.

        Args:
            datasets: list of (X_t, y_t) tensors for t=1,...,s
                      (mutable — new samples are appended in-place)
            device: torch device

        Returns:
            history: dict with RMSE history, sample counts, gate values, etc.
        """
        history = {
            'rmse': [],
            'n_samples': [],
            'batch_sizes': [],
            'gate_q': [],
            'gate_alpha': [],
            'models_selected': [],
        }

        num_models = len(datasets)  # s

        # Initial training (Stage 1 + Stage 2)
        if self.verbose:
            print("\n" + "=" * 60)
            print("GASA: Initial Training")
            print("=" * 60)

        trainer = GAM2Trainer(self.model, self.base_lr, self.lambda_gate)
        trainer.train(
            datasets, device,
            mode=self.train_mode,
            verbose=self.verbose,
            **self.initial_train_kwargs,
        )

        # Record initial state
        q_means, alpha_means = self._get_gate_values(self.model, datasets, device)
        mean_var = self.sample_selector.compute_mean_variance(self.model, device)
        self.batch_sizer.initialize(mean_var, q_means, alpha_means)

        current_rmse = self._evaluate_rmse(self.model, device)
        if current_rmse is not None:
            history['rmse'].append(current_rmse)
            if self.verbose:
                print(f"\nInitial RMSE: {current_rmse:.6f}")
            if self.rmse_target is not None and current_rmse <= self.rmse_target:
                if self.verbose:
                    print(f"Initial model already satisfies RMSE <= {self.rmse_target:.6f}.")
                return history

        history['gate_q'].append(q_means)
        history['gate_alpha'].append(alpha_means)

        retrain_flag = True
        consecutive_below_epsilon = 0

        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"GASA Iteration {iteration}")
                print(f"{'=' * 60}")

            # 1. Compute adaptive batch size (Eq.19-21)
            mean_var = self.sample_selector.compute_mean_variance(
                self.model, device
            )
            q_means, alpha_means = self._get_gate_values(
                self.model, datasets, device
            )
            n_u = self.batch_sizer.compute_batch_size(
                iteration, mean_var, q_means, alpha_means
            )

            # Check if we should switch from retrain to fine-tune
            if retrain_flag and iteration >= 2:
                G = self.batch_sizer._compute_delta_gate(q_means, alpha_means)
                if self.batch_sizer.delta_gate_1 is not None:
                    G_ratio = G / max(self.batch_sizer.delta_gate_1, 1e-10)
                    if G_ratio < self.gamma:
                        retrain_flag = False
                        if self.verbose:
                            print(f"  Gate stability G={G_ratio:.4f} < γ={self.gamma}, "
                                  f"switching to fine-tune only.")

            if self.verbose:
                print(f"  Batch size N_U = {n_u}")

            # 2-4. Select input sample locations
            selected_x = self.sample_selector.select_samples(
                self.model, n_u, device
            )

            if self.verbose:
                print(f"  Selected {len(selected_x)} sample locations")

            # 5. Model selection + evaluation
            model_indices = self.model_selector.select_models_batch(
                selected_x, self.model, datasets, device
            )

            iter_models = []
            for k, (x_k, m_star) in enumerate(zip(selected_x, model_indices)):
                if self.min_hf_per_iteration > 0:
                    hf_count = sum(1 for m in iter_models if m == num_models - 1)
                    if hf_count < self.min_hf_per_iteration and k < self.min_hf_per_iteration:
                        m_star = num_models - 1

                # Evaluate the selected model
                y_k = self.eval_functions[m_star](x_k)
                if np.isscalar(y_k):
                    y_k = np.array([[y_k]])
                elif y_k.ndim == 1:
                    y_k = y_k.reshape(1, -1)

                # Add to dataset
                X_t, Y_t = datasets[m_star]
                x_k_tensor = torch.tensor(x_k.reshape(1, -1), dtype=torch.float32)
                y_k_tensor = torch.tensor(y_k, dtype=torch.float32)
                datasets[m_star] = (
                    torch.cat([X_t, x_k_tensor], dim=0),
                    torch.cat([Y_t, y_k_tensor], dim=0),
                )
                iter_models.append(m_star)

            history['models_selected'].append(iter_models)
            history['batch_sizes'].append(n_u)
            history['n_samples'].append(
                [datasets[t][0].size(0) for t in range(num_models)]
            )

            if self.verbose:
                model_counts = {}
                for m in iter_models:
                    model_counts[m] = model_counts.get(m, 0) + 1
                print(f"  Models evaluated: {model_counts}")
                print(f"  Total samples: {history['n_samples'][-1]}")

            # 6. Retrain or fine-tune
            if retrain_flag:
                if self.verbose:
                    print("  Full retraining (Stage 1 + Stage 2)...")
                # Reset model for full retrain
                self.model = self.model.rebuild_same().to(device)
                trainer = GAM2Trainer(self.model, self.base_lr, self.lambda_gate)
                trainer.train(
                    datasets, device,
                    mode=self.train_mode,
                    verbose=False,
                    **self.retrain_train_kwargs,
                )
            else:
                if self.verbose:
                    print("  Fine-tuning only (Stage 2)...")
                trainer = GAM2Trainer(self.model, self.base_lr, self.lambda_gate)
                if self.train_mode == 'two_stage':
                    stage2_epochs = self.finetune_train_kwargs.get('stage2_epochs', 15)
                    trainer.stage2(datasets, device, epochs=stage2_epochs, verbose=False)
                else:
                    trainer.train(
                        datasets, device,
                        mode=self.train_mode,
                        verbose=False,
                        **self.finetune_train_kwargs,
                    )

            # Record gate values
            q_means, alpha_means = self._get_gate_values(
                self.model, datasets, device
            )
            history['gate_q'].append(q_means)
            history['gate_alpha'].append(alpha_means)

            # Evaluate RMSE
            current_rmse = self._evaluate_rmse(self.model, device)
            if current_rmse is not None:
                history['rmse'].append(current_rmse)
                if self.verbose:
                    print(f"  RMSE: {current_rmse:.6f}")

                if self.rmse_target is not None and current_rmse <= self.rmse_target:
                    if self.verbose:
                        print(f"\n  Stopping: RMSE <= {self.rmse_target:.6f}.")
                    break

                # Check stopping criterion (Eq.28)
                if len(history['rmse']) > self.K:
                    ir = improvement_rate(
                        history['rmse'][-self.K - 1],
                        history['rmse'][-1]
                    )
                    if self.verbose:
                        print(f"  IR: {ir:.6f} (threshold: {self.epsilon})")

                    if ir < self.epsilon:
                        consecutive_below_epsilon += 1
                    else:
                        consecutive_below_epsilon = 0

                    if consecutive_below_epsilon >= self.K:
                        if self.verbose:
                            print(f"\n  Stopping: IR < {self.epsilon} for "
                                  f"{self.K} consecutive iterations.")
                        break

        if self.verbose:
            print(f"\nGASA completed after {iteration} iterations.")
            if history['rmse']:
                print(f"Final RMSE: {history['rmse'][-1]:.6f}")

        return history
