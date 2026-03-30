"""
GEAF Evaluation Model Selection (Eq.23-27)
Gate-Enhanced Acquisition Function for selecting which model to evaluate.
"""

import torch
import numpy as np
from scipy.spatial.distance import cdist


class GEAFModelSelector:
    """
    Gate-Enhanced Acquisition Function (GEAF) for model selection.

    GEAF(x_k, t) = r_g(x_k, t) · CR(t)^p · ρ(x_k, t)  (Eq.23)

    r_g: gate-derived model correlation (Eq.24)
    CR: cost ratio (Eq.25)
    ρ: sample density function (Eq.26)
    M* = argmax_t GEAF(x_k, t)  (Eq.27)
    """

    def __init__(self, costs: list, length_scale: float = None,
                 normalize_rg: bool = False, cost_exponent: float = 1.0,
                 high_dim_threshold: int = 20):
        """
        Args:
            costs: list of computational costs [C_1, ..., C_s]
                   (last element is HF cost C_s)
            length_scale: l parameter for density function.
                          If None, estimated from data.
            normalize_rg: if True, scale LF r_g values by 1/max(q_t)
                          so the best LF expert has r_g=1.0 (comparable to HF).
                          Compensates for softmax dilution with many experts.
            cost_exponent: exponent p applied to CR(t). p>1 amplifies
                           cost advantage of cheaper models.
            high_dim_threshold: above this input dimension, use sample-count-
                                based density instead of distance-based.
        """
        self.costs = costs
        self.length_scale = length_scale
        self.normalize_rg = normalize_rg
        self.cost_exponent = cost_exponent
        self.high_dim_threshold = high_dim_threshold
        self.num_models = len(costs)  # s

    def compute_rg(self, q_values: list, x_k: torch.Tensor,
                   model, device: torch.device) -> list:
        """
        Gate-derived model correlation r_g(x_k, t) (Eq.24).

        r_g(x_k, t) = q_t(x_k) for t=1,...,s-1
        r_g(x_k, s) = 1 (HF model always r_g=1)

        When normalize_rg=True, LF values are scaled by 1/max(q_t) so the
        best expert reaches r_g=1.0, compensating for softmax dilution.
        """
        model.eval()
        with torch.no_grad():
            if x_k.dim() == 1:
                x_k = x_k.unsqueeze(0)
            _, _, _, q_vals, _ = model(x_k.to(device), stage=2)
            rg = [q.item() for q in q_vals]

            if self.normalize_rg and rg:
                q_max = max(rg) if max(rg) > 1e-8 else 1e-8
                rg = [q / q_max for q in rg]

            rg.append(1.0)  # HF model
        return rg

    def compute_cr(self) -> list:
        """
        Cost ratio CR(t) = C_s / C_t  (Eq.25).
        """
        c_s = self.costs[-1]
        return [c_s / max(c_t, 1e-10) for c_t in self.costs]

    def compute_density(self, x_k: np.ndarray,
                        datasets: list) -> list:
        """
        Sample density function ρ(x_k, t) (Eq.26).

        Low-D (d <= high_dim_threshold): distance-based
            ρ(x_k, t) = 1 - exp(-d_min²/(2l²))

        High-D (d > high_dim_threshold): sample-count-based
            ρ(t) = sqrt(n_median / n_t), clamped to [0.1, 3.0]
            Rationale: in high-D, distance-based ρ ≈ 1.0 for all models
            regardless of sample count (curse of dimensionality).

        Args:
            x_k: (d,) input point
            datasets: list of (X_t, y_t) for all models

        Returns:
            list of ρ values for each model
        """
        input_dim = x_k.reshape(-1).shape[0] if x_k.ndim <= 1 else x_k.shape[-1]
        if input_dim > self.high_dim_threshold:
            return self._count_based_density(datasets)
        return self._distance_based_density(x_k, datasets)

    def _count_based_density(self, datasets: list) -> list:
        """
        Sample-count-based density for high-dimensional problems.

        ρ(t) = sqrt(n_median / n_t)  — models with fewer samples get
        higher ρ, encouraging sampling where data is sparse.
        """
        counts = []
        for X_t, _ in datasets:
            n = X_t.shape[0] if isinstance(X_t, torch.Tensor) else len(X_t)
            counts.append(max(n, 1))
        n_median = float(np.median(counts))
        rho_values = [min(max(np.sqrt(n_median / n), 0.1), 3.0) for n in counts]
        return rho_values

    def _distance_based_density(self, x_k: np.ndarray,
                                datasets: list) -> list:
        """Original distance-based density for low-dimensional problems."""
        if x_k.ndim == 1:
            x_k = x_k.reshape(1, -1)

        rho_values = []
        for t, (X_t, _) in enumerate(datasets):
            X_t_np = X_t.cpu().numpy() if isinstance(X_t, torch.Tensor) else X_t
            if len(X_t_np) == 0:
                rho_values.append(1.0)
                continue

            # Compute minimum distance
            dists = cdist(x_k, X_t_np, metric='euclidean')
            d_min = dists.min()

            # Estimate length scale if not provided
            l = self.length_scale
            if l is None:
                # Use mean nearest-neighbor distance as estimate
                if len(X_t_np) > 1:
                    all_dists = cdist(X_t_np, X_t_np)
                    np.fill_diagonal(all_dists, np.inf)
                    l = np.mean(np.min(all_dists, axis=1))
                else:
                    l = 1.0

            rho = 1.0 - np.exp(-d_min ** 2 / (2 * l ** 2))
            rho_values.append(float(rho))

        return rho_values

    def select_model(self, x_k: np.ndarray, q_values: list,
                     datasets: list) -> int:
        """
        Select evaluation model using GEAF (Eq.27).

        M* = argmax_t GEAF(x_k, t)

        Args:
            x_k: (d,) input point
            q_values: list of q_t values at x_k (for LF models)
            datasets: list of (X_t, y_t) for all models

        Returns:
            M_star: index of selected model (0-indexed, s-1 = HF)
        """
        # r_g values (with optional normalization)
        rg = list(q_values)
        if self.normalize_rg and rg:
            q_max = max(rg) if max(rg) > 1e-8 else 1e-8
            rg = [q / q_max for q in rg]
        rg.append(1.0)  # HF model

        # Cost ratios (with exponent)
        cr = self.compute_cr()
        cr = [c ** self.cost_exponent for c in cr]

        # Density values
        rho = self.compute_density(x_k, datasets)

        # GEAF (Eq.23)
        geaf = [rg[t] * cr[t] * rho[t] for t in range(self.num_models)]

        # M* = argmax (Eq.27)
        M_star = int(np.argmax(geaf))
        return M_star

    def select_models_batch(self, x_batch: np.ndarray, model,
                            datasets: list, device: torch.device) -> list:
        """
        Select evaluation models for a batch of input points.

        Args:
            x_batch: (N, d) input points
            model: trained DG-MoE-BNN
            datasets: list of (X_t, y_t) for all models
            device: torch device

        Returns:
            model_indices: list of selected model indices
        """
        model.eval()
        X_tensor = torch.tensor(x_batch, dtype=torch.float32, device=device)

        with torch.no_grad():
            _, _, _, q_vals, _ = model(X_tensor, stage=2)
            # q_vals: list of (N, 1) tensors

        cr = self.compute_cr()
        cr = [c ** self.cost_exponent for c in cr]

        # For high-D count-based density, ρ is the same for all points
        input_dim = x_batch.shape[1] if x_batch.ndim > 1 else x_batch.shape[0]
        use_count_density = input_dim > self.high_dim_threshold
        if use_count_density:
            rho_shared = self._count_based_density(datasets)

        model_indices = []

        for k in range(len(x_batch)):
            x_k = x_batch[k:k+1]
            q_k = [q[k].item() for q in q_vals]
            rg = list(q_k)
            if self.normalize_rg and rg:
                q_max = max(rg) if max(rg) > 1e-8 else 1e-8
                rg = [q / q_max for q in rg]
            rg.append(1.0)

            rho = rho_shared if use_count_density else self.compute_density(x_k, datasets)
            geaf = [rg[t] * cr[t] * rho[t] for t in range(self.num_models)]
            model_indices.append(int(np.argmax(geaf)))

        return model_indices
