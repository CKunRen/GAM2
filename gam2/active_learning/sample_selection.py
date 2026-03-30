"""
Input Sample Selection (Steps 2-4, Eq.22)
Variance-weighted K-means clustering → Voronoi partition → max-variance selection.
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import qmc


class SampleSelector:
    """
    Voronoi-clustering-based sample selection.

    Step 2: Generate candidates via LHS, compute σ²_s
    Step 3: Variance-weighted K-means → Voronoi partition
    Step 4: Select max-variance point per cell (Eq.22)
    """

    def __init__(self, input_dim: int, bounds: np.ndarray,
                 n_candidates: int = 1000):
        """
        Args:
            input_dim: d, input dimensionality
            bounds: (d, 2) array of [lower, upper] bounds per dimension
            n_candidates: number of candidate samples to generate
        """
        self.input_dim = input_dim
        self.bounds = bounds
        self.n_candidates = n_candidates

    def generate_candidates(self) -> np.ndarray:
        """
        Generate candidate samples using Latin Hypercube Sampling.

        Returns:
            candidates: (n_candidates, d) array in the input space
        """
        sampler = qmc.LatinHypercube(d=self.input_dim)
        samples = sampler.random(n=self.n_candidates)
        # Scale to bounds
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        candidates = qmc.scale(samples, lower, upper)
        return candidates

    def select_samples(self, model, n_samples: int,
                       device: torch.device) -> np.ndarray:
        """
        Full sample selection pipeline.

        Args:
            model: trained DG-MoE-BNN model
            n_samples: N_U^(i), number of samples to select
            device: torch device

        Returns:
            selected: (n_samples, d) array of selected input locations
        """
        # Step 2: Generate candidates and compute variances
        candidates = self.generate_candidates()
        X_cand = torch.tensor(candidates, dtype=torch.float32, device=device)

        model.eval()
        with torch.no_grad():
            mu, var, _, _, _ = model(X_cand, stage=2)
            # Sum variance across output dims
            variances = var.sum(dim=-1).cpu().numpy()  # (n_candidates,)

        # Step 3: Variance-weighted K-means clustering
        n_clusters = min(n_samples, len(candidates))
        if n_clusters <= 0:
            return candidates[:1]

        # Weight candidates by variance for K-means
        # Use variance as sample weights
        weights = variances / (variances.sum() + 1e-10)

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(candidates, sample_weight=weights)
        labels = kmeans.labels_

        # Step 4: Select max-variance point per cell (Eq.22)
        selected = []
        for k in range(n_clusters):
            mask = labels == k
            if not mask.any():
                continue
            cell_indices = np.where(mask)[0]
            cell_variances = variances[cell_indices]
            best_idx = cell_indices[np.argmax(cell_variances)]
            selected.append(candidates[best_idx])

        selected = np.array(selected)
        return selected

    def compute_mean_variance(self, model, device: torch.device) -> float:
        """
        Compute mean HF prediction variance over candidates.
        Used for adaptive batch size computation.
        """
        candidates = self.generate_candidates()
        X_cand = torch.tensor(candidates, dtype=torch.float32, device=device)

        model.eval()
        with torch.no_grad():
            _, var, _, _, _ = model(X_cand, stage=2)
            mean_var = var.sum(dim=-1).mean().item()

        return mean_var
