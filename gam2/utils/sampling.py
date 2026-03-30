"""
Sampling Utilities: LHS and OLHS
"""

import numpy as np
from scipy.stats import qmc


def generate_lhs(n_samples: int, dim: int, bounds: np.ndarray,
                 seed: int = None) -> np.ndarray:
    """
    Generate Latin Hypercube Samples.

    Args:
        n_samples: number of samples
        dim: input dimensionality
        bounds: (dim, 2) array of [lower, upper] bounds
        seed: random seed

    Returns:
        samples: (n_samples, dim) array
    """
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    samples = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])
    return samples


def generate_olhs(n_samples: int, dim: int, bounds: np.ndarray,
                  seed: int = None, n_candidates: int = 100) -> np.ndarray:
    """
    Generate Optimized Latin Hypercube Samples.
    Uses maximin criterion to select best from multiple LHS candidates.

    Args:
        n_samples: number of samples
        dim: input dimensionality
        bounds: (dim, 2) array of [lower, upper] bounds
        seed: random seed
        n_candidates: number of candidate LHS designs to compare

    Returns:
        samples: (n_samples, dim) array (best maximin design)
    """
    rng = np.random.default_rng(seed)
    best_samples = None
    best_min_dist = -1

    for _ in range(n_candidates):
        s = rng.integers(0, 2**31)
        sampler = qmc.LatinHypercube(d=dim, seed=s)
        unit_samples = sampler.random(n=n_samples)
        samples = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])

        # Compute minimum pairwise distance (maximin criterion)
        from scipy.spatial.distance import pdist
        if n_samples > 5000:
            # For large n, subsample to avoid O(n^2) memory
            idx = rng.choice(n_samples, size=min(2000, n_samples), replace=False)
            min_dist = pdist(samples[idx]).min()
        elif n_samples > 1:
            min_dist = pdist(samples).min()
        else:
            min_dist = 0

        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_samples = samples

    return best_samples
