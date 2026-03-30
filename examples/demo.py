"""
Demo script for GAM2 (Gated Adaptive Multi-fidelity Metamodeling)
Demonstrates the full pipeline on synthetic multi-fidelity test functions.
"""

import torch
import numpy as np

from models.dg_moe_bnn import DGMoEBNN
from training.trainer import GAM2Trainer
from active_learning.gasa import GASA
from utils.metrics import rmse, maxae
from utils.sampling import generate_lhs


# ============================================================================
# Synthetic Test Functions
# ============================================================================

def forrester_hf(x):
    """Forrester high-fidelity function (1D)."""
    x = np.asarray(x).flatten()
    return ((6 * x - 2) ** 2 * np.sin(12 * x - 4)).reshape(-1, 1)


def forrester_lf1(x):
    """Forrester low-fidelity function 1 (hierarchical)."""
    x = np.asarray(x).flatten()
    hf = ((6 * x - 2) ** 2 * np.sin(12 * x - 4)).flatten()
    return (0.5 * hf + 10 * (x - 0.5) - 5).reshape(-1, 1)


def forrester_lf2(x):
    """Forrester low-fidelity function 2 (non-hierarchical, different form)."""
    x = np.asarray(x).flatten()
    return (3 * np.sin(8 * np.pi * x) + 2 * x).reshape(-1, 1)


def branin_hf(x):
    """Branin high-fidelity function (2D)."""
    x = np.asarray(x).reshape(-1, 2)
    x1, x2 = x[:, 0], x[:, 1]
    a, b, c = 1, 5.1 / (4 * np.pi ** 2), 5 / np.pi
    r, s, t = 6, 10, 1 / (8 * np.pi)
    return (a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 +
            s * (1 - t) * np.cos(x1) + s).reshape(-1, 1)


def branin_lf(x):
    """Branin low-fidelity function (2D)."""
    x = np.asarray(x).reshape(-1, 2)
    return (branin_hf(x).flatten() * 0.7 + 5 * x[:, 0] - 3).reshape(-1, 1)


# --- 20D Dixon-Price multi-fidelity functions ---

def dixon_price_hf(x):
    """Dixon-Price high-fidelity function (dD). x_i ∈ [-10, 10].
    Scaled by 1/d^3 for numerical stability in high dimensions."""
    x = np.asarray(x).reshape(-1, x.shape[-1] if x.ndim > 1 else -1)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    d = x.shape[1]
    term1 = (x[:, 0] - 1) ** 2
    term2 = np.sum(
        [i * (2 * x[:, i] ** 2 - x[:, i - 1]) ** 2 for i in range(1, d)],
        axis=0,
    )
    return ((term1 + term2) / (d ** 3)).reshape(-1, 1)


def dixon_price_lf1(x):
    """
    Dixon-Price LF1 (hierarchical, higher correlation).
    Affine transform of HF.
    """
    x = np.asarray(x).reshape(-1, x.shape[-1] if x.ndim > 1 else -1)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    hf = dixon_price_hf(x).flatten()
    return (0.8 * hf + 0.5 * np.mean(x, axis=1) + 5.0).reshape(-1, 1)


def dixon_price_lf2(x):
    """
    Dixon-Price LF2 (non-hierarchical, moderate correlation ~0.6-0.8).
    Uses HF with multiplicative and additive region-dependent perturbation.
    """
    x = np.asarray(x).reshape(-1, x.shape[-1] if x.ndim > 1 else -1)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    hf = dixon_price_hf(x).flatten()
    # Multiplicative factor varies 0.3–0.9 depending on input region
    mult = 0.6 + 0.3 * np.sin(0.3 * np.sum(x[:, :5], axis=1))
    # Additive distortion scaled to HF magnitude
    add = 0.2 * hf.mean() * np.cos(0.2 * np.sum(x[:, 5:10], axis=1))
    return (mult * hf + add).reshape(-1, 1)


# --- 50D Styblinski-Tang multi-fidelity functions ---

def styblinski_tang_hf(x):
    """Styblinski-Tang high-fidelity function (dD). x_i ∈ [-5, 5]."""
    x = np.asarray(x).reshape(-1, x.shape[-1] if x.ndim > 1 else -1)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    val = 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x, axis=1)
    return val.reshape(-1, 1)


def styblinski_tang_lf1(x):
    """
    Styblinski-Tang LF1 (hierarchical, good correlation).
    Affine scaling of HF + small bias.
    """
    x = np.asarray(x).reshape(-1, x.shape[-1] if x.ndim > 1 else -1)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    hf = styblinski_tang_hf(x).flatten()
    return (0.9 * hf + 2.0 * np.mean(x, axis=1) + 10.0).reshape(-1, 1)


def styblinski_tang_lf2(x):
    """
    Styblinski-Tang LF2 (non-hierarchical, moderate correlation ~0.6-0.7).
    Uses 0.7*HF + spatially-varying perturbation that breaks rank ordering
    with LF1 in different regions.
    """
    x = np.asarray(x).reshape(-1, x.shape[-1] if x.ndim > 1 else -1)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    d = x.shape[1]
    hf = styblinski_tang_hf(x).flatten()
    # Region-dependent perturbation using sin of grouped dims
    perturb = (150.0 * np.sin(0.2 * np.sum(x[:, :10], axis=1))
               + 120.0 * np.cos(0.15 * np.sum(x[:, 10:25], axis=1)))
    return (0.7 * hf + perturb).reshape(-1, 1)


def styblinski_tang_lf3(x):
    """
    Styblinski-Tang LF3 (low quality, weak correlation ~0.2-0.4).
    Only 30% of HF signal + heavy oscillatory distortion.
    """
    x = np.asarray(x).reshape(-1, x.shape[-1] if x.ndim > 1 else -1)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    hf = styblinski_tang_hf(x).flatten()
    distortion = (150.0 * np.sin(0.5 * np.sum(x[:, :15], axis=1))
                  + 100.0 * np.cos(0.3 * np.sum(x[:, 15:35], axis=1)))
    return (0.3 * hf + distortion + 200.0).reshape(-1, 1)


# ============================================================================
# Demo 1: 1D Forrester (2 LF + 1 HF, hybrid fidelity)
# ============================================================================

def demo_forrester():
    """
    Demo with 1D Forrester function.
    - LF1: hierarchical (correlated with HF)
    - LF2: non-hierarchical (weakly correlated)
    - HF: Forrester function
    """
    print("=" * 70)
    print("DEMO 1: 1D Forrester (2 LF + 1 HF, hybrid fidelity)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    np.random.seed(42)
    torch.manual_seed(42)

    # Problem setup
    input_dim = 1
    bounds = np.array([[0.0, 1.0]])

    # Generate initial samples
    X_lf1 = generate_lhs(20, input_dim, bounds, seed=1)
    X_lf2 = generate_lhs(15, input_dim, bounds, seed=2)
    X_hf = generate_lhs(5, input_dim, bounds, seed=3)

    y_lf1 = forrester_lf1(X_lf1)
    y_lf2 = forrester_lf2(X_lf2)
    y_hf = forrester_hf(X_hf)

    # Convert to tensors
    datasets = [
        (torch.tensor(X_lf1, dtype=torch.float32),
         torch.tensor(y_lf1, dtype=torch.float32)),
        (torch.tensor(X_lf2, dtype=torch.float32),
         torch.tensor(y_lf2, dtype=torch.float32)),
        (torch.tensor(X_hf, dtype=torch.float32),
         torch.tensor(y_hf, dtype=torch.float32)),
    ]

    # Test data
    X_test = np.linspace(0, 1, 100).reshape(-1, 1)
    y_test = forrester_hf(X_test)

    # Create model
    model = DGMoEBNN(
        input_dim=input_dim,
        num_lf_models=2,
        base_hidden=32,
        expert_hidden=32,
        hf_hidden=32,
        output_dim=1,
    ).to(device)

    # --- Stage 1 + Stage 2 Training ---
    print("\n--- Training DG-MoE-BNN ---")
    trainer = GAM2Trainer(model, base_lr=0.001, lambda_gate=0.01)
    trainer.train(datasets, device, meta_rounds=50, expert_epochs=100,
                  hf_epochs=100, stage2_epochs=20, verbose=True)

    # Evaluate
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu, var, lf_preds, q_vals, alpha_vals = model(X_test_tensor, stage=2)

    test_rmse = rmse(mu, y_test)
    test_maxae = maxae(mu, y_test)

    print(f"\n--- Results after training ---")
    print(f"RMSE:  {test_rmse:.6f}")
    print(f"MaxAE: {test_maxae:.6f}")
    print(f"Quality gates: q1={q_vals[0].mean().item():.4f}, "
          f"q2={q_vals[1].mean().item():.4f}")
    if alpha_vals:
        print("Routing gates:")
        for (i, j), v in alpha_vals.items():
            print(f"  α_{i+1},{j+1} = {v.item():.4f}")
    print(f"Mean variance: {var.mean().item():.6f}")

    return model, datasets


# ============================================================================
# Demo 2: Full GASA Active Learning on Forrester
# ============================================================================

def demo_gasa_forrester():
    """
    Demo with full GASA active learning loop on Forrester problem.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Full GASA Active Learning on 1D Forrester")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(42)
    torch.manual_seed(42)

    input_dim = 1
    bounds = np.array([[0.0, 1.0]])

    # Small initial samples
    X_lf1 = generate_lhs(10, input_dim, bounds, seed=1)
    X_lf2 = generate_lhs(8, input_dim, bounds, seed=2)
    X_hf = generate_lhs(3, input_dim, bounds, seed=3)

    datasets = [
        (torch.tensor(X_lf1, dtype=torch.float32),
         torch.tensor(forrester_lf1(X_lf1), dtype=torch.float32)),
        (torch.tensor(X_lf2, dtype=torch.float32),
         torch.tensor(forrester_lf2(X_lf2), dtype=torch.float32)),
        (torch.tensor(X_hf, dtype=torch.float32),
         torch.tensor(forrester_hf(X_hf), dtype=torch.float32)),
    ]

    # Test data
    X_test = np.linspace(0, 1, 200).reshape(-1, 1)
    y_test = forrester_hf(X_test)

    # Evaluation functions (simulate calling the actual models)
    eval_functions = [forrester_lf1, forrester_lf2, forrester_hf]
    costs = [1.0, 1.0, 10.0]  # HF is 10x more expensive

    # Create model
    model = DGMoEBNN(
        input_dim=input_dim,
        num_lf_models=2,
        base_hidden=32,
        expert_hidden=32,
        hf_hidden=32,
        output_dim=1,
    ).to(device)

    # Run GASA
    gasa = GASA(
        model=model,
        input_dim=input_dim,
        bounds=bounds,
        costs=costs,
        eval_functions=eval_functions,
        hf_test_X=X_test,
        hf_test_y=y_test,
        K=3,
        epsilon=0.01,
        gamma=0.2,
        max_iterations=10,
        base_lr=0.001,
        lambda_gate=0.01,
        n_candidates=500,
        verbose=True,
    )

    history = gasa.run(datasets, device)

    # Print summary
    print("\n--- GASA Summary ---")
    print(f"RMSE history: {[f'{r:.4f}' for r in history['rmse']]}")
    print(f"Total samples per model: {history['n_samples'][-1]}")
    print(f"Batch sizes: {history['batch_sizes']}")

    return history


# ============================================================================
# Demo 3: 2D Branin (1 LF + 1 HF)
# ============================================================================

def demo_branin():
    """
    Demo with 2D Branin function.
    Simple 1 LF + 1 HF case.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: 2D Branin (1 LF + 1 HF)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(42)
    torch.manual_seed(42)

    input_dim = 2
    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])

    # Generate initial samples
    X_lf = generate_lhs(30, input_dim, bounds, seed=1)
    X_hf = generate_lhs(8, input_dim, bounds, seed=2)

    datasets = [
        (torch.tensor(X_lf, dtype=torch.float32),
         torch.tensor(branin_lf(X_lf), dtype=torch.float32)),
        (torch.tensor(X_hf, dtype=torch.float32),
         torch.tensor(branin_hf(X_hf), dtype=torch.float32)),
    ]

    # Test data
    X_test = generate_lhs(200, input_dim, bounds, seed=99)
    y_test = branin_hf(X_test)

    # Create model
    model = DGMoEBNN(
        input_dim=input_dim,
        num_lf_models=1,
        base_hidden=64,
        expert_hidden=64,
        hf_hidden=64,
        output_dim=1,
    ).to(device)

    # Train
    trainer = GAM2Trainer(model, base_lr=0.001)
    trainer.train(datasets, device, meta_rounds=50, expert_epochs=100,
                  hf_epochs=100, stage2_epochs=20, verbose=True)

    # Evaluate
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu, var, _, q_vals, _ = model(X_test_tensor, stage=2)

    test_rmse = rmse(mu, y_test)
    test_maxae = maxae(mu, y_test)

    print(f"\n--- Results ---")
    print(f"RMSE:  {test_rmse:.6f}")
    print(f"MaxAE: {test_maxae:.6f}")
    print(f"Quality gate q1: {q_vals[0].mean().item():.4f}")
    print(f"Mean variance: {var.mean().item():.6f}")


# ============================================================================
# Demo 4: 20D Dixon-Price (2 LF + 1 HF, hybrid fidelity)
# ============================================================================

def demo_dixon_price_20d():
    """
    Demo with 20D Dixon-Price function.
    - LF1: hierarchical (affine of HF, good correlation)
    - LF2: non-hierarchical (different polynomial, moderate correlation)
    - HF: Dixon-Price function
    High-dimensional test to verify scalability.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: 20D Dixon-Price (2 LF + 1 HF, hybrid fidelity)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    np.random.seed(42)
    torch.manual_seed(42)

    input_dim = 20
    bounds = np.column_stack([np.full(input_dim, -10.0),
                              np.full(input_dim, 10.0)])

    # High-dim needs more samples: ~10d per LF, ~5d for HF
    X_lf1 = generate_lhs(200, input_dim, bounds, seed=1)
    X_lf2 = generate_lhs(200, input_dim, bounds, seed=2)
    X_hf = generate_lhs(60, input_dim, bounds, seed=3)

    datasets = [
        (torch.tensor(X_lf1, dtype=torch.float32),
         torch.tensor(dixon_price_lf1(X_lf1), dtype=torch.float32)),
        (torch.tensor(X_lf2, dtype=torch.float32),
         torch.tensor(dixon_price_lf2(X_lf2), dtype=torch.float32)),
        (torch.tensor(X_hf, dtype=torch.float32),
         torch.tensor(dixon_price_hf(X_hf), dtype=torch.float32)),
    ]

    # Test data
    X_test = generate_lhs(500, input_dim, bounds, seed=99)
    y_test = dixon_price_hf(X_test)

    # Larger network for high-dim
    model = DGMoEBNN(
        input_dim=input_dim,
        num_lf_models=2,
        base_hidden=128,
        expert_hidden=128,
        hf_hidden=128,
        output_dim=1,
    ).to(device)

    print(f"\nSample sizes: LF1={X_lf1.shape[0]}, LF2={X_lf2.shape[0]}, "
          f"HF={X_hf.shape[0]}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n--- Training DG-MoE-BNN ---")
    trainer = GAM2Trainer(model, base_lr=0.001, lambda_gate=0.01)
    trainer.train(datasets, device, meta_rounds=80, expert_epochs=200,
                  hf_epochs=200, stage2_epochs=30, verbose=True)

    # Evaluate
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu, var, lf_preds, q_vals, alpha_vals = model(X_test_tensor, stage=2)

    test_rmse = rmse(mu, y_test)
    test_maxae = maxae(mu, y_test)

    print(f"\n--- Results (20D Dixon-Price) ---")
    print(f"RMSE:  {test_rmse:.6f}")
    print(f"MaxAE: {test_maxae:.6f}")
    print(f"Quality gates: q1={q_vals[0].mean().item():.4f}, "
          f"q2={q_vals[1].mean().item():.4f}")
    if alpha_vals:
        print("Routing gates:")
        for (i, j), v in alpha_vals.items():
            print(f"  α_{i+1},{j+1} = {v.item():.4f}")
    print(f"Mean variance: {var.mean().item():.6f}")


# ============================================================================
# Demo 5: 50D Styblinski-Tang (3 LF + 1 HF, mixed fidelity)
# ============================================================================

def demo_styblinski_tang_50d():
    """
    Demo with 50D Styblinski-Tang function.
    - LF1: hierarchical (affine of HF, high correlation)
    - LF2: non-hierarchical (different structure, moderate correlation)
    - LF3: low quality (weak correlation, should be suppressed by q_t)
    - HF: Styblinski-Tang function
    Tests scalability to 50D and robustness to bad LF sources.
    """
    print("\n" + "=" * 70)
    print("DEMO 5: 50D Styblinski-Tang (3 LF + 1 HF, mixed fidelity)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    np.random.seed(42)
    torch.manual_seed(42)

    input_dim = 50
    bounds = np.column_stack([np.full(input_dim, -5.0),
                              np.full(input_dim, 5.0)])

    # 50D: need substantial samples
    X_lf1 = generate_lhs(500, input_dim, bounds, seed=1)
    X_lf2 = generate_lhs(500, input_dim, bounds, seed=2)
    X_lf3 = generate_lhs(400, input_dim, bounds, seed=3)
    X_hf = generate_lhs(100, input_dim, bounds, seed=4)

    datasets = [
        (torch.tensor(X_lf1, dtype=torch.float32),
         torch.tensor(styblinski_tang_lf1(X_lf1), dtype=torch.float32)),
        (torch.tensor(X_lf2, dtype=torch.float32),
         torch.tensor(styblinski_tang_lf2(X_lf2), dtype=torch.float32)),
        (torch.tensor(X_lf3, dtype=torch.float32),
         torch.tensor(styblinski_tang_lf3(X_lf3), dtype=torch.float32)),
        (torch.tensor(X_hf, dtype=torch.float32),
         torch.tensor(styblinski_tang_hf(X_hf), dtype=torch.float32)),
    ]

    # Test data
    X_test = generate_lhs(500, input_dim, bounds, seed=99)
    y_test = styblinski_tang_hf(X_test)

    # Larger network for 50D
    model = DGMoEBNN(
        input_dim=input_dim,
        num_lf_models=3,
        base_hidden=256,
        expert_hidden=128,
        hf_hidden=128,
        output_dim=1,
    ).to(device)

    print(f"\nSample sizes: LF1={X_lf1.shape[0]}, LF2={X_lf2.shape[0]}, "
          f"LF3={X_lf3.shape[0]}, HF={X_hf.shape[0]}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Train — more epochs for high-dim
    print("\n--- Training DG-MoE-BNN ---")
    trainer = GAM2Trainer(model, base_lr=0.0005, lambda_gate=0.01)
    trainer.train(datasets, device, meta_rounds=100, expert_epochs=300,
                  hf_epochs=300, stage2_epochs=40, verbose=True)

    # Evaluate
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu, var, lf_preds, q_vals, alpha_vals = model(X_test_tensor, stage=2)

    test_rmse = rmse(mu, y_test)
    test_maxae = maxae(mu, y_test)

    print(f"\n--- Results (50D Styblinski-Tang) ---")
    print(f"RMSE:  {test_rmse:.6f}")
    print(f"MaxAE: {test_maxae:.6f}")
    print(f"Quality gates: q1={q_vals[0].mean().item():.4f}, "
          f"q2={q_vals[1].mean().item():.4f}, "
          f"q3={q_vals[2].mean().item():.4f}")
    if alpha_vals:
        print("Routing gates:")
        for (i, j), v in alpha_vals.items():
            print(f"  α_{i+1},{j+1} = {v.item():.4f}")
    print(f"Mean variance: {var.mean().item():.6f}")
    print(f"\nExpected: q1 > q2 > q3 (LF3 is low quality, should be suppressed)")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("GAM2: Gated Adaptive Multi-fidelity Metamodeling")
    print("DG-MoE-BNN Implementation Demo")
    print()

    # Demo 1: Basic training (1D)
    demo_forrester()

    # Demo 2: Full GASA active learning (1D)
    demo_gasa_forrester()

    # Demo 3: 2D problem
    demo_branin()

    # Demo 4: 20D high-dimensional
    demo_dixon_price_20d()

    # Demo 5: 50D high-dimensional with bad LF source
    demo_styblinski_tang_50d()

    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)
