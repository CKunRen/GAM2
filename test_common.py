"""
Common utilities for DG-MoE-BNN per-example test scripts.
=========================================================
Shared helpers: OLHS sampling, error metrics, train loop, .mat I/O.

Each E{i}_test.py imports from here and defines only its own:
  - test functions, setup, network config, example config.
"""

import sys
import os
import time
import numpy as np
import torch
from scipy.io import savemat
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.dg_moe_bnn import DGMoEBNN
from training.trainer import GAM2Trainer
from utils.metrics import rmse
from utils.sampling import generate_olhs

# ============================================================================
# Global Settings
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED_BASE = 42
N_TEST = 1000
NUM_REPEATS = 20

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Sampling
# ============================================================================

def olhs_samples(n, dim, bounds, seed, n_cand=None):
    """OLHS with auto-scaled candidate count for large n."""
    if n_cand is None:
        if n >= 10000:
            n_cand = 2
        elif n >= 1000:
            n_cand = 5
        else:
            n_cand = 50
    return generate_olhs(n, dim, bounds, seed=seed, n_candidates=n_cand)


def olhs_normal(n, dim, mu, sigma, seed, n_cand=50):
    """OLHS in [0,1]^d -> inverse-CDF -> N(mu, sigma^2)."""
    unit = generate_olhs(n, dim, np.tile([0.0, 1.0], (dim, 1)),
                         seed=seed, n_candidates=n_cand)
    unit = np.clip(unit, 1e-6, 1 - 1e-6)
    return norm.ppf(unit, loc=mu, scale=sigma)


# ============================================================================
# Type Helpers
# ============================================================================

def to_t(x):
    return torch.tensor(np.asarray(x, np.float64), dtype=torch.float32)

def make2d(x):
    x = np.asarray(x, np.float64)
    return x.reshape(-1, 1) if x.ndim == 1 else x


# ============================================================================
# Error Metrics
# ============================================================================

def relative_l2_error(y_pred, y_true):
    yp, yt = np.asarray(y_pred).ravel(), np.asarray(y_true).ravel()
    return float(np.linalg.norm(yp - yt) / np.linalg.norm(yt))

def compute_mse(y_pred, y_true):
    yp, yt = np.asarray(y_pred).ravel(), np.asarray(y_true).ravel()
    return float(np.mean((yp - yt) ** 2))

def compute_rmse(y_pred, y_true):
    return rmse(y_pred, y_true)


# ============================================================================
# Parameter Count
# ============================================================================

def calc_n_params(d, h, k):
    """Estimate total parameter count for DGMoEBNN(d, k, h, h, h)."""
    base = 2688 + h * (64 * d + 1)           # Conv1d(1->16->32->64) + Linear(64d -> h)
    experts = k * (3 * h * h + 3 * h + 1)    # per expert: Linear(2h->h->h->1)
    routing = k * max(k - 1, 0)               # alpha_ij scalars
    quality = 49                               # shared MLP (1->16->1)
    hf = 2 * h * h + 5 * h                    # Linear(1+h->h->h) + BayesianLinear(h->1)
    return base + experts + routing + quality + hf


# ============================================================================
# Train & Evaluate (single run)
# ============================================================================

def train_and_evaluate(cfg, datasets_np, X_test_np, y_test_true, seed):
    """Build model, two-stage train, evaluate. Returns result dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    nc = cfg['net_cfg']
    h = nc['hidden']

    # --- Standardize outputs (y -> (y - y_mean) / y_std) ---
    all_y = np.concatenate([y.ravel() for _, y in datasets_np])
    y_mean = float(all_y.mean())
    y_std = float(all_y.std())
    y_std = max(y_std, 1e-8)

    model = DGMoEBNN(input_dim=cfg['d'], num_lf_models=cfg['num_lf'],
                     base_hidden=h, expert_hidden=h, hf_hidden=h,
                     **nc.get('model_kwargs', {})).to(DEVICE)

    datasets = [(to_t(make2d(X)).to(DEVICE),
                 to_t(((y.ravel() - y_mean) / y_std).reshape(-1, 1)).to(DEVICE))
                for X, y in datasets_np]

    trainer = GAM2Trainer(model,
                          base_lr=nc.get('base_lr', 0.001),
                          lambda_gate=nc.get('lambda_gate', 0.01),
                          kl_weight=nc.get('kl_weight', 1.0))

    loss_hist = trainer.train(
        datasets, DEVICE,
        meta_rounds=nc['meta_rounds'],
        expert_epochs=nc['expert_epochs'],
        hf_epochs=nc['hf_epochs'],
        stage2_epochs=nc['stage2_epochs'],
        mode=nc.get('train_mode', 'two_stage'),
        e2e_epochs=nc.get('e2e_epochs', 5000),
        e2e_hf_weight=nc.get('e2e_hf_weight', 5.0),
        e2e_lr=nc.get('e2e_lr', None),
        e2e_warmup=nc.get('e2e_warmup', 0),
        e2e_batch_size=nc.get('e2e_batch_size', 0),
        tp_warmup=nc.get('tp_warmup', 15000),
        tp_frozen=nc.get('tp_frozen', 8000),
        tp_joint=nc.get('tp_joint', 5000),
        tp_hf_weight=nc.get('tp_hf_weight', 8.0),
        tp_lr_warmup=nc.get('tp_lr_warmup', 0.005),
        tp_lr_frozen=nc.get('tp_lr_frozen', 0.003),
        tp_lr_joint=nc.get('tp_lr_joint', 0.001),
        verbose=False,
    )

    # --- test prediction (1000 OLHS) -> de-normalize ---
    model.eval()
    Xt = to_t(make2d(X_test_np)).to(DEVICE)
    with torch.no_grad():
        mu, var, _, q_vals, alpha_vals = model(Xt, stage=2)
    y_pred = mu.cpu().numpy().ravel() * y_std + y_mean
    y_var = var.cpu().numpy().ravel() * (y_std ** 2)

    err = cfg['metric_fn'](y_pred, y_test_true)

    alpha_dict = {f"a{i+1}{j+1}": float(v.item())
                  for (i, j), v in alpha_vals.items()}
    q_means = [float(q.mean().item()) for q in q_vals]

    # --- 1D smooth curve (500 uniform) -> de-normalize ---
    y_curve = y_curve_var = None
    if cfg.get('is_1d'):
        xlo, xhi = cfg['x_range']
        x_uni = np.linspace(xlo, xhi, 500)
        Xu = to_t(make2d(x_uni)).to(DEVICE)
        with torch.no_grad():
            mu_c, var_c, _, _, _ = model(Xu, stage=2)
        y_curve = mu_c.cpu().numpy().ravel() * y_std + y_mean
        y_curve_var = var_c.cpu().numpy().ravel() * (y_std ** 2)

    return dict(error=err, y_pred=y_pred, y_var=y_var,
                y_curve=y_curve, y_curve_var=y_curve_var,
                alpha=alpha_dict, q_means=q_means,
                loss_hist=loss_hist, net_cfg=nc)


# ============================================================================
# Save .mat
# ============================================================================

def save_loss_mat(cfg, all_loss_hists):
    """Save E{i}_loss.mat: 20 runs x training loss curves."""
    mat = {}

    def stack_pad(arrays):
        max_len = max(len(a) for a in arrays)
        out = np.full((len(arrays), max_len), np.nan)
        for i, a in enumerate(arrays):
            out[i, :len(a)] = a
        return out

    mat['stage1_meta'] = stack_pad([h['stage1_meta'] for h in all_loss_hists])
    mat['stage1_hf'] = stack_pad([h['stage1_hf'] for h in all_loss_hists])
    mat['stage2'] = stack_pad([h['stage2'] for h in all_loss_hists])

    for t in range(cfg['num_lf']):
        mat[f'stage1_expert{t+1}'] = stack_pad(
            [h['stage1_expert'][t] for h in all_loss_hists])

    mat_path = os.path.join(OUTPUT_DIR, f"E{cfg['idx']}_loss.mat")
    savemat(mat_path, mat)
    print(f"  >> Loss saved: {mat_path}")


def save_results_mat(cfg, results_list):
    """Save E{i}_results.mat: errors, gates, 1D curves."""
    idx = cfg['idx']
    mn = cfg['metric_name']

    mat = {}
    errors = np.array([r['error'] for r in results_list])
    mat['error_all'] = errors
    mat['error_metric'] = mn
    mat['error_mean'] = np.mean(errors)
    mat['error_std'] = np.std(errors)

    # Gates
    akeys = sorted(set().union(*(r['alpha'].keys() for r in results_list)))
    for k in akeys:
        mat[f'alpha_{k}'] = np.array(
            [r['alpha'].get(k, np.nan) for r in results_list])

    q_all = np.array([r['q_means'] for r in results_list])
    mat['q_all'] = q_all

    # Network info
    nc = results_list[0]['net_cfg']
    for k in ('hidden', 'n_params_est', 'meta_rounds',
              'expert_epochs', 'hf_epochs', 'stage2_epochs'):
        mat[k] = nc[k]

    # 1D curves (500 uniform points)
    if cfg.get('is_1d'):
        xlo, xhi = cfg['x_range']
        xp = np.linspace(xlo, xhi, 500)
        mat['x'] = xp
        mat['y_hf_true'] = cfg['hf_func'](xp)
        preds = np.array([r['y_curve'] for r in results_list])
        varis = np.array([r['y_curve_var'] for r in results_list])
        mat['y_pred_all'] = preds
        mat['y_var_all'] = varis
        mat['y_pred_mean'] = np.mean(preds, axis=0)
        mat['y_pred_std'] = np.std(preds, axis=0)
        for ln, lf in cfg.get('lf_funcs', {}).items():
            mat[ln] = lf(xp)

    mat_path = os.path.join(OUTPUT_DIR, f"E{idx}_results.mat")
    savemat(mat_path, mat)
    print(f"  >> Results saved: {mat_path}")


# ============================================================================
# Run One Example (entry point for each E{i}_test.py)
# ============================================================================

def run_example(cfg):
    """Full 20-run evaluation loop for a single example."""
    idx = cfg['idx']
    name = cfg['name']
    mn = cfg['metric_name']
    nc = cfg['net_cfg']

    print("=" * 70)
    print(f"  {name}")
    print(f"  d={cfg['d']}, s={cfg['num_lf']+1}, "
          f"N_LF={cfg['ns']}, N_HF={cfg['nhf']}, "
          f"h={nc['hidden']}, ~{nc['n_params_est']:,} params")
    print(f"  epochs: meta={nc['meta_rounds']}, expert={nc['expert_epochs']}, "
          f"hf={nc['hf_epochs']}, stage2={nc['stage2_epochs']}")
    print(f"  metric: {mn}")
    print(f"  {NUM_REPEATS} runs | Test: {N_TEST} HF | Device: {DEVICE}")
    print("=" * 70)

    results_list = []
    loss_hists = []

    for rep in range(NUM_REPEATS):
        seed = SEED_BASE + rep * 100
        t0 = time.time()
        print(f"  Run {rep+1:2d}/{NUM_REPEATS} (seed={seed})", end=' ... ', flush=True)

        datasets_np, Xt, yt = cfg['setup'](seed)
        res = train_and_evaluate(cfg, datasets_np, Xt, yt, seed)

        results_list.append(res)
        loss_hists.append(res['loss_hist'])

        print(f"{mn}={res['error']:.6f}  ({time.time()-t0:.1f}s)")

    # Save
    save_results_mat(cfg, results_list)
    save_loss_mat(cfg, loss_hists)

    # Summary
    errs = [r['error'] for r in results_list]
    print(f"\n  {mn}: {np.mean(errs):.6f} +/- {np.std(errs):.6f}")

    akeys = sorted(set().union(*(r['alpha'].keys() for r in results_list)))
    if akeys:
        for k in akeys:
            vs = [r['alpha'].get(k, np.nan) for r in results_list]
            print(f"  alpha_{k[1:]}: {np.mean(vs):.4f} +/- {np.std(vs):.4f}")
    else:
        print("  (s=2, single LF - no routing gates)")

    qa = np.array([r['q_means'] for r in results_list])
    for t in range(qa.shape[1]):
        print(f"  q_{t+1}: {np.mean(qa[:,t]):.4f} +/- {np.std(qa[:,t]):.4f}")

    print(f"\n  Files: results/E{idx}_results.mat, results/E{idx}_loss.mat")
    return results_list
