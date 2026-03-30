"""
E1: Forrester 1D — Hierarchical, s=3
=====================================
  LF1: 0.5*f_H + 10(x-0.5) - 5       N1 = 15
  MF2: 0.8*f_H + 5(x-0.5) - 2        N2 = 9
  HF:  (6x-2)^2 sin(12x-4)            Ns = 6

Metric: Relative L2 Error
Ref: Xiao et al. 2018  (target ≈ 0.0664)

Network: Tanh, input-dependent softmax quality gates, residual,
  e2e training. H=20, hf_det_layers=0 (simpler correction).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from test_common import (olhs_samples, relative_l2_error, calc_n_params,
                         run_example, N_TEST)

# ---- Test functions --------------------------------------------------------
def forrester_hf(x):
    return (6*x - 2)**2 * np.sin(12*x - 4)

def forrester_mf(x):
    return 0.8*forrester_hf(x) + 5*(x - 0.5) - 2

def forrester_lf(x):
    return 0.5*forrester_hf(x) + 10*(x - 0.5) - 5

# ---- Data setup (OLHS, regenerated per run) --------------------------------
def setup(seed):
    b = np.array([[0.0, 1.0]])
    X1 = olhs_samples(15, 1, b, seed).ravel()
    X2 = olhs_samples(9,  1, b, seed + 1000).ravel()
    X3 = olhs_samples(6,  1, b, seed + 2000).ravel()
    ds = [(X1, forrester_lf(X1)),
          (X2, forrester_mf(X2)),
          (X3, forrester_hf(X3))]
    Xt = olhs_samples(N_TEST, 1, b, seed + 9000).ravel()
    return ds, Xt, forrester_hf(Xt)

# ---- Config ----------------------------------------------------------------
D, K, H = 1, 2, 20

n_params = 2700  # approximate

CFG = dict(
    idx=1,
    name='E1 Forrester (H, 1D, s=3)',
    d=D, num_lf=K, ns=[15, 9], nhf=6,
    setup=setup,
    metric_fn=relative_l2_error, metric_name='Rel.Error',
    is_1d=True, x_range=[0.0, 1.0],
    hf_func=forrester_hf,
    lf_funcs={'y1_lf': forrester_lf, 'y2_mf': forrester_mf},
    net_cfg=dict(
        hidden=H,
        meta_rounds=200,
        expert_epochs=500,
        hf_epochs=500,
        stage2_epochs=50,
        # End-to-end training
        train_mode='e2e',
        e2e_epochs=25000,
        e2e_hf_weight=15.0,
        e2e_lr=0.003,
        # Common
        base_lr=0.001,
        lambda_gate=0.005,
        kl_weight=0.001,
        n_params_est=n_params,
        model_kwargs=dict(
            use_conv=False,
            base_layers=2,
            expert_layers=1,
            hf_det_layers=0,        # simpler HF correction
            activation='tanh',
            residual=True,
            quality_mode='softmax_input',
            routing_init=-2.0,
            bayesian_bias=True,
        ),
    ),
)

if __name__ == '__main__':
    run_example(CFG)
