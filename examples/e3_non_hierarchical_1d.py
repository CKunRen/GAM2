# -*- coding: utf-8 -*-
"""Example 3 of the paper: 1-D non-hierarchical benchmark.

Two LF models of comparable accuracy and different functional form are fused to
predict the HF response:

    LF1: (x - 0.5)(x - 4)(x - 9) / 20 + 2
    LF2: sin(x) + 0.2 x + 0.5
    HF : sin(x) + 0.2 x + (x - 5)^2 / 16 + 0.5
    x in [0, 10]

Training samples follow Table 3 of the paper (N1 = 5, N2 = 5, NHF = 4); the
network configuration follows Table 1 (h = 32, MLP stem, tanh, one expert hidden
layer, one deterministic HF layer, lambda_gate = 0.005).

Run:  python examples/e3_non_hierarchical_1d.py [--epochs 25000] [--seed 42]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gam2 import DGMoEBNN, GAM2Trainer                 # noqa: E402
from gam2.utils.sampling import generate_olhs          # noqa: E402


def hf(x):
    return np.sin(x) + 0.2 * x + (x - 5.0) ** 2 / 16.0 + 0.5


def lf1(x):
    return (x - 0.5) * (x - 4.0) * (x - 9.0) / 20.0 + 2.0


def lf2(x):
    return np.sin(x) + 0.2 * x + 0.5


def sample(n, seed):
    return generate_olhs(n, 1, np.array([[0.0, 10.0]]), seed=seed,
                         n_candidates=50).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=25000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--hf-weight', type=float, default=15.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x1, x2, x3 = sample(5, args.seed), sample(5, args.seed + 1000), \
        sample(4, args.seed + 2000)
    x_test = sample(1000, args.seed + 9000)

    y1, y2, y3 = lf1(x1), lf2(x2), hf(x3)
    all_y = np.concatenate([y1, y2, y3])
    mean, std = all_y.mean(), all_y.std()

    def t(x, y):
        return (torch.tensor(x, dtype=torch.float32).unsqueeze(-1).to(device),
                torch.tensor((y - mean) / std, dtype=torch.float32)
                .unsqueeze(-1).to(device))

    datasets = [t(x1, y1), t(x2, y2), t(x3, y3)]

    model = DGMoEBNN(
        input_dim=1, num_lf_models=2, hidden_size=32, output_dim=1,
        use_conv=False, base_layers=2, expert_layers=1, hf_layers=1,
        activation='tanh', gate_hidden=16, routing_exchanges=1,
        routing_init=-2.0, bayesian_bias=True, residual=True,
    ).to(device)
    print('architecture:', model.architecture_summary())

    trainer = GAM2Trainer(model, base_lr=1e-3, lambda_gate=0.005,
                          kl_weight=1e-3)
    trainer.train(datasets, device, mode='e2e', e2e_epochs=args.epochs,
                  e2e_hf_weight=args.hf_weight, e2e_lr=args.lr, verbose=False)

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1).to(device)
        mu, _, _, _, _ = model(xt, stage=2)
    pred = mu.cpu().numpy().ravel() * std + mean
    truth = hf(x_test)
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    print(f'RMSE = {rmse:.4f}   (paper: 0.3267 +/- 0.0547)')


if __name__ == '__main__':
    main()
