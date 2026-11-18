# -*- coding: utf-8 -*-
"""Example 4 of the paper: 32-D Dixon-Price benchmark (single LF source).

    HF : (x_1 - 1)^2 + sum_{i=2}^{32} (2 x_i^2 - x_{i-1})^2
    LF : 0.8 * HF - 0.4 * sum_{i=1}^{31} x_i x_{i+1} - 50
    x in [-3, 3]^32

There is a single low-fidelity source, so the routing gates are absent and the
quality gate has one entry (it acts as a scalar weight on the LF prediction).
Training samples follow Table 3 of the paper (N1 = 200 000, NHF = 2 000); the
network follows Table 1 (h = 128, convolutional stem with kernel 1, LeakyReLU,
two expert layers, two deterministic HF layers, lambda_gate = 0.01, HF residual
skip).

The reported metric is the mean squared error on the standardized HF response.

Run:  python examples/e4_dixon_price_32d.py [--n-lf 200000] [--n-hf 2000]
                                             [--epochs 5000] [--seed 42]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gam2 import DGMoEBNN, GAM2Trainer                 # noqa: E402
from gam2.utils.sampling import generate_olhs          # noqa: E402

DIM = 32
BOUNDS = np.array([[-3.0, 3.0]] * DIM)


def hf(x):
    y = (x[:, 0] - 1.0) ** 2
    for i in range(1, x.shape[1]):
        y = y + (2.0 * x[:, i] ** 2 - x[:, i - 1]) ** 2
    return y


def lf(x):
    cross = np.sum(x[:, :-1] * x[:, 1:], axis=1)
    return 0.8 * hf(x) - 0.4 * cross - 50.0


def sample(n, seed):
    return generate_olhs(n, DIM, BOUNDS, seed=seed, n_candidates=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-lf', type=int, default=200000)
    ap.add_argument('--n-hf', type=int, default=2000)
    ap.add_argument('--epochs', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--hf-weight', type=float, default=1.0)
    ap.add_argument('--batch-size', type=int, default=4096)
    ap.add_argument('--warmup', type=int, default=1000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_lf, x_hf = sample(args.n_lf, args.seed), sample(args.n_hf, args.seed + 1000)
    x_test = sample(10000, args.seed + 9000)
    y_lf, y_hf = lf(x_lf), hf(x_hf)

    all_y = np.concatenate([y_lf, y_hf])
    mean, std = all_y.mean(), all_y.std()

    def t(x, y):
        return (torch.tensor(x, dtype=torch.float32).to(device),
                torch.tensor((y - mean) / std, dtype=torch.float32)
                .unsqueeze(-1).to(device))

    datasets = [t(x_lf, y_lf), t(x_hf, y_hf)]

    model = DGMoEBNN(
        input_dim=DIM, num_lf_models=1, hidden_size=128, output_dim=1,
        use_conv=True, kernel_size=1, conv_padding=0, base_layers=1,
        expert_layers=2, hf_layers=2, activation='leaky_relu',
        gate_hidden=16, routing_exchanges=1, routing_init=-2.0,
        residual=True,
    ).to(device)
    print('architecture:', model.architecture_summary())

    trainer = GAM2Trainer(model, base_lr=1e-3, lambda_gate=0.01,
                          kl_weight=0.01)
    trainer.train(datasets, device, mode='e2e', e2e_epochs=args.epochs,
                  e2e_hf_weight=args.hf_weight, e2e_lr=args.lr,
                  e2e_batch_size=args.batch_size, e2e_warmup=args.warmup,
                  verbose=False)

    model.eval()
    mse = 0.0
    with torch.no_grad():
        for k in range(0, x_test.shape[0], 2000):
            xb = torch.tensor(x_test[k:k + 2000], dtype=torch.float32).to(device)
            mu, _, _, _, _ = model(xb, stage=2)
            mu = mu.cpu().numpy().ravel() * std + mean
            mse += float(np.sum((mu - hf(x_test[k:k + 2000])) ** 2))
    mse /= x_test.shape[0]
    rmse = float(np.sqrt(mse))
    nrmse = rmse / float(np.max(hf(x_test)) - np.min(hf(x_test)))
    print(f'RMSE  = {rmse:.2f}    (paper NRMSE 0.0055 -> RMSE 18.69)')
    print(f'NRMSE = {nrmse:.5f}  (paper: 0.0055)')
    print(f'MSE   = {mse:.2f}')


if __name__ == '__main__':
    main()
