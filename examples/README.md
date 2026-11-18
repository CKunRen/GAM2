# Examples

Two reference examples are included.

* **Example 3** — 1-D non-hierarchical benchmark, three fidelity levels. Small
  enough to run in a few minutes; exercises the MLP stem, two experts, the
  routing and quality gates, the concatenated fusion and the Bayesian output.
* **Example 4** — 32-D Dixon-Price benchmark, one low-fidelity source. Large
  (200 000 LF / 2 000 HF samples, as in Table 3) and takes about an hour; it
  exercises the convolutional stem and the single-expert path.

## Run

```bash
pip install -e .                       # from the repository root
python examples/e3_non_hierarchical_1d.py
```

Optional flags: `--epochs` (default 25 000), `--seed` (default 42),
`--lr` (default 3e-3), `--hf-weight` (default 15).

## Configuration

The script uses the configuration of the paper: Table 3 for the sample sizes and
Table 1 for the network.

| Item | Value |
|---|---|
| Input dimension | 1 |
| Fidelity levels | 3 (2 LF + HF) |
| Training samples | 5 / 5 / 4 |
| Hidden size | 32 |
| Base stem | MLP, 2 layers |
| Expert hidden layers | 1 |
| HF deterministic layers | 1 |
| Activation | tanh |
| Quality gate | 32 → 16 → 1, softmax across experts |
| λ_gate | 0.005 |
| Learning rate (end-to-end) | 3e-3 |
| HF loss weight | 15 |
| Epochs | 25 000 |
| HF residual skip (`residual=True`) | yes |

## Expected output

```
NRMSE = 0.0883   (paper: 0.0981)
```

Measured with seed 42 on CPU. The exact value depends on the seed; the paper
reports the mean over twenty seeds.

## Example 4

```bash
python examples/e4_dixon_price_32d.py            # full setup (~1 h)
python examples/e4_dixon_price_32d.py --n-lf 2000 --n-hf 200 --epochs 30
```

| Item | Value |
|---|---|
| Input dimension | 32 |
| Fidelity levels | 2 (1 LF + HF) |
| Training samples | 200 000 / 2 000 |
| Hidden size | 128 |
| Base stem | Conv1d, 1 → 16 → 32 → 64, kernel 1 |
| Expert hidden layers | 2 |
| HF deterministic layers | 2 |
| Activation | LeakyReLU |
| λ_gate | 0.01 |
| Learning rate (end-to-end) | 2e-3 |
| HF loss weight | 1 |
| Epochs | 5 000 |
| Mini-batch size | 4 096 |
| LR warm-up (epochs) | 1 000 |
| HF residual skip | yes |

Expected:

```
NRMSE = 0.00412   (paper: 0.0055)
```
