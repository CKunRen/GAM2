# GAM2 — Dual-Gated Mixture-of-Experts Bayesian Neural Network

Reference implementation of the network and training procedure described in

> *Unified Fidelity-Agnostic Multi-Fidelity Modeling via Dual-Gated
> Mixture-of-Experts and Bayesian Learning*

This release contains the model, the training procedure and the gate-guided
active-learning sampler, together with the scripts of the ONERA M6 case.
It contains **no benchmark data**: the library is meant to be imported, and
the M6 scripts expect the CFD data described in `m6/README.md`, which are
available from the corresponding author on request. A single reference
example, Example 3 of the paper, is provided in `examples/`; it generates
its own data from the published formulas and reproduces the reported
benchmark at the published configuration.

## What is included

| Component | Module | Equation |
|---|---|---|
| Base class module (shared feature extractor) | `gam2/models/base_module.py` | Eq. (1) |
| Expert modules (one per LF source) | `gam2/models/expert_module.py` | Eq. (2) |
| Inter-expert routing gates | `gam2/models/gating.py` | Eqs. (3)–(4) |
| Expert quality gates | `gam2/models/gating.py` | Eq. (5) |
| HF prediction module and fusion | `gam2/models/hf_module.py` | Eqs. (6)–(8) |
| Bayesian output layer (mean / variance / KL) | `gam2/models/bayesian_linear.py` | Eqs. (9)–(11) |
| Loss functions | `gam2/training/losses.py` | Eqs. (12)–(17) |
| End-to-end trainer | `gam2/training/trainer.py` | Section 2.1.3 |
| Gate-guided active learning (GALSA) | `gam2/active_learning/` | Algorithm 1 |
| ONERA M6 case (HF surrogate, DO/RO, GALSA run) | `m6/` | Section 4 |
| Reference examples (Example 3 and Example 4) | `examples/` | Section 3 |

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

## Model

```python
import torch
from gam2 import DGMoEBNN

model = DGMoEBNN(
    input_dim=10,          # d
    num_lf_models=2,       # s - 1
    output_dim=1,
)

x = torch.randn(8, 10)
mu, var, preds, q, alphas = model(x)
# mu, var : (8, 1)      predictive mean and variance, Eq. (10)-(11)
# preds   : list of s-1 LF predictions
# q       : list of s-1 quality gates, Eq. (5)
# alphas  : {(i, j): alpha_ij}, Eq. (3)
```

The architecture is parameterised entirely by constructor arguments —
`hidden_size`, `kernel_size`, `activation`, `expert_layers`, `hf_layers`,
`gate_hidden`, `routing_exchanges`, `residual`. The values used for each benchmark are
listed per example in Table 1 of the paper; the defaults above are only a
convenience and are not claimed to reproduce any reported number.

### Routing, Eq. (4)

`r_t = Σ_{i≠t} α_it h_i⁻` is a coupled system: the features `h_i⁻` themselves
depend on the routed inputs. The implementation solves it with
`routing_exchanges` Jacobi exchanges (default one): all expert features are
first computed with `r_t = 0`, and the routed inputs are then formed from those
features. One exchange is what the paper reports, and it exchanges information
in both directions for every ordered pair `(i, t)`. `stage=1` evaluates the
experts without routing.

### Fusion, Eq. (6)

The HF module receives the concatenation
`[q_1 y_1, …, q_{s-1} y_{s-1}; h_base]`, i.e. `h + s - 1` inputs for scalar
expert outputs (see `model.architecture_summary()`). With
`residual=True` (the default) its output is added to the fused LF
prediction, so the network represents the correction
`y_s = Σ_t q_t y_t + W_s^out f_HF(...)`; on Example 3 this halves the RMSE.

## Training

```python
from gam2 import GAM2Trainer

datasets = [
    (X_lf1, y_lf1),
    (X_lf2, y_lf2),
    (X_hf,  y_hf),          # the HF dataset is the last element
]
trainer = GAM2Trainer(model, base_lr=1e-3, lambda_gate=0.01, kl_weight=1.0)
history = trainer.train(datasets, torch.device('cpu'), mode='e2e',
                        e2e_epochs=1000, e2e_hf_weight=5.0)
```

The objective is the multi-task loss of Eq. (17): the mean squared error of
each expert, the HF loss of Eq. (14) (`MSE + KL/N`), and the gate-sparsity
penalty of Eq. (16).

## Tests

```bash
python tests/test_architecture.py
```

The checks verify the widths and activations of the configuration table
(`h + s - 1` fusion input, `2h` expert input, one gate per ordered pair), that
the quality gates form a softmax, that the predictive variance is non-negative,
and that an end-to-end training step runs.

## Citation

```bibtex
@article{gam2_2026,
  title   = {Unified Fidelity-Agnostic Multi-Fidelity Modeling via Dual-Gated
             Mixture-of-Experts and Bayesian Learning},
  author  = {Ren, Chengkun and Wang, Bomin and Xiong, Fenfen},
  journal = {Structural and Multidisciplinary Optimization},
  year    = {2026}
}
```

## License

MIT — see `LICENSE`.
