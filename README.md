# GAM2: Gated Adaptive Multi-Fidelity Metamodeling

Official implementation of the paper:

> **Automatic arbitrary-fidelity metamodeling via gated information routing and Bayesian active learning**

## Overview

GAM2 is a multi-fidelity surrogate modeling framework that handles **arbitrary fidelity configurations** (hierarchical, non-hierarchical, and hybrid) **without prior knowledge** of fidelity relationships. It consists of two core components:

- **DG-MoE-BNN** (Dual-Gated Mixture-of-Experts Bayesian Neural Network): A neural network architecture with learnable routing gates and quality gates that automatically discover inter-fidelity information flow and suppress unreliable sources.
- **GASA** (Gate-Guided Active Sampling Algorithm): An active learning algorithm that leverages learned gate values for cost-efficient multi-fidelity sample allocation.

## Repository Structure

```
GAM2-public/
├── gam2/
│   ├── models/
│   │   ├── base_module.py        # Shared feature extractor
│   │   ├── expert_module.py      # Source-specific expert modules
│   │   ├── gating.py             # Dual-level gating (routing + quality)
│   │   ├── hf_module.py          # HF prediction with Bayesian output
│   │   ├── bayesian_linear.py    # Bayesian linear layer
│   │   └── dg_moe_bnn.py        # Full DG-MoE-BNN assembly
│   ├── training/
│   │   ├── losses.py             # Multi-task loss functions
│   │   ├── trainer.py            # End-to-end / multi-stage trainer
│   │   └── meta_learning.py      # Reptile meta-learning for base module
│   ├── active_learning/
│   │   ├── gasa.py               # GASA main loop (Algorithm 1)
│   │   ├── model_selection.py    # GEAF fidelity selection
│   │   ├── sample_selection.py   # Variance-guided sample selection
│   │   └── adaptive_batch.py     # Adaptive batch sizing
│   └── utils/
│       ├── sampling.py           # LHS / Optimized LHS
│       └── metrics.py            # RMSE, MaxAE, Improvement Rate
├── examples/
│   ├── demo.py                   # Quick demo (1D–50D synthetic functions)
│   ├── E1_Forrester.py           # Example 1: 1D Forrester (hierarchical)
│   └── test_common.py            # Shared evaluation utilities
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python >= 3.8 and PyTorch >= 2.0.

## Quick Start

```python
import torch
from gam2.models.dg_moe_bnn import DGMoEBNN
from gam2.training.trainer import Trainer

# Define a 3-fidelity problem (2 LF + 1 HF), input_dim=1
model = DGMoEBNN(
    input_dim=1,
    num_lf=2,
    hidden_dim=20,
    activation='tanh',
    residual=True,
    quality_mode='softmax_input',
    routing_init=-2.0,
    bayesian_bias=True,
)

# Prepare data: list of (X_t, Y_t) for each LF, plus (X_hf, Y_hf)
datasets_lf = [(X_lf1, Y_lf1), (X_lf2, Y_lf2)]
dataset_hf = (X_hf, Y_hf)

# Train end-to-end
trainer = Trainer(model, train_mode='e2e', e2e_epochs=15000, e2e_hf_weight=10.0)
trainer.train(datasets_lf, dataset_hf)

# Predict with uncertainty
mu, var = model.predict(X_test)
```

## Run Examples

```bash
cd examples
python demo.py          # 5 synthetic benchmark demos
python E1_Forrester.py  # Reproduces Example 1 from the paper
```

## Citation

If you find this code useful, please cite:

```bibtex
@article{GAM2_2026,
  title={Arbitrary unknown fidelity metamodeling via dual-gated mixture-of-experts and Bayesian active learning},
  author={},
  journal={},
  year={2026}
}
```

## License

This project is released under the MIT License.
