# ONERA M6 case — model and optimization pipeline

This folder contains the code of the engineering case: the high-fidelity
surrogate of the RANS solver, the deterministic and robust optimization
problems, and the GALSA sampling run. **No data are distributed with it.**

## Files

| File | Purpose |
|---|---|
| `dnn_surrogate.py` | HF surrogate: DNN definition, training, cross-validation, checkpoint I/O |
| `train_hf_dnn.py` | trains the HF surrogate on `hfdata.mat` |
| `load_hf_dnn.py` | loads a trained checkpoint and predicts |
| `m6_thickness.py` | wing geometry and the maximum-thickness constraint |
| `optimize_m6_robust.py` | robust-optimization problem (mean lift / mean drag), Monte Carlo over (Ma, α) |
| `optimize_m6_robust_de_slsqp.py` | robust optimization: differential evolution + SLSQP |
| `optimize_m6_deterministic.py` | deterministic optimization |
| `run_m6_dgmoebnn_gasa.py` | builds the five-fidelity case and runs GALSA on DG-MoEs-BNN |

## Required data (not included)

All scripts read `hfdata.mat` from this folder. The file must contain one 2-D
array with at least 54 columns:

```
columns  0:52   design and operating variables
                50 geometry variables (5 spanwise sections x 10 control points)
                + Mach number + angle of attack
columns 52:54   Cl, Cd
```

The 600 RANS samples used in the paper are not distributed. They — and the
mesh files — are available from the corresponding author on reasonable request,
as stated in the Data availability statement of the paper.

## Running

Install the package first (from the repository root):

```bash
pip install -e .
python m6/train_hf_dnn.py            # trains the HF surrogate
python m6/optimize_m6_deterministic.py
python m6/optimize_m6_robust_de_slsqp.py
python m6/run_m6_dgmoebnn_gasa.py
```

Each script writes its results to `m6_case_results/` inside this folder.
