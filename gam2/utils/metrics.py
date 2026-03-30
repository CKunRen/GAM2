"""
Accuracy Metrics: RMSE, MaxAE, IR (Eq.28-29)
"""

import numpy as np
import torch


def rmse(y_pred, y_true):
    """
    Root Mean Squared Error (Eq.29).

    RMSE = sqrt(1/N * Σ(ŷ_s(x_i) - y_s(x_i))²)
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def maxae(y_pred, y_true):
    """Maximum Absolute Error."""
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()
    return float(np.max(np.abs(y_pred - y_true)))


def improvement_rate(rmse_prev: float, rmse_curr: float) -> float:
    """
    Improvement Rate (Eq.28).

    IR_i = (RMSE_{i-K} - RMSE_i) / RMSE_{i-K}
    """
    if rmse_prev <= 0:
        return 0.0
    return (rmse_prev - rmse_curr) / rmse_prev
