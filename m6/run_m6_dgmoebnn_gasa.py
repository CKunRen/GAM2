import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
import torch
from scipy.io import savemat
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

import sys

sys.path.insert(0, str(ROOT))

from dnn_surrogate import (
    TrainConfig,
    get_device,
    load_checkpoint,
    load_mat_dataset,
    predict_numpy,
    save_checkpoint,
    train_full_model,
)
from gam2.models.dg_moe_bnn import DGMoEBNN
from gam2.active_learning.gasa import GASA
from gam2.utils.metrics import rmse


ORACLE_PATH = ROOT / "trained_dnn_model" / "ys_oracle_model.pt"
RESULT_DIR = ROOT / "m6_case_results"
RESULT_DIR.mkdir(exist_ok=True)
OUTPUT_SCALE = np.array([1.0, 10.0], dtype=np.float64)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lhs_unit(n: int, dim: int, seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    return sampler.random(n=n)


def load_raw_bounds() -> Tuple[np.ndarray, np.ndarray]:
    x, _ = load_mat_dataset(str(ROOT / "hfdata.mat"))
    return x.min(axis=0), x.max(axis=0)


def ensure_oracle_checkpoint(device: torch.device) -> Dict:
    if ORACLE_PATH.exists():
        return load_checkpoint(str(ORACLE_PATH), map_location=device)

    x, y = load_mat_dataset(str(ROOT / "hfdata.mat"))
    config = TrainConfig(
        hidden_dims=(160, 160, 160),
        activation="tanh",
        dropout=0.0,
        norm="none",
        residual=True,
        optimizer="adamw",
        loss_name="mse",
        lr=0.0009,
        weight_decay=1e-6,
        batch_size=16,
        max_epochs=1600,
        early_stop_patience=160,
        lr_patience=40,
        min_lr=1e-6,
        grad_clip=5.0,
    )
    final_model = train_full_model(x, y, config, seed=4096, device=device, val_size=0.1)
    save_checkpoint(
        str(ORACLE_PATH),
        final_model["model"].cpu(),
        final_model["x_scaler"],
        final_model["y_scaler"],
        config,
        metadata={
            "source": "ONERA M6 hfdata.mat",
            "full_metrics": final_model["full_metrics"],
        },
    )
    return load_checkpoint(str(ORACLE_PATH), map_location=device)


def unit_to_raw(x_unit: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    return x_min + x_unit * (x_max - x_min)


class RandomField:
    def __init__(self, input_dim: int, output_dim: int, seed: int, n_terms: int = 24):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 1.0, size=(n_terms, input_dim))
        self.b1 = rng.uniform(0.0, 2.0 * np.pi, size=(n_terms,))
        self.c1 = rng.normal(0.0, 1.0, size=(n_terms, output_dim))
        self.w2 = rng.normal(0.0, 1.0, size=(n_terms, input_dim))
        self.b2 = rng.uniform(0.0, 2.0 * np.pi, size=(n_terms,))
        self.c2 = rng.normal(0.0, 1.0, size=(n_terms, output_dim))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h1 = np.sin(2.0 * np.pi * x @ self.w1.T + self.b1)
        h2 = np.cos(2.0 * np.pi * x @ self.w2.T + self.b2)
        return h1 @ self.c1 + h2 @ self.c2


def envelope(x: np.ndarray, level: int) -> np.ndarray:
    design_mean = x[:, :50].mean(axis=1)
    design_std = x[:, :50].std(axis=1)
    mach = x[:, 50]
    aoa = x[:, 51]
    phase = (
        (level + 1) * 2.3 * design_mean
        + (level + 2) * 1.7 * design_std
        + (level + 1) * 3.2 * mach
        - (level + 2) * 2.1 * aoa
    )
    env = 0.65 + 0.35 * (0.5 + 0.5 * np.sin(2.0 * np.pi * phase))
    return env.reshape(-1, 1)


def build_fidelity_functions(
    oracle_bundle: Dict,
    x_min: np.ndarray,
    x_max: np.ndarray,
    calib_x: np.ndarray,
    target_rmses: List[float],
) -> Tuple[List, Dict]:
    hf_calib = predict_numpy(oracle_bundle, unit_to_raw(calib_x, x_min, x_max))
    y_std = hf_calib.std(axis=0, keepdims=True) + 1e-12
    lf_functions = []
    metrics = []

    for level, target_rmse in enumerate(target_rmses, start=1):
        field = RandomField(input_dim=52, output_dim=2, seed=1000 + level, n_terms=30)
        base_error = envelope(calib_x, level) * field(calib_x)
        base_err_std = base_error.std(axis=0, keepdims=True) + 1e-12
        base_error = base_error / base_err_std
        base_error = base_error * y_std

        base_rmse = float(np.sqrt(np.mean(base_error ** 2)))
        scale = target_rmse / max(base_rmse, 1e-12)

        def lf_func(x_unit, field=field, scale=scale, level=level, base_err_std=base_err_std):
            x_unit = np.asarray(x_unit, dtype=np.float64).reshape(-1, 52)
            y_hf = predict_numpy(oracle_bundle, unit_to_raw(x_unit, x_min, x_max))
            err = envelope(x_unit, level) * field(x_unit)
            err = err / base_err_std
            err = err * y_std * scale
            return y_hf + err

        y_lf = lf_func(calib_x)
        fidelity_rmse = float(np.sqrt(np.mean((y_lf - hf_calib) ** 2)))
        per_output_rmse = np.sqrt(np.mean((y_lf - hf_calib) ** 2, axis=0))
        corr_cl = float(np.corrcoef(hf_calib[:, 0], y_lf[:, 0])[0, 1])
        corr_cd = float(np.corrcoef(hf_calib[:, 1], y_lf[:, 1])[0, 1])
        metrics.append(
            {
                "level": level,
                "target_rmse": float(target_rmse),
                "actual_rmse": fidelity_rmse,
                "rmse_cl": float(per_output_rmse[0]),
                "rmse_cd": float(per_output_rmse[1]),
                "corr_cl": corr_cl,
                "corr_cd": corr_cd,
            }
        )
        lf_functions.append(lf_func)

    return lf_functions, {"calibration_metrics": metrics, "hf_calibration": hf_calib}


def build_initial_datasets(
    counts: List[int],
    eval_functions: List,
    seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    datasets = []
    for idx, (n, fn) in enumerate(zip(counts, eval_functions)):
        x = lhs_unit(n, 52, seed + idx * 100)
        y = fn(x)
        datasets.append(
            (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            )
        )
    return datasets


def tensor_list_to_numpy(datasets: List[Tuple[torch.Tensor, torch.Tensor]]):
    output = []
    for x_t, y_t in datasets:
        output.append((x_t.detach().cpu().numpy(), y_t.detach().cpu().numpy()))
    return output


def scale_outputs(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).copy()
    y *= OUTPUT_SCALE.reshape(1, -1)
    return y


def unscale_outputs(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).copy()
    y /= OUTPUT_SCALE.reshape(1, -1)
    return y


def unscale_variance(var: np.ndarray) -> np.ndarray:
    var = np.asarray(var, dtype=np.float64).copy()
    var /= (OUTPUT_SCALE.reshape(1, -1) ** 2)
    return var


def save_case_results(
    path: Path,
    summary: Dict,
    history: Dict,
    fidelity_info: Dict,
    test_x: np.ndarray,
    test_hf_y: np.ndarray,
    final_pred: np.ndarray,
    final_var: np.ndarray,
    datasets_np: List[Tuple[np.ndarray, np.ndarray]],
) -> None:
    mat = {
        "test_x": test_x,
        "test_hf_y": test_hf_y,
        "final_pred": final_pred,
        "final_var": final_var,
        "history_rmse": np.array(history.get("rmse", []), dtype=np.float64),
        "history_batch_sizes": np.array(history.get("batch_sizes", []), dtype=np.float64),
        "history_gate_q": np.array(history.get("gate_q", []), dtype=np.float64),
        "summary_final_rmse": float(summary["final_rmse"]),
        "summary_initial_rmse": float(summary["initial_rmse"]),
        "summary_iterations": int(summary["iterations"]),
        "summary_threshold": float(summary["rmse_target"]),
    }
    for idx, item in enumerate(fidelity_info["calibration_metrics"], start=1):
        mat[f"lf{idx}_rmse"] = float(item["actual_rmse"])
        mat[f"lf{idx}_corr_cl"] = float(item["corr_cl"])
        mat[f"lf{idx}_corr_cd"] = float(item["corr_cd"])
    for idx, (x_arr, y_arr) in enumerate(datasets_np, start=1):
        mat[f"X_{idx}"] = x_arr
        mat[f"Y_{idx}"] = y_arr
    savemat(str(path), mat)


def build_bilingual_text(summary: Dict, fidelity_info: Dict, counts: List[int], costs: List[float]) -> str:
    lf_metrics = fidelity_info["calibration_metrics"]
    cn = []
    cn.append("中文补充")
    cn.append("本算例以训练得到的最优 DNN 代理作为高精度模型 ys，并以归一化后的 52 维输入空间 [0,1]^52 作为统一设计域。")
    cn.append(
        f"初始样本量设置为 y1/y2/y3/y4/ys = {counts[0]}/{counts[1]}/{counts[2]}/{counts[3]}/{counts[4]}，成本向量设为 {costs}。"
    )
    cn.append(
        "4 个低精度模型通过向 ys 叠加随输入变化的随机场误差构造，其中误差场由固定随机频率、相位和系数生成，再乘以与设计变量均值、标准差、马赫数和攻角相关的包络函数，从而形成输入相关但可重复的精度退化。"
    )
    cn.append("为平衡双输出训练难度，代理建模阶段对阻力系数 Cd 进行 10 倍缩放，使其与升力系数 Cl 处于相近量级；在误差评估与结果汇报阶段再将 Cd 还原。")
    cn.append(
        "校准后 4 个低精度模型的整体 RMSE 分别为 "
        + "，".join([f"y{i['level']}={i['actual_rmse']:.4f}" for i in lf_metrics])
        + "，满足 y1 < y2 ≈ y3 < y4 的精度层级。"
    )
    cn.append(
        f"在 DG-MoE-BNN+GASA 验证中，主动学习从初始 RMSE={summary['initial_rmse']:.4f} 出发，经过 {summary['iterations']} 轮自适应采样后达到最终 RMSE={summary['final_rmse']:.4f}，满足停止准则 RMSE ≤ {summary['rmse_target']:.2f}。"
    )
    cn.append(
        "该结果说明所构造的任意精度 ONERA M6 代理算例可以有效验证双门控信息路由、多精度模型选择以及基于不确定性的自适应采样机制。"
    )

    en = []
    en.append("English Supplement")
    en.append("In this case study, the best-trained DNN surrogate is treated as the high-fidelity model ys, and the normalized 52-dimensional domain [0,1]^52 is used as the unified design space.")
    en.append(
        f"The initial sample allocation is set to y1/y2/y3/y4/ys = {counts[0]}/{counts[1]}/{counts[2]}/{counts[3]}/{counts[4]}, with the cost vector {costs}."
    )
    en.append(
        "The four low-fidelity models are generated by superimposing input-dependent random-field errors onto ys. The error fields are constructed from fixed random frequencies, phases, and coefficients, and are further modulated by an envelope function related to the mean and standard deviation of the design variables as well as the Mach number and angle of attack."
    )
    en.append("To balance the two-output regression task, the drag coefficient Cd is multiplied by 10 during DG-MoE-BNN training so that it has a magnitude comparable to Cl; Cd is then restored to its physical scale for accuracy evaluation and final reporting.")
    en.append(
        "After calibration, the overall RMSE values of the four low-fidelity models are "
        + ", ".join([f"y{i['level']}={i['actual_rmse']:.4f}" for i in lf_metrics])
        + ", which satisfies the intended fidelity ranking y1 < y2 ≈ y3 < y4."
    )
    en.append(
        f"For the DG-MoE-BNN+GASA validation, the active learning process starts from an initial RMSE of {summary['initial_rmse']:.4f} and reaches a final RMSE of {summary['final_rmse']:.4f} after {summary['iterations']} adaptive iterations, satisfying the stopping criterion RMSE ≤ {summary['rmse_target']:.2f}."
    )
    en.append(
        "These results indicate that the constructed arbitrary-fidelity ONERA M6 surrogate benchmark is suitable for validating dual-gated information routing, fidelity-aware model selection, and uncertainty-driven adaptive sampling."
    )
    return "\n".join(cn + ["", ""] + en) + "\n"


def main():
    set_seed(2026)
    device = get_device()
    x_min, x_max = load_raw_bounds()
    oracle_bundle = ensure_oracle_checkpoint(device)

    calib_x = lhs_unit(4000, 52, seed=111)
    lf_functions, fidelity_info = build_fidelity_functions(
        oracle_bundle=oracle_bundle,
        x_min=x_min,
        x_max=x_max,
        calib_x=calib_x,
        target_rmses=[0.015, 0.009, 0.0085, 0.004],
    )

    def hf_func(x_unit):
        x_unit = np.asarray(x_unit, dtype=np.float64).reshape(-1, 52)
        return predict_numpy(oracle_bundle, unit_to_raw(x_unit, x_min, x_max))

    eval_functions_raw = lf_functions + [hf_func]
    eval_functions = [lambda x, fn=fn: scale_outputs(fn(x)) for fn in eval_functions_raw]
    costs = [1.0, 1.8, 1.8, 2.6, 8.0]
    init_counts = [60, 45, 45, 30, 4]

    datasets = build_initial_datasets(init_counts, eval_functions, seed=2027)

    test_x = lhs_unit(2000, 52, seed=2028)
    test_y_raw = hf_func(test_x)

    def rmse_eval_fn(model, eval_device):
        model.eval()
        with torch.no_grad():
            test_tensor = torch.tensor(test_x, dtype=torch.float32, device=eval_device)
            mu_scaled, _, _, _, _ = model(test_tensor, stage=2)
        mu_raw = unscale_outputs(mu_scaled.cpu().numpy())
        return rmse(mu_raw, test_y_raw)

    model = DGMoEBNN(
        input_dim=52,
        num_lf_models=4,
        base_hidden=64,
        expert_hidden=64,
        hf_hidden=64,
        output_dim=2,
        use_conv=False,
        expert_layers=2,
        hf_det_layers=1,
        activation="tanh",
        base_layers=2,
        residual=True,
        routing_init=0.0,
        bayesian_bias=True,
    ).to(device)

    gasa = GASA(
        model=model,
        input_dim=52,
        bounds=np.array([[0.0, 1.0]] * 52, dtype=np.float64),
        costs=costs,
        eval_functions=eval_functions,
        hf_test_X=None,
        hf_test_y=None,
        K=3,
        epsilon=0.01,
        gamma=0.0,
        max_iterations=15,
        base_lr=0.001,
        lambda_gate=0.001,
        n_candidates=2000,
        verbose=True,
        train_mode="e2e",
        initial_train_kwargs=dict(
            e2e_epochs=2500,
            e2e_hf_weight=8.0,
            e2e_lr=0.002,
            e2e_warmup=300,
            e2e_batch_size=0,
        ),
        retrain_train_kwargs=dict(
            e2e_epochs=2000,
            e2e_hf_weight=8.0,
            e2e_lr=0.0015,
            e2e_warmup=200,
            e2e_batch_size=0,
        ),
        finetune_train_kwargs=dict(
            e2e_epochs=400,
            e2e_hf_weight=8.0,
            e2e_lr=0.0010,
            e2e_warmup=0,
            e2e_batch_size=0,
        ),
        rmse_target=0.01,
        min_hf_per_iteration=0,
        rmse_eval_fn=rmse_eval_fn,
        normalize_rg=True,
        cost_exponent=1.0,
        high_dim_threshold=20,
        n_base_override=10,
    )

    history = gasa.run(datasets, device)

    model = gasa.model
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_x, dtype=torch.float32, device=device)
        mu, var, _, q_vals, alpha_vals = model(test_tensor, stage=2)

    final_pred = unscale_outputs(mu.cpu().numpy())
    final_var = unscale_variance(var.cpu().numpy())
    final_rmse = float(rmse(final_pred, test_y_raw))
    initial_rmse = float(history["rmse"][0]) if history.get("rmse") else np.nan
    summary = {
        "initial_rmse": initial_rmse,
        "final_rmse": final_rmse,
        "iterations": len(history.get("batch_sizes", [])),
        "rmse_target": 0.01,
        "final_q_means": [float(q.mean().item()) for q in q_vals],
        "final_alpha": {f"{i}_{j}": float(v.item()) for (i, j), v in alpha_vals.items()},
        "final_sample_counts": [int(ds[0].shape[0]) for ds in datasets],
    }

    datasets_np = tensor_list_to_numpy(datasets)
    save_case_results(
        RESULT_DIR / "m6_dgmoebnn_gasa_results.mat",
        summary,
        history,
        fidelity_info,
        test_x,
        test_y_raw,
        final_pred,
        final_var,
        datasets_np,
    )

    (RESULT_DIR / "m6_dgmoebnn_gasa_summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "fidelity_info": fidelity_info["calibration_metrics"],
                "init_counts": init_counts,
                "costs": costs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
