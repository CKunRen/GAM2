import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import Bounds, NonlinearConstraint, differential_evolution, minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import optimize_m6_robust as base


RESULT_DIR = base.RESULT_DIR
RESULT_DIR.mkdir(exist_ok=True)

SEED = base.SEED + 202
DE_POP = 96
DE_MAXITER = 70
DE_FEAS_MAXITER = 30
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7
SLSQP_MAXITER = 300


class DeterministicEvaluator:
    def __init__(
        self,
        predictor: base.OraclePredictor,
        baseline_tmax: float,
    ):
        self.predictor = predictor
        self.baseline_tmax = float(baseline_tmax)
        self.cache: Dict[Tuple[float, ...], Dict[str, float]] = {}

    def _key(self, x: np.ndarray) -> Tuple[float, ...]:
        return tuple(np.round(np.asarray(x, dtype=np.float64), 10))

    def evaluate(self, x: np.ndarray) -> Dict[str, float]:
        key = self._key(x)
        if key in self.cache:
            return self.cache[key]

        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        full_x = np.hstack(
            [
                x,
                np.full((1, 1), base.MA_MEAN, dtype=np.float64),
                np.full((1, 1), base.AOA_MEAN, dtype=np.float64),
            ]
        )
        pred = self.predictor.predict(full_x)[0]
        thickness = base.compute_m6_max_thickness(x.reshape(-1))["overall_max_thickness"]
        out = {
            "cl": float(pred[0]),
            "cd": float(pred[1]),
            "max_thickness": float(thickness),
            "g_cd": float(pred[1] - base.CD0),
            "g_t": float(self.baseline_tmax - thickness),
        }
        self.cache[key] = out
        return out

    def evaluate_batch(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        full_x = np.hstack(
            [
                X,
                np.full((X.shape[0], 1), base.MA_MEAN, dtype=np.float64),
                np.full((X.shape[0], 1), base.AOA_MEAN, dtype=np.float64),
            ]
        )
        pred = self.predictor.predict(full_x)
        thickness = np.array(
            [base.compute_m6_max_thickness(x)["overall_max_thickness"] for x in X],
            dtype=np.float64,
        )
        return {
            "cl": pred[:, 0].astype(np.float64),
            "cd": pred[:, 1].astype(np.float64),
            "max_thickness": thickness,
            "g_cd": pred[:, 1].astype(np.float64) - base.CD0,
            "g_t": self.baseline_tmax - thickness,
        }

    def objective(self, x: np.ndarray) -> float:
        return -self.evaluate(x)["cl"]

    def violation_objective(self, x: np.ndarray) -> float:
        s = self.evaluate(x)
        v_cd = max(s["g_cd"], 0.0)
        v_t = max(s["g_t"], 0.0)
        return v_cd * v_cd + v_t * v_t

    def constraints_vector(self, x: np.ndarray) -> np.ndarray:
        s = self.evaluate(x)
        return np.array([s["g_cd"], s["g_t"]], dtype=np.float64)

    def constraint_cd(self, x: np.ndarray) -> float:
        return base.CD0 - self.evaluate(x)["cd"]

    def constraint_t(self, x: np.ndarray) -> float:
        return self.evaluate(x)["max_thickness"] - self.baseline_tmax


def load_ro_best_design() -> Optional[np.ndarray]:
    candidates = [
        RESULT_DIR / "m6_robust_optimization_de_slsqp_result.mat",
        RESULT_DIR / "m6_robust_optimization_result.mat",
    ]
    for path in candidates:
        if path.exists():
            data = loadmat(str(path))
            x = np.asarray(data.get("best_design"), dtype=np.float64).reshape(-1)
            if x.size == 50:
                return x
    return None


def build_initial_population(
    evaluator: DeterministicEvaluator,
    design_db: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    size: int,
    ro_best_x: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    rng = np.random.default_rng(SEED)
    stats = evaluator.evaluate_batch(design_db)
    feasible = (stats["g_cd"] <= 0.0) & (stats["g_t"] <= 0.0)
    feasible_idx = np.where(feasible)[0]
    feasible_sorted = feasible_idx[np.argsort(stats["cl"][feasible_idx])[::-1]] if feasible_idx.size > 0 else np.array([], dtype=np.int64)
    best_dataset_seed = design_db[feasible_sorted[0]].copy() if feasible_sorted.size > 0 else None

    init = np.empty((size, design_db.shape[1]), dtype=np.float64)
    cursor = 0

    if ro_best_x is not None:
        init[cursor] = ro_best_x
        cursor += 1

    if best_dataset_seed is not None and cursor < size:
        init[cursor] = best_dataset_seed
        cursor += 1

    if ro_best_x is not None:
        n_local = min(max(size // 5, 8), size - cursor)
        if n_local > 0:
            sigma = 0.08 * (ub - lb)
            local = ro_best_x.reshape(1, -1) + rng.normal(0.0, sigma, size=(n_local, ro_best_x.size))
            init[cursor : cursor + n_local] = np.clip(local, lb, ub)
            cursor += n_local

    take_feasible = min(feasible_sorted.size, max(size // 2, 1), size - cursor)
    if take_feasible > 0:
        init[cursor : cursor + take_feasible] = design_db[feasible_sorted[:take_feasible]]
        cursor += take_feasible

    if cursor < size:
        remaining = size - cursor
        pick = min(remaining // 2, design_db.shape[0])
        if pick > 0:
            idx = rng.choice(design_db.shape[0], size=pick, replace=False)
            init[cursor : cursor + pick] = design_db[idx]
            cursor += pick

    if cursor < size:
        init[cursor:] = rng.uniform(lb, ub, size=(size - cursor, design_db.shape[1]))

    return init, best_dataset_seed


def pick_best_candidate(
    evaluator: DeterministicEvaluator,
    candidates: List[Tuple[str, np.ndarray]],
) -> Tuple[str, np.ndarray, Dict[str, float]]:
    ranked: List[Tuple[float, float, str, np.ndarray, Dict[str, float]]] = []
    for name, x in candidates:
        stats = evaluator.evaluate(x)
        violation = max(stats["g_cd"], 0.0) + max(stats["g_t"], 0.0)
        ranked.append((violation, -stats["cl"], name, np.asarray(x, dtype=np.float64), stats))
    ranked.sort(key=lambda item: (item[0], item[1]))
    _, _, name, x, stats = ranked[0]
    return name, x, stats


def run_slsqp_refinement(
    evaluator: DeterministicEvaluator,
    bounds: Bounds,
    starts: List[Tuple[str, np.ndarray]],
) -> List[Tuple[str, object]]:
    results: List[Tuple[str, object]] = []
    for name, x0 in starts:
        res = minimize(
            evaluator.objective,
            x0=np.asarray(x0, dtype=np.float64),
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {"type": "ineq", "fun": evaluator.constraint_cd},
                {"type": "ineq", "fun": evaluator.constraint_t},
            ],
            options={"maxiter": SLSQP_MAXITER, "ftol": 1.0e-9, "disp": False},
        )
        results.append((name, res))
    return results


def evaluate_mcs_design(
    predictor: base.OraclePredictor,
    design: np.ndarray,
    ma_samples: np.ndarray,
    aoa_samples: np.ndarray,
) -> Dict[str, np.ndarray]:
    design = np.asarray(design, dtype=np.float64).reshape(1, -1)
    n_mc = ma_samples.size
    full_x = np.hstack(
        [
            np.repeat(design, n_mc, axis=0),
            ma_samples.reshape(-1, 1),
            aoa_samples.reshape(-1, 1),
        ]
    )
    pred = predictor.predict(full_x)
    stats = base.robust_stats_from_predictions(pred, 1, n_mc)
    return {
        "cl_samples": pred[:, 0].astype(np.float64),
        "cd_samples": pred[:, 1].astype(np.float64),
        "mu_cl": np.array([stats["mu_cl"][0]], dtype=np.float64),
        "sigma_cl": np.array([stats["sigma_cl"][0]], dtype=np.float64),
        "var_cl": np.array([stats["var_cl"][0]], dtype=np.float64),
        "mu_cd": np.array([stats["mu_cd"][0]], dtype=np.float64),
        "sigma_cd": np.array([stats["sigma_cd"][0]], dtype=np.float64),
        "var_cd": np.array([stats["var_cd"][0]], dtype=np.float64),
        "robust_lift": np.array([stats["robust_lift"][0]], dtype=np.float64),
        "robust_drag": np.array([stats["robust_drag"][0]], dtype=np.float64),
    }


def make_boxplot_figure(
    baseline_mcs: Dict[str, np.ndarray],
    do_mcs: Dict[str, np.ndarray],
    ro_mcs: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=220)
    labels = ["Baseline", "DO", "RO"]

    axes[0].boxplot(
        [baseline_mcs["cl_samples"], do_mcs["cl_samples"], ro_mcs["cl_samples"]],
        labels=labels,
        showmeans=True,
    )
    axes[0].set_ylabel("Cl")
    axes[0].set_title("Lift Coefficient under 1000 MCS")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].boxplot(
        [baseline_mcs["cd_samples"], do_mcs["cd_samples"], ro_mcs["cd_samples"]],
        labels=labels,
        showmeans=True,
    )
    axes[1].set_ylabel("Cd")
    axes[1].set_title("Drag Coefficient under 1000 MCS")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    device = base.get_device()
    predictor = base.OraclePredictor(base.MODEL_PATH, device)
    x_all, _ = base.load_mat_dataset(str(ROOT / "hfdata.mat"))
    design_db = x_all[:, :50].astype(np.float64)
    design_lb = design_db.min(axis=0)
    design_ub = design_db.max(axis=0)
    baseline_x = np.zeros(50, dtype=np.float64)
    baseline_tmax = base.compute_m6_max_thickness(baseline_x)["overall_max_thickness"]

    evaluator = DeterministicEvaluator(
        predictor=predictor,
        baseline_tmax=baseline_tmax,
    )

    ro_best_x = load_ro_best_design()
    init, best_dataset_seed = build_initial_population(
        evaluator=evaluator,
        design_db=design_db,
        lb=design_lb,
        ub=design_ub,
        size=DE_POP,
        ro_best_x=ro_best_x,
    )

    bounds = Bounds(design_lb, design_ub)
    nonlinear_constraints = (
        NonlinearConstraint(
            evaluator.constraints_vector,
            lb=np.array([-np.inf, -np.inf], dtype=np.float64),
            ub=np.array([0.0, 0.0], dtype=np.float64),
        ),
    )

    de_feas_result = differential_evolution(
        evaluator.violation_objective,
        bounds=list(zip(design_lb, design_ub)),
        strategy="best1bin",
        maxiter=DE_FEAS_MAXITER,
        popsize=1,
        tol=1.0e-4,
        mutation=DE_MUTATION,
        recombination=DE_RECOMBINATION,
        seed=SEED,
        polish=False,
        disp=False,
        init=init,
        updating="deferred",
    )

    stage2_init = init.copy()
    stage2_init[0] = de_feas_result.x
    if ro_best_x is not None and stage2_init.shape[0] > 1:
        stage2_init[1] = ro_best_x
    if best_dataset_seed is not None and stage2_init.shape[0] > 2:
        stage2_init[2] = best_dataset_seed

    de_result = differential_evolution(
        evaluator.objective,
        bounds=list(zip(design_lb, design_ub)),
        strategy="best1bin",
        maxiter=DE_MAXITER,
        popsize=1,
        tol=1.0e-4,
        mutation=DE_MUTATION,
        recombination=DE_RECOMBINATION,
        seed=SEED + 1,
        polish=False,
        disp=True,
        init=stage2_init,
        constraints=nonlinear_constraints,
        updating="deferred",
    )

    start_candidates: List[Tuple[str, np.ndarray]] = [
        ("de_stage1", de_feas_result.x),
        ("de_stage2", de_result.x),
        ("baseline", baseline_x),
    ]
    if ro_best_x is not None:
        start_candidates.append(("ro_seed", ro_best_x))
    if best_dataset_seed is not None:
        start_candidates.append(("dataset_seed", best_dataset_seed))

    best_start_name, best_start_x, best_start_stats = pick_best_candidate(evaluator, start_candidates)
    slsqp_results = run_slsqp_refinement(evaluator, bounds, start_candidates)

    final_candidates: List[Tuple[str, np.ndarray]] = [(best_start_name, best_start_x)]
    for name, res in slsqp_results:
        final_candidates.append((f"slsqp_{name}", np.asarray(res.x, dtype=np.float64)))
    final_name, best_x, best_stats = pick_best_candidate(evaluator, final_candidates)

    if ro_best_x is None:
        raise FileNotFoundError("RO best design was not found in m6_case_results.")

    ro_nominal = evaluator.evaluate(ro_best_x)
    baseline_nominal = evaluator.evaluate(baseline_x)

    mcs_ma, mcs_aoa = base.make_uncertainty_samples(base.N_MC, SEED + 77)
    baseline_mcs = evaluate_mcs_design(predictor, baseline_x, mcs_ma, mcs_aoa)
    do_mcs = evaluate_mcs_design(predictor, best_x, mcs_ma, mcs_aoa)
    ro_mcs = evaluate_mcs_design(predictor, ro_best_x, mcs_ma, mcs_aoa)

    fig_path = RESULT_DIR / "m6_baseline_do_ro_boxplot.png"
    make_boxplot_figure(baseline_mcs, do_mcs, ro_mcs, fig_path)

    mat_path = RESULT_DIR / "m6_deterministic_optimization_result.mat"
    savemat(
        str(mat_path),
        {
            "best_design": np.asarray(best_x, dtype=np.float64).reshape(-1, 1),
            "best_nominal_cl": np.array([[best_stats["cl"]]], dtype=np.float64),
            "best_nominal_cd": np.array([[best_stats["cd"]]], dtype=np.float64),
            "best_max_thickness": np.array([[best_stats["max_thickness"]]], dtype=np.float64),
            "best_constraint_cd": np.array([[best_stats["g_cd"]]], dtype=np.float64),
            "best_constraint_t": np.array([[best_stats["g_t"]]], dtype=np.float64),
            "ro_design": np.asarray(ro_best_x, dtype=np.float64).reshape(-1, 1),
            "ro_nominal_cl": np.array([[ro_nominal["cl"]]], dtype=np.float64),
            "ro_nominal_cd": np.array([[ro_nominal["cd"]]], dtype=np.float64),
            "ro_max_thickness": np.array([[ro_nominal["max_thickness"]]], dtype=np.float64),
            "baseline_design": baseline_x.reshape(-1, 1),
            "baseline_nominal_cl": np.array([[baseline_nominal["cl"]]], dtype=np.float64),
            "baseline_nominal_cd": np.array([[baseline_nominal["cd"]]], dtype=np.float64),
            "baseline_max_thickness": np.array([[baseline_nominal["max_thickness"]]], dtype=np.float64),
            "baseline_cl_samples": baseline_mcs["cl_samples"].reshape(-1, 1),
            "baseline_cd_samples": baseline_mcs["cd_samples"].reshape(-1, 1),
            "baseline_mu_cl": baseline_mcs["mu_cl"].reshape(1, 1),
            "baseline_sigma_cl": baseline_mcs["sigma_cl"].reshape(1, 1),
            "baseline_var_cl": baseline_mcs["var_cl"].reshape(1, 1),
            "baseline_mu_cd": baseline_mcs["mu_cd"].reshape(1, 1),
            "baseline_sigma_cd": baseline_mcs["sigma_cd"].reshape(1, 1),
            "baseline_var_cd": baseline_mcs["var_cd"].reshape(1, 1),
            "do_cl_samples": do_mcs["cl_samples"].reshape(-1, 1),
            "do_cd_samples": do_mcs["cd_samples"].reshape(-1, 1),
            "do_mu_cl": do_mcs["mu_cl"].reshape(1, 1),
            "do_sigma_cl": do_mcs["sigma_cl"].reshape(1, 1),
            "do_var_cl": do_mcs["var_cl"].reshape(1, 1),
            "do_mu_cd": do_mcs["mu_cd"].reshape(1, 1),
            "do_sigma_cd": do_mcs["sigma_cd"].reshape(1, 1),
            "do_var_cd": do_mcs["var_cd"].reshape(1, 1),
            "ro_cl_samples": ro_mcs["cl_samples"].reshape(-1, 1),
            "ro_cd_samples": ro_mcs["cd_samples"].reshape(-1, 1),
            "ro_mu_cl": ro_mcs["mu_cl"].reshape(1, 1),
            "ro_sigma_cl": ro_mcs["sigma_cl"].reshape(1, 1),
            "ro_var_cl": ro_mcs["var_cl"].reshape(1, 1),
            "ro_mu_cd": ro_mcs["mu_cd"].reshape(1, 1),
            "ro_sigma_cd": ro_mcs["sigma_cd"].reshape(1, 1),
            "ro_var_cd": ro_mcs["var_cd"].reshape(1, 1),
            "mcs_ma_samples": mcs_ma.reshape(-1, 1),
            "mcs_aoa_samples": mcs_aoa.reshape(-1, 1),
            "cd0": np.array([[base.CD0]], dtype=np.float64),
            "ma_mean": np.array([[base.MA_MEAN]], dtype=np.float64),
            "aoa_mean": np.array([[base.AOA_MEAN]], dtype=np.float64),
            "n_mc": np.array([[base.N_MC]], dtype=np.int32),
            "seed": np.array([[SEED]], dtype=np.int32),
        },
    )

    summary = {
        "mat_path": str(mat_path),
        "boxplot_path": str(fig_path),
        "best_source": final_name,
        "best_start_source": best_start_name,
        "de_feasible_violation": float(evaluator.violation_objective(de_feas_result.x)),
        "de_success": bool(de_result.success),
        "de_message": str(de_result.message),
        "best_nominal_cl": float(best_stats["cl"]),
        "best_nominal_cd": float(best_stats["cd"]),
        "best_max_thickness": float(best_stats["max_thickness"]),
        "best_constraint_cd": float(best_stats["g_cd"]),
        "best_constraint_t": float(best_stats["g_t"]),
        "ro_nominal_cl": float(ro_nominal["cl"]),
        "ro_nominal_cd": float(ro_nominal["cd"]),
        "ro_max_thickness": float(ro_nominal["max_thickness"]),
        "baseline_nominal_cl": float(baseline_nominal["cl"]),
        "baseline_nominal_cd": float(baseline_nominal["cd"]),
        "baseline_max_thickness": float(baseline_nominal["max_thickness"]),
        "baseline_mcs": {
            "mu_cl": float(baseline_mcs["mu_cl"][0]),
            "sigma_cl": float(baseline_mcs["sigma_cl"][0]),
            "var_cl": float(baseline_mcs["var_cl"][0]),
            "mu_cd": float(baseline_mcs["mu_cd"][0]),
            "sigma_cd": float(baseline_mcs["sigma_cd"][0]),
            "var_cd": float(baseline_mcs["var_cd"][0]),
        },
        "do_mcs": {
            "mu_cl": float(do_mcs["mu_cl"][0]),
            "sigma_cl": float(do_mcs["sigma_cl"][0]),
            "var_cl": float(do_mcs["var_cl"][0]),
            "mu_cd": float(do_mcs["mu_cd"][0]),
            "sigma_cd": float(do_mcs["sigma_cd"][0]),
            "var_cd": float(do_mcs["var_cd"][0]),
        },
        "ro_mcs": {
            "mu_cl": float(ro_mcs["mu_cl"][0]),
            "sigma_cl": float(ro_mcs["sigma_cl"][0]),
            "var_cl": float(ro_mcs["var_cl"][0]),
            "mu_cd": float(ro_mcs["mu_cd"][0]),
            "sigma_cd": float(ro_mcs["sigma_cd"][0]),
            "var_cd": float(ro_mcs["var_cd"][0]),
        },
        "slsqp_runs": [
            {
                "start": name,
                "success": bool(res.success),
                "message": str(res.message),
                "nominal_cl": float(evaluator.evaluate(res.x)["cl"]),
                "nominal_cd": float(evaluator.evaluate(res.x)["cd"]),
                "constraint_cd": float(evaluator.evaluate(res.x)["g_cd"]),
                "constraint_t": float(evaluator.evaluate(res.x)["g_t"]),
            }
            for name, res in slsqp_results
        ],
    }
    summary_path = RESULT_DIR / "m6_deterministic_optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
