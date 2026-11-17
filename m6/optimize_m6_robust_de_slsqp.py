import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import Bounds, NonlinearConstraint, differential_evolution, minimize


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import optimize_m6_robust as base


RESULT_DIR = base.RESULT_DIR
RESULT_DIR.mkdir(exist_ok=True)

SEED = base.SEED + 101
DE_POP = 96
DE_MAXITER = 70
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7
SLSQP_MAXITER = 300


class RobustEvaluator:
    def __init__(self, problem: base.M6RobustOptimizationProblem):
        self.problem = problem
        self.cache: Dict[tuple, Dict[str, float]] = {}

    def _key(self, x: np.ndarray) -> tuple:
        return tuple(np.round(np.asarray(x, dtype=np.float64), 10))

    def evaluate(self, x: np.ndarray) -> Dict[str, float]:
        key = self._key(x)
        if key in self.cache:
            return self.cache[key]

        stats = self.problem.evaluate_designs(np.asarray(x, dtype=np.float64).reshape(1, -1))
        out = {
            "robust_lift": float(stats["robust_lift"][0]),
            "robust_drag": float(stats["robust_drag"][0]),
            "mu_cl": float(stats["mu_cl"][0]),
            "sigma_cl": float(stats["sigma_cl"][0]),
            "var_cl": float(stats["var_cl"][0]),
            "mu_cd": float(stats["mu_cd"][0]),
            "sigma_cd": float(stats["sigma_cd"][0]),
            "var_cd": float(stats["var_cd"][0]),
            "max_thickness": float(stats["max_thickness"][0]),
            "g_cd": float(stats["g_cd"][0]),
            "g_t": float(stats["g_t"][0]),
        }
        self.cache[key] = out
        return out

    def objective(self, x: np.ndarray) -> float:
        return -self.evaluate(x)["robust_lift"]

    def violation_objective(self, x: np.ndarray) -> float:
        s = self.evaluate(x)
        v_cd = max(s["g_cd"], 0.0)
        v_t = max(s["g_t"], 0.0)
        return v_cd * v_cd + v_t * v_t

    def constraints_vector(self, x: np.ndarray) -> np.ndarray:
        s = self.evaluate(x)
        return np.array([s["g_cd"], s["g_t"]], dtype=np.float64)

    def constraint_cd(self, x: np.ndarray) -> float:
        return base.CD0 - self.evaluate(x)["robust_drag"]

    def constraint_t(self, x: np.ndarray) -> float:
        return self.evaluate(x)["max_thickness"] - self.problem.baseline_tmax


def load_ga_best_design() -> Optional[np.ndarray]:
    ga_mat = RESULT_DIR / "m6_robust_optimization_result.mat"
    if not ga_mat.exists():
        return None
    data = loadmat(str(ga_mat))
    x = np.asarray(data.get("best_design"), dtype=np.float64).reshape(-1)
    if x.size != 50:
        return None
    return x


def build_de_initial_population(
    design_db: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    size: int,
    ga_best_x: Optional[np.ndarray],
) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    init = np.empty((size, design_db.shape[1]), dtype=np.float64)
    cursor = 0

    if ga_best_x is not None:
        init[cursor] = ga_best_x
        cursor += 1

        n_local = min(max(size // 4, 8), size - cursor)
        if n_local > 0:
            sigma = 0.08 * (ub - lb)
            local = ga_best_x.reshape(1, -1) + rng.normal(0.0, sigma, size=(n_local, ga_best_x.size))
            init[cursor : cursor + n_local] = np.clip(local, lb, ub)
            cursor += n_local

    take = min(design_db.shape[0], max(size // 2, 1), size - cursor)
    idx = rng.choice(design_db.shape[0], size=take, replace=False)
    init[cursor : cursor + take] = design_db[idx]
    cursor += take

    if cursor < size:
        init[cursor:] = rng.uniform(lb, ub, size=(size - cursor, design_db.shape[1]))
    return init


def pick_best_candidate(
    evaluator: RobustEvaluator,
    candidates: List[Tuple[str, np.ndarray]],
) -> Tuple[str, np.ndarray, Dict[str, float]]:
    ranked: List[Tuple[float, float, str, np.ndarray, Dict[str, float]]] = []
    for name, x in candidates:
        stats = evaluator.evaluate(x)
        violation = max(stats["g_cd"], 0.0) + max(stats["g_t"], 0.0)
        ranked.append((violation, -stats["robust_lift"], name, x, stats))
    ranked.sort(key=lambda item: (item[0], item[1]))
    _, _, name, x, stats = ranked[0]
    return name, x, stats


def run_slsqp_refinement(
    evaluator: RobustEvaluator,
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


def main():
    device = base.get_device()
    predictor = base.OraclePredictor(base.MODEL_PATH, device)
    x_all, _ = base.load_mat_dataset(str(base.ROOT / "hfdata.mat"))
    design_db = x_all[:, :50].astype(np.float64)
    design_lb = design_db.min(axis=0)
    design_ub = design_db.max(axis=0)
    baseline_tmax = base.compute_m6_max_thickness(np.zeros(50, dtype=np.float64))["overall_max_thickness"]
    ma_samples, aoa_samples = base.make_uncertainty_samples(base.N_MC, base.SEED)

    problem = base.M6RobustOptimizationProblem(
        predictor=predictor,
        design_lb=design_lb,
        design_ub=design_ub,
        ma_samples=ma_samples,
        aoa_samples=aoa_samples,
        baseline_tmax=baseline_tmax,
    )
    evaluator = RobustEvaluator(problem)

    bounds = Bounds(design_lb, design_ub)
    ga_best_x = load_ga_best_design()
    init = build_de_initial_population(design_db, design_lb, design_ub, DE_POP, ga_best_x)
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
        maxiter=max(30, DE_MAXITER // 2),
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
    if ga_best_x is not None and stage2_init.shape[0] > 1:
        stage2_init[1] = ga_best_x

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
        ("baseline", np.zeros(50, dtype=np.float64)),
    ]
    if ga_best_x is not None:
        start_candidates.append(("ga_seed", ga_best_x))

    best_start_name, best_start_x, best_start_stats = pick_best_candidate(evaluator, start_candidates)
    slsqp_results = run_slsqp_refinement(evaluator, bounds, start_candidates)

    de_stats = evaluator.evaluate(de_result.x)
    final_candidates: List[Tuple[str, np.ndarray]] = [(best_start_name, best_start_x)]
    for name, res in slsqp_results:
        final_candidates.append((f"slsqp_{name}", np.asarray(res.x, dtype=np.float64)))
    final_name, final_x, final_stats = pick_best_candidate(evaluator, final_candidates)
    final_stats = evaluator.evaluate(final_x)

    ga_summary_path = RESULT_DIR / "m6_robust_optimization_summary.json"
    ga_summary = None
    if ga_summary_path.exists():
        ga_summary = json.loads(ga_summary_path.read_text(encoding="utf-8"))

    mat_path = RESULT_DIR / "m6_robust_optimization_de_slsqp_result.mat"
    savemat(
        str(mat_path),
        {
            "best_design": np.asarray(final_x, dtype=np.float64).reshape(-1, 1),
            "de_design": np.asarray(de_result.x, dtype=np.float64).reshape(-1, 1),
            "best_objective": np.array([[final_stats["robust_lift"]]], dtype=np.float64),
            "best_mu_cl": np.array([[final_stats["mu_cl"]]], dtype=np.float64),
            "best_sigma_cl": np.array([[final_stats["sigma_cl"]]], dtype=np.float64),
            "best_var_cl": np.array([[final_stats["var_cl"]]], dtype=np.float64),
            "best_mu_cd": np.array([[final_stats["mu_cd"]]], dtype=np.float64),
            "best_sigma_cd": np.array([[final_stats["sigma_cd"]]], dtype=np.float64),
            "best_var_cd": np.array([[final_stats["var_cd"]]], dtype=np.float64),
            "best_robust_drag": np.array([[final_stats["robust_drag"]]], dtype=np.float64),
            "best_max_thickness": np.array([[final_stats["max_thickness"]]], dtype=np.float64),
            "best_constraint_cd": np.array([[final_stats["g_cd"]]], dtype=np.float64),
            "best_constraint_t": np.array([[final_stats["g_t"]]], dtype=np.float64),
            "de_stage1_design": np.asarray(de_feas_result.x, dtype=np.float64).reshape(-1, 1),
            "de_stage1_violation": np.array([[evaluator.violation_objective(de_feas_result.x)]], dtype=np.float64),
            "de_objective": np.array([[de_stats["robust_lift"]]], dtype=np.float64),
            "de_constraint_cd": np.array([[de_stats["g_cd"]]], dtype=np.float64),
            "de_constraint_t": np.array([[de_stats["g_t"]]], dtype=np.float64),
            "baseline_max_thickness": np.array([[baseline_tmax]], dtype=np.float64),
            "design_lower_bounds": design_lb.reshape(-1, 1),
            "design_upper_bounds": design_ub.reshape(-1, 1),
            "ma_samples": ma_samples.reshape(-1, 1),
            "aoa_samples": aoa_samples.reshape(-1, 1),
            "cd0": np.array([[base.CD0]], dtype=np.float64),
            "n_mc": np.array([[base.N_MC]], dtype=np.int32),
            "de_population": np.array([[DE_POP]], dtype=np.int32),
            "de_maxiter": np.array([[DE_MAXITER]], dtype=np.int32),
            "seed": np.array([[SEED]], dtype=np.int32),
        },
    )

    summary = {
        "mat_path": str(mat_path),
        "ga_seed_used": ga_best_x is not None,
        "de_stage1_violation_objective": float(evaluator.violation_objective(de_feas_result.x)),
        "de_stage1_constraint_cd": float(evaluator.evaluate(de_feas_result.x)["g_cd"]),
        "de_stage1_constraint_t": float(evaluator.evaluate(de_feas_result.x)["g_t"]),
        "de_success": bool(de_result.success),
        "de_message": str(de_result.message),
        "best_source": final_name,
        "best_start_source": best_start_name,
        "de_objective": float(de_stats["robust_lift"]),
        "de_mu_cl": float(de_stats["mu_cl"]),
        "de_sigma_cl": float(de_stats["sigma_cl"]),
        "de_mu_cd": float(de_stats["mu_cd"]),
        "de_sigma_cd": float(de_stats["sigma_cd"]),
        "de_robust_drag": float(de_stats["robust_drag"]),
        "de_max_thickness": float(de_stats["max_thickness"]),
        "de_constraint_cd": float(de_stats["g_cd"]),
        "de_constraint_t": float(de_stats["g_t"]),
        "slsqp_runs": [
            {
                "start": name,
                "success": bool(res.success),
                "message": str(res.message),
                "objective": float(evaluator.evaluate(res.x)["robust_lift"]),
                "constraint_cd": float(evaluator.evaluate(res.x)["g_cd"]),
                "constraint_t": float(evaluator.evaluate(res.x)["g_t"]),
            }
            for name, res in slsqp_results
        ],
        "best_start_objective": float(best_start_stats["robust_lift"]),
        "best_start_constraint_cd": float(best_start_stats["g_cd"]),
        "best_start_constraint_t": float(best_start_stats["g_t"]),
        "best_objective": float(final_stats["robust_lift"]),
        "best_mu_cl": float(final_stats["mu_cl"]),
        "best_sigma_cl": float(final_stats["sigma_cl"]),
        "best_var_cl": float(final_stats["var_cl"]),
        "best_mu_cd": float(final_stats["mu_cd"]),
        "best_sigma_cd": float(final_stats["sigma_cd"]),
        "best_var_cd": float(final_stats["var_cd"]),
        "best_robust_drag": float(final_stats["robust_drag"]),
        "best_max_thickness": float(final_stats["max_thickness"]),
        "constraint_cd": float(final_stats["g_cd"]),
        "constraint_t": float(final_stats["g_t"]),
    }
    if ga_summary is not None:
        summary["ga_best_objective"] = ga_summary.get("best_objective")
        summary["ga_best_robust_drag"] = ga_summary.get("best_robust_drag")
        summary["ga_best_max_thickness"] = ga_summary.get("best_max_thickness")
    summary_path = RESULT_DIR / "m6_robust_optimization_de_slsqp_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
