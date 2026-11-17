import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from scipy.io import savemat


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from dnn_surrogate import get_device, load_checkpoint, load_mat_dataset
from m6_thickness import compute_m6_max_thickness


MODEL_PATH = ROOT / "trained_dnn_model" / "ys_oracle_model.pt"
RESULT_DIR = ROOT / "m6_case_results"
RESULT_DIR.mkdir(exist_ok=True)

CD0 = 0.01809
CD_SIGMA_WEIGHT = 1.0
MA_MEAN = 0.8395
MA_DELTA = 0.02
AOA_MEAN = 3.06
AOA_DELTA = 0.2
N_MC = 1000
SEED = 20260329
POP_SIZE = 80
N_GEN = 60
BATCH_SIZE = 16384


class OraclePredictor:
    def __init__(self, model_path: Path, device: torch.device):
        self.bundle = load_checkpoint(str(model_path), map_location=device)
        self.device = device
        self.model = self.bundle["model"].to(device)
        self.model.eval()
        self.x_mean = self.bundle["x_mean"].astype(np.float32)
        self.x_scale = self.bundle["x_scale"].astype(np.float32)
        self.y_mean = self.bundle["y_mean"].astype(np.float32)
        self.y_scale = self.bundle["y_scale"].astype(np.float32)

    def predict(self, x: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        outputs = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                stop = min(start + batch_size, x.shape[0])
                xb = (x[start:stop] - self.x_mean) / self.x_scale
                xt = torch.tensor(xb, dtype=torch.float32, device=self.device)
                y_scaled = self.model(xt).cpu().numpy()
                outputs.append(y_scaled * self.y_scale + self.y_mean)
        return np.vstack(outputs)


def robust_stats_from_predictions(pred: np.ndarray, n_design: int, n_mc: int) -> Dict[str, np.ndarray]:
    pred = pred.reshape(n_design, n_mc, 2)
    cl = pred[:, :, 0]
    cd = pred[:, :, 1]
    mu_cl = cl.mean(axis=1)
    mu_cd = cd.mean(axis=1)
    var_cl = cl.var(axis=1)
    var_cd = cd.var(axis=1)
    sigma_cl = np.sqrt(var_cl)
    sigma_cd = np.sqrt(var_cd)
    robust_lift = mu_cl - 3.0 * sigma_cl
    robust_drag = mu_cd + CD_SIGMA_WEIGHT * sigma_cd
    return {
        "mu_cl": mu_cl,
        "mu_cd": mu_cd,
        "var_cl": var_cl,
        "var_cd": var_cd,
        "sigma_cl": sigma_cl,
        "sigma_cd": sigma_cd,
        "robust_lift": robust_lift,
        "robust_drag": robust_drag,
    }


class M6RobustOptimizationProblem(Problem):
    def __init__(
        self,
        predictor: OraclePredictor,
        design_lb: np.ndarray,
        design_ub: np.ndarray,
        ma_samples: np.ndarray,
        aoa_samples: np.ndarray,
        baseline_tmax: float,
    ):
        super().__init__(
            n_var=50,
            n_obj=1,
            n_ieq_constr=2,
            xl=design_lb,
            xu=design_ub,
        )
        self.predictor = predictor
        self.ma_samples = ma_samples.astype(np.float32)
        self.aoa_samples = aoa_samples.astype(np.float32)
        self.n_mc = ma_samples.size
        self.baseline_tmax = float(baseline_tmax)

    def evaluate_designs(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        n_design = X.shape[0]
        design_rep = np.repeat(X, self.n_mc, axis=0)
        ma_rep = np.tile(self.ma_samples, n_design).reshape(-1, 1)
        aoa_rep = np.tile(self.aoa_samples, n_design).reshape(-1, 1)
        full_x = np.hstack([design_rep, ma_rep, aoa_rep])
        pred = self.predictor.predict(full_x)
        stats = robust_stats_from_predictions(pred, n_design, self.n_mc)

        thickness = np.array(
            [compute_m6_max_thickness(x)["overall_max_thickness"] for x in X],
            dtype=np.float64,
        )
        stats["max_thickness"] = thickness
        stats["g_cd"] = stats["robust_drag"] - CD0
        stats["g_t"] = self.baseline_tmax - thickness
        return stats

    def _evaluate(self, X: np.ndarray, out: Dict, *args, **kwargs):
        stats = self.evaluate_designs(X)
        out["F"] = (-stats["robust_lift"]).reshape(-1, 1)
        out["G"] = np.column_stack([stats["g_cd"], stats["g_t"]])


def make_uncertainty_samples(n_mc: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ma = rng.uniform(MA_MEAN - MA_DELTA, MA_MEAN + MA_DELTA, size=n_mc)
    aoa = rng.uniform(AOA_MEAN - AOA_DELTA, AOA_MEAN + AOA_DELTA, size=n_mc)
    return ma, aoa


def build_initial_population(problem: M6RobustOptimizationProblem, design_db: np.ndarray, pop_size: int) -> np.ndarray:
    stats = problem.evaluate_designs(design_db)
    feasible = (stats["g_cd"] <= 0.0) & (stats["g_t"] <= 0.0)
    violation = np.maximum(stats["g_cd"], 0.0) + np.maximum(stats["g_t"], 0.0)
    score = stats["robust_lift"]

    feasible_idx = np.where(feasible)[0]
    infeasible_idx = np.where(~feasible)[0]
    feasible_sorted = feasible_idx[np.argsort(score[feasible_idx])[::-1]]
    infeasible_sorted = infeasible_idx[np.argsort(violation[infeasible_idx])]

    ordered = np.concatenate([feasible_sorted, infeasible_sorted], axis=0)
    seeds = design_db[ordered[: min(pop_size, ordered.size)]]

    if seeds.shape[0] < pop_size:
        rng = np.random.default_rng(SEED + 7)
        rand = rng.uniform(problem.xl, problem.xu, size=(pop_size - seeds.shape[0], problem.n_var))
        seeds = np.vstack([seeds, rand])

    return seeds


def main():
    device = get_device()
    predictor = OraclePredictor(MODEL_PATH, device)
    x_all, _ = load_mat_dataset(str(ROOT / "hfdata.mat"))
    design_db = x_all[:, :50].astype(np.float64)
    design_lb = design_db.min(axis=0)
    design_ub = design_db.max(axis=0)

    baseline_tmax = compute_m6_max_thickness(np.zeros(50, dtype=np.float64))["overall_max_thickness"]
    ma_samples, aoa_samples = make_uncertainty_samples(N_MC, SEED)

    problem = M6RobustOptimizationProblem(
        predictor=predictor,
        design_lb=design_lb,
        design_ub=design_ub,
        ma_samples=ma_samples,
        aoa_samples=aoa_samples,
        baseline_tmax=baseline_tmax,
    )

    sampling = build_initial_population(problem, design_db, POP_SIZE)
    algorithm = GA(pop_size=POP_SIZE, sampling=sampling, eliminate_duplicates=True)
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", N_GEN),
        seed=SEED,
        verbose=True,
    )

    best_x = np.asarray(result.X, dtype=np.float64).reshape(1, -1)
    best_stats = problem.evaluate_designs(best_x)

    baseline_design = np.zeros((1, 50), dtype=np.float64)
    baseline_stats = problem.evaluate_designs(baseline_design)

    mat_data = {
        "best_design": best_x.reshape(-1, 1),
        "best_objective": np.array([[best_stats["robust_lift"][0]]], dtype=np.float64),
        "best_mu_cl": np.array([[best_stats["mu_cl"][0]]], dtype=np.float64),
        "best_sigma_cl": np.array([[best_stats["sigma_cl"][0]]], dtype=np.float64),
        "best_var_cl": np.array([[best_stats["var_cl"][0]]], dtype=np.float64),
        "best_mu_cd": np.array([[best_stats["mu_cd"][0]]], dtype=np.float64),
        "best_sigma_cd": np.array([[best_stats["sigma_cd"][0]]], dtype=np.float64),
        "best_var_cd": np.array([[best_stats["var_cd"][0]]], dtype=np.float64),
        "best_robust_drag": np.array([[best_stats["robust_drag"][0]]], dtype=np.float64),
        "best_max_thickness": np.array([[best_stats["max_thickness"][0]]], dtype=np.float64),
        "baseline_design": baseline_design.reshape(-1, 1),
        "baseline_mu_cl": np.array([[baseline_stats["mu_cl"][0]]], dtype=np.float64),
        "baseline_sigma_cl": np.array([[baseline_stats["sigma_cl"][0]]], dtype=np.float64),
        "baseline_var_cl": np.array([[baseline_stats["var_cl"][0]]], dtype=np.float64),
        "baseline_mu_cd": np.array([[baseline_stats["mu_cd"][0]]], dtype=np.float64),
        "baseline_sigma_cd": np.array([[baseline_stats["sigma_cd"][0]]], dtype=np.float64),
        "baseline_var_cd": np.array([[baseline_stats["var_cd"][0]]], dtype=np.float64),
        "baseline_robust_lift": np.array([[baseline_stats["robust_lift"][0]]], dtype=np.float64),
        "baseline_robust_drag": np.array([[baseline_stats["robust_drag"][0]]], dtype=np.float64),
        "baseline_max_thickness": np.array([[baseline_tmax]], dtype=np.float64),
        "design_lower_bounds": design_lb.reshape(-1, 1),
        "design_upper_bounds": design_ub.reshape(-1, 1),
        "ma_samples": ma_samples.reshape(-1, 1),
        "aoa_samples": aoa_samples.reshape(-1, 1),
        "cd0": np.array([[CD0]], dtype=np.float64),
        "n_mc": np.array([[N_MC]], dtype=np.int32),
        "population_size": np.array([[POP_SIZE]], dtype=np.int32),
        "n_generations": np.array([[N_GEN]], dtype=np.int32),
        "seed": np.array([[SEED]], dtype=np.int32),
    }

    mat_path = RESULT_DIR / "m6_robust_optimization_result.mat"
    savemat(str(mat_path), mat_data)

    summary = {
        "mat_path": str(mat_path),
        "best_objective": float(best_stats["robust_lift"][0]),
        "best_mu_cl": float(best_stats["mu_cl"][0]),
        "best_sigma_cl": float(best_stats["sigma_cl"][0]),
        "best_var_cl": float(best_stats["var_cl"][0]),
        "best_mu_cd": float(best_stats["mu_cd"][0]),
        "best_sigma_cd": float(best_stats["sigma_cd"][0]),
        "best_var_cd": float(best_stats["var_cd"][0]),
        "best_robust_drag": float(best_stats["robust_drag"][0]),
        "best_max_thickness": float(best_stats["max_thickness"][0]),
        "baseline_max_thickness": float(baseline_tmax),
        "constraint_cd": float(best_stats["g_cd"][0]),
        "constraint_t": float(best_stats["g_t"][0]),
    }
    (RESULT_DIR / "m6_robust_optimization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
