import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


ROOT = Path(__file__).resolve().parent
BASELINE_PATH = ROOT / "M6_1_Modify.dat"
SECTION_BASELINE_FILES = [ROOT / f"M6_{i}_Modify.dat" for i in range(1, 6)]
SECTION_SCALES = np.array([1.00, 0.89, 0.78, 0.67, 0.56], dtype=np.float64)
SECTION_Z = np.array([0.0, 299.075, 598.15, 897.225, 1196.3], dtype=np.float64)
FCT1 = np.log(0.5) / np.log(np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64))
FCT2 = np.array([3.0, 3.0, 3.0, 3.0, 1.0], dtype=np.float64)


def _shape_delta(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    basis = np.sin(np.pi * x[:, None] ** FCT1[None, :]) ** FCT2[None, :]
    return basis @ alpha


def _load_baseline(path: Path = BASELINE_PATH) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline coordinate file not found: {path}. "
            "The Matlab functions require M6_1_Modify.dat to reconstruct the airfoil sections."
        )
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected baseline data shape: {data.shape}")
    return data


def _split_surface(mat_crd: np.ndarray):
    num_up = mat_crd.shape[0] // 2
    x_up = mat_crd[:num_up, 0].copy()
    y_up = mat_crd[:num_up, 1].copy()
    x_low = mat_crd[num_up:, 0].copy()
    y_low = mat_crd[num_up:, 1].copy()
    return x_up, y_up, x_low, y_low


def _sorted_surface(x: np.ndarray, y: np.ndarray):
    order = np.argsort(x)
    return x[order], y[order]


def build_section(section_alpha: np.ndarray, section_idx: int, baseline_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
    if section_alpha.shape != (10,):
        raise ValueError(f"Each section must have 10 design variables, got {section_alpha.shape}")

    if baseline_path is None:
        baseline_path = SECTION_BASELINE_FILES[section_idx]
    mat_crd = _load_baseline(baseline_path)
    x_up, y_up, x_low, y_low = _split_surface(mat_crd)

    alpha_up = section_alpha[:5]
    alpha_low = section_alpha[5:10]

    y_up_new = y_up + _shape_delta(x_up, alpha_up)
    y_low_new = y_low + _shape_delta(x_low, alpha_low)
    y_up_new[0] = 0.0
    if len(y_low_new) >= 72 and len(y_up_new) >= 72:
        y_low_new[71] = y_up_new[71]

    scale = SECTION_SCALES[section_idx]
    z_coord = SECTION_Z[section_idx]

    x_up_scaled = x_up * scale
    y_up_scaled = y_up_new * scale
    x_low_scaled = x_low * scale
    y_low_scaled = y_low_new * scale

    x_up_sorted, y_up_sorted = _sorted_surface(x_up_scaled, y_up_scaled)
    x_low_sorted, y_low_sorted = _sorted_surface(x_low_scaled, y_low_scaled)

    x_common = np.unique(np.concatenate([x_up_sorted, x_low_sorted]))
    x_min = max(x_up_sorted.min(), x_low_sorted.min())
    x_max = min(x_up_sorted.max(), x_low_sorted.max())
    x_common = x_common[(x_common >= x_min) & (x_common <= x_max)]

    y_up_interp = np.interp(x_common, x_up_sorted, y_up_sorted)
    y_low_interp = np.interp(x_common, x_low_sorted, y_low_sorted)
    thickness = y_up_interp - y_low_interp

    return {
        "x_up": x_up_scaled,
        "y_up": y_up_scaled,
        "x_low": x_low_scaled,
        "y_low": y_low_scaled,
        "x_common": x_common,
        "thickness": thickness,
        "max_thickness": float(np.max(thickness)),
        "max_thickness_x": float(x_common[np.argmax(thickness)]),
        "z": float(z_coord),
    }


def compute_m6_max_thickness(design_vars: np.ndarray, baseline_path: Optional[Path] = None) -> Dict:
    design_vars = np.asarray(design_vars, dtype=np.float64).reshape(-1)
    if design_vars.size != 50:
        raise ValueError(f"Expected 50 design variables, got {design_vars.size}")

    sections: List[Dict] = []
    for i in range(5):
        section_alpha = design_vars[i * 10:(i + 1) * 10]
        section_path = None if baseline_path is None else baseline_path
        sections.append(build_section(section_alpha, i, section_path))

    section_tmax = np.array([sec["max_thickness"] for sec in sections], dtype=np.float64)
    idx = int(np.argmax(section_tmax))
    return {
        "design_vars": design_vars,
        "section_max_thickness": section_tmax,
        "overall_max_thickness": float(section_tmax[idx]),
        "overall_section_index": idx,
        "overall_section_name": f"section_{idx + 1}",
        "overall_x_location": float(sections[idx]["max_thickness_x"]),
        "sections": sections,
    }


def _load_design_vector(design_json: Optional[str]) -> np.ndarray:
    if design_json is None:
        return np.zeros(50, dtype=np.float64)
    path = Path(design_json)
    data = json.loads(path.read_text(encoding="utf-8"))
    arr = np.asarray(data, dtype=np.float64).reshape(-1)
    if arr.size != 50:
        raise ValueError(f"Expected 50 design variables in {path}, got {arr.size}")
    return arr


def main():
    parser = argparse.ArgumentParser(description="Compute ONERA M6 maximum thickness from 50 design variables.")
    parser.add_argument("--design-json", help="Path to a JSON file containing 50 design variables.")
    parser.add_argument("--baseline", default=str(BASELINE_PATH), help="Path to M6_1_Modify.dat")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    design_vars = _load_design_vector(args.design_json)
    zero_vars = np.zeros(50, dtype=np.float64)

    current = compute_m6_max_thickness(design_vars, baseline_path)
    baseline = compute_m6_max_thickness(zero_vars, baseline_path)

    print("Current design overall max thickness:", current["overall_max_thickness"])
    print("Current design controlling section:", current["overall_section_name"])
    print("Current design section tmax:", current["section_max_thickness"])
    print("Zero design overall max thickness:", baseline["overall_max_thickness"])
    print("Zero design controlling section:", baseline["overall_section_name"])
    print("Zero design section tmax:", baseline["section_max_thickness"])


if __name__ == "__main__":
    main()
