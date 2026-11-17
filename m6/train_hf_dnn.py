import time
from pathlib import Path

from dnn_surrogate import (
    TrainConfig,
    cross_validate,
    final_train_with_holdout,
    get_device,
    load_mat_dataset,
    save_checkpoint,
    train_full_model,
    write_json,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "hfdata.mat"
OUTPUT_DIR = ROOT / "trained_dnn_model"
OUTPUT_DIR.mkdir(exist_ok=True)


def build_stage1_search_space():
    common = dict(
        optimizer="adamw",
        batch_size=64,
        max_epochs=800,
        early_stop_patience=80,
        lr_patience=20,
        min_lr=1e-5,
        grad_clip=5.0,
    )
    return [
        TrainConfig(hidden_dims=(128, 128, 64), activation="relu", dropout=0.00, norm="none", residual=False, loss_name="mse", lr=2e-3, weight_decay=1e-5, **common),
        TrainConfig(hidden_dims=(256, 128, 64), activation="relu", dropout=0.02, norm="none", residual=False, loss_name="mse", lr=2e-3, weight_decay=1e-5, **common),
        TrainConfig(hidden_dims=(256, 256, 128), activation="relu", dropout=0.05, norm="layernorm", residual=False, loss_name="mse", lr=1.5e-3, weight_decay=2e-5, **common),
        TrainConfig(hidden_dims=(128, 128, 128), activation="tanh", dropout=0.00, norm="none", residual=True, loss_name="mse", lr=1.5e-3, weight_decay=1e-5, **common),
        TrainConfig(hidden_dims=(256, 256, 256), activation="tanh", dropout=0.00, norm="layernorm", residual=True, loss_name="mse", lr=1e-3, weight_decay=1e-5, **common),
        TrainConfig(hidden_dims=(256, 256, 128), activation="leaky_relu", dropout=0.02, norm="layernorm", residual=False, loss_name="mse", lr=1.5e-3, weight_decay=2e-5, **common),
        TrainConfig(hidden_dims=(256, 256, 256), activation="leaky_relu", dropout=0.03, norm="layernorm", residual=True, loss_name="mse", lr=1.2e-3, weight_decay=2e-5, **common),
        TrainConfig(hidden_dims=(192, 192, 192, 96), activation="gelu", dropout=0.02, norm="layernorm", residual=False, loss_name="mse", lr=1.2e-3, weight_decay=2e-5, **common),
        TrainConfig(hidden_dims=(256, 256, 256), activation="gelu", dropout=0.03, norm="layernorm", residual=True, loss_name="mse", lr=1e-3, weight_decay=3e-5, **common),
        TrainConfig(hidden_dims=(320, 320, 160), activation="silu", dropout=0.02, norm="layernorm", residual=False, loss_name="mse", lr=1e-3, weight_decay=2e-5, **common),
        TrainConfig(hidden_dims=(256, 256, 256, 128), activation="silu", dropout=0.03, norm="layernorm", residual=True, loss_name="mse", lr=8e-4, weight_decay=3e-5, **common),
        TrainConfig(hidden_dims=(256, 256, 128, 64), activation="silu", dropout=0.00, norm="layernorm", residual=False, loss_name="smooth_l1", lr=1e-3, weight_decay=1e-5, **common),
    ]


def build_stage2_search_space():
    common = dict(
        activation="tanh",
        dropout=0.0,
        norm="none",
        residual=True,
        loss_name="mse",
        batch_size=32,
        max_epochs=1200,
        early_stop_patience=120,
        lr_patience=30,
        min_lr=5e-6,
        grad_clip=5.0,
    )
    return [
        TrainConfig(hidden_dims=(128, 128, 128), optimizer="adamw", lr=1e-3, weight_decay=0.0, **common),
        TrainConfig(hidden_dims=(128, 128, 128), optimizer="adamw", lr=1.2e-3, weight_decay=5e-6, **common),
        TrainConfig(hidden_dims=(128, 128, 128), optimizer="adamw", lr=1.5e-3, weight_decay=1e-5, **common),
        TrainConfig(hidden_dims=(160, 160, 160), optimizer="adamw", lr=1.2e-3, weight_decay=5e-6, **common),
        TrainConfig(hidden_dims=(192, 192, 192), optimizer="adamw", lr=1e-3, weight_decay=5e-6, **common),
        TrainConfig(hidden_dims=(128, 128, 128, 128), optimizer="adamw", lr=1e-3, weight_decay=5e-6, **common),
        TrainConfig(hidden_dims=(160, 160, 160), optimizer="adam", lr=1e-3, weight_decay=0.0, **common),
        TrainConfig(hidden_dims=(128, 128, 128), optimizer="adam", lr=1.2e-3, weight_decay=0.0, **common),
    ]


def run_search(x, y, configs, device, seed_base, title):
    results = []
    print("=" * 90)
    print(title)
    print(f"Configs: {len(configs)}")
    for idx, config in enumerate(configs, start=1):
        print("=" * 90)
        print(f"[{idx}/{len(configs)}] CV search: {config.to_dict()}")
        t0 = time.time()
        result = cross_validate(x, y, config, seed=seed_base + idx * 100, device=device, num_folds=5)
        elapsed = time.time() - t0
        result["elapsed_sec"] = elapsed
        results.append(result)
        summary = result["summary"]
        print(
            "mean score={:.6f} | rmse_cl={:.6f} | rmse_cd={:.6f} | r2_cl={:.6f} | r2_cd={:.6f} | {:.1f}s".format(
                summary["mean_score"],
                summary["mean_rmse_cl"],
                summary["mean_rmse_cd"],
                summary["mean_r2_cl"],
                summary["mean_r2_cd"],
                elapsed,
            )
        )
    results.sort(key=lambda item: item["summary"]["mean_score"])
    return results


def main():
    device = get_device()
    x, y = load_mat_dataset(str(DATA_PATH))
    print(f"Device: {device}")
    print(f"Dataset: X={x.shape}, Y={y.shape}")

    stage1_results = run_search(x, y, build_stage1_search_space(), device, seed_base=42, title="Stage 1 broad search")
    stage2_results = run_search(x, y, build_stage2_search_space(), device, seed_base=500, title="Stage 2 refined search")

    all_search_results = stage1_results + stage2_results
    all_search_results.sort(key=lambda item: item["summary"]["mean_score"])
    best_search = all_search_results[0]
    best_config = TrainConfig(
        hidden_dims=tuple(best_search["config"]["hidden_dims"]),
        activation=best_search["config"]["activation"],
        dropout=best_search["config"]["dropout"],
        norm=best_search["config"]["norm"],
        residual=best_search["config"]["residual"],
        optimizer=best_search["config"]["optimizer"],
        loss_name=best_search["config"]["loss_name"],
        lr=best_search["config"]["lr"],
        weight_decay=best_search["config"]["weight_decay"],
        batch_size=best_search["config"]["batch_size"],
        max_epochs=best_search["config"]["max_epochs"],
        early_stop_patience=best_search["config"]["early_stop_patience"],
        lr_patience=best_search["config"]["lr_patience"],
        min_lr=best_search["config"]["min_lr"],
        grad_clip=best_search["config"]["grad_clip"],
    )

    print("=" * 90)
    print("Best CV config:")
    print(best_config.to_dict())
    print(best_search["summary"])

    holdout = final_train_with_holdout(x, y, best_config, seed=2026, device=device)
    print("=" * 90)
    print("Holdout benchmark:")
    print(holdout["test_metrics"])

    final_model = train_full_model(x, y, best_config, seed=2048, device=device)
    checkpoint_path = OUTPUT_DIR / "best_dnn_surrogate.pt"
    save_checkpoint(
        str(checkpoint_path),
        final_model["model"].cpu(),
        final_model["x_scaler"],
        final_model["y_scaler"],
        best_config,
        metadata={
            "data_path": str(DATA_PATH),
            "cv_summary": best_search["summary"],
            "holdout_metrics": holdout["test_metrics"],
            "full_metrics": final_model["full_metrics"],
        },
    )

    report = {
        "data_path": str(DATA_PATH),
        "num_samples": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "output_dim": int(y.shape[1]),
        "best_config": best_config.to_dict(),
        "stage1_search_results": stage1_results,
        "stage2_search_results": stage2_results,
        "search_results": all_search_results,
        "best_cv_summary": best_search["summary"],
        "holdout": holdout,
        "full_train_metrics": final_model["full_metrics"],
        "checkpoint_path": str(checkpoint_path),
    }
    write_json(str(OUTPUT_DIR / "training_report.json"), report)
    write_json(str(OUTPUT_DIR / "best_config.json"), best_config.to_dict())

    print("=" * 90)
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved report: {OUTPUT_DIR / 'training_report.json'}")


if __name__ == "__main__":
    main()
