import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


TARGET_NAMES = ["cl", "cd"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mat_dataset(mat_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = sio.loadmat(mat_path)
    key = next(k for k in data.keys() if not k.startswith("__"))
    array = np.asarray(data[key], dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 54:
        raise ValueError(f"Expected a 2D matrix with at least 54 columns, got {array.shape}.")
    x = array[:, :52]
    y = array[:, 52:54]
    return x, y


class ActivationFactory:
    @staticmethod
    def create(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "leaky_relu":
            return nn.LeakyReLU(0.01)
        if name == "tanh":
            return nn.Tanh()
        if name == "gelu":
            return nn.GELU()
        if name == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported activation: {name}")


def build_norm(name: str, width: int) -> nn.Module:
    name = name.lower()
    if name == "none":
        return nn.Identity()
    if name == "layernorm":
        return nn.LayerNorm(width)
    if name == "batchnorm":
        return nn.BatchNorm1d(width)
    raise ValueError(f"Unsupported norm: {name}")


class ResidualBlock(nn.Module):
    def __init__(self, width: int, activation: str, dropout: float, norm: str):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.norm1 = build_norm(norm, width)
        self.act1 = ActivationFactory.create(activation)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width, width)
        self.norm2 = build_norm(norm, width)
        self.act2 = ActivationFactory.create(activation)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.fc1(x)
        h = self.norm1(h)
        h = self.act1(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.norm2(h)
        h = h + residual
        h = self.act2(h)
        h = self.drop2(h)
        return h


class FeedForwardBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: str, dropout: float, norm: str):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = build_norm(norm, out_dim)
        self.act = ActivationFactory.create(activation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class RegressionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        activation: str,
        dropout: float,
        norm: str,
        residual: bool,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims)
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for width in hidden_dims:
            if residual and prev_dim == width:
                if not layers:
                    layers.append(FeedForwardBlock(prev_dim, width, activation, dropout, norm))
                layers.append(ResidualBlock(width, activation, dropout, norm))
            else:
                layers.append(FeedForwardBlock(prev_dim, width, activation, dropout, norm))
            prev_dim = width

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


@dataclass
class TrainConfig:
    hidden_dims: Tuple[int, ...]
    activation: str
    dropout: float
    norm: str
    residual: bool
    optimizer: str
    loss_name: str
    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    early_stop_patience: int
    lr_patience: int
    min_lr: float
    grad_clip: float

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["hidden_dims"] = list(self.hidden_dims)
        return data


def make_model(config: TrainConfig, input_dim: int = 52, output_dim: int = 2) -> nn.Module:
    return RegressionMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=config.hidden_dims,
        activation=config.activation,
        dropout=config.dropout,
        norm=config.norm,
        residual=config.residual,
    )


def make_loss(loss_name: str) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss(beta=0.5)
    raise ValueError(f"Unsupported loss: {loss_name}")


def make_optimizer(config: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def create_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def inverse_transform_targets(y_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(y_scaled)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    error = y_pred - y_true
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    mae = np.mean(np.abs(error), axis=0)
    r2_num = np.sum(error ** 2, axis=0)
    r2_den = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0) + 1e-12
    r2 = 1.0 - r2_num / r2_den
    target_std = y_true.std(axis=0) + 1e-12
    nrmse = rmse / target_std

    metrics = {
        "rmse_cl": float(rmse[0]),
        "rmse_cd": float(rmse[1]),
        "mae_cl": float(mae[0]),
        "mae_cd": float(mae[1]),
        "r2_cl": float(r2[0]),
        "r2_cd": float(r2[1]),
        "nrmse_cl": float(nrmse[0]),
        "nrmse_cd": float(nrmse[1]),
        "score": float(nrmse.mean()),
    }
    return metrics


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    model.eval()
    preds = []
    targets = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            if criterion is not None:
                loss = criterion(pred, yb)
                total_loss += loss.item() * xb.size(0)
            total_count += xb.size(0)
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())

    pred_scaled = np.vstack(preds)
    true_scaled = np.vstack(targets)
    pred = inverse_transform_targets(pred_scaled, y_scaler)
    true = inverse_transform_targets(true_scaled, y_scaler)
    metrics = regression_metrics(true, pred)
    if criterion is not None:
        metrics["scaled_loss"] = total_loss / max(total_count, 1)
    return metrics


def fit_single_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
    seed: int,
    device: torch.device,
) -> Dict:
    set_seed(seed)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    y_train_scaled = y_scaler.fit_transform(y_train)
    x_val_scaled = x_scaler.transform(x_val)
    y_val_scaled = y_scaler.transform(y_val)

    train_loader = create_dataloader(x_train_scaled, y_train_scaled, config.batch_size, True)
    val_loader = create_dataloader(x_val_scaled, y_val_scaled, config.batch_size, False)

    model = make_model(config).to(device)
    optimizer = make_optimizer(config, model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.lr_patience,
        min_lr=config.min_lr,
    )
    criterion = make_loss(config.loss_name)

    best_state = None
    best_metrics = None
    best_val_score = math.inf
    epochs_no_improve = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_count += xb.size(0)

        train_loss = total_loss / max(total_count, 1)
        val_metrics = evaluate_model(model, val_loader, x_scaler, y_scaler, device, criterion)
        scheduler.step(val_metrics["scaled_loss"])

        history.append(
            {
                "epoch": epoch,
                "train_scaled_loss": train_loss,
                "val_scaled_loss": val_metrics["scaled_loss"],
                "val_score": val_metrics["score"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_metrics["score"] < best_val_score:
            best_val_score = val_metrics["score"]
            best_metrics = val_metrics
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.early_stop_patience:
            break

    model.load_state_dict(best_state)
    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "metrics": best_metrics,
        "history": history,
    }


def cross_validate(
    x: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
    seed: int,
    device: torch.device,
    num_folds: int = 5,
) -> Dict:
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x), start=1):
        result = fit_single_split(
            x_train=x[train_idx],
            y_train=y[train_idx],
            x_val=x[val_idx],
            y_val=y[val_idx],
            config=config,
            seed=seed + fold_idx,
            device=device,
        )
        metrics = dict(result["metrics"])
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

    summary = {}
    for key in ["rmse_cl", "rmse_cd", "mae_cl", "mae_cd", "r2_cl", "r2_cd", "nrmse_cl", "nrmse_cd", "score"]:
        values = [m[key] for m in fold_metrics]
        summary[f"mean_{key}"] = float(np.mean(values))
        summary[f"std_{key}"] = float(np.std(values))

    return {
        "config": config.to_dict(),
        "fold_metrics": fold_metrics,
        "summary": summary,
    }


def final_train_with_holdout(
    x: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
    seed: int,
    device: torch.device,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> Dict:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, shuffle=True
    )
    val_ratio = val_size / (1.0 - test_size)
    x_subtrain, x_val, y_subtrain, y_val = train_test_split(
        x_train, y_train, test_size=val_ratio, random_state=seed + 1, shuffle=True
    )
    result = fit_single_split(x_subtrain, y_subtrain, x_val, y_val, config, seed + 2, device)

    x_scaler = result["x_scaler"]
    y_scaler = result["y_scaler"]
    test_loader = create_dataloader(
        x_scaler.transform(x_test), y_scaler.transform(y_test), config.batch_size, False
    )
    test_metrics = evaluate_model(result["model"], test_loader, x_scaler, y_scaler, device, make_loss(config.loss_name))

    return {
        "val_metrics": result["metrics"],
        "test_metrics": test_metrics,
        "split_sizes": {
            "train": int(len(x_subtrain)),
            "val": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "history": result["history"],
    }


def train_full_model(
    x: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
    seed: int,
    device: torch.device,
    val_size: float = 0.1,
) -> Dict:
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=val_size, random_state=seed, shuffle=True
    )
    result = fit_single_split(x_train, y_train, x_val, y_val, config, seed + 1, device)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(y)
    full_loader = create_dataloader(x_scaled, y_scaled, config.batch_size, True)

    set_seed(seed + 10)
    model = make_model(config).to(device)
    optimizer = make_optimizer(config, model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(4, config.lr_patience),
        min_lr=config.min_lr,
    )
    criterion = make_loss(config.loss_name)

    best_state = None
    best_score = math.inf
    epochs_no_improve = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for xb, yb in full_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_count += xb.size(0)

        scaled_loss = total_loss / max(total_count, 1)

        probe_loader = create_dataloader(
            result["x_scaler"].transform(x_val),
            result["y_scaler"].transform(y_val),
            config.batch_size,
            False,
        )
        val_metrics = evaluate_model(model, probe_loader, result["x_scaler"], result["y_scaler"], device, criterion)
        scheduler.step(val_metrics["scaled_loss"])

        history.append(
            {
                "epoch": epoch,
                "train_scaled_loss": scaled_loss,
                "probe_val_score": val_metrics["score"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_metrics["score"] < best_score:
            best_score = val_metrics["score"]
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.early_stop_patience:
            break

    model.load_state_dict(best_state)
    full_metrics = evaluate_model(
        model,
        create_dataloader(x_scaled, y_scaled, config.batch_size, False),
        x_scaler,
        y_scaler,
        device,
        criterion,
    )

    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "history": history,
        "full_metrics": full_metrics,
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    config: TrainConfig,
    metadata: Dict,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "input_dim": 52,
        "output_dim": 2,
        "x_scaler_mean": x_scaler.mean_.astype(np.float32),
        "x_scaler_scale": x_scaler.scale_.astype(np.float32),
        "y_scaler_mean": y_scaler.mean_.astype(np.float32),
        "y_scaler_scale": y_scaler.scale_.astype(np.float32),
        "target_names": TARGET_NAMES,
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict:
    checkpoint = torch.load(path, map_location=map_location or "cpu")
    config_dict = checkpoint["config"]
    config = TrainConfig(
        hidden_dims=tuple(config_dict["hidden_dims"]),
        activation=config_dict["activation"],
        dropout=config_dict["dropout"],
        norm=config_dict["norm"],
        residual=config_dict["residual"],
        optimizer=config_dict["optimizer"],
        loss_name=config_dict["loss_name"],
        lr=config_dict["lr"],
        weight_decay=config_dict["weight_decay"],
        batch_size=config_dict["batch_size"],
        max_epochs=config_dict["max_epochs"],
        early_stop_patience=config_dict["early_stop_patience"],
        lr_patience=config_dict["lr_patience"],
        min_lr=config_dict["min_lr"],
        grad_clip=config_dict["grad_clip"],
    )
    model = make_model(config, checkpoint["input_dim"], checkpoint["output_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "config": config,
        "x_mean": checkpoint["x_scaler_mean"],
        "x_scale": checkpoint["x_scaler_scale"],
        "y_mean": checkpoint["y_scaler_mean"],
        "y_scale": checkpoint["y_scaler_scale"],
        "target_names": checkpoint.get("target_names", TARGET_NAMES),
        "metadata": checkpoint.get("metadata", {}),
    }


def predict_numpy(bundle: Dict, x: np.ndarray, device: Optional[torch.device] = None) -> np.ndarray:
    device = device or get_device()
    model = bundle["model"].to(device)
    x = np.asarray(x, dtype=np.float32)
    x_scaled = (x - bundle["x_mean"]) / bundle["x_scale"]
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        y_scaled = model(x_tensor).cpu().numpy()
    y = y_scaled * bundle["y_scale"] + bundle["y_mean"]
    return y


def write_json(path: str, obj: Dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
