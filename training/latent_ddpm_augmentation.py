from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    pairwise_distances,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.ddpm import ConditionalDDPM
from models.unet import DenoiseUNet
from utils.latent_extraction import extract_latents_for_ddpm


DEFAULT_AUGMENTATION_RATIOS = (0.25, 0.50, 1.00, 2.00)
DEFAULT_AUGMENTATION_MODES = ("both_classes",)
CONDITION_UNCONDITIONAL_TOKEN = 0
CONDITION_LABEL_OFFSET = 1
CONDITION_NUM_CLASSES = 3


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: _json_ready(v) for k, v in row.items()} for row in rows])


def _as_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype)
    return array


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ratio_name(ratio: float) -> str:
    return f"ratio_{int(round(float(ratio) * 100))}"


def _mode_name(mode: str) -> str:
    return str(mode).strip().lower().replace("-", "_")


def _checkpoint_output_id(checkpoint_path: Path, checkpoint: Dict[str, Any]) -> str:
    rank = checkpoint.get("rank_by_gvae_checkpoint_metric") or checkpoint.get("rank_by_val_auc")
    rank_part = f"rank_{rank}" if rank is not None else "best"
    digest = hashlib.sha1(str(checkpoint_path).encode("utf-8")).hexdigest()[:8]
    return f"{rank_part}_{digest}"


def _normalize_augmentation_modes(modes: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if modes is None:
        return DEFAULT_AUGMENTATION_MODES
    normalized = tuple(_mode_name(mode) for mode in modes)
    valid = {"minority_only", "responder_only", "both_classes"}
    unknown = sorted(set(normalized) - valid)
    if unknown:
        raise ValueError(
            f"Unsupported augmentation mode(s): {unknown}. "
            f"Choose from {sorted(valid)}."
        )
    return normalized


def _classes_for_augmentation(labels_np: np.ndarray, mode: str) -> Tuple[int, ...]:
    counts = {label: int((labels_np == label).sum()) for label in (0, 1)}
    if mode == "both_classes":
        return tuple(label for label in (0, 1) if counts[label] > 0)
    if mode == "responder_only":
        return (1,) if counts[1] > 0 else tuple()
    if mode == "minority_only":
        positive_counts = {label: count for label, count in counts.items() if count > 0}
        if not positive_counts:
            return tuple()
        min_count = min(positive_counts.values())
        return tuple(label for label, count in positive_counts.items() if count == min_count)
    raise ValueError(f"Unsupported augmentation mode: {mode}")


def _shift_labels_for_classifier_free_guidance(labels: torch.Tensor) -> torch.Tensor:
    """Map binary labels 0/1 to 1/2, reserving 0 for unconditional guidance."""
    return labels.long() + CONDITION_LABEL_OFFSET


def class_distribution(labels: Any) -> Dict[str, Any]:
    labels_np = _as_numpy(labels, dtype=np.int64).reshape(-1)
    counts = {str(int(label)): int((labels_np == label).sum()) for label in sorted(np.unique(labels_np))}
    positives = int((labels_np == 1).sum())
    negatives = int((labels_np == 0).sum())
    total = int(labels_np.size)
    return {
        "total": total,
        "counts": counts,
        "negative": negatives,
        "positive": positives,
        "positive_rate": float(positives / total) if total else None,
    }


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    return float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    return float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None


def binary_classification_metrics(
    y_true: Any,
    y_score: Any,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_true_np = _as_numpy(y_true, dtype=np.int64).reshape(-1)
    y_score_np = _as_numpy(y_score, dtype=np.float64).reshape(-1)
    y_pred = (y_score_np >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true_np, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": float(threshold),
        "roc_auc": _safe_auc(y_true_np, y_score_np),
        "auc": _safe_auc(y_true_np, y_score_np),
        "pr_auc": _safe_pr_auc(y_true_np, y_score_np),
        "f1": float(f1_score(y_true_np, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true_np, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred)),
        "precision": float(precision_score(y_true_np, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "brier_score": float(brier_score_loss(y_true_np, y_score_np)),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "class_distribution": class_distribution(y_true_np),
    }


def _candidate_thresholds(y_score: np.ndarray) -> np.ndarray:
    scores = np.unique(np.clip(y_score.astype(np.float64), 0.0, 1.0))
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    return candidates


def select_best_threshold(
    y_true: Any,
    y_score: Any,
    metric: str = "balanced_accuracy",
) -> Dict[str, Any]:
    y_true_np = _as_numpy(y_true, dtype=np.int64).reshape(-1)
    y_score_np = _as_numpy(y_score, dtype=np.float64).reshape(-1)
    best: Optional[Dict[str, Any]] = None
    for threshold in _candidate_thresholds(y_score_np):
        metrics = binary_classification_metrics(y_true_np, y_score_np, float(threshold))
        score = metrics.get(metric)
        if score is None:
            continue
        if best is None or float(score) > float(best[metric]):
            best = metrics
    if best is None:
        best = binary_classification_metrics(y_true_np, y_score_np, 0.5)
    best["threshold_selection_metric"] = metric
    return best


def train_downstream_classifier(
    train_latents: Any,
    train_labels: Any,
    val_latents: Any,
    val_labels: Any,
    classifier_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    classifier_config = classifier_config or {}
    classifier_type = classifier_config.get("type", "logistic_regression")
    random_state = int(classifier_config.get("random_state", 42))
    scale_features = bool(
        classifier_config.get(
            "scale_features",
            classifier_type == "logistic_regression",
        )
    )

    x_train = _as_numpy(train_latents, dtype=np.float32)
    y_train = _as_numpy(train_labels, dtype=np.int64).reshape(-1)
    x_val = _as_numpy(val_latents, dtype=np.float32)
    y_val = _as_numpy(val_labels, dtype=np.int64).reshape(-1)

    if classifier_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=int(classifier_config.get("n_estimators", 300)),
            max_depth=classifier_config.get("max_depth"),
            class_weight=classifier_config.get("class_weight", "balanced"),
            random_state=random_state,
            n_jobs=int(classifier_config.get("n_jobs", -1)),
        )
    elif classifier_type == "logistic_regression":
        clf = LogisticRegression(
            max_iter=int(classifier_config.get("max_iter", 5000)),
            C=float(classifier_config.get("C", 1.0)),
            class_weight=classifier_config.get("class_weight", "balanced"),
            solver=classifier_config.get("solver", "liblinear"),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported downstream classifier type: {classifier_type}")

    if scale_features:
        scaler = StandardScaler()
        x_train_fit = scaler.fit_transform(x_train)
        x_val_fit = scaler.transform(x_val)
        scaler_fit_split = "train"
    else:
        x_train_fit = x_train
        x_val_fit = x_val
        scaler_fit_split = None

    clf.fit(x_train_fit, y_train)
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(x_val_fit)[:, 1]
    else:
        decision = clf.decision_function(x_val_fit)
        y_score = (decision - decision.min()) / (decision.max() - decision.min() + 1e-12)

    return {
        "classifier_type": classifier_type,
        "feature_scaler_fit_split": scaler_fit_split,
        "train_class_distribution": class_distribution(y_train),
        "val_class_distribution": class_distribution(y_val),
        "threshold_0_5": binary_classification_metrics(y_val, y_score, 0.5),
        "best_f1_threshold": select_best_threshold(y_val, y_score, "f1"),
        "best_balanced_accuracy_threshold": select_best_threshold(
            y_val,
            y_score,
            "balanced_accuracy",
        ),
        "predicted_probabilities": y_score.tolist(),
        "labels": y_val.tolist(),
    }


def train_conditional_ddpm_on_latents(
    train_latents: Any,
    train_labels: Any,
    ddpm_config: Dict[str, Any],
    device: torch.device,
    seed: int = 42,
) -> Tuple[ConditionalDDPM, StandardScaler, List[Dict[str, Any]]]:
    _seed_everything(seed)

    latents_np = _as_numpy(train_latents, dtype=np.float32)
    labels_np = _as_numpy(train_labels, dtype=np.int64).reshape(-1)
    latent_dim = int(latents_np.shape[1])

    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents_np).astype(np.float32)
    x_tensor = torch.tensor(latents_scaled, dtype=torch.float32)
    y_tensor = _shift_labels_for_classifier_free_guidance(
        torch.tensor(labels_np, dtype=torch.long)
    )

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=int(ddpm_config.get("batch_size", 32)),
        shuffle=True,
        drop_last=False,
    )

    denoising_net = DenoiseUNet(
        latent_dim=latent_dim,
        num_classes=CONDITION_NUM_CLASSES,
        dim_mults=tuple(ddpm_config.get("dim_mults", (1, 2))),
        dropout_prob=float(ddpm_config.get("dropout_prob", 0.2)),
    ).to(device)
    ddpm_model = ConditionalDDPM(
        denoise_fn=denoising_net,
        latent_dim=latent_dim,
        timesteps=int(ddpm_config.get("timesteps", 250)),
        beta_schedule=ddpm_config.get("beta_schedule", "linear"),
        cond_drop_prob=float(ddpm_config.get("cond_drop_prob", 0.1)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        ddpm_model.parameters(),
        lr=float(ddpm_config.get("lr", 3e-4)),
        weight_decay=float(ddpm_config.get("weight_decay", 1e-4)),
    )
    grad_clip_norm = float(ddpm_config.get("grad_clip_norm", 1.0))
    epochs = int(ddpm_config.get("epochs", 120))
    history: List[Dict[str, Any]] = []

    progress = tqdm(range(epochs), desc="Training conditional latent DDPM")
    for epoch in progress:
        ddpm_model.train()
        total_loss = 0.0
        total_items = 0
        for x0, y in loader:
            optimizer.zero_grad()
            x0 = x0.to(device)
            y = y.to(device)
            loss = ddpm_model.loss(x0, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm_model.parameters(), grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item()) * int(x0.shape[0])
            total_items += int(x0.shape[0])
        avg_loss = total_loss / max(total_items, 1)
        history.append({"epoch": epoch + 1, "train_noise_mse": avg_loss})
        progress.set_postfix(train_noise_mse=f"{avg_loss:.4f}")

    return ddpm_model, scaler, history


@torch.no_grad()
def generate_synthetic_latents(
    ddpm_model: ConditionalDDPM,
    scaler: StandardScaler,
    train_labels: Any,
    ratio: float,
    device: torch.device,
    augmentation_mode: str = "both_classes",
    guidance_scale: float = 2.0,
    batch_size: int = 64,
    seed: int = 42,
) -> Dict[str, Any]:
    _seed_everything(seed)
    labels_np = _as_numpy(train_labels, dtype=np.int64).reshape(-1)
    augmentation_mode = _mode_name(augmentation_mode)
    classes_to_generate = set(_classes_for_augmentation(labels_np, augmentation_mode))
    generated_batches: List[np.ndarray] = []
    generated_labels: List[np.ndarray] = []
    per_class_counts: Dict[str, int] = {}
    ddpm_model.eval()

    for raw_label in (0, 1):
        real_count = int((labels_np == raw_label).sum())
        n_generate = (
            int(round(real_count * float(ratio)))
            if raw_label in classes_to_generate
            else 0
        )
        per_class_counts[str(raw_label)] = n_generate
        if n_generate <= 0:
            continue
        remaining = n_generate
        while remaining > 0:
            current_batch = min(int(batch_size), remaining)
            y = torch.full(
                (current_batch,),
                raw_label + CONDITION_LABEL_OFFSET,
                dtype=torch.long,
                device=device,
            )
            samples_scaled = ddpm_model.sample(y, guidance_scale=guidance_scale)
            samples_np = samples_scaled.detach().cpu().numpy()
            generated_batches.append(scaler.inverse_transform(samples_np).astype(np.float32))
            generated_labels.append(np.full(current_batch, raw_label, dtype=np.int64))
            remaining -= current_batch

    if generated_batches:
        latents = np.vstack(generated_batches)
        labels = np.concatenate(generated_labels)
    else:
        latents = np.empty((0, ddpm_model.latent_dim), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int64)

    return {
        "latents": latents,
        "labels": labels,
        "ratio": float(ratio),
        "augmentation_mode": augmentation_mode,
        "guidance_scale": float(guidance_scale),
        "per_class_counts": per_class_counts,
        "class_distribution": class_distribution(labels),
    }


def _subsample_rows(x: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if x.shape[0] <= max_rows:
        return x
    rng = np.random.default_rng(seed)
    indices = rng.choice(x.shape[0], size=max_rows, replace=False)
    return x[indices]


def mmd_rbf(
    real: Any,
    generated: Any,
    max_samples: int = 1000,
    seed: int = 42,
) -> Optional[float]:
    x = _as_numpy(real, dtype=np.float64)
    y = _as_numpy(generated, dtype=np.float64)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return None
    x = _subsample_rows(x, max_samples, seed)
    y = _subsample_rows(y, max_samples, seed + 1)
    combined = np.vstack([x, y])
    dist2 = pairwise_distances(combined, metric="sqeuclidean")
    positive_dist2 = dist2[dist2 > 0]
    median_dist2 = float(np.median(positive_dist2)) if positive_dist2.size else 1.0
    gamma = 1.0 / (2.0 * max(median_dist2, 1e-12))

    k_xx = np.exp(-gamma * pairwise_distances(x, x, metric="sqeuclidean")).mean()
    k_yy = np.exp(-gamma * pairwise_distances(y, y, metric="sqeuclidean")).mean()
    k_xy = np.exp(-gamma * pairwise_distances(x, y, metric="sqeuclidean")).mean()
    return float(max(k_xx + k_yy - 2.0 * k_xy, 0.0))


def _mean_pairwise_distance(x: np.ndarray, max_samples: int = 1000, seed: int = 42) -> Optional[float]:
    if x.shape[0] < 2:
        return None
    x = _subsample_rows(x, max_samples, seed)
    distances = pairwise_distances(x, metric="euclidean")
    upper = distances[np.triu_indices_from(distances, k=1)]
    return float(upper.mean()) if upper.size else None


def _nearest_neighbor_stats(
    reference: np.ndarray,
    query: np.ndarray,
    real_real_threshold: Optional[float],
) -> Dict[str, Any]:
    if reference.shape[0] == 0 or query.shape[0] == 0:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "p05": None,
            "p95": None,
            "near_duplicate_fraction": None,
        }
    distances = pairwise_distances(query, reference, metric="euclidean").min(axis=1)
    near_duplicate_fraction = None
    if real_real_threshold is not None:
        near_duplicate_fraction = float((distances <= real_real_threshold).mean())
    return {
        "k": 1,
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
        "min": float(distances.min()),
        "p05": float(np.quantile(distances, 0.05)),
        "p95": float(np.quantile(distances, 0.95)),
        "near_duplicate_fraction": near_duplicate_fraction,
    }


def _real_real_threshold(real: np.ndarray, quantile: float = 0.05) -> Optional[float]:
    if real.shape[0] < 2:
        return None
    distances = pairwise_distances(real, real, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    nn = distances.min(axis=1)
    return float(np.quantile(nn, quantile))


def _coverage(real: np.ndarray, generated: np.ndarray, threshold_quantile: float = 0.95) -> Optional[float]:
    if real.shape[0] < 2 or generated.shape[0] == 0:
        return None
    real_distances = pairwise_distances(real, real, metric="euclidean")
    np.fill_diagonal(real_distances, np.inf)
    threshold = float(np.quantile(real_distances.min(axis=1), threshold_quantile))
    nearest_generated = pairwise_distances(real, generated, metric="euclidean").min(axis=1)
    return float((nearest_generated <= threshold).mean())


def latent_distribution_statistics(real: Any, generated: Any) -> Dict[str, Any]:
    x = _as_numpy(real, dtype=np.float64)
    y = _as_numpy(generated, dtype=np.float64)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return {}
    real_mean = x.mean(axis=0)
    gen_mean = y.mean(axis=0)
    real_std = x.std(axis=0)
    gen_std = y.std(axis=0)
    mean_l2_distance = float(np.linalg.norm(real_mean - gen_mean))
    stats = {
        "mean_l2_distance": mean_l2_distance,
        "mean_distance": mean_l2_distance,
        "mean_abs_mean_difference": float(np.abs(real_mean - gen_mean).mean()),
        "max_abs_mean_difference": float(np.abs(real_mean - gen_mean).max()),
        "mean_abs_std_difference": float(np.abs(real_std - gen_std).mean()),
        "max_abs_std_difference": float(np.abs(real_std - gen_std).max()),
        "real_global_mean": float(x.mean()),
        "generated_global_mean": float(y.mean()),
        "real_global_std": float(x.std()),
        "generated_global_std": float(y.std()),
    }
    if x.shape[0] > 1 and y.shape[0] > 1:
        real_cov = np.cov(x, rowvar=False)
        gen_cov = np.cov(y, rowvar=False)
        covariance_distance = float(np.linalg.norm(real_cov - gen_cov))
        stats["covariance_frobenius_difference"] = covariance_distance
        stats["covariance_distance"] = covariance_distance
    return stats


def filter_synthetic_latents_by_knn(
    real_latents: Any,
    real_labels: Any,
    generated_latents: Any,
    generated_labels: Any,
    quantile: float = 0.95,
) -> Dict[str, Any]:
    """Keep generated samples close to same-class train latents.

    The distance threshold is fitted from train-fold real-to-real nearest
    neighbors within each class. Validation/test latents are not used.
    """
    real_x = _as_numpy(real_latents, dtype=np.float64)
    real_y = _as_numpy(real_labels, dtype=np.int64).reshape(-1)
    gen_x = _as_numpy(generated_latents, dtype=np.float64)
    gen_y = _as_numpy(generated_labels, dtype=np.int64).reshape(-1)

    if gen_x.shape[0] == 0:
        return {
            "latents": gen_x.astype(np.float32),
            "labels": gen_y.astype(np.int64),
            "keep_mask": np.zeros((0,), dtype=bool),
            "filter": {
                "enabled": True,
                "method": "same_class_train_knn_distance",
                "threshold_quantile": float(quantile),
                "fit_split": "train",
            },
            "per_class": {},
            "kept_count": 0,
            "removed_count": 0,
            "kept_fraction": None,
        }

    keep_mask = np.zeros(gen_x.shape[0], dtype=bool)
    per_class: Dict[str, Any] = {}
    for label in (0, 1):
        real_class = real_x[real_y == label]
        gen_idx = np.where(gen_y == label)[0]
        gen_class = gen_x[gen_idx]
        threshold = _real_real_threshold(real_class, quantile=quantile)
        if gen_idx.size == 0:
            per_class[str(label)] = {
                "generated_count": 0,
                "kept_count": 0,
                "removed_count": 0,
                "threshold": threshold,
            }
            continue
        if threshold is None or real_class.shape[0] == 0:
            keep_mask[gen_idx] = True
            distances = np.full(gen_idx.size, np.nan)
        else:
            distances = pairwise_distances(gen_class, real_class, metric="euclidean").min(axis=1)
            keep_mask[gen_idx] = distances <= threshold
        kept = int(keep_mask[gen_idx].sum())
        finite_distances = distances[np.isfinite(distances)]
        per_class[str(label)] = {
            "generated_count": int(gen_idx.size),
            "kept_count": kept,
            "removed_count": int(gen_idx.size - kept),
            "threshold": threshold,
            "distance_mean": float(finite_distances.mean()) if finite_distances.size else None,
            "distance_median": float(np.median(finite_distances)) if finite_distances.size else None,
            "distance_p95": float(np.quantile(finite_distances, 0.95)) if finite_distances.size else None,
        }

    kept_latents = gen_x[keep_mask].astype(np.float32)
    kept_labels = gen_y[keep_mask].astype(np.int64)
    kept_count = int(keep_mask.sum())
    removed_count = int(gen_x.shape[0] - kept_count)
    return {
        "latents": kept_latents,
        "labels": kept_labels,
        "keep_mask": keep_mask,
        "filter": {
            "enabled": True,
            "method": "same_class_train_knn_distance",
            "threshold_quantile": float(quantile),
            "fit_split": "train",
        },
        "per_class": per_class,
        "kept_count": kept_count,
        "removed_count": removed_count,
        "kept_fraction": float(kept_count / gen_x.shape[0]) if gen_x.shape[0] else None,
    }


def evaluate_generated_latent_quality(
    real_latents: Any,
    real_labels: Any,
    generated_latents: Any,
    generated_labels: Any,
    seed: int = 42,
) -> Dict[str, Any]:
    real_x = _as_numpy(real_latents, dtype=np.float64)
    real_y = _as_numpy(real_labels, dtype=np.int64).reshape(-1)
    gen_x = _as_numpy(generated_latents, dtype=np.float64)
    gen_y = _as_numpy(generated_labels, dtype=np.int64).reshape(-1)

    threshold = _real_real_threshold(real_x)
    quality = {
        "mmd": mmd_rbf(real_x, gen_x, seed=seed),
        "coverage": _coverage(real_x, gen_x),
        "diversity": {
            "real_mean_pairwise_distance": _mean_pairwise_distance(real_x, seed=seed),
            "generated_mean_pairwise_distance": _mean_pairwise_distance(gen_x, seed=seed + 1),
        },
        "nearest_neighbor_generated_to_real": _nearest_neighbor_stats(real_x, gen_x, threshold),
        "latent_distribution_statistics": latent_distribution_statistics(real_x, gen_x),
        "class_distribution": {
            "real": class_distribution(real_y),
            "generated": class_distribution(gen_y),
        },
        "class_wise": {},
    }

    for label in (0, 1):
        real_class = real_x[real_y == label]
        gen_class = gen_x[gen_y == label]
        class_threshold = _real_real_threshold(real_class)
        quality["class_wise"][str(label)] = {
            "mmd": mmd_rbf(real_class, gen_class, seed=seed + label),
            "coverage": _coverage(real_class, gen_class),
            "diversity": {
                "real_mean_pairwise_distance": _mean_pairwise_distance(real_class, seed=seed),
                "generated_mean_pairwise_distance": _mean_pairwise_distance(gen_class, seed=seed + 1),
            },
            "nearest_neighbor_generated_to_real": _nearest_neighbor_stats(
                real_class,
                gen_class,
                class_threshold,
            ),
            "latent_distribution_statistics": latent_distribution_statistics(real_class, gen_class),
            "real_count": int(real_class.shape[0]),
            "generated_count": int(gen_class.shape[0]),
        }

    return quality


def _save_projection_artifacts(
    output_dir: Path,
    method: str,
    coordinates: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{method}_projection.csv"
    rows = [
        {
            "x": float(coordinates[i, 0]),
            "y": float(coordinates[i, 1]),
            "label": int(labels[i]),
            "source": str(sources[i]),
        }
        for i in range(coordinates.shape[0])
    ]
    _write_csv(csv_path, rows)
    artifacts = {"csv": str(csv_path)}

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        for source in sorted(set(sources.tolist())):
            mask = sources == source
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=18,
                alpha=0.7,
                label=source,
            )
        ax.set_title(f"{method.upper()} latent projection")
        ax.set_xlabel(f"{method}_1")
        ax.set_ylabel(f"{method}_2")
        ax.legend()
        fig.tight_layout()
        png_path = output_dir / f"{method}_projection.png"
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        artifacts["png"] = str(png_path)
    except Exception as exc:  # pragma: no cover - optional visualization dependency
        artifacts["plot_error"] = str(exc)

    return artifacts


def save_latent_projection_visualizations(
    real_latents: Any,
    real_labels: Any,
    generated_latents: Any,
    generated_labels: Any,
    output_dir: Path,
    seed: int = 42,
    max_points_per_source: int = 600,
) -> Dict[str, Any]:
    real_x = _as_numpy(real_latents, dtype=np.float32)
    real_y = _as_numpy(real_labels, dtype=np.int64).reshape(-1)
    gen_x = _as_numpy(generated_latents, dtype=np.float32)
    gen_y = _as_numpy(generated_labels, dtype=np.int64).reshape(-1)

    rng = np.random.default_rng(seed)
    if real_x.shape[0] > max_points_per_source:
        idx = rng.choice(real_x.shape[0], size=max_points_per_source, replace=False)
        real_x, real_y = real_x[idx], real_y[idx]
    if gen_x.shape[0] > max_points_per_source:
        idx = rng.choice(gen_x.shape[0], size=max_points_per_source, replace=False)
        gen_x, gen_y = gen_x[idx], gen_y[idx]

    x = np.vstack([real_x, gen_x])
    labels = np.concatenate([real_y, gen_y])
    sources = np.asarray(["real_train"] * real_x.shape[0] + ["generated"] * gen_x.shape[0])
    artifacts: Dict[str, Any] = {}

    if x.shape[0] < 2:
        return {"error": "Not enough points for projection."}

    pca_coords = PCA(n_components=2, random_state=seed).fit_transform(x)
    artifacts["pca"] = _save_projection_artifacts(output_dir, "pca", pca_coords, labels, sources)

    if x.shape[0] >= 5:
        perplexity = min(30, max(2, (x.shape[0] - 1) // 3))
        try:
            tsne_coords = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=seed,
            ).fit_transform(x)
            artifacts["tsne"] = _save_projection_artifacts(
                output_dir,
                "tsne",
                tsne_coords,
                labels,
                sources,
            )
        except Exception as exc:
            artifacts["tsne"] = {"error": str(exc)}

    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=seed)
        umap_coords = reducer.fit_transform(x)
        artifacts["umap"] = _save_projection_artifacts(
            output_dir,
            "umap",
            umap_coords,
            labels,
            sources,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        artifacts["umap"] = {"available": False, "reason": str(exc)}

    return artifacts


def save_conditional_ddpm_checkpoint(
    path: Path,
    ddpm_model: ConditionalDDPM,
    scaler: StandardScaler,
    ddpm_config: Dict[str, Any],
    train_history: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": ddpm_model.state_dict(),
            "ddpm_config": _json_ready(ddpm_config),
            "latent_dim": ddpm_model.latent_dim,
            "condition_num_classes": CONDITION_NUM_CLASSES,
            "condition_unconditional_token": CONDITION_UNCONDITIONAL_TOKEN,
            "condition_label_offset": CONDITION_LABEL_OFFSET,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "scaler_var": scaler.var_.tolist(),
            "train_history": train_history,
        },
        path,
    )


def apply_train_only_pca(
    train_latents: Any,
    val_latents: Any,
    n_components: Optional[int],
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    train_tensor = torch.as_tensor(train_latents, dtype=torch.float32)
    val_tensor = torch.as_tensor(val_latents, dtype=torch.float32)
    metadata: Dict[str, Any] = {
        "enabled": False,
        "fit_split": "train",
        "requested_components": n_components,
        "original_dim": int(train_tensor.shape[1]),
    }
    if n_components is None:
        return train_tensor, val_tensor, metadata

    train_np = train_tensor.detach().cpu().numpy()
    val_np = val_tensor.detach().cpu().numpy()
    max_components = min(train_np.shape[0], train_np.shape[1])
    final_components = int(min(max(1, int(n_components)), max_components))
    pca = PCA(n_components=final_components, random_state=seed)
    train_pca = pca.fit_transform(train_np).astype(np.float32)
    val_pca = pca.transform(val_np).astype(np.float32)
    metadata.update(
        {
            "enabled": True,
            "requested_components": int(n_components),
            "n_components": final_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        }
    )
    return (
        torch.tensor(train_pca, dtype=torch.float32),
        torch.tensor(val_pca, dtype=torch.float32),
        metadata,
    )


def run_conditional_latent_augmentation_for_checkpoint(
    checkpoint_path: str | Path,
    full_data: Any,
    output_dir: str | Path,
    ratios: Iterable[float] = DEFAULT_AUGMENTATION_RATIOS,
    augmentation_modes: Iterable[str] = DEFAULT_AUGMENTATION_MODES,
    latent_key: str = "concat_mu",
    pca_components: Optional[int] = None,
    filter_config: Optional[Dict[str, Any]] = None,
    ddpm_config: Optional[Dict[str, Any]] = None,
    classifier_config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device | str] = None,
    overwrite_latents: bool = False,
    sample_seed: int = 42,
) -> Dict[str, Any]:
    ddpm_config = dict(ddpm_config or {})
    classifier_config = dict(classifier_config or {})
    filter_config = dict(filter_config or {})
    augmentation_modes = _normalize_augmentation_modes(augmentation_modes)
    ratios = tuple(float(ratio) for ratio in ratios)
    filter_enabled = bool(filter_config.get("enabled", False))
    filter_quantile = float(filter_config.get("quantile", 0.95))
    output_dir = Path(output_dir)
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fold = checkpoint.get("fold", "unknown")
    fold_dir = output_dir / f"fold_{fold}" / _checkpoint_output_id(checkpoint_path, checkpoint)
    latents_path = fold_dir / "real_latents_for_ddpm.pt"
    latents_path = extract_latents_for_ddpm(
        checkpoint_path=checkpoint_path,
        full_data=full_data,
        output_dir=output_dir / "real_latents",
        device=device,
        sample_seed=sample_seed,
        overwrite=overwrite_latents,
        artifact_path=latents_path,
        extra_metadata={
            "pipeline": "conditional_latent_ddpm_augmentation",
            "latent_key": latent_key,
            "ddpm_is_classifier": False,
        },
    )
    latent_artifact = torch.load(latents_path, map_location="cpu", weights_only=False)
    splits = latent_artifact["splits"]
    if "train" not in splits:
        raise ValueError(f"Latent artifact is missing a train split: {latents_path}")
    validation_split = "val" if "val" in splits else "test"
    if validation_split not in splits:
        raise ValueError(f"Latent artifact must contain val or test split: {latents_path}")

    raw_train_latents = splits["train"][latent_key]
    train_labels = splits["train"]["labels"]
    raw_val_latents = splits[validation_split][latent_key]
    val_labels = splits[validation_split]["labels"]
    train_latents, val_latents, pca_metadata = apply_train_only_pca(
        raw_train_latents,
        raw_val_latents,
        pca_components,
        seed=sample_seed,
    )
    latent_dim = int(train_latents.shape[1])
    ddpm_config["latent_dim"] = latent_dim
    effective_latent_key = (
        f"{latent_key}_pca_{pca_metadata['n_components']}"
        if pca_metadata.get("enabled")
        else latent_key
    )

    baseline_classifier = train_downstream_classifier(
        train_latents=train_latents,
        train_labels=train_labels,
        val_latents=val_latents,
        val_labels=val_labels,
        classifier_config=classifier_config,
    )

    ddpm_model, scaler, train_history = train_conditional_ddpm_on_latents(
        train_latents=train_latents,
        train_labels=train_labels,
        ddpm_config=ddpm_config,
        device=device,
        seed=sample_seed,
    )
    ddpm_checkpoint_path = fold_dir / "conditional_ddpm" / "conditional_ddpm.pt"
    save_conditional_ddpm_checkpoint(
        ddpm_checkpoint_path,
        ddpm_model,
        scaler,
        ddpm_config,
        train_history,
    )
    _write_json(fold_dir / "conditional_ddpm" / "train_history.json", {"history": train_history})

    fold_results: Dict[str, Any] = {
        "fold": fold,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_metrics": checkpoint.get("validation_metrics"),
        "latent_key": effective_latent_key,
        "source_latent_key": latent_key,
        "latent_dim": latent_dim,
        "validation_split": validation_split,
        "real_latents_path": str(latents_path),
        "conditional_ddpm_checkpoint": str(ddpm_checkpoint_path),
        "preprocessing": {
            "pca": pca_metadata,
            "ddpm_scaler_fit_split": "train",
            "downstream_scaler_fit_split": (
                baseline_classifier.get("feature_scaler_fit_split")
            ),
        },
        "augmentation_modes": list(augmentation_modes),
        "filter_config": {
            "enabled": filter_enabled,
            "method": "same_class_train_knn_distance" if filter_enabled else None,
            "quantile": filter_quantile if filter_enabled else None,
            "fit_split": "train" if filter_enabled else None,
        },
        "class_distribution": {
            "train": class_distribution(train_labels),
            validation_split: class_distribution(val_labels),
        },
        "ablation_gvae_only": {
            "source": "checkpoint_validation_metrics",
            "metrics": checkpoint.get("validation_metrics"),
            "validation_auc": checkpoint.get("validation_auc"),
            "note": (
                "GVAE-only metrics come from the checkpoint classifier head; "
                "DDPM is not used as a predictor."
            ),
        },
        "ddpm_training": {
            "objective": "noise_prediction_mse",
            "is_classifier": False,
            "condition": "binary_label_shifted_to_1_2_with_0_unconditional",
            "final_train_noise_mse": train_history[-1]["train_noise_mse"] if train_history else None,
        },
        "downstream_real_only": baseline_classifier,
        "augmentation_results": [],
    }

    comparison_rows: List[Dict[str, Any]] = []
    real_best_bal = baseline_classifier["best_balanced_accuracy_threshold"]
    real_best_f1 = baseline_classifier["best_f1_threshold"]
    comparison_rows.append(
        {
            "fold": fold,
            "source": "real_only",
            "augmentation_mode": "none",
            "augmentation_ratio": 0.0,
            "filtered": False,
            "pca_enabled": bool(pca_metadata.get("enabled")),
            "pca_components": pca_metadata.get("n_components"),
            "roc_auc": real_best_bal.get("roc_auc"),
            "pr_auc": real_best_bal.get("pr_auc"),
            "f1_best_balanced_accuracy_threshold": real_best_bal.get("f1"),
            "balanced_accuracy_best_threshold": real_best_bal.get("balanced_accuracy"),
            "f1_best_f1_threshold": real_best_f1.get("f1"),
            "threshold_best_balanced_accuracy": real_best_bal.get("threshold"),
            "threshold_best_f1": real_best_f1.get("threshold"),
            "mmd": None,
            "coverage": None,
            "synthetic_count": 0,
        }
    )

    for augmentation_mode in augmentation_modes:
        for ratio in ratios:
            ratio = float(ratio)
            ratio_dir = fold_dir / "generated" / augmentation_mode / _ratio_name(ratio)
            generated = generate_synthetic_latents(
                ddpm_model=ddpm_model,
                scaler=scaler,
                train_labels=train_labels,
                ratio=ratio,
                device=device,
                augmentation_mode=augmentation_mode,
                guidance_scale=float(ddpm_config.get("guidance_scale", 2.0)),
                batch_size=int(ddpm_config.get("sample_batch_size", ddpm_config.get("batch_size", 32))),
                seed=(
                    sample_seed
                    + int(round(ratio * 1000))
                    + sum(ord(ch) for ch in augmentation_mode)
                ),
            )
            generated_path = ratio_dir / "generated_latents.pt"
            ratio_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "latents": torch.tensor(generated["latents"], dtype=torch.float32),
                    "labels": torch.tensor(generated["labels"], dtype=torch.long),
                    "ratio": ratio,
                    "augmentation_mode": augmentation_mode,
                    "guidance_scale": generated["guidance_scale"],
                    "per_class_counts": generated["per_class_counts"],
                    "latent_key": effective_latent_key,
                    "source_latent_key": latent_key,
                    "source_checkpoint": str(checkpoint_path),
                    "condition_label_offset": CONDITION_LABEL_OFFSET,
                    "condition_unconditional_token": CONDITION_UNCONDITIONAL_TOKEN,
                    "pca": pca_metadata,
                },
                generated_path,
            )

            quality = evaluate_generated_latent_quality(
                real_latents=train_latents,
                real_labels=train_labels,
                generated_latents=generated["latents"],
                generated_labels=generated["labels"],
                seed=sample_seed,
            )
            projections = save_latent_projection_visualizations(
                real_latents=train_latents,
                real_labels=train_labels,
                generated_latents=generated["latents"],
                generated_labels=generated["labels"],
                output_dir=ratio_dir / "projections",
                seed=sample_seed,
            )
            augmented_train_latents = torch.cat(
                [train_latents, torch.tensor(generated["latents"], dtype=train_latents.dtype)],
                dim=0,
            )
            augmented_train_labels = torch.cat(
                [train_labels.long(), torch.tensor(generated["labels"], dtype=torch.long)],
                dim=0,
            )
            downstream_augmented = train_downstream_classifier(
                train_latents=augmented_train_latents,
                train_labels=augmented_train_labels,
                val_latents=val_latents,
                val_labels=val_labels,
                classifier_config=classifier_config,
            )

            augmentation_result = {
                "ratio": ratio,
                "augmentation_mode": augmentation_mode,
                "generated_latents_path": str(generated_path),
                "generated_class_distribution": generated["class_distribution"],
                "quality": quality,
                "projection_artifacts": projections,
                "downstream_real_plus_generated": downstream_augmented,
            }
            fold_results["augmentation_results"].append(augmentation_result)
            _write_json(ratio_dir / "quality_metrics.json", quality)
            _write_json(ratio_dir / "projection_artifacts.json", projections)
            _write_json(ratio_dir / "downstream_classifier_metrics.json", downstream_augmented)

            aug_best_bal = downstream_augmented["best_balanced_accuracy_threshold"]
            aug_best_f1 = downstream_augmented["best_f1_threshold"]
            comparison_rows.append(
                {
                    "fold": fold,
                    "source": "real_plus_generated",
                    "augmentation_mode": augmentation_mode,
                    "augmentation_ratio": ratio,
                    "filtered": False,
                    "pca_enabled": bool(pca_metadata.get("enabled")),
                    "pca_components": pca_metadata.get("n_components"),
                    "roc_auc": aug_best_bal.get("roc_auc"),
                    "pr_auc": aug_best_bal.get("pr_auc"),
                    "f1_best_balanced_accuracy_threshold": aug_best_bal.get("f1"),
                    "balanced_accuracy_best_threshold": aug_best_bal.get("balanced_accuracy"),
                    "f1_best_f1_threshold": aug_best_f1.get("f1"),
                    "threshold_best_balanced_accuracy": aug_best_bal.get("threshold"),
                    "threshold_best_f1": aug_best_f1.get("threshold"),
                    "mmd": quality.get("mmd"),
                    "coverage": quality.get("coverage"),
                    "synthetic_count": int(len(generated["labels"])),
                }
            )

            if filter_enabled:
                filtered = filter_synthetic_latents_by_knn(
                    real_latents=train_latents,
                    real_labels=train_labels,
                    generated_latents=generated["latents"],
                    generated_labels=generated["labels"],
                    quantile=filter_quantile,
                )
                filtered_path = ratio_dir / "filtered_generated_latents.pt"
                torch.save(
                    {
                        "latents": torch.tensor(filtered["latents"], dtype=torch.float32),
                        "labels": torch.tensor(filtered["labels"], dtype=torch.long),
                        "keep_mask": torch.tensor(filtered["keep_mask"], dtype=torch.bool),
                        "ratio": ratio,
                        "augmentation_mode": augmentation_mode,
                        "latent_key": effective_latent_key,
                        "filter": filtered["filter"],
                        "per_class": filtered["per_class"],
                    },
                    filtered_path,
                )
                filtered_quality = evaluate_generated_latent_quality(
                    real_latents=train_latents,
                    real_labels=train_labels,
                    generated_latents=filtered["latents"],
                    generated_labels=filtered["labels"],
                    seed=sample_seed,
                )
                filtered_train_latents = torch.cat(
                    [
                        train_latents,
                        torch.tensor(filtered["latents"], dtype=train_latents.dtype),
                    ],
                    dim=0,
                )
                filtered_train_labels = torch.cat(
                    [
                        train_labels.long(),
                        torch.tensor(filtered["labels"], dtype=torch.long),
                    ],
                    dim=0,
                )
                downstream_filtered = train_downstream_classifier(
                    train_latents=filtered_train_latents,
                    train_labels=filtered_train_labels,
                    val_latents=val_latents,
                    val_labels=val_labels,
                    classifier_config=classifier_config,
                )
                augmentation_result["filtered"] = {
                    "filtered_latents_path": str(filtered_path),
                    "filter_metrics": {
                        k: v
                        for k, v in filtered.items()
                        if k not in {"latents", "labels", "keep_mask"}
                    },
                    "quality": filtered_quality,
                    "downstream_real_plus_filtered_generated": downstream_filtered,
                }
                _write_json(ratio_dir / "filter_metrics.json", augmentation_result["filtered"]["filter_metrics"])
                _write_json(ratio_dir / "filtered_quality_metrics.json", filtered_quality)
                _write_json(
                    ratio_dir / "filtered_downstream_classifier_metrics.json",
                    downstream_filtered,
                )

                filt_best_bal = downstream_filtered["best_balanced_accuracy_threshold"]
                filt_best_f1 = downstream_filtered["best_f1_threshold"]
                comparison_rows.append(
                    {
                        "fold": fold,
                        "source": "real_plus_filtered_generated",
                        "augmentation_mode": augmentation_mode,
                        "augmentation_ratio": ratio,
                        "filtered": True,
                        "pca_enabled": bool(pca_metadata.get("enabled")),
                        "pca_components": pca_metadata.get("n_components"),
                        "roc_auc": filt_best_bal.get("roc_auc"),
                        "pr_auc": filt_best_bal.get("pr_auc"),
                        "f1_best_balanced_accuracy_threshold": filt_best_bal.get("f1"),
                        "balanced_accuracy_best_threshold": filt_best_bal.get("balanced_accuracy"),
                        "f1_best_f1_threshold": filt_best_f1.get("f1"),
                        "threshold_best_balanced_accuracy": filt_best_bal.get("threshold"),
                        "threshold_best_f1": filt_best_f1.get("threshold"),
                        "mmd": filtered_quality.get("mmd"),
                        "coverage": filtered_quality.get("coverage"),
                        "synthetic_count": int(len(filtered["labels"])),
                    }
                )

    _write_json(fold_dir / "fold_summary.json", fold_results)
    _write_csv(fold_dir / "augmentation_comparison.csv", comparison_rows)
    fold_results["comparison_csv"] = str(fold_dir / "augmentation_comparison.csv")
    return fold_results


def summarize_augmentation_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for fold_result in results:
        fold = fold_result["fold"]
        pca_metadata = fold_result.get("preprocessing", {}).get("pca", {})
        baseline = fold_result["downstream_real_only"]["best_balanced_accuracy_threshold"]
        rows.append(
            {
                "fold": fold,
                "augmentation_ratio": 0.0,
                "augmentation_mode": "none",
                "source": "real_only",
                "filtered": False,
                "pca_enabled": bool(pca_metadata.get("enabled")),
                "pca_components": pca_metadata.get("n_components"),
                "roc_auc": baseline.get("roc_auc"),
                "pr_auc": baseline.get("pr_auc"),
                "f1": baseline.get("f1"),
                "balanced_accuracy": baseline.get("balanced_accuracy"),
                "precision": baseline.get("precision"),
                "recall": baseline.get("recall"),
                "specificity": baseline.get("specificity"),
                "mmd": None,
                "coverage": None,
                "synthetic_count": 0,
            }
        )
        for aug in fold_result["augmentation_results"]:
            metrics = aug["downstream_real_plus_generated"]["best_balanced_accuracy_threshold"]
            rows.append(
                {
                    "fold": fold,
                    "augmentation_ratio": aug["ratio"],
                    "augmentation_mode": aug.get("augmentation_mode", "both_classes"),
                    "source": "real_plus_generated",
                    "filtered": False,
                    "pca_enabled": bool(pca_metadata.get("enabled")),
                    "pca_components": pca_metadata.get("n_components"),
                    "roc_auc": metrics.get("roc_auc"),
                    "pr_auc": metrics.get("pr_auc"),
                    "f1": metrics.get("f1"),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "specificity": metrics.get("specificity"),
                    "mmd": aug["quality"].get("mmd"),
                    "coverage": aug["quality"].get("coverage"),
                    "synthetic_count": (
                        aug.get("generated_class_distribution", {}).get("total")
                    ),
                }
            )
            filtered = aug.get("filtered")
            if filtered:
                filtered_metrics = filtered[
                    "downstream_real_plus_filtered_generated"
                ]["best_balanced_accuracy_threshold"]
                rows.append(
                    {
                        "fold": fold,
                        "augmentation_ratio": aug["ratio"],
                        "augmentation_mode": aug.get("augmentation_mode", "both_classes"),
                        "source": "real_plus_filtered_generated",
                        "filtered": True,
                        "pca_enabled": bool(pca_metadata.get("enabled")),
                        "pca_components": pca_metadata.get("n_components"),
                        "roc_auc": filtered_metrics.get("roc_auc"),
                        "pr_auc": filtered_metrics.get("pr_auc"),
                        "f1": filtered_metrics.get("f1"),
                        "balanced_accuracy": filtered_metrics.get("balanced_accuracy"),
                        "precision": filtered_metrics.get("precision"),
                        "recall": filtered_metrics.get("recall"),
                        "specificity": filtered_metrics.get("specificity"),
                        "mmd": filtered["quality"].get("mmd"),
                        "coverage": filtered["quality"].get("coverage"),
                        "synthetic_count": (
                            filtered.get("filter_metrics", {}).get("kept_count")
                        ),
                    }
                )

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            f"{row['source']}_{row['augmentation_mode']}"
            f"_ratio_{row['augmentation_ratio']}"
            f"_filtered_{row.get('filtered', False)}"
            f"_pca_{row.get('pca_components') if row.get('pca_enabled') else 'none'}"
        )
        grouped.setdefault(key, []).append(row)

    summary: Dict[str, Any] = {
        "objective": "conditional_latent_space_augmentation",
        "ddpm_is_classifier": False,
        "ddpm_metrics_removed": [
            "loss_based_roc_auc",
            "loss_based_pr_auc",
            "loss_based_f1",
            "loss_based_accuracy",
            "loss_based_balanced_accuracy",
        ],
        "condition_scheme": {
            "unconditional_token": CONDITION_UNCONDITIONAL_TOKEN,
            "negative_label_token": 1,
            "positive_label_token": 2,
        },
        "ablation_labels": {
            "A": "GVAE classifier head only (from checkpoint validation metrics)",
            "B": "Downstream classifier on real concat_mu only",
            "C": "Real concat_mu plus DDPM synthetic latents",
            "D": "Real concat_mu plus filtered DDPM synthetic latents",
            "E": "PCA concat_mu plus DDPM synthetic latents when pca_components is set",
        },
        "rows": rows,
        "aggregate": {},
    }
    numeric_keys = [
        "roc_auc",
        "pr_auc",
        "f1",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "mmd",
        "coverage",
        "synthetic_count",
    ]
    for key, group_rows in grouped.items():
        aggregate: Dict[str, Any] = {}
        for metric in numeric_keys:
            values = [
                float(row[metric])
                for row in group_rows
                if row.get(metric) is not None and math.isfinite(float(row[metric]))
            ]
            aggregate[f"mean_{metric}"] = float(np.mean(values)) if values else None
            aggregate[f"std_{metric}"] = float(np.std(values)) if values else None
        summary["aggregate"][key] = aggregate
    return summary


def run_conditional_latent_augmentation_pipeline(
    checkpoint_paths: Sequence[str | Path],
    full_data: Any,
    output_dir: str | Path,
    ratios: Iterable[float] = DEFAULT_AUGMENTATION_RATIOS,
    augmentation_modes: Iterable[str] = DEFAULT_AUGMENTATION_MODES,
    latent_key: str = "concat_mu",
    pca_components: Optional[int] = None,
    filter_config: Optional[Dict[str, Any]] = None,
    ddpm_config: Optional[Dict[str, Any]] = None,
    classifier_config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device | str] = None,
    overwrite_latents: bool = False,
    sample_seed: int = 42,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = tuple(float(ratio) for ratio in ratios)
    augmentation_modes = _normalize_augmentation_modes(augmentation_modes)
    fold_results = []
    for checkpoint_path in checkpoint_paths:
        fold_result = run_conditional_latent_augmentation_for_checkpoint(
            checkpoint_path=checkpoint_path,
            full_data=full_data,
            output_dir=output_dir,
            ratios=ratios,
            augmentation_modes=augmentation_modes,
            latent_key=latent_key,
            pca_components=pca_components,
            filter_config=filter_config,
            ddpm_config=ddpm_config,
            classifier_config=classifier_config,
            device=device,
            overwrite_latents=overwrite_latents,
            sample_seed=sample_seed,
        )
        fold_results.append(fold_result)

    summary = summarize_augmentation_results(fold_results)
    summary["augmentation_modes"] = list(_normalize_augmentation_modes(augmentation_modes))
    summary["augmentation_ratios"] = [float(ratio) for ratio in ratios]
    summary["pca_components"] = pca_components
    summary["filter_config"] = dict(filter_config or {})
    summary["fold_results"] = fold_results
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "augmentation_comparison.csv", summary["rows"])
    return summary
