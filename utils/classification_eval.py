import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    return roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else np.nan


def safe_pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    return average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else np.nan


def safe_brier_score(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return brier_score_loss(labels, scores)
    except ValueError:
        return np.nan


def class_distribution(labels: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    total = int(labels.size)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    return {
        "n": total,
        "positive": positives,
        "negative": negatives,
        "positive_rate": float(positives / total) if total else np.nan,
        "negative_rate": float(negatives / total) if total else np.nan,
    }


def confusion_matrix_dict(labels: np.ndarray, preds: np.ndarray) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "labels": ["negative", "positive"],
        "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def metrics_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores > threshold).astype(int)
    cm = confusion_matrix_dict(labels, preds)
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    metrics = {
        "threshold": float(threshold),
        "auc": safe_binary_auc(labels, scores),
        "roc_auc": safe_binary_auc(labels, scores),
        "pr_auc": safe_pr_auc(labels, scores),
        "f1": f1_score(labels, preds, zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier_score": safe_brier_score(labels, scores),
    }
    return metrics, cm


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.array([0.5], dtype=float)

    unique_scores = np.unique(scores)
    candidates = [0.0, 0.5, 1.0]
    eps = 1e-12
    candidates.append(max(0.0, float(unique_scores[0]) - eps))
    candidates.append(min(1.0, float(unique_scores[-1]) + eps))
    candidates.extend(float(x) for x in unique_scores)
    if unique_scores.size > 1:
        candidates.extend(float(x) for x in (unique_scores[:-1] + unique_scores[1:]) / 2.0)
    return np.array(sorted(set(candidates)), dtype=float)


def best_threshold_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    selection_metric: str = "balanced_accuracy",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    best_metrics = None
    best_cm = None
    best_key = None
    for threshold in threshold_candidates(scores):
        metrics, cm = metrics_at_threshold(labels, scores, float(threshold))
        metric_value = metrics.get(selection_metric)
        if metric_value is None or np.isnan(metric_value):
            continue
        key = (
            metric_value,
            metrics.get("balanced_accuracy", np.nan),
            metrics.get("f1", np.nan),
            metrics.get("specificity", np.nan),
            -abs(float(threshold) - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = metrics
            best_cm = cm

    if best_metrics is None:
        best_metrics, best_cm = metrics_at_threshold(labels, scores, 0.5)
    best_metrics["threshold_selection_metric"] = selection_metric
    return best_metrics, best_cm


def score_distribution(scores: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return {}
    return {
        "count": int(scores.size),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "median": float(np.median(scores)),
        "q25": float(np.quantile(scores, 0.25)),
        "q75": float(np.quantile(scores, 0.75)),
        "num_above_0_5": int(np.sum(scores > 0.5)),
        "num_at_or_below_0_5": int(np.sum(scores <= 0.5)),
        "all_above_0_5": bool(np.all(scores > 0.5)),
    }


def curve_data(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) <= 1:
        return {
            "roc": {"fpr": [], "tpr": [], "thresholds": []},
            "pr": {"precision": [], "recall": [], "thresholds": []},
        }
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    return {
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        "pr": {"precision": precision, "recall": recall, "thresholds": pr_thresholds},
    }


def build_binary_classification_diagnostics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    train_labels: Optional[np.ndarray] = None,
    threshold_selection_metric: str = "balanced_accuracy",
    metadata: Optional[Dict[str, Any]] = None,
    probability_note: Optional[str] = None,
) -> Dict[str, Any]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    metrics_0_5, cm_0_5 = metrics_at_threshold(labels, scores, 0.5)
    best_metrics, best_cm = best_threshold_metrics(
        labels,
        scores,
        selection_metric=threshold_selection_metric,
    )
    best_f1_metrics, best_f1_cm = best_threshold_metrics(
        labels,
        scores,
        selection_metric="f1",
    )
    best_bal_metrics, best_bal_cm = best_threshold_metrics(
        labels,
        scores,
        selection_metric="balanced_accuracy",
    )

    val_dist = class_distribution(labels)
    class_dist = {"val": val_dist}
    if train_labels is not None:
        class_dist["train"] = class_distribution(train_labels)

    prevalence = val_dist["positive_rate"]
    pr_auc = metrics_0_5["pr_auc"]
    normalized_ap = (
        (pr_auc - prevalence) / (1.0 - prevalence)
        if not np.isnan(pr_auc) and prevalence < 1.0
        else np.nan
    )

    return {
        "metadata": metadata or {},
        "class_distribution": class_dist,
        "prediction_distribution": score_distribution(scores),
        "threshold_0_5_metrics": metrics_0_5,
        "best_threshold_metrics": best_metrics,
        "best_f1_threshold_metrics": best_f1_metrics,
        "best_balanced_accuracy_threshold_metrics": best_bal_metrics,
        "confusion_matrix_threshold_0_5": cm_0_5,
        "confusion_matrix_best_threshold": best_cm,
        "confusion_matrix_best_f1_threshold": best_f1_cm,
        "confusion_matrix_best_balanced_accuracy_threshold": best_bal_cm,
        "curves": curve_data(labels, scores),
        "pr_auc_baseline_positive_prevalence": prevalence,
        "normalized_average_precision": normalized_ap,
        "probability_note": probability_note,
    }


def save_binary_classification_artifacts(
    eval_dir: Path,
    diagnostics: Dict[str, Any],
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    indices: Optional[Any] = None,
    patient_ids: Optional[List[Any]] = None,
    score_column: str = "predicted_probability",
    extra_columns: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, str]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    best_threshold = diagnostics["best_threshold_metrics"]["threshold"]

    write_json(eval_dir / "metrics.json", diagnostics)
    write_json(eval_dir / "class_distribution.json", diagnostics["class_distribution"])
    write_json(
        eval_dir / "confusion_matrix_threshold_0_5.json",
        diagnostics["confusion_matrix_threshold_0_5"],
    )
    write_json(
        eval_dir / "confusion_matrix_best_threshold.json",
        diagnostics["confusion_matrix_best_threshold"],
    )

    curves = diagnostics["curves"]
    roc_rows = [
        {"fpr": float(fpr), "tpr": float(tpr), "threshold": float(threshold)}
        for fpr, tpr, threshold in zip(
            curves["roc"]["fpr"],
            curves["roc"]["tpr"],
            curves["roc"]["thresholds"],
        )
    ]
    write_csv(eval_dir / "roc_curve.csv", roc_rows, ["fpr", "tpr", "threshold"])

    pr_thresholds = list(curves["pr"]["thresholds"])
    pr_rows = []
    for idx, (precision, recall) in enumerate(
        zip(curves["pr"]["precision"], curves["pr"]["recall"])
    ):
        threshold = pr_thresholds[idx] if idx < len(pr_thresholds) else np.nan
        pr_rows.append(
            {
                "precision": float(precision),
                "recall": float(recall),
                "threshold": float(threshold) if not np.isnan(threshold) else "",
            }
        )
    write_csv(eval_dir / "pr_curve.csv", pr_rows, ["precision", "recall", "threshold"])

    if indices is None:
        index_values = list(range(labels.size))
    elif isinstance(indices, torch.Tensor):
        index_values = indices.detach().cpu().numpy().tolist()
    else:
        index_values = np.asarray(indices).tolist()

    extra_columns = extra_columns or {}
    rows = []
    for row_idx, (idx, label, score) in enumerate(zip(index_values, labels, scores)):
        row = {
            "patient_index": int(idx),
            "label": int(label),
            score_column: float(score),
            "pred_threshold_0_5": int(score > 0.5),
            "pred_best_threshold": int(score > best_threshold),
        }
        if patient_ids is not None:
            row["patient_id"] = patient_ids[row_idx]
        for name, values in extra_columns.items():
            row[name] = json_ready(np.asarray(values)[row_idx])
        rows.append(row)

    fieldnames = ["patient_index", "patient_id", "label", score_column]
    fieldnames.extend(k for k in extra_columns.keys())
    fieldnames.extend(["pred_threshold_0_5", "pred_best_threshold"])
    fieldnames = [name for name in fieldnames if rows and name in rows[0]]
    write_csv(eval_dir / "predicted_probabilities.csv", rows, fieldnames)

    files = {
        "metrics_json": str(eval_dir / "metrics.json"),
        "class_distribution_json": str(eval_dir / "class_distribution.json"),
        "confusion_matrix_threshold_0_5_json": str(
            eval_dir / "confusion_matrix_threshold_0_5.json"
        ),
        "confusion_matrix_best_threshold_json": str(
            eval_dir / "confusion_matrix_best_threshold.json"
        ),
        "roc_curve_csv": str(eval_dir / "roc_curve.csv"),
        "pr_curve_csv": str(eval_dir / "pr_curve.csv"),
        "predicted_probabilities_csv": str(eval_dir / "predicted_probabilities.csv"),
    }
    diagnostics["files"] = files
    write_json(eval_dir / "metrics.json", diagnostics)
    return files
