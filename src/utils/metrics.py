"""
Shared evaluation utilities for TransactionGuard models.

Reports the metrics that matter for imbalanced fraud detection:
- Precision, Recall, F1 (threshold-based)
- PR-AUC (Area Under Precision-Recall Curve) — threshold-independent
- Confusion matrix summary

WHY NOT ROC-AUC OR ACCURACY?
------------------------------
With <1% fraud rate (class imbalance ~1:800):
  - Accuracy: A naive "predict everything as legitimate" classifier gets
    99.87% accuracy while catching exactly ZERO frauds. Useless.
  - ROC-AUC: The ROC curve plots TPR vs. FPR. At very low fraud rates,
    FPR (false positive rate = FP / (FP + TN)) is dominated by the massive
    number of true negatives. Even a poor model looks good on ROC because
    it can push many legit transactions correctly to "not fraud" at the cost
    of missing lots of actual fraud. ROC-AUC of 0.95 can still mean we catch
    only 40% of frauds.
  - PR-AUC directly measures the tradeoff between precision (of all our
    fraud alerts, how many are real?) and recall (of all real frauds, how
    many did we catch?). It degrades sharply when the model fails on the
    minority class — exactly the signal we need.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate a fraud detection model and print a formatted report.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_pred : array-like of {0, 1}  (binary predictions at given threshold)
    y_prob : array-like of float   (fraud probability scores)
    model_name : str
    threshold : float

    Returns
    -------
    dict with keys: precision, recall, f1, pr_auc
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    pr_auc    = average_precision_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*65}")
    print(f" {model_name} — Evaluation Results")
    print(f"{'='*65}")
    print(f"  Threshold   : {threshold:.2f}")
    print(f"  Precision   : {precision:.4f}  (of flagged txs, {precision*100:.1f}% are real fraud)")
    print(f"  Recall      : {recall:.4f}  (caught {recall*100:.1f}% of all actual frauds)")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  PR-AUC      : {pr_auc:.4f}  ← primary metric (higher = better at any threshold)")
    print(f"")
    print(f"  Confusion Matrix:")
    print(f"    True Negatives  (legit, predicted legit): {tn:>8,}")
    print(f"    False Positives (legit, predicted fraud): {fp:>8,}")
    print(f"    False Negatives (fraud, predicted legit): {fn:>8,}")
    print(f"    True Positives  (fraud, predicted fraud): {tp:>8,}")
    print(f"{'='*65}")

    return {
        "model": model_name,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "pr_auc":    round(pr_auc, 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[metrics] Saved → {path}")


def print_comparison_table(baseline: dict, graph: dict) -> str:
    """
    Print and return a clean comparison table between baseline and graph model.
    This is the key artifact of Phase 3.
    """
    lines = [
        "",
        "╔══════════════════════════════════════════════════════════╗",
        "║         PHASE 2 vs PHASE 3 — MODEL COMPARISON           ║",
        "╠══════════════╦═════════════╦═════════════╦══════════════╣",
        "║ Metric       ║  Baseline   ║   + Graph   ║    Change    ║",
        "╠══════════════╬═════════════╬═════════════╬══════════════╣",
    ]

    metrics = ["precision", "recall", "f1", "pr_auc"]
    labels  = ["Precision", "Recall", "F1 Score", "PR-AUC"]

    for metric, label in zip(metrics, labels):
        base_val  = baseline.get(metric, 0)
        graph_val = graph.get(metric, 0)
        delta     = graph_val - base_val
        sign      = "+" if delta >= 0 else ""
        lines.append(
            f"║ {label:<12} ║ {base_val:>11.4f} ║ {graph_val:>11.4f} ║ {sign}{delta:>+11.4f} ║"
        )

    lines += [
        "╚══════════════╩═════════════╩═════════════╩══════════════╝",
        "",
    ]

    table = "\n".join(lines)
    print(table)
    return table
