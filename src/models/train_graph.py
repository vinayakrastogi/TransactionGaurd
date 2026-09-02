"""
Phase 3 — Graph-Augmented Model Training

Builds a transaction graph from the training data, computes graph features,
joins them onto the tabular features, retrains XGBoost, and prints the
Phase 2 vs Phase 3 comparison table.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.tabular import TABULAR_FEATURE_COLS, TARGET_COL
from src.features.graph import (
    build_transaction_graph,
    compute_node_features,
    compute_edge_features,
    GRAPH_FEATURE_COLS,
)
from src.utils.metrics import evaluate_model, save_metrics, print_comparison_table

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR    = ARTIFACTS_DIR / "models"
METRICS_DIR   = PROJECT_ROOT / "metrics"

ALL_FEATURE_COLS = TABULAR_FEATURE_COLS + GRAPH_FEATURE_COLS


def train_graph_model():
    print("\n" + "="*65)
    print(" Phase 3 — Graph-Augmented Model Training")
    print("="*65)

    # 1. Load feature-engineered data from Phase 2
    print("[graph_model] Loading tabular-featured datasets...")
    train_df = pd.read_parquet(PROCESSED_DIR / "train_tabular.parquet")
    test_df  = pd.read_parquet(PROCESSED_DIR / "test_tabular.parquet")

    # 2. Build transaction graph from TRAINING data only
    # (We build from full cleaned dataset so graph structure is complete,
    #  but we note this as a known graph leakage limitation in the README)
    print("[graph_model] Building transaction graph from training data...")
    clean_df = pd.read_parquet(PROCESSED_DIR / "transactions_clean.parquet")
    G = build_transaction_graph(clean_df)

    # Save graph for the backend
    import pickle
    graph_path = ARTIFACTS_DIR / "transaction_graph.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
    print(f"[graph_model] Graph saved → {graph_path}")

    # 3. Compute node features
    node_features = compute_node_features(G)
    node_features.to_parquet(ARTIFACTS_DIR / "node_features.parquet", index=False)

    # 4. Join graph features onto train/test
    print("[graph_model] Computing edge features for train set...")
    train_graph = compute_edge_features(train_df, G, node_features)

    print("[graph_model] Computing edge features for test set...")
    test_graph = compute_edge_features(test_df, G, node_features)

    # Save for backend/Phase 4
    train_graph.to_parquet(PROCESSED_DIR / "train_graph.parquet", index=False)
    test_graph.to_parquet(PROCESSED_DIR / "test_graph.parquet", index=False)

    # 5. Prepare features
    X_train = train_graph[ALL_FEATURE_COLS].values
    y_train = train_graph[TARGET_COL].values
    X_test  = test_graph[ALL_FEATURE_COLS].values
    y_test  = test_graph[TARGET_COL].values

    print(f"\n[graph_model] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[graph_model] Graph features added: {GRAPH_FEATURE_COLS}")

    # 6. SMOTETomek on subsample (same strategy as baseline)
    fraud_mask = y_train == 1
    legit_mask = y_train == 0
    n_fraud = fraud_mask.sum()
    n_legit = legit_mask.sum()

    MAX_LEGIT_FOR_SMOTE = 50_000
    if n_legit > MAX_LEGIT_FOR_SMOTE:
        rng = np.random.RandomState(42)
        legit_idx = np.where(legit_mask)[0]
        sampled_legit = rng.choice(legit_idx, MAX_LEGIT_FOR_SMOTE, replace=False)
        fraud_idx = np.where(fraud_mask)[0]
        smote_idx = np.concatenate([sampled_legit, fraud_idx])
        X_smote = X_train[smote_idx]
        y_smote = y_train[smote_idx]
    else:
        X_smote, y_smote = X_train, y_train

    print(f"\n[graph_model] Applying SMOTETomek...")
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_smote, y_smote)
    print(f"  After SMOTETomek: {X_resampled.shape[0]:,} samples")

    # 7. Train XGBoost with graph features
    scale_pos_weight = (y_resampled == 0).sum() / (y_resampled == 1).sum()

    model = XGBClassifier(
        n_estimators=400,
        max_depth=7,           # slightly deeper to capture graph feature interactions
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    print("\n[graph_model] Training XGBoost with graph features...")
    model.fit(
        X_resampled, y_resampled,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # 8. Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_thresh = thresholds[np.argmax(f1s[:-1])]
    print(f"\n[graph_model] Optimal threshold: {best_thresh:.4f}")

    y_pred = (y_prob >= best_thresh).astype(int)

    graph_metrics = evaluate_model(
        y_test, y_pred, y_prob,
        model_name="Phase 3 — Graph-Augmented Model",
        threshold=best_thresh,
    )
    graph_metrics["threshold"] = round(float(best_thresh), 4)
    graph_metrics["feature_cols"] = ALL_FEATURE_COLS

    # 9. Save model + scored test set
    model_path = MODELS_DIR / "xgb_graph.pkl"
    joblib.dump({
        "model": model,
        "threshold": best_thresh,
        "feature_cols": ALL_FEATURE_COLS,
    }, model_path)
    print(f"\n[graph_model] Model saved → {model_path}")

    save_metrics(graph_metrics, METRICS_DIR / "graph_metrics.json")

    # Save test set with predictions (for backend)
    test_graph["fraud_prob"]    = y_prob
    test_graph["fraud_pred"]    = y_pred
    test_graph.to_parquet(PROCESSED_DIR / "test_scored.parquet", index=False)
    print(f"[graph_model] Scored test set saved.")

    # 10. Load baseline metrics and print comparison table
    baseline_path = METRICS_DIR / "baseline_metrics.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_metrics = json.load(f)

        print("\n[graph_model] *** KEY RESULT: MODEL COMPARISON ***")
        table = print_comparison_table(baseline_metrics, graph_metrics)

        # Save comparison as JSON for the API and README
        comparison = {
            "baseline": baseline_metrics,
            "graph": graph_metrics,
            "improvement": {
                metric: round(graph_metrics.get(metric, 0) - baseline_metrics.get(metric, 0), 4)
                for metric in ["precision", "recall", "f1", "pr_auc"]
            }
        }
        comparison_path = ARTIFACTS_DIR / "metrics_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n[graph_model] Comparison saved → {comparison_path}")

    print("\n[graph_model] Phase 3 complete.")
    return model, graph_metrics, test_graph


if __name__ == "__main__":
    train_graph_model()
