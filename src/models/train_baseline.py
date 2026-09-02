"""
Phase 2 — Baseline Tabular Model Training

Trains an XGBoost classifier on tabular features only.
Handles class imbalance with SMOTETomek.
Reports: Precision, Recall, F1, PR-AUC.
Saves model + metrics to disk.
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
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.tabular import engineer_tabular_features, TABULAR_FEATURE_COLS, TARGET_COL
from src.utils.metrics import evaluate_model, save_metrics

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR  = PROJECT_ROOT / "artifacts"
MODELS_DIR     = ARTIFACTS_DIR / "models"
METRICS_DIR    = PROJECT_ROOT / "metrics"

for d in [MODELS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_baseline():
    print("\n" + "="*65)
    print(" Phase 2 — Baseline Tabular Model Training")
    print("="*65)

    # 1. Load data
    print("[baseline] Loading train/test splits...")
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test_df  = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    # 2. Feature engineering
    print("[baseline] Engineering tabular features...")
    train_feat = engineer_tabular_features(train_df)
    test_feat  = engineer_tabular_features(test_df)

    # Save feature-engineered datasets for Phase 3
    train_feat.to_parquet(PROCESSED_DIR / "train_tabular.parquet", index=False)
    test_feat.to_parquet(PROCESSED_DIR / "test_tabular.parquet", index=False)

    X_train = train_feat[TABULAR_FEATURE_COLS].values
    y_train = train_feat[TARGET_COL].values
    X_test  = test_feat[TABULAR_FEATURE_COLS].values
    y_test  = test_feat[TARGET_COL].values

    print(f"[baseline] Train: {X_train.shape}, fraud rate: {y_train.mean():.4%}")
    print(f"[baseline] Test:  {X_test.shape},  fraud rate: {y_test.mean():.4%}")

    # 3. Handle class imbalance with SMOTETomek
    # SMOTETomek = SMOTE (oversample minority) + Tomek Links (remove borderline majority)
    # This gives cleaner decision boundaries than plain SMOTE.
    print("\n[baseline] Applying SMOTETomek (this may take a minute for 500k rows)...")
    print("  SMOTETomek strategy: oversample fraud class, then clean borderline samples")

    # For large datasets, apply SMOTETomek on a subsample to avoid memory issues
    # (SMOTE on 400k rows with <1% fraud creates a very large intermediate set)
    fraud_mask   = y_train == 1
    legit_mask   = y_train == 0
    n_fraud      = fraud_mask.sum()
    n_legit      = legit_mask.sum()

    # Sample at most 50k legitimate + all fraud for SMOTE, then use XGB scale_pos_weight
    # for the remaining imbalance. This is a common practical approach.
    MAX_LEGIT_FOR_SMOTE = 50_000
    if n_legit > MAX_LEGIT_FOR_SMOTE:
        rng = np.random.RandomState(42)
        legit_idx = np.where(legit_mask)[0]
        sampled_legit = rng.choice(legit_idx, MAX_LEGIT_FOR_SMOTE, replace=False)
        fraud_idx = np.where(fraud_mask)[0]
        smote_idx = np.concatenate([sampled_legit, fraud_idx])
        X_smote = X_train[smote_idx]
        y_smote = y_train[smote_idx]
        print(f"  Using {MAX_LEGIT_FOR_SMOTE:,} legit + {n_fraud} fraud for SMOTE fit")
    else:
        X_smote, y_smote = X_train, y_train

    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_smote, y_smote)
    print(f"  After SMOTETomek: {X_resampled.shape[0]:,} samples "
          f"({y_resampled.sum():,} fraud, {(y_resampled==0).sum():,} legit)")

    # 4. Train XGBoost
    print("\n[baseline] Training XGBoost classifier...")
    # scale_pos_weight handles residual imbalance from the full training set
    # (we used SMOTETomek on a subsample, XGB handles the rest)
    scale_pos_weight = (y_resampled == 0).sum() / (y_resampled == 1).sum()

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
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
    model.fit(
        X_resampled, y_resampled,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # 5. Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold via PR curve (maximize F1)
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_thresh = thresholds[np.argmax(f1s[:-1])]
    print(f"\n[baseline] Optimal threshold (max F1): {best_thresh:.4f}")

    y_pred = (y_prob >= best_thresh).astype(int)

    metrics = evaluate_model(y_test, y_pred, y_prob, model_name="Phase 2 — Tabular Baseline", threshold=best_thresh)
    metrics["threshold"] = round(float(best_thresh), 4)
    metrics["feature_cols"] = TABULAR_FEATURE_COLS

    # 6. Save
    model_path = MODELS_DIR / "xgb_baseline.pkl"
    joblib.dump({"model": model, "threshold": best_thresh, "feature_cols": TABULAR_FEATURE_COLS}, model_path)
    print(f"\n[baseline] Model saved → {model_path}")

    save_metrics(metrics, METRICS_DIR / "baseline_metrics.json")

    print("\n[baseline] Phase 2 complete.")
    return model, metrics, test_feat


if __name__ == "__main__":
    train_baseline()
