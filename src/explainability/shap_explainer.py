"""
Phase 4 — SHAP Explainability

Uses TreeExplainer (exact, not approximate) for XGBoost to provide:
  1. Global feature importance summary plot
  2. Per-transaction explanation: fraud probability + top 5 SHAP features
     + a plain-English one-line reason

WHY SHAP OVER LIME OR OTHER METHODS?
--------------------------------------
- SHAP TreeExplainer is EXACT for tree-based models (XGBoost). LIME uses
  local linear approximations which are sensitive to kernel bandwidth.
- SHAP satisfies game-theoretic fairness axioms (efficiency, symmetry,
  dummy, linearity). LIME does not.
- SHAP values are consistent: if a model relies more on feature X for a
  prediction, X's SHAP value is always higher. LIME can be inconsistent
  across nearby samples.
- Industry standard: most fraud/credit ML teams use SHAP for regulatory
  explainability (e.g., adverse action reasons for loan denials under ECOA).
"""

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = ARTIFACTS_DIR / "models"


# ---------------------------------------------------------------------------
# Load model and test data
# ---------------------------------------------------------------------------

def load_model_and_data():
    """Load the graph-augmented model and scored test set."""
    model_bundle = joblib.load(MODELS_DIR / "xgb_graph.pkl")
    model         = model_bundle["model"]
    threshold     = model_bundle["threshold"]
    feature_cols  = model_bundle["feature_cols"]

    test_df = pd.read_parquet(PROCESSED_DIR / "test_scored.parquet")
    X_test  = test_df[feature_cols].values

    return model, threshold, feature_cols, test_df, X_test


# ---------------------------------------------------------------------------
# Global SHAP summary plot
# ---------------------------------------------------------------------------

def compute_shap_values(model, X: np.ndarray, feature_cols: list[str]) -> shap.Explanation:
    """Compute SHAP values using TreeExplainer (exact for XGBoost)."""
    print("[shap] Computing SHAP values with TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    # For large test sets, sample 5000 for the summary plot
    if len(X) > 5000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), 5000, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    shap_values = explainer.shap_values(X_sample)
    print(f"[shap] SHAP values computed. Shape: {shap_values.shape}")
    return explainer, shap_values, X_sample


def plot_shap_summary(shap_values: np.ndarray, X_sample: np.ndarray, feature_cols: list[str]) -> None:
    """Generate and save global SHAP summary plot."""
    print("[shap] Generating global SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_cols,
        show=False,
        max_display=20,
        plot_type="dot",
    )
    plt.title("Global Feature Importance (SHAP)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = ARTIFACTS_DIR / "shap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap] Summary plot saved → {out}")


# ---------------------------------------------------------------------------
# Per-prediction explanation
# ---------------------------------------------------------------------------

# Plain-English templates for top features
FEATURE_DESCRIPTIONS = {
    # Tabular features
    "orig_zero_balance_after":    "originator account was completely drained to zero",
    "dest_had_zero_before":       "destination account had zero balance before receiving funds",
    "error_balance_orig":         "originator balance change doesn't match the amount sent",
    "error_balance_dest":         "destination balance change doesn't match the amount received",
    "amount":                     "transaction amount",
    "amount_to_orig_balance":     "transaction amount is a large fraction of originator's balance",
    "balance_delta_orig":         "originator's balance dropped sharply",
    "type_TRANSFER":              "transaction is a TRANSFER (highest fraud-risk type)",
    "type_CASH_OUT":              "transaction is a CASH_OUT (high fraud-risk type)",
    "orig_tx_count_24h":          "high originator transaction frequency in the past 24 hours",
    "orig_vol_24h":               "high originator transaction volume in the past 24 hours",
    "dest_tx_count_24h":          "high destination transaction frequency in the past 24 hours",
    # Graph features
    "shared_neighbors":           "unusually high number of shared counterparties between sender and receiver",
    "community_same":             "sender and receiver belong to different transaction communities (cross-community transfer)",
    "orig_pagerank":              "originator has abnormally high network influence (PageRank)",
    "dest_pagerank":              "destination has abnormally high network influence (PageRank)",
    "orig_out_degree":            "originator is sending to an unusually large number of accounts",
    "dest_in_degree":             "destination is receiving from an unusually large number of accounts",
    "orig_clustering":            "originator is embedded in a tightly connected cluster (ring signature)",
    "dest_clustering":            "destination is embedded in a tightly connected cluster (ring signature)",
}


def _get_feature_description(feat: str, shap_val: float) -> str:
    """Get a plain-English description of a feature and its direction."""
    base = FEATURE_DESCRIPTIONS.get(feat, feat.replace("_", " "))
    direction = "increased" if shap_val > 0 else "decreased"
    return f"{base} (pushed fraud score {direction})"


def explain_transaction(
    transaction_id: str,
    test_df: pd.DataFrame,
    model,
    explainer,
    feature_cols: list[str],
    threshold: float,
    top_n: int = 5,
) -> dict:
    """
    Explain a single transaction prediction.

    Returns
    -------
    dict with:
      - transaction_id
      - fraud_prob
      - is_fraud_predicted
      - top_features: list of {feature, shap_value, direction, description}
      - plain_english_reason: one-line explanation string
      - actual_label (if available in test set)
    """
    row_mask = test_df["transaction_id"] == transaction_id
    if not row_mask.any():
        return {"error": f"Transaction {transaction_id} not found in test set."}

    row = test_df[row_mask].iloc[0]
    X_row = row[feature_cols].values.reshape(1, -1)

    # Prediction
    fraud_prob = float(model.predict_proba(X_row)[0, 1])
    is_predicted_fraud = fraud_prob >= threshold

    # SHAP values for this single transaction
    shap_vals = explainer.shap_values(X_row)[0]  # shape: (n_features,)

    # Sort features by |SHAP value| descending
    abs_shap = np.abs(shap_vals)
    top_indices = np.argsort(abs_shap)[::-1][:top_n]

    top_features = []
    for idx in top_indices:
        feat  = feature_cols[idx]
        val   = float(shap_vals[idx])
        top_features.append({
            "feature":     feat,
            "shap_value":  round(val, 4),
            "direction":   "increases_fraud_score" if val > 0 else "decreases_fraud_score",
            "feature_value": float(row[feat]),
            "description": FEATURE_DESCRIPTIONS.get(feat, feat.replace("_", " ")),
        })

    # Build plain-English reason from the top 2-3 features
    reason = _build_plain_english_reason(top_features, fraud_prob)

    return {
        "transaction_id":      transaction_id,
        "fraud_prob":          round(fraud_prob, 4),
        "is_fraud_predicted":  bool(is_predicted_fraud),
        "threshold_used":      round(threshold, 4),
        "actual_label":        int(row.get("isFraud", -1)),
        "top_features":        top_features,
        "plain_english_reason": reason,
        "transaction_info": {
            "type":   row.get("type", ""),
            "amount": float(row.get("amount", 0)),
            "orig":   row.get("nameOrig", ""),
            "dest":   row.get("nameDest", ""),
        }
    }


def _build_plain_english_reason(top_features: list[dict], fraud_prob: float) -> str:
    """
    Generate a single human-readable sentence explaining the prediction.
    Focuses on fraud-increasing features (positive SHAP) in plain language.
    """
    # Filter to fraud-increasing features
    fraud_factors = [f for f in top_features if f["direction"] == "increases_fraud_score"]

    if not fraud_factors:
        return (f"No strong fraud indicators found (probability {fraud_prob:.1%}); "
                "prediction based on combination of weak signals.")

    top = fraud_factors[0]
    desc1 = FEATURE_DESCRIPTIONS.get(top["feature"], top["feature"].replace("_", " "))

    if len(fraud_factors) >= 2:
        sec = fraud_factors[1]
        desc2 = FEATURE_DESCRIPTIONS.get(sec["feature"], sec["feature"].replace("_", " "))
        reason = (f"Flagged primarily due to: {desc1}, and secondarily: {desc2}. "
                  f"Combined fraud probability: {fraud_prob:.1%}.")
    else:
        reason = f"Flagged primarily due to: {desc1}. Fraud probability: {fraud_prob:.1%}."

    return reason


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase4():
    print("\n" + "="*65)
    print(" Phase 4 — SHAP Explainability")
    print("="*65)

    model, threshold, feature_cols, test_df, X_test = load_model_and_data()

    explainer, shap_values, X_sample = compute_shap_values(model, X_test, feature_cols)
    plot_shap_summary(shap_values, X_sample, feature_cols)

    # Save explainer for the backend
    explainer_bundle = {
        "explainer":    explainer,
        "feature_cols": feature_cols,
        "threshold":    threshold,
    }
    import pickle
    explainer_path = ARTIFACTS_DIR / "shap_explainer.pkl"
    with open(explainer_path, "wb") as f:
        pickle.dump(explainer_bundle, f)
    print(f"[shap] Explainer saved → {explainer_path}")

    # Demo: explain a known fraud transaction
    fraud_ids = test_df[test_df["isFraud"] == 1]["transaction_id"]
    legit_ids = test_df[test_df["isFraud"] == 0]["transaction_id"]

    if len(fraud_ids) > 0:
        fraud_ex = explain_transaction(
            fraud_ids.iloc[0], test_df, model, explainer, feature_cols, threshold
        )
        print(f"\n[shap] Sample FRAUD explanation:")
        print(f"  Transaction: {fraud_ex['transaction_id']}")
        print(f"  Fraud prob:  {fraud_ex['fraud_prob']:.4f}")
        print(f"  Reason: {fraud_ex['plain_english_reason']}")
        print(f"  Top features:")
        for feat in fraud_ex["top_features"][:3]:
            print(f"    {feat['feature']:35s} SHAP={feat['shap_value']:+.4f}")

    if len(legit_ids) > 0:
        legit_ex = explain_transaction(
            legit_ids.iloc[100], test_df, model, explainer, feature_cols, threshold
        )
        print(f"\n[shap] Sample LEGITIMATE explanation:")
        print(f"  Transaction: {legit_ex['transaction_id']}")
        print(f"  Fraud prob:  {legit_ex['fraud_prob']:.4f}")
        print(f"  Reason: {legit_ex['plain_english_reason']}")

    print("\n[shap] Phase 4 complete.")
    return explainer, feature_cols, threshold


if __name__ == "__main__":
    run_phase4()
