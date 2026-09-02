"""
POST /score — Score a transaction and return SHAP explanation.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.models import TransactionInput, ScoreResponse, FeatureExplanation
from backend.startup import state
from src.features.tabular import engineer_tabular_features, TABULAR_FEATURE_COLS
from src.features.graph import GRAPH_FEATURE_COLS

router = APIRouter()

ALL_FEATURE_COLS = TABULAR_FEATURE_COLS + GRAPH_FEATURE_COLS


def _score_from_test_set(tx_id: str) -> dict:
    """Score a transaction that exists in the test set (uses precomputed graph features)."""
    row_mask = state.test_df["transaction_id"] == tx_id
    if not row_mask.any():
        return None

    row = state.test_df[row_mask].iloc[0]
    X_row = row[state.feature_cols].values.reshape(1, -1)
    fraud_prob = float(state.model.predict_proba(X_row)[0, 1])
    is_predicted_fraud = fraud_prob >= state.threshold

    # SHAP explanation
    shap_vals = state.explainer.shap_values(X_row)[0]
    abs_shap = np.abs(shap_vals)
    top_indices = np.argsort(abs_shap)[::-1][:5]

    from src.explainability.shap_explainer import FEATURE_DESCRIPTIONS, _build_plain_english_reason
    top_features = []
    for idx in top_indices:
        feat = state.feature_cols[idx]
        val  = float(shap_vals[idx])
        top_features.append({
            "feature":       feat,
            "shap_value":    round(val, 4),
            "direction":     "increases_fraud_score" if val > 0 else "decreases_fraud_score",
            "feature_value": float(row[feat]),
            "description":   FEATURE_DESCRIPTIONS.get(feat, feat.replace("_", " ")),
        })

    reason = _build_plain_english_reason(top_features, fraud_prob)

    return {
        "transaction_id":       tx_id,
        "fraud_prob":           round(fraud_prob, 4),
        "is_fraud_predicted":   bool(is_predicted_fraud),
        "threshold_used":       round(state.threshold, 4),
        "plain_english_reason": reason,
        "top_features":         top_features,
        "transaction_info": {
            "type":   str(row.get("type", "")),
            "amount": float(row.get("amount", 0)),
            "orig":   str(row.get("nameOrig", "")),
            "dest":   str(row.get("nameDest", "")),
        }
    }


def _score_new_transaction(tx: TransactionInput) -> dict:
    """Score a new transaction not in the test set (uses live graph lookups)."""
    import uuid
    tx_id = tx.transaction_id or f"TX_LIVE_{uuid.uuid4().hex[:8].upper()}"

    # Build a single-row DataFrame and engineer tabular features
    row_dict = {
        "transaction_id": tx_id,
        "step": tx.step or 1,
        "type": tx.type,
        "amount": tx.amount,
        "nameOrig": tx.nameOrig,
        "oldbalanceOrg": tx.oldbalanceOrg,
        "newbalanceOrig": tx.newbalanceOrig,
        "nameDest": tx.nameDest,
        "oldbalanceDest": tx.oldbalanceDest,
        "newbalanceDest": tx.newbalanceDest,
        "isFraud": 0,  # unknown, placeholder
    }
    df_row = pd.DataFrame([row_dict])

    # Tabular features
    df_feat = engineer_tabular_features(df_row)

    # Graph features — look up from precomputed node features
    nf = state.node_features_idx
    def get_nf(account, feat, default=0.0):
        return nf.get(account, {}).get(feat, default)

    orig, dest = tx.nameOrig, tx.nameDest

    # Shared neighbors in graph
    try:
        G_und = state.graph.to_undirected()
        n_orig = set(G_und.neighbors(orig)) if orig in state.graph else set()
        n_dest = set(G_und.neighbors(dest)) if dest in state.graph else set()
        shared_nbrs = len(n_orig & n_dest)
    except Exception:
        shared_nbrs = 0

    orig_comm = get_nf(orig, "community_id", -1)
    dest_comm = get_nf(dest, "community_id", -1)
    community_same = 1 if (orig_comm != -1 and orig_comm == dest_comm) else 0

    df_feat["orig_in_degree"]    = get_nf(orig, "in_degree", 0)
    df_feat["orig_out_degree"]   = get_nf(orig, "out_degree", 0)
    df_feat["orig_pagerank"]     = get_nf(orig, "pagerank", 0.0)
    df_feat["orig_clustering"]   = get_nf(orig, "clustering_coeff", 0.0)
    df_feat["orig_community_id"] = orig_comm
    df_feat["dest_in_degree"]    = get_nf(dest, "in_degree", 0)
    df_feat["dest_out_degree"]   = get_nf(dest, "out_degree", 0)
    df_feat["dest_pagerank"]     = get_nf(dest, "pagerank", 0.0)
    df_feat["dest_clustering"]   = get_nf(dest, "clustering_coeff", 0.0)
    df_feat["dest_community_id"] = dest_comm
    df_feat["shared_neighbors"]  = shared_nbrs
    df_feat["community_same"]    = community_same

    X_row = df_feat[state.feature_cols].values.reshape(1, -1)
    fraud_prob = float(state.model.predict_proba(X_row)[0, 1])
    is_predicted_fraud = fraud_prob >= state.threshold

    shap_vals = state.explainer.shap_values(X_row)[0]
    abs_shap = np.abs(shap_vals)
    top_indices = np.argsort(abs_shap)[::-1][:5]

    from src.explainability.shap_explainer import FEATURE_DESCRIPTIONS, _build_plain_english_reason
    top_features = []
    for idx in top_indices:
        feat = state.feature_cols[idx]
        val  = float(shap_vals[idx])
        top_features.append({
            "feature":       feat,
            "shap_value":    round(val, 4),
            "direction":     "increases_fraud_score" if val > 0 else "decreases_fraud_score",
            "feature_value": float(df_feat[feat].iloc[0]),
            "description":   FEATURE_DESCRIPTIONS.get(feat, feat.replace("_", " ")),
        })

    reason = _build_plain_english_reason(top_features, fraud_prob)

    return {
        "transaction_id":       tx_id,
        "fraud_prob":           round(fraud_prob, 4),
        "is_fraud_predicted":   bool(is_predicted_fraud),
        "threshold_used":       round(state.threshold, 4),
        "plain_english_reason": reason,
        "top_features":         top_features,
        "transaction_info": {
            "type":   tx.type,
            "amount": tx.amount,
            "orig":   tx.nameOrig,
            "dest":   tx.nameDest,
        }
    }


@router.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_transaction(tx: TransactionInput):
    """
    Score a transaction for fraud probability with SHAP explanation.

    - If `transaction_id` matches an entry in the test set, uses precomputed features.
    - Otherwise, engineers features on-the-fly and looks up graph node features.
    """
    # Try test set lookup first
    if tx.transaction_id:
        result = _score_from_test_set(tx.transaction_id)
        if result:
            return ScoreResponse(**result)

    # Fall back to live scoring
    result = _score_new_transaction(tx)
    return ScoreResponse(**result)
