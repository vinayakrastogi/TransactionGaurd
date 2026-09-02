"""
Application state loaded once at startup.
Holds the model, SHAP explainer, transaction graph, and scored test set.
"""

import pickle
import json
from pathlib import Path
from typing import Optional

import joblib
import networkx as nx
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR   = PROJECT_ROOT / "metrics"


class AppState:
    model = None
    explainer = None
    threshold: float = 0.5
    feature_cols: list[str] = []
    graph: Optional[nx.DiGraph] = None
    test_df: Optional[pd.DataFrame] = None
    node_features: Optional[pd.DataFrame] = None
    node_features_idx: Optional[dict] = None
    baseline_metrics: Optional[dict] = None
    graph_metrics: Optional[dict] = None
    comparison: Optional[dict] = None


state = AppState()


def load_all():
    """Load model, explainer, graph, and data at startup."""
    print("[startup] Loading graph-augmented model...")
    model_bundle = joblib.load(ARTIFACTS_DIR / "models" / "xgb_graph.pkl")
    state.model        = model_bundle["model"]
    state.threshold    = model_bundle["threshold"]
    state.feature_cols = model_bundle["feature_cols"]
    print(f"[startup]   Model loaded. Threshold: {state.threshold:.4f}")

    print("[startup] Loading SHAP explainer...")
    with open(ARTIFACTS_DIR / "shap_explainer.pkl", "rb") as f:
        explainer_bundle = pickle.load(f)
    state.explainer = explainer_bundle["explainer"]
    print("[startup]   Explainer loaded.")

    print("[startup] Loading transaction graph...")
    with open(ARTIFACTS_DIR / "transaction_graph.pkl", "rb") as f:
        state.graph = pickle.load(f)
    print(f"[startup]   Graph: {state.graph.number_of_nodes():,} nodes, {state.graph.number_of_edges():,} edges")

    print("[startup] Loading node features...")
    state.node_features = pd.read_parquet(ARTIFACTS_DIR / "node_features.parquet")
    state.node_features_idx = state.node_features.set_index("account").to_dict("index")

    print("[startup] Loading scored test set...")
    state.test_df = pd.read_parquet(PROCESSED_DIR / "test_scored.parquet")
    print(f"[startup]   Test set: {len(state.test_df):,} transactions, "
          f"{state.test_df.get('fraud_pred', pd.Series([0])).sum() if 'fraud_pred' in state.test_df else 'N/A'} flagged")

    print("[startup] Loading metrics...")
    comparison_path = ARTIFACTS_DIR / "metrics_comparison.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            state.comparison = json.load(f)
        state.baseline_metrics = state.comparison.get("baseline")
        state.graph_metrics    = state.comparison.get("graph")

    print("[startup] All assets loaded. API ready.")
