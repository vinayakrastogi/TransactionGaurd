"""
GET /metrics — Return the Phase 2 vs Phase 3 model comparison metrics.
"""

from fastapi import APIRouter, HTTPException
from backend.models import MetricsResponse, MetricSet
from backend.startup import state

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics():
    """Return Phase 2 (baseline) vs Phase 3 (graph-augmented) comparison metrics."""
    if not state.comparison:
        raise HTTPException(status_code=503, detail="Metrics not available. Run training pipeline first.")

    def _parse(m: dict, name: str) -> MetricSet:
        return MetricSet(
            model     = m.get("model", name),
            precision = m.get("precision", 0.0),
            recall    = m.get("recall", 0.0),
            f1        = m.get("f1", 0.0),
            pr_auc    = m.get("pr_auc", 0.0),
            tp        = m.get("tp", 0),
            fp        = m.get("fp", 0),
            tn        = m.get("tn", 0),
            fn        = m.get("fn", 0),
            threshold = m.get("threshold", 0.5),
        )

    return MetricsResponse(
        baseline    = _parse(state.baseline_metrics, "Phase 2 — Tabular Baseline"),
        graph       = _parse(state.graph_metrics,    "Phase 3 — Graph-Augmented"),
        improvement = state.comparison.get("improvement", {}),
    )
