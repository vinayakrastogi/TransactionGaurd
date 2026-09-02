"""
GET /transactions/flagged — Return paginated flagged transactions.
"""

from fastapi import APIRouter, Query
from backend.models import FlaggedTransactionsResponse, FlaggedTransaction
from backend.startup import state

router = APIRouter()


@router.get("/transactions/flagged", response_model=FlaggedTransactionsResponse, tags=["Transactions"])
async def get_flagged_transactions(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    min_prob: float = Query(default=0.0, ge=0.0, le=1.0),
):
    """
    Return recent flagged transactions (fraud_pred=1 or fraud_prob >= min_prob),
    sorted by fraud probability descending.
    """
    df = state.test_df

    # Filter: predicted as fraud OR high probability
    flagged = df[
        (df["fraud_pred"] == 1) | (df["fraud_prob"] >= min_prob)
    ].copy()

    # Sort by fraud probability descending
    flagged = flagged.sort_values("fraud_prob", ascending=False)

    total = len(flagged)
    start = (page - 1) * limit
    end   = start + limit
    page_df = flagged.iloc[start:end]

    transactions = []
    for _, row in page_df.iterrows():
        transactions.append(FlaggedTransaction(
            transaction_id     = str(row["transaction_id"]),
            fraud_prob         = round(float(row["fraud_prob"]), 4),
            is_fraud_predicted = bool(row["fraud_pred"] == 1),
            actual_label       = int(row.get("isFraud", -1)),
            type               = str(row.get("type", "")),
            amount             = float(row.get("amount", 0)),
            nameOrig           = str(row.get("nameOrig", "")),
            nameDest           = str(row.get("nameDest", "")),
            step               = int(row["step"]) if "step" in row else None,
        ))

    return FlaggedTransactionsResponse(
        total=total,
        page=page,
        limit=limit,
        transactions=transactions,
    )
