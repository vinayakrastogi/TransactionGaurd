"""
Pydantic schemas for the TransactionGuard API.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Input schema for POST /score"""
    transaction_id: Optional[str] = None
    step: Optional[int] = None
    type: str = Field(..., description="CASH_IN | CASH_OUT | DEBIT | PAYMENT | TRANSFER")
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float


class FeatureExplanation(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_fraud_score" | "decreases_fraud_score"
    feature_value: float
    description: str


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_prob: float
    is_fraud_predicted: bool
    threshold_used: float
    plain_english_reason: str
    top_features: list[FeatureExplanation]
    transaction_info: dict[str, Any]


class FlaggedTransaction(BaseModel):
    transaction_id: str
    fraud_prob: float
    is_fraud_predicted: bool
    actual_label: int
    type: str
    amount: float
    nameOrig: str
    nameDest: str
    step: Optional[int] = None


class FlaggedTransactionsResponse(BaseModel):
    total: int
    page: int
    limit: int
    transactions: list[FlaggedTransaction]


class GraphNode(BaseModel):
    id: str
    is_flagged: bool
    degree: int
    pagerank: float
    community_id: int


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float
    is_fraud: bool


class GraphResponse(BaseModel):
    account_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class MetricSet(BaseModel):
    model: str
    precision: float
    recall: float
    f1: float
    pr_auc: float
    tp: int
    fp: int
    tn: int
    fn: int
    threshold: float


class MetricsResponse(BaseModel):
    baseline: MetricSet
    graph: MetricSet
    improvement: dict[str, float]
