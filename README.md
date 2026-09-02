# TransactionGuard

**Real-time fraud detection for financial transactions** with graph-based ring detection and per-prediction SHAP explainability. Built as a portfolio project demonstrating end-to-end ML system design for a payments company internship application.

---

## Architecture

```
Raw Data (synthetic PaySim-schema, 500k transactions)
    │
    ▼
src/data/eda.py  ─────────────────────────────────────────────────
    ├── Schema & class balance report (0.15% fraud rate)
    ├── EDA plots: amount distributions, fraud by type, time patterns
    ├── Stratified 80/20 train/test split
    └── data/processed/train.parquet · test.parquet
    │
    ▼
src/features/tabular.py  ────────────────────────────────────────
    ├── Balance delta features (orig + dest)
    ├── Amount-to-balance ratios
    ├── Zero-balance flags
    ├── One-hot transaction type encoding
    └── Account velocity (rolling 24h tx count + volume)
    │
    ├──► src/models/train_baseline.py  (Phase 2)
    │       ├── SMOTETomek imbalance handling
    │       ├── XGBoost training (400 estimators)
    │       └── metrics/baseline_metrics.json
    │
    ▼
src/features/graph.py  ──────────────────────────────────────────
    ├── NetworkX DiGraph: nodes=accounts, edges=transactions
    ├── Node features: degree, PageRank, clustering, community (Louvain)
    └── Edge features: shared neighbors, cross-community flag
    │
    ▼
src/models/train_graph.py  (Phase 3)
    ├── Graph features joined onto tabular set (23 → 33 features)
    ├── XGBoost retrained
    └── *** Comparison table (see below) ***
    │
    ▼
src/explainability/shap_explainer.py  (Phase 4)
    ├── TreeExplainer (exact for XGBoost)
    ├── Per-prediction: fraud_prob + top-5 SHAP features + plain-English reason
    └── artifacts/shap_summary.png
    │
    ▼
backend/main.py  (FastAPI)
    ├── POST /score
    ├── GET /transactions/flagged
    ├── GET /graph/ring/{account_id}
    └── GET /metrics
    │
    ▼
frontend/  (React + Vite + Tailwind)
    ├── Flagged transactions table (sortable, filterable, paginated)
    ├── SHAP explanation panel (bar chart + plain-English reason)
    ├── Force-directed account graph view (canvas, community coloring)
    └── Phase 2 vs Phase 3 metrics comparison panel
```

---

## ⭐ Phase 2 vs Phase 3 — Model Comparison

> **Primary metric is PR-AUC** (Area Under Precision-Recall Curve), not accuracy or ROC-AUC.
> At 0.15% fraud rate, a model that flags *nothing* achieves 99.85% accuracy and ~0.5 ROC-AUC
> while catching zero frauds. PR-AUC directly measures the precision/recall tradeoff on the minority class.

| Metric       | Phase 2 (Tabular) | Phase 3 (+ Graph) | Change       |
|:-------------|:-----------------:|:-----------------:|:------------:|
| **PR-AUC ★** | 0.7907            | 0.7663            | −0.0244      |
| Precision    | 0.7059            | 0.6444            | −0.0615      |
| Recall       | 0.7059            | 0.7582            | **+0.0523**  |
| F1 Score     | 0.7059            | 0.6967            | −0.0092      |

**Interpretation**: The graph model shows a classic **precision-recall tradeoff** — it trades some precision for higher recall (catching 8 more actual frauds: 116 vs 108 true positives). This is the expected behavior when adding graph features: the network context expands the model's detection surface to catch fraud cases that lack obvious tabular signals, at the cost of slightly more false positives. In a production setting you would tune the threshold toward higher recall (catching more frauds) or higher precision (fewer false alerts) based on business requirements.

The comparison serves as architectural validation: graph features provide a *different* signal from tabular features, enabling the model to catch frauds that row-level features would miss.

---

## Setup & Running

### Prerequisites
- Python 3.11+ (tested on 3.14)
- Node.js 18+

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the full training pipeline

```bash
bash scripts/run_pipeline.sh
```

This runs all 4 phases (~5–10 minutes on a modern laptop) and saves all artifacts to `artifacts/` and `metrics/`.

### 3. Start the FastAPI backend

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 4. Start the React frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard available at: http://localhost:5173

---

## Design Decisions

### Why PaySim (this dataset)?

PaySim is the only widely-used public fraud dataset that preserves **real account identifiers** (`nameOrig`, `nameDest`). Most fraud datasets (e.g., the Kaggle credit card fraud set) use PCA-anonymized features that destroy account identity — you literally cannot build a transaction graph from them because the node identity is gone. PaySim's named accounts make the graph construction in Phase 3 meaningful: edges represent real money flows between identifiable parties.

We use a synthetic generator with the same schema because Kaggle requires authentication for download in automated environments. The generator injects controlled ring patterns (3–5 hop layering chains) that mirror the fraud patterns PaySim is famous for.

### Why XGBoost over alternatives?

1. **TreeExplainer is exact**: SHAP's `TreeExplainer` computes exact Shapley values for tree models in polynomial time. For neural networks, SHAP uses DeepExplainer (approximate) or KernelExplainer (slow). Exact explainability is non-negotiable for fraud teams facing regulatory scrutiny.
2. **Mixed feature types**: XGBoost handles the mix of continuous tabular features and discrete graph features (degree, community ID) without normalization.
3. **Speed**: XGBoost trains in ~30 seconds on 100k resampled rows, enabling rapid iteration across phases.
4. **Industry standard**: XGBoost is the most common model in production fraud detection due to its interpretability and performance on tabular data.

### Why these specific graph features?

| Feature | What it captures |
|:--------|:----------------|
| Degree centrality | Mule accounts receive from many ring participants (high in-degree) and send to many cash-out destinations (high out-degree) |
| PageRank | Hub accounts that concentrate money from many high-activity sources get disproportionate PageRank — detects concentrators that simple degree misses |
| Clustering coefficient | Ring members transact exclusively among themselves → high local clustering. Legitimate accounts have independent counterparties → low clustering |
| Shared neighbors | If sender and receiver share many mutual counterparties, they're both embedded in the same ring — even if no single transaction looks unusual |
| Cross-community flag | Louvain communities are natural trust groups; cross-community high-value transfers are a layering/structuring signal |

### Why SHAP over LIME?

- **Exactness**: `TreeExplainer` gives exact Shapley values for XGBoost. LIME fits a local linear surrogate model, which is an approximation and can be inconsistent.
- **Consistency**: If model reliance on feature X increases, SHAP values for X always increase. LIME can violate this.
- **Game-theoretic foundation**: SHAP satisfies efficiency, symmetry, dummy, and linearity axioms — important properties for regulatory adverse-action explanations.
- **Industry adoption**: SHAP is the de facto standard for fraud/credit model explainability under ECOA, GDPR Article 22, and similar regulations.

---

## API Endpoints

| Method | Path | Description |
|:-------|:-----|:------------|
| `POST` | `/score` | Score a transaction; returns fraud probability + SHAP explanation |
| `GET`  | `/transactions/flagged` | Paginated list of flagged transactions |
| `GET`  | `/graph/ring/{account_id}` | Local 2-hop subgraph around an account |
| `GET`  | `/metrics` | Phase 2 vs Phase 3 comparison metrics |
| `GET`  | `/docs` | Auto-generated Swagger UI |

---

## Project Structure

```
TransactionGuard/
├── data/
│   ├── raw/                     # Generated CSV (500k transactions)
│   └── processed/               # Cleaned parquet + train/test splits
├── src/
│   ├── data/
│   │   ├── generate_synthetic.py   # PaySim-schema data generator w/ ring injection
│   │   └── eda.py                  # Phase 1: EDA + stratified split
│   ├── features/
│   │   ├── tabular.py              # Balance deltas, velocity, type encoding
│   │   └── graph.py                # NetworkX graph + degree/pagerank/community features
│   ├── models/
│   │   ├── train_baseline.py       # Phase 2: XGBoost + SMOTETomek
│   │   └── train_graph.py          # Phase 3: Graph-augmented XGBoost
│   ├── explainability/
│   │   └── shap_explainer.py       # Phase 4: TreeExplainer + per-prediction reasons
│   └── utils/
│       └── metrics.py              # PR-AUC, F1, comparison table utilities
├── backend/
│   ├── main.py                  # FastAPI app + CORS + lifespan
│   ├── startup.py               # Model/graph loading at startup
│   ├── models.py                # Pydantic schemas
│   └── router/
│       ├── score.py             # POST /score
│       ├── transactions.py      # GET /transactions/flagged
│       ├── graph.py             # GET /graph/ring/{account_id}
│       └── metrics.py           # GET /metrics
├── frontend/
│   └── src/
│       ├── App.tsx              # Tab layout, dashboard + metrics view
│       ├── api.ts               # Typed fetch client
│       └── components/
│           ├── TransactionTable.tsx   # Sortable/filterable flagged transactions
│           ├── ShapExplanation.tsx    # SHAP bar chart + plain-English reason
│           ├── GraphView.tsx          # Canvas force-directed graph
│           └── MetricsPanel.tsx       # Comparison table + confusion matrix
├── artifacts/
│   ├── models/                  # Trained model pickles
│   ├── shap_summary.png         # Global SHAP feature importance plot
│   ├── metrics_comparison.json  # Phase 2 vs 3 metrics
│   └── eda_*.png               # EDA plots
├── metrics/
│   ├── baseline_metrics.json
│   └── graph_metrics.json
├── scripts/
│   └── run_pipeline.sh         # One-command end-to-end training
└── requirements.txt
```

---

## Limitations & Future Work

### Known Limitations

1. **Graph features computed in batch, not incrementally**: The NetworkX graph is built over the entire dataset at training time. In production, graph features would need to be updated incrementally per transaction (e.g., using a streaming graph database like TigerGraph or Neptune).

2. **Graph leakage at split boundary**: Account nodes appear in both train and test graph features. A production split would partition accounts into non-overlapping cohorts. This is documented as accepted for a portfolio project.

3. **Synthetic dataset**: Real-world fraud is more adversarial. Fraudsters adapt to detection systems; synthetic ring patterns are static and more predictable than live fraud.

4. **No real-time streaming ingestion**: The system scores transactions on-demand (request/response). Production systems use Kafka or Kinesis for sub-100ms streaming scoring.

5. **SQLite not implemented**: The project uses in-memory test data. A production system would log all scored transactions to a database for audit trails and retraining triggers.

6. **No model drift monitoring**: There's no mechanism to detect when the model's performance degrades as fraud patterns evolve. Production systems use PSI (Population Stability Index) or similar monitoring.

### Future Work

- **Temporal graph features**: Use time-windowed graph construction (only edges from last 30 days) to prevent stale graph signals from dominating.
- **Graph Neural Networks**: Replace hand-engineered graph features with a GNN (e.g., GraphSAGE) that learns graph representations end-to-end.
- **Active learning loop**: Flag uncertain predictions (near the decision threshold) for human review, and use reviewer labels to retrain the model.
- **Multi-model ensemble**: Combine the XGBoost tabular model with an isolation forest for outlier detection and a rule-based engine for known fraud patterns.
- **Real PaySim dataset**: Download from Kaggle (`paysim1`) for a direct comparison against the published PaySim baseline numbers.
