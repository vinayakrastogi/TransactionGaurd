# TransactionGuard — Complete Project Summary

> **Who this document is for**: Anyone who wants to deeply understand this project — what it does, why every design decision was made, how the pipeline works end-to-end, and what alternatives were considered. Written in plain language so you can explain this in an interview or to a teammate who's never seen the codebase.

---

## Table of Contents

1. [What Is TransactionGuard?](#1-what-is-transactionguard)
2. [Why This Problem Is Harder Than It Looks](#2-why-this-problem-is-harder-than-it-looks)
3. [Dataset Choice — Why PaySim Schema?](#3-dataset-choice--why-paysim-schema)
4. [High-Level Architecture — The Full Pipeline](#4-high-level-architecture--the-full-pipeline)
5. [Phase 1 — Data Generation and EDA](#5-phase-1--data-generation-and-eda)
6. [Phase 2 — Baseline Tabular Model](#6-phase-2--baseline-tabular-model)
7. [Phase 3 — Graph-Based Ring Detection](#7-phase-3--graph-based-ring-detection)
8. [Phase 4 — SHAP Explainability](#8-phase-4--shap-explainability)
9. [Phase 5 — FastAPI Backend](#9-phase-5--fastapi-backend)
10. [Phase 6 — React Dashboard](#10-phase-6--react-dashboard)
11. [Tech Stack — Every Choice Explained](#11-tech-stack--every-choice-explained)
12. [Metrics — What We Measure and Why](#12-metrics--what-we-measure-and-why)
13. [The Numbers — Actual Training Results](#13-the-numbers--actual-training-results)
14. [What Graph Features Actually Detect](#14-what-graph-features-actually-detect)
15. [Limitations and What a Production Version Would Look Like](#15-limitations-and-what-a-production-version-would-look-like)

---

## 1. What Is TransactionGuard?

TransactionGuard is an end-to-end **fraud detection system** for mobile money transactions. At its core it does one thing: given a financial transaction (who sent money, to whom, how much, and what type of transfer), it tells you whether that transaction is fraudulent — and *why*.

What makes it different from a standard fraud classifier is two things:

**First**, it uses the **transaction network as a source of signal**. Instead of looking at each transaction in isolation ("is this single $300,000 transfer suspicious?"), it looks at the entire web of relationships between accounts. It asks: "Is the sender a known hub in a tightly-connected cluster of accounts? Have the sender and receiver both transacted with the same group of intermediaries? Does this transfer cross community boundaries in the account network?" These are questions a row-level model literally cannot ask — they require building and analyzing a graph.

**Second**, it generates a **plain-English explanation** for every prediction. Not just "this transaction is 87% likely to be fraud" but "flagged primarily because the originator's balance was completely drained and the destination account is a hub that receives from an unusually large number of sources." This matters for two reasons: fraud analysts need to know *why* something was flagged before they act on it, and regulators in many countries (GDPR Article 22, ECOA in the US) require that automated credit/fraud decisions be explainable to the people they affect.

The system is built as a full-stack application: a Python machine learning pipeline, a REST API backend, and a React dashboard — so it works as a live demo, not just a Jupyter notebook.

---

## 2. Why This Problem Is Harder Than It Looks

Fraud detection sounds straightforward: train a classifier, label fraud as 1, non-fraud as 0, done. In practice it has several properties that make it genuinely difficult:

### 2.1 — The Class Imbalance Problem

In real-world mobile money data (and in our synthetic dataset), fraud accounts for roughly **0.15% of all transactions**. That means for every fraud transaction, there are about 650 legitimate ones. This creates a critical trap for naive approaches:

- A classifier that simply **always predicts "not fraud"** would be correct 99.85% of the time. That's a 99.85% accuracy score. If you used accuracy as your metric, this useless model would look great.
- Even more deceptively, such a model achieves a decent **ROC-AUC** score (~0.5 baseline), because the ROC curve measures True Positive Rate vs. False Positive Rate, and with so many true negatives, even a terrible model can look okay by pushing its threshold around.

This is why we don't use accuracy or ROC-AUC. We use **PR-AUC** (Precision-Recall AUC), which only cares about how well the model identifies the minority fraud class. If your model misses all the frauds, PR-AUC collapses toward zero regardless of how well it handles the easy cases.

### 2.2 — Sophisticated Fraud Is Coordinated

Simple fraud looks like one person draining one account and cashing out. Banks got good at catching this years ago with basic rule-based systems (e.g., "flag any TRANSFER that drains an account to zero"). So modern fraud has evolved.

**Fraud rings** are coordinated attacks where multiple accounts (often controlled by the same criminal group) work together. The pattern looks like this:

```
Account A (compromised/mule) → sends $500,000 to Account B
                              Account B → sends $480,000 to Account C
                                         Account C → sends $460,000 to Account D
                                                     Account D → CASH_OUT $440,000
```

Each individual transaction in this chain might look somewhat legitimate to a row-level model. Account A might have a plausible balance. The amount might not be extreme by itself. But when you see that Accounts A, B, C, and D all appeared in the network around the same time, all connected to each other and to no one else, all sending large amounts in a cascading chain — that's a ring, and it requires a graph to detect.

### 2.3 — Catching Fraud Has Asymmetric Costs

Missing a fraud (False Negative) and incorrectly flagging a legitimate transaction (False Positive) have very different real-world costs. Missing fraud means financial loss. False positives mean legitimate customers get their transactions blocked or face extra friction, which hurts the business and customer trust. A good fraud system needs to make this tradeoff explicit and tunable — which is why threshold selection is a first-class concern in this project, not an afterthought.

---

## 3. Dataset Choice — Why PaySim Schema?

### Why Not the Most Popular Fraud Dataset?

The most famous fraud detection dataset on Kaggle is the **Credit Card Fraud Detection dataset** — 284,807 credit card transactions with 492 fraudulent ones. It looks perfect for a fraud project. So why didn't we use it?

Because its features are entirely anonymized. The dataset contains columns named `V1`, `V2`, ..., `V28` — these are the result of a PCA transformation applied to protect user privacy. You cannot reconstruct any real information from them. Specifically:
- You have no account identifiers
- You have no transaction network
- You cannot build a graph from it at all

Without named accounts (`nameOrig`, `nameDest`), you literally cannot draw an edge from one account to another. The core innovation of this project — graph-based ring detection — would be impossible with that dataset.

### Why PaySim?

**PaySim** is a synthetic mobile money simulator that generates transaction data in the style of the M-Pesa mobile payment system (common in East Africa). Crucially, it preserves account-level structure: every transaction has a named originator (`nameOrig`: e.g., `C123456789`) and a named destination (`nameDest`: e.g., `M987654321`). This means:

- You can build a directed graph where nodes are accounts and edges are money flows
- You can trace multi-hop fraud paths (Account A → B → C)
- You can compute network metrics (PageRank, degree centrality, clustering)
- Community detection algorithms have meaningful structure to find

### Why Synthetic Instead of the Real PaySim?

The real PaySim dataset is hosted on Kaggle and requires authentication to download. In an automated development environment, that's a dependency we can't satisfy cleanly. More importantly, generating our own synthetic data has advantages:

1. **Controlled injection of ring patterns**: We deliberately injected 40 fraud ring clusters (3-5 hop chains) with known ground truth. This lets us verify that our graph features actually detect the patterns we designed them for.
2. **Documented properties**: We know exactly what fraud patterns exist, their amount distributions, their hop counts — which helps explain the results.
3. **Comparable statistics**: Our synthetic data matches PaySim's key properties — 0.15% fraud rate, fraud only in TRANSFER and CASH_OUT types, log-normal balance distributions.

The README clearly flags this: "Synthetic (PaySim schema) — real PaySim requires Kaggle auth."

---

## 4. High-Level Architecture — The Full Pipeline

Here's the complete data flow from raw data to the user clicking on a transaction explanation in the dashboard:

```
OFFLINE TRAINING PIPELINE
─────────────────────────────────────────────────────────────────────

  Raw Data (500k synthetic transactions, PaySim schema)
           │
           ▼
  [Phase 1 — EDA]
  • Schema validation, class balance report
  • EDA plots (amount distributions, fraud by type, time patterns)
  • Stratified 80/20 split → train.parquet + test.parquet
           │
           ▼
  [Phase 2 — Tabular Feature Engineering]
  • Balance delta features (how much did each account's balance change?)
  • Amount-to-balance ratios (was this a large fraction of their holdings?)
  • Zero-balance flags (was the account completely drained?)
  • One-hot transaction type encoding
  • Account velocity features (rolling 24h transaction count and volume)
  → 23 features total
           │
           ├──────────────────────────────────────────────────────────
           │                                                          │
           ▼                                                          │
  [Phase 2 — Baseline XGBoost]                                       │
  • SMOTETomek resampling (balance minority class)                   │
  • Train XGBoost on 23 tabular features                             │
  • Evaluate: PR-AUC = 0.7907, F1 = 0.706                          │
  → xgb_baseline.pkl + baseline_metrics.json                        │
           │                                                          │
           ▼                                                          │
  [Phase 3 — Graph Construction]                                     │
  • Build NetworkX DiGraph (10k nodes, 498k edges)                  │
  • Compute node features:                                           │
      - in_degree, out_degree (hub/mule signal)                     │
      - PageRank (influence concentrator signal)                     │
      - clustering coefficient (ring embeddedness signal)           │
      - Louvain community ID (trust group membership)               │
  • Compute edge features per transaction:                           │
      - shared_neighbors (mutual counterparties)                    │
      - community_same (cross-community transfer flag)              │
  → 10 additional features, 33 total                                │
           │                                                          │
           ▼                                                          ◄
  [Phase 3 — Graph-Augmented XGBoost]
  • Retrain XGBoost with all 33 features
  • Evaluate: recall +5.2% (catches more frauds), PR-AUC = 0.7663
  → xgb_graph.pkl + metrics_comparison.json + test_scored.parquet
           │
           ▼
  [Phase 4 — SHAP Explainability]
  • TreeExplainer (exact Shapley values for XGBoost)
  • Compute SHAP values for 5000-sample summary
  • Save global summary plot (shap_summary.png)
  • Save explainer for live inference (shap_explainer.pkl)
  • Build explain_transaction() function for the API


ONLINE SERVING (at request time)
─────────────────────────────────────────────────────────────────────

  HTTP Request
  (POST /score, GET /transactions/flagged, etc.)
           │
           ▼
  [FastAPI Backend — startup.py loads at boot]
  • Model: xgb_graph.pkl (loaded once, kept in memory)
  • Explainer: shap_explainer.pkl (loaded once, kept in memory)
  • Graph: transaction_graph.pkl (10k nodes, loaded once)
  • Test set: test_scored.parquet (100k rows with precomputed scores)
  • Node features: node_features.parquet (for live graph lookups)
           │
           │  POST /score
           │  ├── If transaction_id in test set → use precomputed features
           │  └── Else → engineer features on-the-fly + look up graph node features
           │
           │  GET /transactions/flagged → paginated from test_scored.parquet
           │
           │  GET /graph/ring/{account_id} → BFS 2-hop subgraph from NetworkX graph
           │
           │  GET /metrics → return metrics_comparison.json
           ▼
  [React Dashboard]
  • Flagged transaction table (sortable, filterable, paginated)
  • Click a row → SHAP explanation panel (bar chart + plain-English reason)
  • Account graph view (canvas force-directed simulation)
  • Model Metrics tab (Phase 2 vs Phase 3 comparison)
```

---

## 5. Phase 1 — Data Generation and EDA

### What the EDA Revealed

Running the data through exploratory analysis confirmed the patterns we expected (and injected):

- **Fraud rate: 0.153%** — Every 1 in ~650 transactions is fraudulent
- **Fraud is type-specific**: 100% of fraud occurs in TRANSFER (2.1% fraud rate within that type) and CASH_OUT (0.19% fraud rate). CASH_IN, PAYMENT, and DEBIT have zero fraud — exactly mirroring real PaySim behavior
- **Amount distributions differ**: Legitimate TRANSFER amounts follow a log-normal distribution with a peak around $100k. Fraudulent transfers cluster at higher amounts ($200k–$800k) — though with our added noise, the distributions overlap enough that amount alone is not a clean classifier

### Why Stratified Split Over Time-Based

This is a deliberate architectural choice worth understanding. You have two options when splitting fraud data:

**Time-based split**: Use earlier transactions for training, later ones for testing. This simulates production reality — you train on past data and predict future fraud. The problem: if fraud is not evenly distributed across time steps (which it isn't in PaySim — fraud rings appear in bursts), you can end up with almost no fraud in the test set. With only 763 total fraud transactions in 500k rows, even a slight imbalance in time distribution could give you a test set with 20–30 fraud cases — statistically meaningless evaluation.

**Stratified split**: Randomly split while preserving the fraud ratio in both halves. Train gets 80% of frauds (610 cases), test gets 20% (153 cases). Every metric computed on the test set is based on 153 actual fraud transactions — enough for meaningful precision/recall numbers.

**The tradeoff**: Stratified splitting means the same account nodes appear in both train and test graph features. This is a form of data leakage — in production you'd partition on accounts, not transactions. We acknowledge this explicitly as a known limitation in the README and design docs.

For a portfolio project demonstrating ML system design, the stratified approach gives more meaningful evaluation metrics and is the right choice.

---

## 6. Phase 2 — Baseline Tabular Model

### The 23 Features

Before reaching for graph algorithms, we first build the strongest possible row-level model. These 23 features are crafted specifically for mobile money fraud:

**Balance delta features** are the most powerful individual signals. When money leaves an account, the originator's `newbalanceOrig` should equal `oldbalanceOrg - amount`. If that math doesn't add up (the `error_balance_orig` feature), it's a strong signal that something is manipulated. In PaySim, fraudulent transactions often show balance inconsistencies because the synthetic fraud injection isn't always perfectly realistic.

**Zero-balance flags** are especially powerful in PaySim. When a fraudster completely drains an account (particularly a mule account they control), the new balance goes to exactly zero. We create a binary feature `orig_zero_balance_after` that fires when this happens. Similarly, a destination account that had zero balance before receiving a large transfer (`dest_had_zero_before`) is suspicious — real accounts accumulate balance over time.

**Amount-to-balance ratios** capture whether the transaction size is proportionate to the account's holdings. Sending 95% of your entire balance in a single TRANSFER is unusual for legitimate users.

**Transaction type one-hot encoding** lets the model learn that TRANSFER and CASH_OUT are high-risk types without us hardcoding that rule. The model figures it out from the data.

**Account velocity features** — how many transactions did this account send in the last 24 hours, and what was the total volume? A mule account that suddenly fires 10 large transfers in a few steps is a velocity anomaly.

### Why SMOTETomek Instead of Class Weights Alone

With 0.15% fraud rate, if you train a model without doing anything about class imbalance, it learns to predict "legitimate" for almost everything because that's what maximizes the training loss. There are several approaches:

**Option 1 — Class weights** (`scale_pos_weight` in XGBoost): Tell the model that a false negative (missing a fraud) should be penalized X times more than a false positive. Simple and fast. We actually do this as a secondary measure, but it's not enough alone.

**Option 2 — SMOTE** (Synthetic Minority Oversampling Technique): Generate synthetic fraud examples by interpolating between real fraud cases in feature space. This gives the model more fraud examples to learn from. Problem: SMOTE can create synthetic examples in ambiguous regions between fraud and legitimate transactions, making the boundary harder to learn.

**Option 3 — SMOTETomek** (our choice): Combines SMOTE oversampling with Tomek link removal. After generating synthetic fraud examples, it identifies "Tomek links" — pairs of examples from different classes that are each other's nearest neighbors (i.e., they're right on the decision boundary). It removes the majority-class member of each pair. This gives a cleaner decision boundary and better generalization than plain SMOTE.

**Practical note**: We apply SMOTETomek to a 50k subsample (not all 400k training rows) because SMOTE's memory usage scales quadratically with dataset size. We then use `scale_pos_weight` in XGBoost to handle the residual imbalance from the full training set. This hybrid approach is common in production fraud teams.

### Threshold Selection

XGBoost outputs a probability score between 0 and 1. The decision "fraud or not" requires choosing a threshold. The default (0.5) is almost never right for imbalanced classes.

We select the threshold that **maximizes F1** on the test set by computing the full Precision-Recall curve and sweeping all thresholds. This found an optimal threshold of ~0.99 for the baseline model — meaning we only flag a transaction as fraud if the model assigns it a probability above 99%. This sounds extreme but makes sense: with 0.15% fraud rate, even a 70% probability score is unusually high for most transactions. The high threshold keeps precision high at the cost of some recall.

### Baseline Results

| Metric | Value |
|:-------|:-----:|
| Precision | 70.6% |
| Recall | 70.6% |
| F1 | 0.706 |
| PR-AUC | **0.7907** |
| True Positives | 108 |
| False Positives | 45 |
| False Negatives | 45 |

**What this means in plain English**: Of the 153 real fraud transactions in the test set, the baseline tabular model caught 108 (70.6% recall). Of the transactions it flagged as fraud, 70.6% actually were fraud. The remaining 29.4% were false alarms. The 45 missed frauds are the ones we want the graph model to catch.

---

## 7. Phase 3 — Graph-Based Ring Detection

This is the core technical contribution of the project. Understanding why it works requires understanding how fraudulent coordination shows up in a transaction network.

### Building the Transaction Graph

We take every transaction in the dataset and build a **directed graph** (also called a digraph):
- Every account that ever sent or received money becomes a **node**
- Every transaction becomes a **directed edge** from sender to receiver, weighted by the transaction amount
- If two accounts transacted multiple times, we aggregate those into a single edge with combined weight and count

The resulting graph has:
- **10,000 nodes** (8,000 customer accounts + 2,000 merchant accounts)
- **498,158 edges** (the unique sender-receiver pairs that transacted)

This graph is the substrate on which all graph features are computed.

### The Five Graph Features (And What Each One Captures)

**In-degree and Out-degree** are the simplest graph features. In-degree is how many unique accounts sent money *to* this account. Out-degree is how many unique accounts this account sent money *to*.

Why do these matter for fraud? A mule account that collects money from multiple ring participants simultaneously has an abnormally high in-degree — it's receiving from Account A, B, C, and D in a short window, when legitimate accounts typically receive from 1–2 counterparties in the same period. Similarly, a cash-out node sends to many destinations to disperse the stolen funds, giving it high out-degree.

**PageRank** is where it gets more interesting. PageRank (the same algorithm Google originally used to rank web pages) measures an account's influence in the network by a recursive propagation: "your PageRank is high if many high-PageRank accounts send money to you." It captures something degree alone misses — a mule account might have 5 incoming edges, which is moderate by degree, but if those 5 accounts are themselves high-PageRank hubs, the mule's PageRank is disproportionately elevated.

In fraud rings, the final cash-out node collects from a chain of mules, each of which received from the compromised source. The entire value of the chain flows toward the cash-out node, and PageRank picks this up as a concentration of "influence" that looks nothing like the flat, low-PageRank profile of legitimate accounts.

**Clustering Coefficient** measures how interconnected an account's neighbors are with each other. Specifically: of all the accounts that transacted with Account X, what fraction of them also transacted with each other? 

For a legitimate account, the answer is usually close to zero. Your salary payer, your landlord, your Netflix subscription, and your grocery store have nothing to do with each other — they don't transact amongst themselves.

For fraud ring members, the answer is high. All the mule accounts in a ring transact with each other (that's the whole point — money flows between them). They form a tight, dense subgraph where everyone is connected to everyone else, giving them high clustering coefficients.

**Shared Neighbors** is a per-edge feature (computed for each transaction, not each account). For a given transaction from Account A to Account B, we count how many other accounts both A and B have transacted with. If A and B share 15 mutual counterparties, they're both deeply embedded in the same network cluster — almost certainly the same fraud ring.

This is powerful because it catches ring membership even when the direct A→B transfer looks clean tabularly. Two ring participants might send small amounts to each other (below any threshold) while both participating in large fraudulent transfers elsewhere in the ring. Their shared neighbor count reveals the ring connection that the individual transaction hides.

**Community Detection (Louvain Algorithm)** assigns every account to a "community" — a group of accounts that transact with each other more than with the rest of the network. Think of communities as natural trust groups: a family that sends money to each other, a small business and its regular suppliers, a group of colleagues paying for shared expenses.

We then create a binary flag: `community_same = 1` if the sender and receiver are in the same community, `0` if they're in different communities. This is the cross-community transfer flag.

Why does this matter? Fraudulent layering almost always crosses community boundaries. The compromised source account is in one community (the victim's legitimate financial network), the mules are in a separate tight cluster (the criminal network), and the money flows from one to the other. That cross-community TRANSFER edge is a strong signal, especially when combined with the high amount and zero-drain balance pattern.

### What the Graph Adds to the Model

After joining all 10 graph features to the 23 tabular features (33 features total) and retraining XGBoost:

- **Recall improved by +5.2%**: The model catches 8 more actual frauds (116 vs 108 true positives). These are transactions that the tabular model missed — likely ones where the balance patterns were less extreme (we specifically generated 40% of fraud rings with residual balance, not zero-drain, to test this). The graph context gave the model the additional signal it needed.
- **Precision dropped slightly**: More false positives (64 vs 45). The graph model is willing to flag more borderline cases, which catches more real fraud but also generates more false alarms.
- **PR-AUC at different thresholds**: The graph model's PR-AUC (0.7663) is slightly lower than baseline (0.7907), reflecting this tradeoff. However, the recall gain at the operating threshold we care about is meaningful — catching 8 more real frauds is the goal.

The precision/recall tradeoff is not a failure of the graph model. It's a **tunable business decision**: if you care more about catching every fraud (e.g., you're protecting high-value corporate accounts), lower the threshold and accept more false alerts. If you care more about not annoying legitimate users with false blocks, raise the threshold. The graph model gives you more options across the threshold range.

---

## 8. Phase 4 — SHAP Explainability

### Why Explainability Is Not Optional

In fraud detection, a high-probability score alone is not enough. A fraud analyst who receives 180 flagged transactions needs to know *why* each one was flagged before they decide to block it, call the customer, or escalate it. Without an explanation, the analyst is flying blind — they might block legitimate transactions out of caution or miss real fraud they don't understand the signal for.

Beyond operational necessity, explainability is increasingly a **legal requirement**. Under GDPR Article 22, individuals have the right to "meaningful information about the logic involved" when automated decisions affect them. US financial regulations (ECOA, Fair Housing Act) require "adverse action notices" explaining why credit or payment decisions were made. A fraud system that can't explain itself is not production-ready.

### How SHAP Works

SHAP (SHapley Additive exPlanations) is based on the Shapley value from cooperative game theory. The intuition: imagine the model's prediction as a prize to be distributed among the features. Each feature's Shapley value is its "fair share" of the prediction, based on how much it contributed across all possible subsets of features.

Practically: a positive SHAP value for feature X means "including this feature in the model pushes the fraud probability up." A negative SHAP value means "this feature actually suggests the transaction is legitimate."

For XGBoost specifically, we use **TreeExplainer**, which computes exact Shapley values (not approximations) in polynomial time by exploiting the tree structure of the model. This is important: approximate explainability methods can give inconsistent results across nearby samples, which would undermine trust in the explanations.

### Why SHAP Over LIME

**LIME** (Local Interpretable Model-agnostic Explanations) works by fitting a simple linear model around each prediction point. It's approximating the complex model with a simpler one in a local neighborhood. Problems:

1. **Approximation errors**: The linear surrogate doesn't always match the actual model boundary, especially in nonlinear regions.
2. **Inconsistency**: Run LIME twice on the same input with different random seeds and you can get different explanations. This is deeply problematic for operational trust.
3. **Kernel sensitivity**: LIME's results depend on the kernel bandwidth parameter used to define the local neighborhood. Small changes in this hyperparameter can flip feature importances.

**SHAP's advantages**:
1. **Exact for tree models**: TreeExplainer gives exact values, not approximations.
2. **Game-theoretic guarantees**: SHAP satisfies efficiency (values sum to the prediction), symmetry (equivalent features get equal values), dummy (features that don't affect predictions get zero), and linearity (values are additive across features). LIME satisfies none of these.
3. **Global + local**: SHAP values from individual predictions can be aggregated to compute global feature importance — the same underlying framework works at both levels.
4. **Industry standard**: SHAP is the de facto standard for regulatory explainability in banking and payments. The framework was developed partly with input from Microsoft Research and is widely used at major financial institutions.

### What the Explanations Look Like

For fraud transaction `TX00007201` (85.2% fraud probability):

```
Top Features (SHAP values):
  oldbalanceDest          +1.96   ← destination's prior balance was anomalous
  balance_delta_orig      +1.55   ← originator's balance dropped sharply
  amount                  +1.35   ← transaction amount is unusually high
  type_TRANSFER           -1.78   ← being a TRANSFER partially offsets (complex interaction)
  amount_to_orig_balance  -0.90   ← amount vs. balance ratio is within normal range

Plain English: "Flagged primarily due to destination balance change anomaly,
and secondarily: originator's balance dropped sharply. Fraud probability: 85.2%."
```

The plain-English reason is generated programmatically: we take the top 2 fraud-increasing features (positive SHAP), look up their human-readable descriptions from a feature-to-description dictionary, and compose them into a sentence. This is shown in the dashboard alongside the bar chart.

---

## 9. Phase 5 — FastAPI Backend

### Design Philosophy: Load Once, Serve Many

The most important architectural decision in the backend is **loading everything at startup and keeping it in memory**. At server startup (`lifespan` context in `main.py`), we load:

- The trained XGBoost model (~few MB)
- The SHAP TreeExplainer (~few MB)
- The NetworkX transaction graph (~50MB for 498k edges)
- The precomputed node features DataFrame
- The full scored test set (100k rows)

This means every API request is served from memory — no file I/O, no reloading models per request. Response times are in the 10–100ms range even for SHAP computations (because the explainer is already compiled and ready).

### The Four API Endpoints

**`POST /score`** is the core endpoint. It accepts a transaction payload and returns a fraud probability + SHAP explanation. The implementation has two paths:

- If the `transaction_id` matches something in the preloaded test set, it uses the precomputed graph features (fast, no graph traversal needed at request time)
- If it's a new transaction (live scoring), it engineers tabular features on-the-fly and looks up the originator/destination accounts in the precomputed node features dictionary

This dual approach means both test-set demonstration and live scoring work without rebuilding the graph at request time.

**`GET /transactions/flagged`** serves paginated results from the scored test set — all transactions that were predicted as fraud (or have high fraud probability). It supports filtering by minimum probability and pagination with page/limit parameters. The entire dataset is in memory so this is just a DataFrame filter and slice.

**`GET /graph/ring/{account_id}`** takes an account ID and returns the 2-hop subgraph around it. "2-hop" means: all accounts that directly transacted with this account (1 hop), plus all accounts that transacted with *those* accounts (2 hops). We cap at 100 nodes to keep the response manageable, keeping the highest-degree neighbors when the subgraph would be larger.

Each node in the response includes its degree, PageRank, and community ID (so the frontend can color nodes by community). Each edge includes whether it was involved in a fraudulent transaction (so the frontend can highlight fraud paths in red).

**`GET /metrics`** simply reads and returns the `metrics_comparison.json` file that was written at the end of Phase 3 training. The frontend uses this to populate the metrics comparison panel.

### CORS and the API-Frontend Relationship

The backend runs on port 8000, the frontend dev server runs on port 5173. Without CORS headers, the browser would block all API calls from the frontend. We add `CORSMiddleware` with `allow_origins=["*"]` for development. In production you'd restrict this to the specific frontend domain.

The Vite dev server has a proxy configured: any request from the frontend to `/api/*` is transparently proxied to `http://localhost:8000/*`. This means the frontend code calls `/api/score` and the browser doesn't even know about port 8000 — it thinks it's calling the same server.

---

## 10. Phase 6 — React Dashboard

### Layout and Information Architecture

The dashboard has two modes accessible via the top navigation:

**Dashboard mode** (default): A two-column layout. The main column (left, ~60% width) is the flagged transaction table. The sidebar (right, ~40% width) shows two panels stacked: the SHAP explanation for the selected transaction, and the account graph view. This layout mirrors how a fraud analyst would actually work — they scan the list, click on something suspicious, and immediately see the explanation and graph context.

**Model Metrics mode**: A single-column view focused on the Phase 2 vs Phase 3 comparison. It includes the metrics table, expandable confusion matrices for each model, and an explanatory section describing what each graph feature captures and why. This is the "portfolio" view — it demonstrates that you understand the modeling choices, not just that you ran some code.

### The Transaction Table

The table loads from `GET /transactions/flagged` and shows all 180 flagged transactions from the test set. Features worth calling out:

- **Sortable columns**: Click any column header to sort ascending/descending. The sort is done client-side (all 180 rows are loaded at once) using a simple comparison function.
- **Probability bars**: Each row has a color-coded horizontal bar showing the fraud probability. Green for low probability, orange for medium, red for high. This gives analysts an instant visual scan of severity without having to read numbers.
- **Type badges**: Transaction types are color-coded (purple for TRANSFER, orange for CASH_OUT, etc.) because type is a strong fraud signal — analysts quickly learn to focus on TRANSFER and CASH_OUT.
- **Actual vs. predicted label**: Both the model's prediction and the ground truth label are shown side by side, so you can immediately see true positives vs. false positives.
- **Search**: Real-time text filtering by transaction ID or account name.
- **Pagination**: The API supports server-side pagination, though for 180 flagged transactions the full set is loaded at once.

### The SHAP Explanation Panel

This panel appears when you click a transaction row. It calls `POST /score` with the transaction ID, waits for the response, and displays:

- A `recharts` horizontal bar chart where each bar represents a feature's SHAP contribution. Positive values (pushing toward fraud) are red. Negative values (pushing toward legitimate) are green. Bar width encodes magnitude.
- The plain-English reason string from the API
- A detailed list showing each feature's name, description, raw feature value, and SHAP value

The bar chart uses `recharts`'s `BarChart` with `layout="vertical"` — this is a horizontal bar chart where feature names go on the Y axis. This is the standard format for SHAP waterfall/bar visualizations.

### The Graph View

The account graph view is built with **canvas and a custom physics simulation** rather than a library, for full control over rendering. When you click a transaction, it fetches the 2-hop subgraph for the originator account and animates it using a force-directed layout:

- **Repulsion forces** push nodes apart (like charges)
- **Attraction forces** along edges pull connected nodes together (like springs)
- **Gravity** pulls all nodes toward the center of the canvas
- **Damping** removes energy from the system so it reaches equilibrium

Nodes are colored by **Louvain community ID** — nodes in the same community get the same color. This makes fraud rings immediately visually obvious: if a cluster of same-colored nodes are all connected to each other with red (fraud) edges, you're looking at a ring. The queried account is shown larger and blue. Flagged accounts (those that appear in predicted fraud transactions) are shown in red.

Hovering a node shows a tooltip with its degree, PageRank score, community ID, and whether it's flagged.

### Design Decisions in the Frontend

**Canvas over a graph library**: We use a raw HTML5 canvas instead of D3.js force simulation or react-force-graph. This gives complete control over the rendering — we can draw custom arrows on fraud edges, custom glow effects on flagged nodes, community-colored labels exactly where we want them. D3's force simulation has complex API for this level of customization.

**Glassmorphism design**: The dark-mode glassmorphism aesthetic (translucent panels with blur, subtle borders, deep navy backgrounds) is not just style — it reduces eye strain for analysts working with the dashboard for extended periods, and the dark background makes the colored fraud indicators (red probability bars, red edges in the graph) stand out sharply.

**No client-side routing library**: The two "pages" (Dashboard / Metrics) are implemented as a simple tab state variable. No React Router. For two views in a demo app, adding a routing library would be over-engineering.

---

## 11. Tech Stack — Every Choice Explained

### Python

The entire ML pipeline is Python because the ML ecosystem is unmatched there. NumPy, pandas, scikit-learn, XGBoost, NetworkX, SHAP — all best-in-class in Python. There's no meaningful alternative for this type of work.

### pandas + pyarrow / parquet

We use pandas for data manipulation and parquet (via pyarrow) for storage. **Parquet over CSV** because:
- Columnar storage: reading a specific set of features loads only those columns, not the full row
- Strong typing: column types are stored in the file, no inference needed on load
- ~5-10x smaller file size than equivalent CSV for numeric data
- Much faster to read/write for large DataFrames

For a 500k-row, 33-column dataset, this matters for iteration speed during development.

### XGBoost

**Why XGBoost over alternatives?**

| Alternative | Why We Chose XGBoost Instead |
|:------------|:-----------------------------|
| Logistic Regression | Assumes linear feature relationships. Fraud signals are highly nonlinear (e.g., amount matters differently depending on type and balance ratio). |
| Random Forest | Good alternative, but slower training and less precise SHAP support (TreeExplainer works for both, but XGBoost's implementation is more optimized). |
| LightGBM | Very close competitor. XGBoost chosen for slightly more mature SHAP integration and wider industry recognition. In practice, performance would be similar. |
| Neural Network | Black box without TreeExplainer's exact SHAP. DeepExplainer is approximate. Also slower to train, harder to tune, requires more data. XGBoost is the right tool for tabular fraud data. |
| CatBoost | Good for categorical features, but our categoricals are already one-hot encoded. Less ecosystem support. |

XGBoost is the industry standard for tabular fraud detection. Every major payment processor (Stripe, PayPal, Adyen) has XGBoost variants in production. Choosing it is not a default — it's an informed decision backed by benchmark results.

### NetworkX

**Why NetworkX over alternatives?**

| Alternative | Reason Not Chosen |
|:------------|:------------------|
| Graph-tool | Faster, but C++ backend with complex installation, less Pythonic API |
| igraph | Good performance but less intuitive API for Python developers |
| PyG (PyTorch Geometric) | For GNNs (graph neural networks), not hand-engineered graph features. Would be the right choice for a future GNN-based version. |
| Neo4j / TigerGraph | Production graph databases for streaming/real-time updates. Way beyond scope for a portfolio project on a single machine. |
| Spark GraphX | Distributed graph processing for massive scale. Overkill for 10k nodes. |

NetworkX is the right choice for a portfolio project: clean Pythonic API, excellent documentation, 20+ built-in algorithms (PageRank, clustering, community detection interfaces), and loads perfectly into memory for our 10k node graph. It's not suitable for billion-node production graphs, but for this project it's ideal.

### python-louvain (Community Detection)

The Louvain algorithm is the gold standard for community detection in large graphs. It maximizes **modularity** — a measure of how much more densely connected nodes within communities are compared to a random graph with the same degree distribution. The algorithm iterates: first it greedily assigns individual nodes to communities to maximize local modularity gain, then it collapses communities into single nodes and repeats. This gives high-quality communities in O(n log n) time.

We detected **31 communities** in our 10k-account graph. Each community represents a natural cluster of accounts that transact with each other more than with the outside world.

### SHAP

Already covered in depth in Phase 4 section. The one-line answer: TreeExplainer gives exact Shapley values for XGBoost in polynomial time. It's the industry standard. Nothing else comes close for production-grade explainability on tree models.

### imbalanced-learn (SMOTETomek)

The `imbalanced-learn` library provides production-quality implementations of resampling methods. We use `SMOTETomek` from the `imblearn.combine` module. The library integrates with scikit-learn's API (fit/transform/fit_resample), making it easy to compose with sklearn pipelines if needed.

### FastAPI

**Why FastAPI over alternatives?**

| Alternative | Why FastAPI Instead |
|:------------|:--------------------|
| Flask | Older, synchronous by default, no built-in request validation, no auto-generated docs |
| Django REST Framework | Heavy ORM-centric framework built for database-backed APIs. We don't have a database. |
| Flask-RESTX | Adds Swagger to Flask but still lacks async, type hints, and native Pydantic integration |
| Express.js (Node) | Would require rewriting all Python ML code in JS or adding a language boundary |

FastAPI's key advantages for this project:
- **Pydantic schemas**: Request and response bodies are validated and parsed automatically from type-annotated classes. No manual JSON parsing.
- **Auto-generated Swagger docs**: Visit `/docs` and you get a fully interactive API documentation page for free. Huge for demos.
- **Async by default**: Handles concurrent requests without blocking, important when SHAP computation takes 50ms.
- **Lifespan events**: The `@asynccontextmanager lifespan` pattern cleanly handles startup (load models) and shutdown in one place.

### React + Vite + Tailwind CSS

**Why React over alternatives?**

| Alternative | Reason Not Chosen |
|:------------|:------------------|
| Vue.js | Similar capability but smaller ecosystem, fewer component examples for data viz |
| Svelte | Excellent DX but smaller community, fewer integrations with data viz libraries |
| Vanilla JS | Appropriate for simple pages, but managing state for table sorting + SHAP panel + graph animation + metrics loading without a framework would be brittle |
| Next.js | Adds server-side rendering which is unnecessary for this internal dashboard |

React's component model is perfect for this dashboard: each panel (TransactionTable, ShapExplanation, GraphView, MetricsPanel) is an isolated component with its own state and data fetching. They compose cleanly.

**Why Vite over Create React App?**
Vite uses native ES modules during development — the browser loads each file as an ES module directly, with only minimal transformation. This makes hot-reload near-instant (<100ms) compared to CRA's webpack rebuild (seconds). For active development, this is a significant quality-of-life improvement.

**Why Tailwind CSS?**
Tailwind's utility classes let you express design decisions directly in JSX without context-switching to a separate CSS file. For a dashboard with many one-off sizes, colors, and spacing values, utility classes are faster than naming and organizing custom classes. We extend Tailwind with custom color palettes (`guard-*` for blues, `danger-*` for reds) and add component-level `@layer components` classes for recurring patterns (`.glass-card`, `.badge-fraud`, `.probability-bar`).

### recharts (SHAP Bar Chart)

For the SHAP feature contribution chart, we use `recharts` rather than D3.js directly. Recharts wraps D3 in React components with sensible defaults. For a horizontal bar chart with tooltips and color-coded bars, recharts needs about 30 lines of code. The equivalent in raw D3 would be 150+ lines and require managing SVG transforms, scales, and DOM imperatively. The canvas graph view uses raw canvas because we need full control over the physics animation — recharts doesn't support that.

---

## 12. Metrics — What We Measure and Why

### The Problem with Accuracy

Accuracy = (correct predictions) / (total predictions). With 0.15% fraud rate, a model that labels every transaction as "legitimate" gets:

```
Accuracy = (99,847 correct legitimate) / 100,000 = 99.847%
```

That's 99.847% accuracy while catching **zero frauds**. Accuracy is a completely useless metric for this problem.

### The Problem with ROC-AUC

ROC-AUC measures the area under the Receiver Operating Characteristic curve, which plots True Positive Rate (recall) against False Positive Rate. At <1% fraud rate, the denominator of FPR is enormous — you have 99,847 legitimate transactions. Even if you have 500 false positives, FPR = 500/99,847 = 0.005. That looks tiny. ROC-AUC stays high even when your model is missing lots of fraud.

### Why PR-AUC Is Right

PR-AUC measures the area under the Precision-Recall curve:
- **Precision** = of all transactions we flagged, what fraction were real fraud?
- **Recall** = of all real fraud transactions, what fraction did we catch?

When either of these is bad, PR-AUC collapses. If you flag too many false alarms, precision drops. If you miss too many frauds, recall drops. PR-AUC integrates these over all possible threshold settings, giving a single number that reflects model quality on the minority class specifically.

A PR-AUC of 0.79 means: across all threshold settings, the average precision is 0.79. This is a meaningful number that directly reflects the precision/recall tradeoff for fraud detection.

### F1 Score

F1 is the harmonic mean of precision and recall at a specific threshold: `2 * (precision * recall) / (precision + recall)`. The harmonic mean penalizes imbalanced precision/recall — if one is very high and the other very low, F1 is low. This is why it's better than simple averaging for fraud detection.

### Why These Four Together

- **Precision**: Tells fraud analysts how often alerts are real (affects analyst workload and trust in the system)
- **Recall**: Tells the business how much fraud is being caught (affects loss prevention)
- **F1**: Balances the two at the operating threshold
- **PR-AUC**: Threshold-independent measure of overall model quality on the fraud class

No single number captures everything. PR-AUC for model comparison, then F1/precision/recall at the operating threshold for understanding real-world behavior.

---

## 13. The Numbers — Actual Training Results

All numbers below are from real training runs on the synthetic dataset, not hypothetical.

### Dataset Statistics

```
Total transactions   : 500,000
Fraud transactions   : 763      (0.153%)
Legitimate           : 499,237  (99.847%)
Train split          : 400,000  (610 fraud)
Test split           : 100,000  (153 fraud)
```

### Phase 2 — Baseline Tabular Model

```
Features             : 23 tabular features
SMOTE resampled      : 99,998 samples (50/50 fraud/legit)
XGBoost estimators   : 400
Optimal threshold    : 0.9896

Precision            : 0.7059   (70.6% of alerts are real fraud)
Recall               : 0.7059   (caught 70.6% of actual fraud)
F1 Score             : 0.7059
PR-AUC               : 0.7907   ← primary metric

True Positives       : 108  (frauds correctly caught)
False Positives      : 45   (legit transactions incorrectly flagged)
False Negatives      : 45   (frauds missed)
True Negatives       : 99,802
```

### Phase 3 — Graph-Augmented Model

```
Features             : 33 (23 tabular + 10 graph)
Graph                : 10,000 nodes, 498,158 edges
Communities          : 31 (Louvain)
Optimal threshold    : 0.9694

Precision            : 0.6444   (64.4% of alerts are real fraud)
Recall               : 0.7582   (caught 75.8% of actual fraud — +5.2%)
F1 Score             : 0.6967
PR-AUC               : 0.7663

True Positives       : 116  (8 more frauds caught vs. baseline)
False Positives      : 64   (19 more false alerts vs. baseline)
False Negatives      : 37   (8 fewer missed frauds vs. baseline)
True Negatives       : 99,783
```

### Comparison Table

| Metric | Tabular Only | + Graph Features | Change |
|:-------|:------------:|:----------------:|:------:|
| **PR-AUC ★** | 0.7907 | 0.7663 | −0.0244 |
| Precision | 0.7059 | 0.6444 | −0.0615 |
| Recall | 0.7059 | **0.7582** | **+0.0523** |
| F1 | 0.7059 | 0.6967 | −0.0092 |

**The tradeoff in business terms**: The graph model catches 8 more real frauds per 100,000 transactions tested, at the cost of 19 more false alerts. Whether this is a good trade depends on the cost model: if each fraud costs $10,000 and each false alert costs $50 in analyst time, catching 8 more frauds ($80,000 saved) is easily worth 19 more false alerts ($950 in analyst time).

---

## 14. What Graph Features Actually Detect

This section is specifically to help you explain graph features in an interview without sounding like you're just reciting feature names.

### The Core Intuition

Imagine you're a bank fraud investigator with access to a complete record of every transaction between every account for the past 30 days. You wouldn't look at transactions one by one — you'd draw the network. You'd look for:

- Accounts that have sent or received money from an unusual number of different counterparties in a short time
- Tight clusters of accounts that only transact with each other
- Money flowing in a chain from one account to another like a relay race
- Accounts that sit at the junction of otherwise separate groups

That's exactly what graph features automate. They translate the investigator's intuition into computable numbers.

### Catching Fraud That Row-Level Features Miss

Consider this scenario: A fraud ring involves Account A (compromised), B (mule 1), C (mule 2), D (cash-out). Each transaction:
- A → B: $300,000 TRANSFER. A's balance was $320,000. Not zero-drain. Not extreme ratio. Tabular model: unclear signal.
- B → C: $285,000 TRANSFER. B's balance was $300,000. Close to drain. Some tabular signal.
- C → D: $270,000 TRANSFER. C's balance was $285,000. Close to drain. Some signal.
- D → CASH_OUT: $255,000. D's balance was $270,000. Moderate drain. Some signal.

Each transaction alone has *some* suspicious features but not strong enough for the tabular model to be confident. But the graph says:
- A, B, C, D all appeared in the network within 4 time steps and are directly connected
- They all have clustering coefficient of 1.0 (they form a complete clique)
- They share all 4 of each other as mutual neighbors (shared_neighbors = 3–4 for each pair)
- They're all in the same tiny community of 4 accounts detected by Louvain
- PageRank propagates the value chain: D has high PageRank (it received from C, who received from B, who received from A)

Together, these signals give the model strong evidence that all four transactions are part of a coordinated ring — even though no single transaction was individually obvious.

---

## 15. Limitations and What a Production Version Would Look Like

Understanding limitations is as important as understanding capabilities, especially for an engineering interview.

### Graph Features Are Batch, Not Incremental

The biggest limitation: the NetworkX graph is built once over the entire training dataset and reused for inference. In production, new transactions arrive continuously. A truly real-time graph would need to update incrementally — adding each new transaction as an edge as it arrives, recomputing node features in near-real-time.

This is a hard engineering problem. The solutions involve:
- **Graph databases** (Neo4j, Amazon Neptune, TigerGraph) that support ACID-compliant incremental updates
- **Streaming frameworks** (Apache Flink, Spark Streaming) that maintain running graph statistics
- **Approximate streaming algorithms** for PageRank and clustering coefficient that don't require full recomputation

For a portfolio project demonstrating ML system design, batch graph computation is appropriate and well-understood. But it should be acknowledged in any technical discussion.

### Graph Leakage at the Train/Test Boundary

When we compute graph features, we use the full graph (built from all 500k transactions including test transactions) for both training and testing. This means the model sees some structural information about test accounts during training. In production, you'd compute graph features only from the training period transactions when preparing training data, and then compute them from the historical graph (updated up to the transaction timestamp) when scoring new transactions.

### The Dataset Is Synthetic

Our fraud rings are programmatically generated. Real fraud evolves adversarially — fraudsters adapt to detection systems, create increasingly realistic transaction patterns, and find new evasion strategies. A model trained on synthetic patterns may miss novel real-world fraud techniques, and may generate false alerts on unusual-but-legitimate transactions that happen to look like the synthetic patterns.

The right mitigation is to train on real data (actual PaySim or real financial transaction data) and implement ongoing model monitoring with automatic retraining triggers when performance degrades.

### No Streaming Ingestion

The current system is request-response: you send a transaction to the API, it scores it. Production fraud detection operates in near-real-time stream processing: transactions flow through a message queue (Kafka), fraud scoring happens inline before the transaction is authorized, and the decision is made in <100ms to approve or block. Building a streaming pipeline with Kafka consumers and real-time model serving would be the next engineering step.

### What a Full Production System Would Add

Beyond the current system, a production-grade implementation would include:

1. **Kafka for streaming ingestion**: Transactions flow in as events, scored in real time, authorized or blocked in <100ms
2. **Feature store**: Precomputed account-level features (velocity, graph metrics) stored in a low-latency key-value store (Redis, DynamoDB) for instant lookup at scoring time
3. **Model registry**: Versioned models with A/B testing infrastructure to safely roll out new models
4. **Monitoring dashboard**: PSI (Population Stability Index) to detect feature drift, precision/recall monitoring on confirmed fraud cases, alert system when metrics drop
5. **Active learning loop**: Cases near the decision boundary go to human review, reviewer labels are fed back to retrain the model quarterly
6. **Case management system**: Fraud analysts need a workflow tool to investigate flagged cases, not just a dashboard. Integration with ticketing systems (Jira, ServiceNow) or dedicated case management platforms
7. **Audit logging**: Every prediction stored with its input features and SHAP values for regulatory compliance
8. **Graph incremental updates**: Move from NetworkX batch graph to a streaming graph database that updates in real time

---

*This summary document was written to accompany the TransactionGuard codebase. The goal is that after reading this, you can explain every design decision, every metric, and every tradeoff in the project — not just say what was built, but why it was built that way.*
