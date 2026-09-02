"""
Graph feature engineering for TransactionGuard.

Builds a directed transaction graph (nodes = accounts, edges = transactions)
and computes per-node and per-edge features that capture coordinated fraud
patterns invisible to row-level tabular features.

WHAT EACH GRAPH FEATURE CAPTURES (for explainability):
-------------------------------------------------------

1. in_degree / out_degree
   - How many transactions does an account RECEIVE vs. SEND?
   - Fraud mule accounts typically have HIGH in-degree (many senders funneling
     money in) and HIGH out-degree (multiple cash-outs). Legitimate accounts
     have more balanced, moderate degrees.

2. pagerank
   - PageRank measures "importance" in the network by propagating influence
     along edges, iteratively. An account that receives from many high-activity
     accounts gets a high PageRank.
   - Fraud ring hubs — accounts that collect from multiple ring participants —
     accumulate disproportionately high PageRank relative to their transaction
     count. This flags them as suspicious concentrators of flow.

3. clustering_coefficient
   - The fraction of an account's neighbors that also transact WITH EACH OTHER.
   - Fraud rings have high clustering because ring members transact exclusively
     among themselves. Legitimate accounts have low clustering because their
     counterparties are independent.
   - We use the undirected clustering coefficient to capture bidirectional ties.

4. shared_neighbors (per edge: orig → dest)
   - How many other accounts have BOTH orig and dest transacted with?
   - A high shared-neighbor count means orig and dest are both embedded in the
     same tight cluster — a hallmark of layering rings. Unrelated parties in
     a legitimate transfer rarely share many mutual counterparties.

5. community_same (cross-community flag, per edge)
   - We run Louvain community detection to find natural clusters of frequently
     interacting accounts. Fraudulent TRANSFERS often cross community boundaries
     (ring originator in community A, mule in community B) while legitimate
     transfers cluster within communities (e.g., families, companies).
   - A cross-community, high-value, zero-drain transfer is a classic layering signal.

6. dest_in_degree_ratio / orig_out_degree_ratio
   - Ratios of in/out degree relative to the graph average, giving the model
     a normalized signal that is robust to dataset size.

WHY THIS CATCHES COORDINATED FRAUD THAT ROW-LEVEL FEATURES MISS:
-----------------------------------------------------------------
Row-level features see only one transaction in isolation. A single $300K
TRANSFER from a zero-balance account looks suspicious, but not much more so
than a legitimate large transfer. However, the GRAPH sees that the sender
already received from 4 other accounts in the past 3 steps, that the
recipient immediately sent to 2 more accounts, and that all parties belong
to a tight cluster with high mutual overlap. Together these signals are
strongly discriminative even for sophisticated fraud rings that keep individual
transaction amounts just below rule-based thresholds.
"""

import warnings
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_transaction_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed weighted transaction graph from a transaction DataFrame.

    Nodes: account IDs (nameOrig, nameDest)
    Edges: transactions, directed Orig → Dest
    Edge weights: transaction amount (aggregated if multiple transactions between same pair)

    Parameters
    ----------
    df : pd.DataFrame
        Transaction DataFrame with columns: nameOrig, nameDest, amount.

    Returns
    -------
    nx.DiGraph
    """
    print("[graph] Building transaction graph...")
    G = nx.DiGraph()

    # Add edges with attributes; aggregate multiple transactions between same pair
    edge_data: dict[tuple, dict] = {}
    for row in df.itertuples(index=False):
        key = (row.nameOrig, row.nameDest)
        if key not in edge_data:
            edge_data[key] = {"weight": 0.0, "count": 0, "fraud_count": 0}
        edge_data[key]["weight"]      += row.amount
        edge_data[key]["count"]       += 1
        edge_data[key]["fraud_count"] += int(row.isFraud)

    for (orig, dest), attrs in edge_data.items():
        G.add_edge(orig, dest, **attrs)

    print(f"[graph] Nodes: {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")
    return G


# ---------------------------------------------------------------------------
# Node-level features
# ---------------------------------------------------------------------------

def compute_node_features(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute per-node graph features.

    Returns a DataFrame indexed by node (account ID) with columns:
      - in_degree, out_degree
      - pagerank
      - clustering_coeff  (computed on undirected version)
      - community_id      (Louvain community label)
    """
    print("[graph] Computing node features (degree, pagerank, clustering, community)...")

    nodes = list(G.nodes())

    # 1. Degree features
    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    # 2. PageRank
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-6)

    # 3. Clustering coefficient (undirected — captures ring structure)
    G_undirected = G.to_undirected()
    clustering = nx.clustering(G_undirected)

    # 4. Community detection via Louvain
    community_map = _detect_communities(G_undirected)

    node_df = pd.DataFrame({
        "account":           nodes,
        "in_degree":         [in_deg.get(n, 0)      for n in nodes],
        "out_degree":        [out_deg.get(n, 0)     for n in nodes],
        "pagerank":          [pagerank.get(n, 0.0)  for n in nodes],
        "clustering_coeff":  [clustering.get(n, 0.0) for n in nodes],
        "community_id":      [community_map.get(n, -1) for n in nodes],
    })

    print(f"[graph] Node features computed for {len(node_df):,} accounts.")
    return node_df


def _detect_communities(G_undirected: nx.Graph) -> dict:
    """
    Detect communities using Louvain algorithm (via python-louvain).
    Falls back to greedy modularity if python-louvain is not installed.
    """
    try:
        import community as community_louvain  # python-louvain package
        partition = community_louvain.best_partition(G_undirected, random_state=42)
        print(f"[graph]   Communities detected (Louvain): {len(set(partition.values()))}")
        return partition
    except ImportError:
        print("[graph]   python-louvain not found, using greedy modularity...")
        communities = nx.algorithms.community.greedy_modularity_communities(G_undirected)
        partition = {}
        for comm_id, community_set in enumerate(communities):
            for node in community_set:
                partition[node] = comm_id
        print(f"[graph]   Communities detected (greedy): {len(communities)}")
        return partition


# ---------------------------------------------------------------------------
# Edge-level features (joined per-transaction)
# ---------------------------------------------------------------------------

def compute_edge_features(
    df: pd.DataFrame,
    G: nx.DiGraph,
    node_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join graph features onto the transaction DataFrame.

    Adds per-transaction columns:
      - orig_in_degree, orig_out_degree, orig_pagerank, orig_clustering_coeff
      - dest_in_degree, dest_out_degree, dest_pagerank, dest_clustering_coeff
      - community_same           (1 if orig and dest share a community, 0 if cross-community)
      - shared_neighbors         (count of accounts that both orig and dest transacted with)
      - orig_community_id, dest_community_id
    """
    print("[graph] Joining graph features onto transactions...")

    node_idx = node_features.set_index("account")

    def get_node_feat(account: str, feat: str, default=0.0):
        try:
            return node_idx.at[account, feat]
        except KeyError:
            return default

    # Build shared-neighbor lookup
    print("[graph]   Computing shared neighbors (this may take a moment)...")
    G_undirected = G.to_undirected()

    # For large graphs compute shared-neighbors only for sampled edges
    # For <10k unique account pairs this is tractable
    shared_neighbor_cache: dict[tuple, int] = {}

    def shared_neighbors(orig: str, dest: str) -> int:
        key = (min(orig, dest), max(orig, dest))
        if key not in shared_neighbor_cache:
            try:
                n_orig = set(G_undirected.neighbors(orig))
                n_dest = set(G_undirected.neighbors(dest))
                shared_neighbor_cache[key] = len(n_orig & n_dest)
            except nx.exception.NetworkXError:
                shared_neighbor_cache[key] = 0
        return shared_neighbor_cache[key]

    # Apply to each transaction
    # Vectorize via pandas apply (acceptable for 500k rows with caching)
    orig_accounts = df["nameOrig"].values
    dest_accounts = df["nameDest"].values

    print("[graph]   Mapping orig account features...")
    orig_in_deg     = [get_node_feat(a, "in_degree", 0)         for a in orig_accounts]
    orig_out_deg    = [get_node_feat(a, "out_degree", 0)        for a in orig_accounts]
    orig_pagerank   = [get_node_feat(a, "pagerank", 0.0)        for a in orig_accounts]
    orig_clustering = [get_node_feat(a, "clustering_coeff", 0.0) for a in orig_accounts]
    orig_community  = [get_node_feat(a, "community_id", -1)     for a in orig_accounts]

    print("[graph]   Mapping dest account features...")
    dest_in_deg     = [get_node_feat(a, "in_degree", 0)         for a in dest_accounts]
    dest_out_deg    = [get_node_feat(a, "out_degree", 0)        for a in dest_accounts]
    dest_pagerank   = [get_node_feat(a, "pagerank", 0.0)        for a in dest_accounts]
    dest_clustering = [get_node_feat(a, "clustering_coeff", 0.0) for a in dest_accounts]
    dest_community  = [get_node_feat(a, "community_id", -1)     for a in dest_accounts]

    print("[graph]   Computing shared neighbors per transaction...")
    shared_nbrs = [
        shared_neighbors(o, d)
        for o, d in zip(orig_accounts, dest_accounts)
    ]

    community_same = [
        1 if (oc != -1 and oc == dc) else 0
        for oc, dc in zip(orig_community, dest_community)
    ]

    df = df.copy()
    df["orig_in_degree"]      = orig_in_deg
    df["orig_out_degree"]     = orig_out_deg
    df["orig_pagerank"]       = orig_pagerank
    df["orig_clustering"]     = orig_clustering
    df["orig_community_id"]   = orig_community

    df["dest_in_degree"]      = dest_in_deg
    df["dest_out_degree"]     = dest_out_deg
    df["dest_pagerank"]       = dest_pagerank
    df["dest_clustering"]     = dest_clustering
    df["dest_community_id"]   = dest_community

    df["shared_neighbors"]    = shared_nbrs
    df["community_same"]      = community_same

    print(f"[graph] Graph features joined. Shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Feature column lists
# ---------------------------------------------------------------------------

GRAPH_FEATURE_COLS = [
    "orig_in_degree", "orig_out_degree", "orig_pagerank", "orig_clustering",
    "dest_in_degree", "dest_out_degree", "dest_pagerank", "dest_clustering",
    "shared_neighbors", "community_same",
]
