"""
GET /graph/ring/{account_id} — Return local transaction subgraph around an account.
Returns nodes and edges within 2 hops, for force-directed visualization.
"""

from fastapi import APIRouter, HTTPException
import networkx as nx

from backend.models import GraphResponse, GraphNode, GraphEdge
from backend.startup import state

router = APIRouter()

MAX_HOPS   = 2
MAX_NODES  = 100   # cap to keep response manageable


@router.get("/graph/ring/{account_id}", response_model=GraphResponse, tags=["Graph"])
async def get_account_subgraph(account_id: str):
    """
    Return the local transaction subgraph within 2 hops of an account.
    Nodes are enriched with degree, PageRank, and community membership.
    Edges are flagged if they involve a fraudulent transaction.
    """
    G = state.graph
    if account_id not in G:
        raise HTTPException(
            status_code=404,
            detail=f"Account '{account_id}' not found in transaction graph."
        )

    # BFS to collect nodes within MAX_HOPS
    G_und = G.to_undirected()
    subgraph_nodes = set()
    subgraph_nodes.add(account_id)

    frontier = {account_id}
    for hop in range(MAX_HOPS):
        next_frontier = set()
        for node in frontier:
            neighbors = set(G_und.neighbors(node))
            new_nodes = neighbors - subgraph_nodes
            next_frontier.update(new_nodes)
            subgraph_nodes.update(new_nodes)
            if len(subgraph_nodes) >= MAX_NODES:
                break
        frontier = next_frontier
        if len(subgraph_nodes) >= MAX_NODES:
            break

    # Trim to MAX_NODES (keep account_id plus highest-degree neighbors)
    if len(subgraph_nodes) > MAX_NODES:
        degrees = dict(G_und.degree(subgraph_nodes))
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)[:MAX_NODES]
        subgraph_nodes = set(sorted_nodes)
        subgraph_nodes.add(account_id)  # always include the queried account

    # Build subgraph
    sub = G.subgraph(subgraph_nodes)
    nf  = state.node_features_idx

    # Determine which accounts are flagged (appear in test set as fraud_pred=1)
    flagged_accounts: set[str] = set()
    if state.test_df is not None and "fraud_pred" in state.test_df.columns:
        flagged_mask = state.test_df["fraud_pred"] == 1
        flagged_accounts.update(state.test_df[flagged_mask]["nameOrig"].tolist())
        flagged_accounts.update(state.test_df[flagged_mask]["nameDest"].tolist())

    # Build node objects
    nodes = []
    for node_id in sub.nodes():
        ndata = nf.get(node_id, {})
        nodes.append(GraphNode(
            id           = node_id,
            is_flagged   = node_id in flagged_accounts,
            degree       = int(G_und.degree(node_id)) if node_id in G_und else 0,
            pagerank     = round(float(ndata.get("pagerank", 0.0)), 6),
            community_id = int(ndata.get("community_id", -1)),
        ))

    # Build edge objects
    edges = []
    fraud_edges: set[tuple] = set()
    if state.test_df is not None:
        fraud_txs = state.test_df[state.test_df["isFraud"] == 1]
        for _, row in fraud_txs.iterrows():
            fraud_edges.add((row["nameOrig"], row["nameDest"]))

    for src, dst, edata in sub.edges(data=True):
        edges.append(GraphEdge(
            source  = src,
            target  = dst,
            weight  = round(float(edata.get("weight", 0.0)), 2),
            is_fraud = (src, dst) in fraud_edges,
        ))

    return GraphResponse(
        account_id = account_id,
        nodes      = nodes,
        edges      = edges,
    )
