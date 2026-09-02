"""
Synthetic transaction data generator with PaySim schema.

Generates a dataset that mirrors the PaySim dataset structure:
  - step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
    nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

Key design decisions:
  - Fraud only occurs in TRANSFER and CASH_OUT transactions (same as PaySim).
  - ~0.13% base fraud rate to match PaySim.
  - 40 injected "fraud ring" clusters: chains of 3-5 accounts that funnel
    money through intermediate hops before cashing out. These rings represent
    a layering/mule pattern that row-level features alone will miss.
  - Ring transactions have distinct balance-draining signatures.
"""

import numpy as np
import pandas as pd
import random
import string
import argparse
from pathlib import Path

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Account name generation
# ---------------------------------------------------------------------------

def _make_account_id(prefix: str, index: int) -> str:
    """Generate an account ID like 'C123456789' or 'M987654321'."""
    suffix = str(index).zfill(9)
    return f"{prefix}{suffix}"


def generate_accounts(n_customers: int = 8_000, n_merchants: int = 2_000):
    """Return lists of customer and merchant account IDs."""
    customers = [_make_account_id("C", i) for i in range(1, n_customers + 1)]
    merchants = [_make_account_id("M", i) for i in range(1, n_merchants + 1)]
    return customers, merchants


# ---------------------------------------------------------------------------
# Normal (non-fraud) transaction generation
# ---------------------------------------------------------------------------

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
TYPE_WEIGHTS      = [0.23,       0.35,       0.04,    0.34,      0.04]

AMOUNT_PARAMS = {
    "CASH_IN":   dict(mean=100_000,  std=180_000),
    "CASH_OUT":  dict(mean=150_000,  std=200_000),
    "DEBIT":     dict(mean=50_000,   std=100_000),
    "PAYMENT":   dict(mean=10_000,   std=20_000),
    "TRANSFER":  dict(mean=180_000,  std=250_000),
}


def _sample_amount(tx_type: str) -> float:
    p = AMOUNT_PARAMS[tx_type]
    amount = abs(np.random.normal(p["mean"], p["std"]))
    return round(max(1.0, amount), 2)


def _make_normal_tx(
    step: int,
    tx_type: str,
    orig: str,
    dest: str,
    orig_balance: float,
    dest_balance: float,
) -> dict:
    amount = _sample_amount(tx_type)
    # Clamp so accounts don't go wildly negative
    amount = min(amount, orig_balance + 1.0)
    new_orig = max(0.0, round(orig_balance - amount, 2))
    new_dest = round(dest_balance + amount, 2)
    return {
        "step": step,
        "type": tx_type,
        "amount": amount,
        "nameOrig": orig,
        "oldbalanceOrg": orig_balance,
        "newbalanceOrig": new_orig,
        "nameDest": dest,
        "oldbalanceDest": dest_balance,
        "newbalanceDest": new_dest,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }


# ---------------------------------------------------------------------------
# Fraud ring generation
# ---------------------------------------------------------------------------
# A fraud ring works as follows:
#   1. A "source" account receives money (CASH_IN or already funded).
#   2. Money is layered through 3-5 intermediate "mule" accounts via TRANSFER.
#   3. Final account cashes out via CASH_OUT.
#
# This creates a hub-and-spoke signature in the graph that the Louvain
# community detector will pick up as cross-community transfers.


def generate_fraud_rings(
    customers: list,
    step_range: tuple,
    n_rings: int = 40,
    min_hops: int = 3,
    max_hops: int = 5,
) -> list[dict]:
    """Generate fraudulent ring-pattern transactions with realistic noise."""
    ring_txs = []
    used = set()

    for ring_id in range(n_rings):
        n_hops = random.randint(min_hops, max_hops)
        candidates = [c for c in customers if c not in used]
        if len(candidates) < n_hops + 1:
            candidates = customers

        ring_accounts = random.sample(candidates, n_hops + 1)
        used.update(ring_accounts)

        base_amount = round(random.uniform(200_000, 800_000), 2)
        step = random.randint(*step_range)

        # Add some legitimate-looking balance (not always zero-drain)
        orig_extra_balance = round(random.uniform(0, 50_000), 2)

        for hop in range(n_hops):
            orig = ring_accounts[hop]
            dest = ring_accounts[hop + 1]
            amount = round(base_amount * random.uniform(0.80, 1.0), 2)
            tx_type = "TRANSFER" if hop < n_hops - 1 else "CASH_OUT"

            # 40% of ring txs keep some residual balance (harder to detect tabularly)
            residual = round(random.uniform(0, 30_000), 2) if random.random() < 0.4 else 0.0
            orig_balance = amount + residual + orig_extra_balance
            new_orig_balance = residual  # not always zero

            ring_txs.append({
                "step": step + hop,
                "type": tx_type,
                "amount": amount,
                "nameOrig": orig,
                "oldbalanceOrg": round(orig_balance, 2),
                "newbalanceOrig": round(new_orig_balance, 2),
                "nameDest": dest,
                "oldbalanceDest": round(random.uniform(0, 20_000), 2),
                "newbalanceDest": round(amount + random.uniform(0, 20_000), 2),
                "isFraud": 1,
                "isFlaggedFraud": 0,
            })

    return ring_txs


# ---------------------------------------------------------------------------
# Isolated fraud (non-ring) — single-hop TRANSFER/CASH_OUT frauds
# ---------------------------------------------------------------------------

def generate_isolated_fraud(
    customers: list,
    step_range: tuple,
    n_frauds: int = 600,
) -> list[dict]:
    """Generate isolated (non-ring) fraudulent transactions with varied balance patterns."""
    fraud_txs = []
    for _ in range(n_frauds):
        orig, dest = random.sample(customers, 2)
        tx_type = random.choice(["TRANSFER", "CASH_OUT"])
        amount = round(random.uniform(50_000, 500_000), 2)
        step = random.randint(*step_range)

        # 60% zero-drain, 40% partial drain (harder to catch with tabular features alone)
        if random.random() < 0.6:
            orig_old = round(amount * random.uniform(1.0, 1.3), 2)
            orig_new = round(orig_old - amount, 2)
        else:
            orig_old = round(amount + random.uniform(10_000, 100_000), 2)
            orig_new = round(orig_old - amount, 2)

        dest_old = round(random.uniform(0, 50_000), 2)
        dest_new = round(dest_old + amount, 2)

        fraud_txs.append({
            "step": step,
            "type": tx_type,
            "amount": amount,
            "nameOrig": orig,
            "oldbalanceOrg": orig_old,
            "newbalanceOrig": orig_new,
            "nameDest": dest,
            "oldbalanceDest": dest_old,
            "newbalanceDest": dest_new,
            "isFraud": 1,
            "isFlaggedFraud": 0,
        })
    return fraud_txs


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_dataset(n_transactions: int = 500_000, output_path: str | None = None) -> pd.DataFrame:
    """
    Generate a synthetic PaySim-schema dataset.

    Parameters
    ----------
    n_transactions : int
        Total number of transactions (fraud + non-fraud).
    output_path : str or None
        If provided, save the raw CSV here.

    Returns
    -------
    pd.DataFrame
    """
    print(f"[generate] Generating {n_transactions:,} transactions...")

    customers, merchants = generate_accounts()
    all_accounts = customers + merchants

    # Starting balances: log-normal, matching PaySim
    balance_map = {
        acc: round(abs(np.random.lognormal(mean=11.0, sigma=2.0)), 2)
        for acc in all_accounts
    }

    # --- Generate fraud transactions first ---
    step_range = (1, 743)  # PaySim uses 743 steps (1 step = 1 hour, 31 days)

    ring_txs = generate_fraud_rings(customers, step_range, n_rings=40)
    isolated_fraud_txs = generate_isolated_fraud(customers, step_range, n_frauds=600)
    fraud_txs = ring_txs + isolated_fraud_txs
    n_fraud = len(fraud_txs)

    # --- Generate normal transactions ---
    n_normal = n_transactions - n_fraud
    normal_txs = []

    for i in range(n_normal):
        tx_type = np.random.choice(TRANSACTION_TYPES, p=TYPE_WEIGHTS)
        step = np.random.randint(1, 744)
        orig = random.choice(customers)
        # Merchants only appear as destination for PAYMENT/DEBIT
        if tx_type in ("PAYMENT", "DEBIT"):
            dest = random.choice(merchants)
        else:
            dest = random.choice(customers)
            while dest == orig:
                dest = random.choice(customers)

        orig_bal = balance_map.get(orig, 10_000.0)
        dest_bal = balance_map.get(dest, 10_000.0)

        tx = _make_normal_tx(step, tx_type, orig, dest, orig_bal, dest_bal)
        # Update balances for next time this account appears
        balance_map[orig] = tx["newbalanceOrig"]
        balance_map[dest] = tx["newbalanceDest"]
        normal_txs.append(tx)

        if i % 100_000 == 0 and i > 0:
            print(f"  [generate] {i:,} normal transactions generated...")

    # --- Combine and sort by step ---
    all_txs = normal_txs + fraud_txs
    df = pd.DataFrame(all_txs)
    df = df.sort_values("step").reset_index(drop=True)

    # Add a transaction ID
    df.insert(0, "transaction_id", [f"TX{str(i).zfill(8)}" for i in range(len(df))])

    # --- Report ---
    n_total = len(df)
    n_fraud_actual = df["isFraud"].sum()
    fraud_pct = n_fraud_actual / n_total * 100
    print(f"[generate] Done. Total: {n_total:,} | Fraud: {n_fraud_actual:,} ({fraud_pct:.3f}%)")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[generate] Saved to {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic TransactionGuard dataset")
    parser.add_argument("--n", type=int, default=500_000, help="Number of transactions")
    parser.add_argument("--output", type=str, default="data/raw/transactions.csv")
    args = parser.parse_args()
    generate_dataset(n_transactions=args.n, output_path=args.output)
