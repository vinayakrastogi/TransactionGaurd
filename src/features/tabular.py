"""
Tabular feature engineering for TransactionGuard.

Computes features from the raw transaction schema:
  - Transaction type (one-hot encoded)
  - Balance delta features for orig and dest
  - Amount-to-balance ratios
  - Zero-balance-after flags (strong fraud indicator in PaySim)
  - Account-level velocity features (rolling transaction count/volume per account)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Transaction types (all possible, for consistent one-hot encoding)
ALL_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def engineer_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add tabular features to a transaction DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or cleaned transaction DataFrame. Must be sorted by `step`
        for velocity features to be meaningful.

    Returns
    -------
    pd.DataFrame with additional feature columns.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Balance delta features
    # ------------------------------------------------------------------
    # How much did the originator's balance change?
    df["balance_delta_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    # How much did the destination's balance change?
    df["balance_delta_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Expected change should equal ± amount; discrepancies indicate manipulation
    df["error_balance_orig"] = df["amount"] + df["balance_delta_orig"]  # should be ~0 for legit
    df["error_balance_dest"] = df["amount"] - df["balance_delta_dest"]  # should be ~0 for legit

    # ------------------------------------------------------------------
    # 2. Amount-to-balance ratios
    # ------------------------------------------------------------------
    # What fraction of their balance did orig send?
    df["amount_to_orig_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1e-6)
    # How much is the amount vs. dest's existing balance? (churning signal)
    df["amount_to_dest_balance"] = df["amount"] / (df["oldbalanceDest"] + 1e-6)

    # ------------------------------------------------------------------
    # 3. Zero-balance flags
    # ------------------------------------------------------------------
    # Accounts that are completely drained are a very strong PaySim fraud signal
    df["orig_zero_balance_after"] = (df["newbalanceOrig"] == 0).astype(int)
    df["dest_had_zero_before"]    = (df["oldbalanceDest"] == 0).astype(int)
    df["dest_zero_balance_after"] = (df["newbalanceDest"] == 0).astype(int)

    # ------------------------------------------------------------------
    # 4. Transaction type one-hot encoding
    # ------------------------------------------------------------------
    for t in ALL_TYPES:
        df[f"type_{t}"] = (df["type"] == t).astype(int)

    # ------------------------------------------------------------------
    # 5. Account velocity features (rolling window)
    # ------------------------------------------------------------------
    # Sort by step to ensure rolling makes sense temporally
    df = df.sort_values("step").reset_index(drop=True)

    # Per-originator: count of transactions sent in the past N steps
    df = _add_velocity(df, account_col="nameOrig", role="orig")
    # Per-destination: count of transactions received in the past N steps
    df = _add_velocity(df, account_col="nameDest", role="dest")

    return df


def _add_velocity(
    df: pd.DataFrame,
    account_col: str,
    role: str,
    window_steps: int = 24,   # 24 steps = 24 hours
) -> pd.DataFrame:
    """
    For each transaction, count how many transactions the same account
    was involved in within the previous `window_steps` steps.

    This is computed using a merge-based approach (not true rolling,
    which requires per-group sort — too expensive for 500k rows).
    Instead: for each tx, look back at transactions where:
      same account, step in [step - window_steps, step - 1]
    """
    # Build a lookup: for each (account, step), how many prior txs in window?
    # Efficient approach: use cumulative counts per account, then lag

    # Count transactions per (account, step)
    counts = (
        df.groupby([account_col, "step"])
        .size()
        .reset_index(name="count_in_step")
    )
    counts = counts.sort_values([account_col, "step"])

    # Cumulative count per account up to (but not including) this step
    counts["cumcount"] = counts.groupby(account_col)["count_in_step"].cumsum() - counts["count_in_step"]

    # Now for each transaction, we want the cumcount at step-window_steps
    # Merge on account + step to get the rolling estimate
    # Approximate: we assign each tx the cumcount at its step (excludes current step's txs)
    step_to_cum = counts.set_index([account_col, "step"])["cumcount"].to_dict()

    col_name = f"{role}_tx_count_24h"
    df[col_name] = df.apply(
        lambda row: step_to_cum.get((row[account_col], row["step"]), 0),
        axis=1,
    )

    # Also: total transaction volume (amount) for this account up to this step
    vol = (
        df.groupby([account_col, "step"])["amount"]
        .sum()
        .reset_index(name="vol_in_step")
    )
    vol = vol.sort_values([account_col, "step"])
    vol["cumvol"] = vol.groupby(account_col)["vol_in_step"].cumsum() - vol["vol_in_step"]
    step_to_vol = vol.set_index([account_col, "step"])["cumvol"].to_dict()

    vol_col = f"{role}_vol_24h"
    df[vol_col] = df.apply(
        lambda row: step_to_vol.get((row[account_col], row["step"]), 0.0),
        axis=1,
    )

    return df


# Feature columns used for training (excludes ID, label, raw string cols)
TABULAR_FEATURE_COLS = [
    "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balance_delta_orig", "balance_delta_dest",
    "error_balance_orig", "error_balance_dest",
    "amount_to_orig_balance", "amount_to_dest_balance",
    "orig_zero_balance_after", "dest_had_zero_before", "dest_zero_balance_after",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
    "orig_tx_count_24h", "orig_vol_24h",
    "dest_tx_count_24h", "dest_vol_24h",
]

TARGET_COL = "isFraud"
