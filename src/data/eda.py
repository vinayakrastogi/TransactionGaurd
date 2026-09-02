"""
Phase 1 — Exploratory Data Analysis

Loads the raw transaction dataset (generating it if absent), reports:
  - Schema and data types
  - Class balance (fraud vs. non-fraud ratio)
  - Transaction amount distributions by type and fraud label
  - Time (step) patterns
  - Train/test stratified split

Saves:
  - data/processed/transactions_clean.parquet  (full cleaned dataset)
  - data/processed/train.parquet
  - data/processed/test.parquet
  - artifacts/eda_amount_by_type.png
  - artifacts/eda_fraud_by_type.png
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generate_synthetic import generate_dataset

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
RAW_PATH       = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR  = PROJECT_ROOT / "artifacts"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------
# 1. Load (or generate) dataset
# -----------------------------------------------------------------------

def load_or_generate() -> pd.DataFrame:
    if RAW_PATH.exists():
        print(f"[EDA] Loading existing dataset from {RAW_PATH}")
        df = pd.read_csv(RAW_PATH)
    else:
        print("[EDA] Raw dataset not found — generating synthetic data...")
        df = generate_dataset(n_transactions=500_000, output_path=str(RAW_PATH))
    return df


# -----------------------------------------------------------------------
# 2. Schema & basic description
# -----------------------------------------------------------------------

def describe_schema(df: pd.DataFrame) -> None:
    print("\n" + "="*70)
    print("SCHEMA")
    print("="*70)
    print(df.dtypes.to_string())
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() if missing.any() else "  None")


# -----------------------------------------------------------------------
# 3. Class balance
# -----------------------------------------------------------------------

def report_class_balance(df: pd.DataFrame) -> dict:
    n_total = len(df)
    n_fraud = int(df["isFraud"].sum())
    n_legit = n_total - n_fraud
    fraud_pct = n_fraud / n_total * 100

    print("\n" + "="*70)
    print("CLASS BALANCE")
    print("="*70)
    print(f"  Total transactions : {n_total:>10,}")
    print(f"  Legitimate         : {n_legit:>10,}  ({100 - fraud_pct:.3f}%)")
    print(f"  Fraudulent         : {n_fraud:>10,}  ({fraud_pct:.4f}%)")
    print()
    print("  ⚠  With <1% fraud rate, Accuracy and ROC-AUC are misleading metrics.")
    print("     A model that predicts 'not fraud' for every transaction achieves")
    print(f"    ~{100-fraud_pct:.1f}% accuracy and ~0.5 ROC-AUC baseline.")
    print("     We use Precision, Recall, F1, and PR-AUC instead.")

    return {"n_total": n_total, "n_fraud": n_fraud, "fraud_pct": round(fraud_pct, 4)}


# -----------------------------------------------------------------------
# 4. EDA plots
# -----------------------------------------------------------------------

def plot_amount_distributions(df: pd.DataFrame) -> None:
    """Log-scale amount distribution by transaction type, separated by fraud label."""
    print("\n[EDA] Plotting amount distributions by type...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Transaction Amount Distribution by Type\n(log scale, fraud vs. legitimate)",
                 fontsize=14, fontweight="bold")

    tx_types = df["type"].unique()
    axes_flat = axes.flatten()

    for i, tx_type in enumerate(sorted(tx_types)):
        ax = axes_flat[i]
        subset = df[df["type"] == tx_type]
        legit = subset[subset["isFraud"] == 0]["amount"]
        fraud = subset[subset["isFraud"] == 1]["amount"]

        ax.hist(np.log1p(legit), bins=60, alpha=0.6, color="#4C6EF5", label="Legitimate", density=True)
        if len(fraud) > 0:
            ax.hist(np.log1p(fraud), bins=60, alpha=0.8, color="#FF4D4D", label="Fraud", density=True)

        ax.set_title(f"{tx_type} (n={len(subset):,})", fontsize=11)
        ax.set_xlabel("log(1 + amount)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplot
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out = ARTIFACTS_DIR / "eda_amount_by_type.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_fraud_by_type(df: pd.DataFrame) -> None:
    """Fraud count and rate by transaction type."""
    print("[EDA] Plotting fraud by transaction type...")

    fraud_by_type = (
        df.groupby("type")["isFraud"]
        .agg(total="count", fraud_count="sum")
        .assign(fraud_rate=lambda x: x["fraud_count"] / x["total"] * 100)
        .reset_index()
        .sort_values("fraud_rate", ascending=False)
    )

    print("\nFraud by transaction type:")
    print(fraud_by_type.to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fraud Distribution by Transaction Type", fontsize=13, fontweight="bold")

    colors = ["#FF4D4D" if r > 0 else "#4C6EF5" for r in fraud_by_type["fraud_rate"]]
    ax1.bar(fraud_by_type["type"], fraud_by_type["fraud_count"], color=colors)
    ax1.set_title("Fraud Count by Type")
    ax1.set_xlabel("Transaction Type")
    ax1.set_ylabel("Fraud Transactions")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(fraud_by_type["type"], fraud_by_type["fraud_rate"], color=colors)
    ax2.set_title("Fraud Rate by Type (%)")
    ax2.set_xlabel("Transaction Type")
    ax2.set_ylabel("Fraud Rate (%)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = ARTIFACTS_DIR / "eda_fraud_by_type.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_time_patterns(df: pd.DataFrame) -> None:
    """Transaction count and fraud count by step (hour-of-day proxy)."""
    print("[EDA] Plotting time patterns...")

    hourly = (
        df.groupby("step")["isFraud"]
        .agg(total="count", fraud="sum")
        .reset_index()
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Transaction Volume & Fraud by Step (Hour-of-Day Proxy)", fontsize=13)

    ax1.plot(hourly["step"], hourly["total"], color="#4C6EF5", linewidth=0.8, alpha=0.9)
    ax1.set_ylabel("Total Transactions")
    ax1.grid(alpha=0.3)
    ax1.set_title("Total Transaction Volume")

    ax2.fill_between(hourly["step"], hourly["fraud"], color="#FF4D4D", alpha=0.7)
    ax2.set_ylabel("Fraud Transactions")
    ax2.set_xlabel("Step (1 step = 1 hour)")
    ax2.grid(alpha=0.3)
    ax2.set_title("Fraud Transaction Volume")

    plt.tight_layout()
    out = ARTIFACTS_DIR / "eda_time_patterns.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# -----------------------------------------------------------------------
# 5. Clean & split
# -----------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning:
    - Drop isFlaggedFraud (PaySim's flawed built-in rule, not our predictor).
    - Drop any duplicates.
    - Ensure amount is positive.
    """
    df = df.drop(columns=["isFlaggedFraud"], errors="ignore")
    df = df.drop_duplicates(subset=["transaction_id"], keep="first")
    df = df[df["amount"] > 0].copy()
    return df


def stratified_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split on isFraud label.

    Why stratified (not time-based)?
    ---------------------------------
    PaySim uses synthetic time steps; fraud is distributed somewhat uniformly
    across steps. A pure time-based split could yield almost zero fraud in the
    test set if fraud density is low in the later half. Stratification
    guarantees that both train and test maintain the ~0.13% fraud ratio,
    giving meaningful evaluation metrics in Phase 2/3.

    Trade-off: some account nodes appear in both train and test graphs.
    This is documented as a known limitation in the README. For a production
    system one would split on account-level cohorts.
    """
    train, test = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["isFraud"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    df_raw = load_or_generate()

    describe_schema(df_raw)
    balance_stats = report_class_balance(df_raw)

    plot_amount_distributions(df_raw)
    plot_fraud_by_type(df_raw)
    plot_time_patterns(df_raw)

    df_clean = clean(df_raw)
    train_df, test_df = stratified_split(df_clean)

    # Save
    clean_path = PROCESSED_DIR / "transactions_clean.parquet"
    train_path = PROCESSED_DIR / "train.parquet"
    test_path  = PROCESSED_DIR / "test.parquet"

    df_clean.to_parquet(clean_path, index=False)
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\n[EDA] Saved clean dataset → {clean_path}")
    print(f"[EDA] Train split: {len(train_df):,} rows  ({train_df['isFraud'].sum()} fraud)")
    print(f"[EDA] Test split:  {len(test_df):,} rows  ({test_df['isFraud'].sum()} fraud)")

    # Save EDA metadata for later reference
    meta = {
        "dataset": "Synthetic (PaySim schema)",
        **balance_stats,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_fraud": int(train_df["isFraud"].sum()),
        "test_fraud": int(test_df["isFraud"].sum()),
        "split": "stratified (80/20)",
    }
    meta_path = ARTIFACTS_DIR / "eda_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[EDA] Metadata → {meta_path}")

    print("\n" + "="*70)
    print("Phase 1 complete. Proceed to Phase 2 (Baseline Model).")
    print("="*70)


if __name__ == "__main__":
    main()
