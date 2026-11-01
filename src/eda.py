"""Exploratory data analysis for the Lending Club loan dataset."""
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TARGET = "not.fully.paid"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing.")
    return df


def run_eda(df: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "target_counts": {str(k): int(v) for k, v in df[TARGET].value_counts(dropna=False).items()},
        "target_rates": {str(k): float(v) for k, v in df[TARGET].value_counts(normalize=True, dropna=False).items()},
        "numeric_describe": df.describe(include="number").round(3).to_dict(),
    }
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x=TARGET, ax=ax)
    ax.set(title="Target class distribution", xlabel="Not fully paid (1 = yes)", ylabel="Loan count")
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=160)
    plt.close(fig)

    purpose_rate = df.groupby("purpose", dropna=False)[TARGET].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    purpose_rate.plot(kind="bar", ax=ax)
    ax.set(title="Default / not-fully-paid rate by loan purpose", xlabel="Purpose", ylabel="Rate")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "target_rate_by_purpose.png", dpi=160)
    plt.close(fig)

    numeric = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Numeric feature correlation matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    for column in ["int.rate", "fico", "dti", "revol.util", "log.annual.inc"]:
        if column in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(data=df, x=TARGET, y=column, ax=ax)
            ax.set_title(f"{column} by target class")
            fig.tight_layout()
            fig.savefig(output_dir / f"{column.replace('.', '_')}_by_target.png", dpi=160)
            plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Data/loan_data.csv")
    parser.add_argument("--output", default="reports/figures")
    args = parser.parse_args()
    df = load_data(args.data)
    summary = run_eda(df, Path(args.output))
    print(f"EDA complete: {summary['rows']:,} rows, {summary['columns']} columns.")
    print(f"Target distribution: {summary['target_counts']}")


if __name__ == "__main__":
    main()
