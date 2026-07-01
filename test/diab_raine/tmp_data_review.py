"""TEMPORARY data review script — not part of the test suite, safe to delete.

Reviews diab_raine, glucose, and derived HOMA-IR from
test_data/Health/bmi/G2_data.csv across the ages where both diab_raine and
glucose were measured (14, 17, 20, 22, 27, 28 - age 8 only has glucose, no
diab_raine, so it is excluded to keep diab_raine/glucose/HOMA-IR comparable).

Produces, per variable (diab_raine, glucose, homa_ir):
  - one box-and-whisker plot across ages
  - one histogram figure faceted by age (shared bins per variable, picked
    with the Freedman-Diaconis rule on the pooled data so panels are
    visually comparable)
plus one diab_raine-vs-glucose scatter plot faceted by age.

Run:
    python test/diab_raine/tmp_data_review.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_HERE = os.path.dirname(os.path.abspath(__file__))
G2_PATH = os.path.join(_HERE, "..", "..", "test_data", "Health", "bmi", "G2_data.csv")
OUT_DIR = os.path.join(_HERE, "tmp_data_review_output")
os.makedirs(OUT_DIR, exist_ok=True)

# age -> (glucose column, diab_raine column); column names per G2_data_dictionary.csv
VISIT_COLS = {
    14: ("G214_B2", "G214_B12_0"),
    17: ("G217_B2", "G217_B12_0"),
    20: ("G220_B2", "G220_B12_0"),
    22: ("G222_B2", "G222_B12"),
    27: ("G227_B2", "G227_B12"),
    28: ("G228_B2", "G228_B12"),
}

AGES = sorted(VISIT_COLS)


def load_long_df() -> pd.DataFrame:
    df = pd.read_csv(G2_PATH, low_memory=False)
    rows = []
    for age, (glu_col, ins_col) in VISIT_COLS.items():
        sub = df[[glu_col, ins_col]].apply(pd.to_numeric, errors="coerce")
        sub = sub.dropna()
        sub.columns = ["glucose", "diab_raine"]
        sub["age"] = age
        rows.append(sub)
    long_df = pd.concat(rows, ignore_index=True)
    # HOMA-IR (SI units): glucose mmol/L * diab_raine mU/L / 22.5
    long_df["homa_ir"] = long_df["glucose"] * long_df["diab_raine"] / 22.5
    return long_df


def fd_bin_edges(
    values: np.ndarray,
    min_bins: int = 10,
    max_bins: int = 60,
    pct_range: tuple = (0.5, 99.5),
) -> np.ndarray:
    """Freedman-Diaconis bin edges over the [pct_range] core of the data.

    Insulin/HOMA-IR are heavily right-skewed with a few extreme outliers
    (e.g. diab_raine up to ~270 mU/L, HOMA-IR up to ~220), so sizing bins off
    the raw min/max collapses the bulk of the distribution into one or two
    bars. Instead, the bin range and Freedman-Diaconis width are computed
    on the 0.5th-99.5th percentile core; values outside that range are
    clipped into the outermost bin when plotting (see histogram_by_age).
    """
    values = values[~np.isnan(values)]
    lo, hi = np.percentile(values, pct_range)
    core = values[(values >= lo) & (values <= hi)]
    if len(core) < 2 or hi <= lo:
        return np.histogram_bin_edges(values, bins=min_bins)
    iqr = np.subtract(*np.percentile(core, [75, 25]))
    if iqr <= 0:
        return np.linspace(lo, hi, min_bins + 1)
    h = 2 * iqr / (len(core) ** (1 / 3))
    if h <= 0:
        return np.linspace(lo, hi, min_bins + 1)
    n_bins = int(np.ceil((hi - lo) / h))
    n_bins = int(np.clip(n_bins, min_bins, max_bins))
    return np.linspace(lo, hi, n_bins + 1)


def print_summary(long_df: pd.DataFrame) -> None:
    for var in ["diab_raine", "glucose", "homa_ir"]:
        print(f"\n--- {var} ---")
        print(long_df.groupby("age")[var].describe()[["count", "mean", "std", "min", "50%", "max"]])


def boxplot(long_df: pd.DataFrame, var: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="age", y=var, data=long_df, color="lightsteelblue")
    plt.title(f"{ylabel} by age")
    plt.xlabel("Age (years)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"box_{var}.png"), dpi=150)
    plt.close()


def histogram_by_age(long_df: pd.DataFrame, var: str, xlabel: str) -> None:
    edges = fd_bin_edges(long_df[var].to_numpy())
    lo, hi = edges[0], edges[-1]
    n_ages = len(AGES)
    ncols = 3
    nrows = int(np.ceil(n_ages / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, age in zip(axes, AGES):
        data = long_df.loc[long_df["age"] == age, var]
        n_clipped = int(((data < lo) | (data > hi)).sum())
        clipped = data.clip(lo, hi)
        ax.hist(clipped, bins=edges, color="steelblue", edgecolor="white")
        title = f"age {age} (n={len(data)})"
        if n_clipped:
            title += f"\n{n_clipped} outliers clipped to edge bins"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
    for ax in axes[n_ages:]:
        ax.axis("off")
    fig.suptitle(
        f"{xlabel} histogram by age (Freedman-Diaconis bins on 0.5-99.5 pctile, "
        f"n_bins={len(edges) - 1}; extreme outliers clipped into edge bins)"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"hist_{var}.png"), dpi=150)
    plt.close(fig)


def scatter_insulin_glucose(long_df: pd.DataFrame) -> None:
    n_ages = len(AGES)
    ncols = 3
    nrows = int(np.ceil(n_ages / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, age in zip(axes, AGES):
        sub = long_df[long_df["age"] == age]
        ax.scatter(sub["diab_raine"], sub["glucose"], s=12, alpha=0.5, color="darkorange")
        ax.set_title(f"age {age} (n={len(sub)})")
        ax.set_xlabel("Insulin (mU/L)")
        ax.set_ylabel("Glucose (mmol/L)")
    for ax in axes[n_ages:]:
        ax.axis("off")
    fig.suptitle("Insulin vs glucose by age")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "scatter_insulin_glucose.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    long_df = load_long_df()
    print_summary(long_df)

    boxplot(long_df, "diab_raine", "Insulin (mU/L)")
    boxplot(long_df, "glucose", "Glucose (mmol/L)")
    boxplot(long_df, "homa_ir", "HOMA-IR")

    histogram_by_age(long_df, "diab_raine", "Insulin (mU/L)")
    histogram_by_age(long_df, "glucose", "Glucose (mmol/L)")
    histogram_by_age(long_df, "homa_ir", "HOMA-IR")

    scatter_insulin_glucose(long_df)

    print(f"\nPlots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
