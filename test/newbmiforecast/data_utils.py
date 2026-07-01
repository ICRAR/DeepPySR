"""Data utilities for newbmiforecast — uses merged.csv + G1 + G2 data."""

import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Shared helpers from bmiforecast_utils ─────────────────────────────────────
_bmiforecast_dir = str(Path(__file__).parents[1] / "bmiforecast")
if _bmiforecast_dir not in sys.path:
    sys.path.insert(0, _bmiforecast_dir)
from bmiforecast_utils import (
    apply_formula, fill_missing_bmi_with_formula, fill_missing_bmi_with_model,
    save_rolling_dataset_with_predictions, load_forecast_data,
    load_forecast_data_for_model, evaluate_predictions,
    smart_impute, drop_low_variance_and_extra, replace_99_with_nan,
    _load_full_baseline_models, _extract_variables,
    get_best_formula_for_year, get_best_pysr_formula_for_year,
    get_best_baseline_model, save_baseline_models, load_baseline_models,
    _is_categorical_col,
)

# Reuse rename/preprocess logic from diab_raine data_utils (loaded via importlib to avoid name clash)
import importlib.util as _ilu
_insulin_spec = _ilu.spec_from_file_location(
    "insulin_data_utils",
    str(Path(__file__).parents[1] / "diab_raine" / "data_utils.py"),
)
_insulin_du = _ilu.module_from_spec(_insulin_spec)
_insulin_spec.loader.exec_module(_insulin_du)
_build_rename_map = _insulin_du._build_rename_map
_preprocess = _insulin_du._preprocess

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parents[2] / "test_data" / "Health"
_RAINE_PATH = _BASE / "raine" / "merged.csv"
_BMI_PATH = _BASE / "bmi"

# BMI timepoints present in merged.csv (age in years = column year suffix)
YEARS = [8, 10, 14, 17, 20, 22, 27]

# Cache directory for preprocessed datasets (under test/newbmiforecast/ like bmiforecast)
_CACHE_DIR = Path(__file__).parent / "results_newbmiforecast"

# Column used for the earliest (base) BMI prediction
_BASE_BMI_YEAR = 8


def actual_age(year: int) -> int:
    """Return the actual age for a forecast year (identity for this dataset)."""
    return year


_BMI_TARGET_COLS = {f'y{y}bmi' for y in YEARS}


def _is_bmi_col(c: str) -> bool:
    """True if column is a BMI target column (one of the forecast years in YEARS)."""
    return c in _BMI_TARGET_COLS


_COL_AGE_RE = re.compile(r'(?:^|_)yr?(\d+)')


def _col_age(c: str):
    """Return the age (in years) a column was measured at, or None if age-independent."""
    m = _COL_AGE_RE.search(c)
    return int(m.group(1)) if m else None


def get_age_filtered_feature_cols(non_bmi_cols: list, target_year: int) -> list:
    """Restrict non-BMI feature columns to those measured at an age < target_year.

    Prevents leaking variables observed at or after the forecast age into the
    feature set used to predict y{target_year}bmi.
    """
    return [c for c in non_bmi_cols
            if (age := _col_age(c)) is None or age < target_year]


def _compute_early_bmiz(df: pd.DataFrame) -> pd.DataFrame:
    """Add birthbmiz, y1bmiz (WHO LMS), and y5bmiz (CDC extended LMS) columns.

    Uses local data files only — no pygrowup dependency:
      - birthbmiz / y1bmiz: bmizscore/bmifa_{boys,girls}_0_2_zscores.json (WHO)
      - y5bmiz: bmizscore/bmi-age-2022.csv (CDC extended BMI-for-age)

    birth_weight is in grams, heights in cm, y1/y5 weights in kg.
    """
    import json
    from scipy.stats import norm as _norm

    _bmizscore_dir = Path(__file__).parent / 'bmizscore'

    df = df.copy()

    sex_col = next((c for c in ('sex_x', 'sex') if c in df.columns), None)
    if sex_col is None:
        print("  [early_bmiz] WARNING: no sex column found; skipping")
        return df

    def _sex_int(val):
        try:
            v = int(float(val))
            return v if v in (1, 2) else None
        except (TypeError, ValueError):
            return None

    # ── WHO LMS helpers ───────────────────────────────────────────────────────
    def _load_who_lms(sex_int):
        fname = 'bmifa_boys_0_2_zscores.json' if sex_int == 1 else 'bmifa_girls_0_2_zscores.json'
        rows = json.loads((Path(_bmizscore_dir) / fname).read_text())
        # Build dict: month (int) → {L, M, S}
        return {int(r['Month']): {'L': float(r['L']), 'M': float(r['M']), 'S': float(r['S'])}
                for r in rows}

    _who_lms_cache = {}

    def _who_bmiz(bmi, agemos, sex_int):
        """Standard WHO LMS z-score; interpolates between integer months."""
        if sex_int not in _who_lms_cache:
            _who_lms_cache[sex_int] = _load_who_lms(sex_int)
        table = _who_lms_cache[sex_int]
        lo, hi = int(agemos), min(int(agemos) + 1, 24)
        if lo not in table:
            return np.nan
        if lo == hi or hi not in table:
            L, M, S = table[lo]['L'], table[lo]['M'], table[lo]['S']
        else:
            frac = agemos - lo
            L = table[lo]['L'] + frac * (table[hi]['L'] - table[lo]['L'])
            M = table[lo]['M'] + frac * (table[hi]['M'] - table[lo]['M'])
            S = table[lo]['S'] + frac * (table[hi]['S'] - table[lo]['S'])
        if abs(L) >= 0.01:
            z = ((bmi / M) ** L - 1) / (L * S)
        else:
            z = math.log(bmi / M) / S
        return float(z)

    # ── CDC extended LMS helpers (bmi-age-2022.csv) ───────────────────────────
    _cdc_ext = {}

    def _load_cdc_ext(sex_int):
        if sex_int not in _cdc_ext:
            path = _bmizscore_dir / 'bmi-age-2022.csv'
            df_ref = pd.read_csv(path)
            _cdc_ext[sex_int] = df_ref[df_ref['sex'] == sex_int].sort_values('agemos').reset_index(drop=True)
        return _cdc_ext[sex_int]

    def _interp_row(ref, agemos):
        """Linearly interpolate L, M, S, sigma, P95 at the given age in months."""
        idx = (ref['agemos'] - agemos).abs().idxmin()
        r = ref.loc[idx]
        if abs(r['agemos'] - agemos) < 0.01:
            return r
        # Find bracketing rows
        lo_rows = ref[ref['agemos'] <= agemos]
        hi_rows = ref[ref['agemos'] >= agemos]
        if lo_rows.empty or hi_rows.empty:
            return r
        r_lo = ref.loc[lo_rows.index[-1]]
        r_hi = ref.loc[hi_rows.index[0]]
        if r_lo['agemos'] == r_hi['agemos']:
            return r_lo
        frac = (agemos - r_lo['agemos']) / (r_hi['agemos'] - r_lo['agemos'])
        interp = {}
        for col in ('L', 'M', 'S', 'sigma', 'P95'):
            interp[col] = r_lo[col] + frac * (r_hi[col] - r_lo[col])
        return interp

    def _cdc_ext_bmiz(bmi, agemos, sex_int):
        """CDC extended BMI-for-age z-score from bmi-age-2022.csv."""
        ref = _load_cdc_ext(sex_int)
        row = _interp_row(ref, agemos)
        L, M, S = float(row['L']), float(row['M']), float(row['S'])
        sigma, p95 = float(row['sigma']), float(row['P95'])
        # Standard LMS z-score
        if abs(L) >= 0.01:
            z_lms = ((bmi / M) ** L - 1) / (L * S)
        else:
            z_lms = math.log(bmi / M) / S
        pct = float(_norm.cdf(z_lms)) * 100.0
        # Extended z-score for BMI above the 95th percentile
        if bmi > p95:
            pct = 90.0 + 10.0 * float(_norm.cdf((bmi - p95) / sigma))
        pct = min(pct, 99.9999999)
        return float(_norm.ppf(pct / 100.0))

    # ── birthbmiz ─────────────────────────────────────────────────────────────
    if 'birth_weight' in df.columns and 'birth_length' in df.columns:
        birth_bmiz = []
        for _, row in df.iterrows():
            try:
                bw = float(row['birth_weight'])   # grams
                bl = float(row['birth_length'])   # cm
                sx = _sex_int(row[sex_col])
                if pd.isna(bw) or pd.isna(bl) or sx is None or bl <= 0:
                    birth_bmiz.append(np.nan)
                    continue
                bmi = (bw / 1000.0) / (bl / 100.0) ** 2
                birth_bmiz.append(_who_bmiz(bmi, 0, sx))
            except Exception:
                birth_bmiz.append(np.nan)
        df['birthbmiz'] = birth_bmiz
        print(f"  birthbmiz: {int(pd.Series(birth_bmiz).notna().sum())}/{len(df)} valid")
    else:
        print("  [early_bmiz] WARNING: birth_weight or birth_length missing; skipping birthbmiz")

    # ── y1bmiz ────────────────────────────────────────────────────────────────
    if 'y1_a1' in df.columns and 'y1_a2' in df.columns:
        age1_col = next((c for c in ('height_age12', 'weight_age12') if c in df.columns), None)
        y1_bmiz = []
        for _, row in df.iterrows():
            try:
                w1 = float(row['y1_a1'])   # kg
                h1 = float(row['y1_a2'])   # cm
                sx = _sex_int(row[sex_col])
                agemos = (float(row[age1_col])
                          if age1_col and pd.notna(row.get(age1_col)) else 12.0)
                if pd.isna(w1) or pd.isna(h1) or sx is None or h1 <= 0:
                    y1_bmiz.append(np.nan)
                    continue
                bmi = w1 / (h1 / 100.0) ** 2
                y1_bmiz.append(_who_bmiz(bmi, agemos, sx))
            except Exception:
                y1_bmiz.append(np.nan)
        df['y1bmiz'] = y1_bmiz
        print(f"  y1bmiz: {int(pd.Series(y1_bmiz).notna().sum())}/{len(df)} valid")
    else:
        print("  [early_bmiz] WARNING: y1_a1 or y1_a2 missing; skipping y1bmiz")

    # ── y5bmiz ────────────────────────────────────────────────────────────────
    if 'y5_a1' in df.columns and 'y5_a2' in df.columns and 'y5_age' in df.columns:
        y5_bmiz = []
        for _, row in df.iterrows():
            try:
                w5 = float(row['y5_a1'])    # kg
                h5 = float(row['y5_a2'])    # cm
                agey = float(row['y5_age']) # years
                sx = _sex_int(row[sex_col])
                if pd.isna(w5) or pd.isna(h5) or pd.isna(agey) or sx is None or h5 <= 0:
                    y5_bmiz.append(np.nan)
                    continue
                bmi = w5 / (h5 / 100.0) ** 2
                y5_bmiz.append(_cdc_ext_bmiz(bmi, agey * 12.0, sx))
            except Exception:
                y5_bmiz.append(np.nan)
        df['y5bmiz'] = y5_bmiz
        print(f"  y5bmiz: {int(pd.Series(y5_bmiz).notna().sum())}/{len(df)} valid")
    else:
        print("  [early_bmiz] WARNING: y5_a1, y5_a2, or y5_age missing; skipping y5bmiz")

    return df


def _build_merged_bmi() -> pd.DataFrame:
    """Load merged.csv + G1 + G2 and return a preprocessed DataFrame."""
    raine = pd.read_csv(_RAINE_PATH, low_memory=False)

    g1_rename = _build_rename_map(_BMI_PATH / "G1_data_dictionary.csv", "g1")
    g1 = pd.read_csv(_BMI_PATH / "G1_data.csv", low_memory=False).rename(
        columns={**g1_rename, "ID": "mother_id"}
    )

    g2_rename = _build_rename_map(_BMI_PATH / "G2_data_dictionary.csv", "g2")
    g2 = pd.read_csv(_BMI_PATH / "G2_data.csv", low_memory=False).rename(
        columns={**g2_rename, "ID": "child_id"}
    )

    merged = raine.merge(g1, on="mother_id", how="left")
    merged = merged.merge(g2, on="child_id", how="left")
    merged = merged.dropna(subset=["child_id"])

    # Consolidate y8bmi: prefer y8bmi_x (more non-null), fill gaps from y8bmi_y
    if "y8bmi_x" in merged.columns:
        merged["y8bmi"] = merged["y8bmi_x"]
        if "y8bmi_y" in merged.columns:
            mask = merged["y8bmi"].isna() & merged["y8bmi_y"].notna()
            merged.loc[mask, "y8bmi"] = merged.loc[mask, "y8bmi_y"]
        merged = merged.drop(columns=[c for c in ["y8bmi_x", "y8bmi_y"] if c in merged.columns])

    # Drop decile columns (not useful as features or targets)
    decile_cols = [c for c in merged.columns if c.endswith("_decile")]
    merged = merged.drop(columns=decile_cols)

    # Deduplicate by child_id, keep row with most non-null values
    if "child_id" in merged.columns:
        before = len(merged)
        merged = (merged
                  .assign(_n_valid=merged.notna().sum(axis=1))
                  .sort_values("_n_valid", ascending=False)
                  .drop_duplicates(subset="child_id", keep="first")
                  .drop(columns="_n_valid"))
        dropped = before - len(merged)
        if dropped:
            print(f"[dedup by child_id] dropped {dropped} duplicate rows")

    merged = _preprocess(merged)

    # BMI of 0.0 is not physiologically possible; treat as missing.
    bmi_cols_all = [c for c in merged.columns if _is_bmi_col(c)]
    for c in bmi_cols_all:
        merged.loc[merged[c] == 0.0, c] = np.nan

    # Drop rows with no usable BMI signal at all (age >= _BASE_BMI_YEAR).
    forecast_bmi_cols = [c for c in bmi_cols_all
                         if (age := _col_age(c)) is not None and age >= _BASE_BMI_YEAR]
    if forecast_bmi_cols:
        before = len(merged)
        merged = merged.dropna(subset=forecast_bmi_cols, how="all")
        dropped = before - len(merged)
        if dropped:
            print(f"[drop all-NaN BMI] dropped {dropped} rows with no BMI observed at age >= {_BASE_BMI_YEAR}")

    return merged


def drop_highly_correlated(df, cols, threshold=0.98):
    """Drop columns from cols with |r| > threshold with any earlier column, keeping the first.

    y5bmi and PGS variables are always kept regardless of correlation.
    """
    numeric_cols = [c for c in cols if c in df.select_dtypes(include="number").columns]
    if len(numeric_cols) < 2:
        return df, cols
    protected = {c for c in numeric_cols
                 if c == 'y5bmi' or 'pgs' in c.lower()}
    corr = df[numeric_cols].corr().abs()
    upper = corr.where(
        pd.DataFrame([[i < j for j in range(len(corr.columns))]
                      for i in range(len(corr.columns))],
                     index=corr.index, columns=corr.columns)
    )
    to_drop = [c for c in upper.columns
               if c not in protected and (upper[c] > threshold).any()]
    if to_drop:
        print(f"\n[drop_highly_correlated] dropping {len(to_drop)} cols with |r|>{threshold}")
        for col in to_drop:
            partners = upper.index[(upper[col] > threshold)].tolist()
            print(f"  {col} (corr>{threshold} with {partners[:3]})")
    df = df.drop(columns=to_drop)
    cols = [c for c in cols if c not in to_drop]
    return df, cols



def prepare_base_dataset():
    """Load merged + G1 + G2 data, impute non-BMI features.

    Returns:
        merged (DataFrame): fully imputed dataset.
        non_bmi_cols (list): non-BMI, non-id feature column names.
    """
    cache_path = _CACHE_DIR / "base_dataset.csv"

    if cache_path.exists():
        print(f"\n=== Loading cached base dataset from {cache_path} ===")
        merged = pd.read_csv(cache_path, low_memory=False)
        non_bmi_cols = [c for c in merged.columns
                        if c != "child_id" and not _is_bmi_col(c) and not c.endswith("_pred")]
        return merged, non_bmi_cols

    print("\n=== Building merged dataset from merged.csv + G1 + G2 ===")
    merged = _build_merged_bmi()

    # Identify non-BMI feature columns (exclude child_id and all bmi cols)
    non_bmi_cols = [c for c in merged.columns
                    if c != "child_id" and not _is_bmi_col(c)]

    # Remove duplicates before imputation
    n_before = len(merged)
    merged = merged.drop_duplicates()
    n_dropped = n_before - len(merged)
    if n_dropped:
        print(f"  Dropped {n_dropped} duplicate rows.")

    # Print BMI missing rates and drop rows with all BMI targets missing
    bmi_cols = [c for c in merged.columns if _is_bmi_col(c)]
    print("\n[BMI column missing rates]")
    for col in bmi_cols:
        print(f"  {col}: {merged[col].isna().mean():.3f}")
    n_before = len(merged)
    merged = merged.dropna(subset=bmi_cols, how="all")
    print(f"  Dropped {n_before - len(merged)} rows with all BMI targets missing ({len(merged)} remaining)")
    non_bmi_cols = [c for c in non_bmi_cols if c in merged.columns]

    # Drop date columns
    date_cols = [c for c in non_bmi_cols
                 if c in merged.select_dtypes(include=["datetime", "datetimetz"]).columns]
    if date_cols:
        print(f"\n[drop date columns] dropping {len(date_cols)}: {date_cols}")
    merged = merged.drop(columns=date_cols)
    non_bmi_cols = [c for c in non_bmi_cols if c not in date_cols]

    # Drop columns with >60% missingness (all-NaN is a special case)
    high_nan_cols = [c for c in non_bmi_cols if merged[c].isna().mean() > 0.60]
    if high_nan_cols:
        print(f"\n[drop high-NaN cols] dropping {len(high_nan_cols)} columns with >60% missing")
        merged = merged.drop(columns=high_nan_cols)
        non_bmi_cols = [c for c in non_bmi_cols if c not in high_nan_cols]

    # Drop highly correlated non-BMI features before imputation
    merged, non_bmi_cols = drop_highly_correlated(merged, non_bmi_cols)

    # Move y{age}bmi columns to the end
    bmi_cols = [c for c in merged.columns if _is_bmi_col(c)]
    other_cols = [c for c in merged.columns if c not in bmi_cols]
    merged = merged[other_cols + bmi_cols]

    print(f"\n=== Imputing non-BMI variables ({len(non_bmi_cols)} cols) ===")
    merged = smart_impute(merged, non_bmi_cols)

    print("\n=== Computing early BMI z-scores (birthbmiz, y1bmiz, y5bmiz) ===")
    merged = _compute_early_bmiz(merged)
    new_bmiz_cols = [c for c in ('birthbmiz', 'y1bmiz', 'y5bmiz') if c in merged.columns]
    non_bmi_cols = non_bmi_cols + [c for c in new_bmiz_cols if c not in non_bmi_cols]

    merged = merged.copy()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path, index=False)
    print(f"\n=== Cached base dataset to {cache_path} ===")

    return merged, non_bmi_cols
