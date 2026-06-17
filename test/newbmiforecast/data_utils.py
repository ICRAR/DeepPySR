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

# Reuse rename/preprocess logic from insulin data_utils (loaded via importlib to avoid name clash)
import importlib.util as _ilu
_insulin_spec = _ilu.spec_from_file_location(
    "insulin_data_utils",
    str(Path(__file__).parents[1] / "insulin" / "data_utils.py"),
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


def _is_bmi_col(c: str) -> bool:
    """True if column is a BMI target column (y{year}bmi)."""
    return bool(re.match(r'^y\d+bmi$', c))


_COL_AGE_RE = re.compile(r'^yr?(\d+)')


def _col_age(c: str):
    """Return the age (in years) a column was measured at, or None if age-independent."""
    m = _COL_AGE_RE.match(c)
    return int(m.group(1)) if m else None


def get_age_filtered_feature_cols(non_bmi_cols: list, target_year: int) -> list:
    """Restrict non-BMI feature columns to those measured at an age < target_year.

    Prevents leaking variables observed at or after the forecast age into the
    feature set used to predict y{target_year}bmi.
    """
    return [c for c in non_bmi_cols
            if (age := _col_age(c)) is None or age < target_year]


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


def drop_highly_correlated(df, cols, threshold=0.95):
    """Drop columns from cols with |r| > threshold with any earlier column, keeping the first."""
    numeric_cols = [c for c in cols if c in df.select_dtypes(include="number").columns]
    if len(numeric_cols) < 2:
        return df, cols
    corr = df[numeric_cols].corr().abs()
    upper = corr.where(
        pd.DataFrame([[i < j for j in range(len(corr.columns))]
                      for i in range(len(corr.columns))],
                     index=corr.index, columns=corr.columns)
    )
    to_drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    if to_drop:
        print(f"\n[drop_highly_correlated] dropping {len(to_drop)} cols with |r|>{threshold}")
        for col in to_drop:
            partners = upper.index[(upper[col] > threshold)].tolist()
            print(f"  {col} (corr>{threshold} with {partners[:3]})")
    df = df.drop(columns=to_drop)
    cols = [c for c in cols if c not in to_drop]
    return df, cols


def _get_newbmi_base_feature_cols(merged_df: pd.DataFrame, non_bmi_cols: list) -> list:
    """Return feature column list used for y8bmi base model training."""
    return [c for c in non_bmi_cols if c in merged_df.columns]


_base_feature_cols_cache = None


def _get_bmi_base_feature_cols(merged_df=None, non_bmi_cols=None):
    global _base_feature_cols_cache
    if _base_feature_cols_cache is None:
        if merged_df is not None and non_bmi_cols is not None:
            _base_feature_cols_cache = _get_newbmi_base_feature_cols(merged_df, non_bmi_cols)
        else:
            _base_feature_cols_cache = []
    return _base_feature_cols_cache


def prepare_base_dataset():
    """Load merged + G1 + G2 data, impute non-BMI features, create y8bmi pred columns.

    Returns:
        merged (DataFrame): fully imputed dataset with model-specific y8bmi pred columns.
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

    # Drop highly correlated non-BMI features before imputation
    merged, non_bmi_cols = drop_highly_correlated(merged, non_bmi_cols)

    # Move y{age}bmi columns to the end
    bmi_cols = [c for c in merged.columns if _is_bmi_col(c)]
    other_cols = [c for c in merged.columns if c not in bmi_cols]
    merged = merged[other_cols + bmi_cols]

    print(f"\n=== Imputing non-BMI variables ({len(non_bmi_cols)} cols) ===")
    merged = smart_impute(merged, non_bmi_cols)

    bmi8_col = "y8bmi"
    if bmi8_col not in merged.columns:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        merged.to_csv(cache_path, index=False)
        return merged, non_bmi_cols

    print("\n=== Creating model-specific y8bmi prediction columns ===")
    real_mask = merged[bmi8_col].notna()
    missing_mask = ~real_mask
    n_missing = int(missing_mask.sum())
    print(f"  y8bmi: {int(real_mask.sum())} observed, {n_missing} missing")

    bmi8_median = merged.loc[real_mask, bmi8_col].median()
    age8_fcols = [c for c in non_bmi_cols if c in merged.columns]

    # Set cache for later use
    global _base_feature_cols_cache
    _base_feature_cols_cache = age8_fcols

    import sys as _sys
    import joblib as _jl
    _model_utils_dir = str(Path(__file__).parents[1])
    if _model_utils_dir not in _sys.path:
        _sys.path.insert(0, _model_utils_dir)

    try:
        from model_utils import get_baseline_models as _get_bl_models
    except Exception as _ie:
        print(f"  WARNING: Could not import get_baseline_models: {_ie}")
        _get_bl_models = None

    _bl_save_dir = str(_CACHE_DIR / "age_8_baselines")
    os.makedirs(_bl_save_dir, exist_ok=True)

    if _get_bl_models is not None and age8_fcols and n_missing > 0:
        X_train8 = merged.loc[real_mask, age8_fcols].values
        y_train8 = merged.loc[real_mask, bmi8_col].values
        X_miss8 = merged.loc[missing_mask, age8_fcols].values

        _bl_model_instances = _get_bl_models(task="regression", input_dim=len(age8_fcols))
        for model_name, model_template in _bl_model_instances.items():
            pred_col = f"{bmi8_col}_{model_name}_pred"
            merged[pred_col] = merged[bmi8_col].copy()

            fitted_model = None
            _jl_path = os.path.join(_bl_save_dir, f"{model_name}.joblib")
            if model_name != "KAN" and os.path.exists(_jl_path):
                try:
                    fitted_model = _jl.load(_jl_path)
                    print(f"  Loaded saved model: {model_name}")
                except Exception as _le:
                    print(f"  Could not load {model_name}, retraining: {_le}")

            if fitted_model is None:
                try:
                    from sklearn.base import clone as _clone
                    m = _clone(model_template) if hasattr(model_template, "get_params") else model_template
                    m.fit(X_train8, y_train8)
                    fitted_model = m
                    if model_name != "KAN":
                        _jl.dump(m, _jl_path)
                        print(f"  Trained and saved: {model_name}")
                except Exception as _te:
                    print(f"  WARNING: Failed to train {model_name}: {_te}")

            if fitted_model is not None:
                try:
                    preds = fitted_model.predict(X_miss8)
                    merged.loc[missing_mask, pred_col] = preds
                except Exception as _pe:
                    print(f"  WARNING: {model_name}.predict failed: {_pe}")

            still_nan = merged[pred_col].isna()
            if still_nan.any():
                merged.loc[still_nan, pred_col] = bmi8_median
            n_filled = int(merged.loc[missing_mask, pred_col].notna().sum())
            print(f"  {model_name}: {n_filled}/{n_missing} missing filled")

    # Fill base y8bmi
    if n_missing > 0:
        pred_cols = [c for c in merged.columns
                     if c.startswith(f"{bmi8_col}_") and c.endswith("_pred")]
        for pc in pred_cols:
            still_missing = merged[bmi8_col].isna()
            if not still_missing.any():
                break
            merged.loc[still_missing, bmi8_col] = merged.loc[still_missing, pc]
        still_missing = merged[bmi8_col].isna().sum()
        if still_missing:
            merged[bmi8_col].fillna(bmi8_median, inplace=True)
            print(f"  Median fallback for {still_missing} remaining rows.")

    merged = merged.copy()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path, index=False)
    print(f"\n=== Cached base dataset to {cache_path} ===")

    return merged, non_bmi_cols
