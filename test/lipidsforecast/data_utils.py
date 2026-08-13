"""Data utilities for lipidsforecast -- rolling lipid forecast (age 14 -> 28).

Reuses the RAINE merge/preprocess pipeline from lipids_raine/data_utils.py (the
'recent' input_type: PGS + all raw features collected before the target age,
including cross-lipid blood history) and the generic rolling-pipeline helpers
from bmiforecast_utils.py (formula application, full-model loading).

Unlike lipids_raine's per-age standalone loaders, this module builds ONE shared
base dataset with explicit rolling target columns for all 4 lipid targets at all
5 ages, so a chain of models (age 14 -> 17 -> 20 -> 22 -> 28) can be built per
target, each step using the previous step's best model to fill missing target
values -- mirroring bmiforecast_utils's y{year}bmi rolling design, generalized
to a named target column instead of a fixed 'y{year}bmi' pattern.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# lipids_raine's module is also named 'data_utils.py' -- a plain sys.path +
# `import data_utils` here would self-collide with this very module (whichever
# gets cached in sys.modules first under that name wins for both). Load it
# under a distinct name via importlib instead, same trick newbmiforecast/data_utils.py
# uses to pull in diab_raine's identically-named data_utils.py.
import importlib.util as _ilu
_lipids_raine_spec = _ilu.spec_from_file_location(
    "lipids_raine_data_utils",
    str(Path(__file__).parents[1] / "lipids_raine" / "data_utils.py"),
)
_lipids_raine_du = _ilu.module_from_spec(_lipids_raine_spec)
_lipids_raine_spec.loader.exec_module(_lipids_raine_du)

_build_merged = _lipids_raine_du._build_merged
_lipid_target_series = _lipids_raine_du._lipid_target_series
_get_target_col = _lipids_raine_du._get_target_col
_high_age_suffixes = _lipids_raine_du._high_age_suffixes
_recent_feature_cutoff = _lipids_raine_du._recent_feature_cutoff
_clean_and_impute = _lipids_raine_du._clean_and_impute
_dedup_by_child_id = _lipids_raine_du._dedup_by_child_id
_LIPID_TARGETS = _lipids_raine_du._LIPID_TARGETS
_LIPIDS_AGES = _lipids_raine_du._LIPIDS_AGES

_bmiforecast_dir = str(Path(__file__).parents[1] / "bmiforecast")
if _bmiforecast_dir not in sys.path:
    sys.path.insert(0, _bmiforecast_dir)
from bmiforecast_utils import apply_formula, _extract_variables, _load_full_baseline_models

AGES = list(_LIPIDS_AGES)       # [14, 17, 20, 22, 28]
TARGETS = list(_LIPID_TARGETS)  # ['cholesterol', 'triglyceride', 'hdl', 'ldl']

# Raw RAINE waves a lipid panel was measured at. 27 is included even though it's
# no longer a standalone forecast age (see _LIPIDS_AGES) -- its raw column still
# needs excluding from the generic feature set for the same reason age28's cutoff
# excludes it (age28 IS the merged yr27/yr28 wave; see _lipid_target_series).
_LIPID_WAVES = [14, 17, 20, 22, 27, 28]

_CACHE_DIR = Path(__file__).parent / "results_lipidsforecast"


def actual_age(age: int) -> int:
    """Identity: forecast ages already match the real ages (14, 17, 20, 22, 28)."""
    return age


def target_col(target: str, age: int) -> str:
    """Rolling-pipeline target column name, e.g. 'age14_cholesterol'."""
    return f"age{age}_{target}"


_ALL_TARGET_COLS = [target_col(t, a) for t in TARGETS for a in AGES]


def _is_target_col(c: str) -> bool:
    return c in _ALL_TARGET_COLS


def _target_raw_cols(merged: pd.DataFrame, target: str) -> set:
    """Raw RAINE column name(s) carrying `target`'s own history at every lipid-panel
    wave. Excluded from that target's feature set (in favour of the rolled
    age{A}_{target}[_pred] column) so its own past isn't duplicated as a blindly-
    imputed raw copy sitting alongside the model-consistent rolled feature -- other
    targets' raw history is deliberately left in place ('recent' includes cross-lipid
    blood history, unlike 'nblood').

    Waves whose raw column was already dropped by _clean_and_impute (e.g. too much
    missingness) are skipped rather than raising -- there's nothing left to exclude."""
    cols = set()
    for wave in _LIPID_WAVES:
        try:
            cols.add(_get_target_col(merged, target, wave))
        except ValueError:
            continue
    return cols


def get_age_filtered_feature_cols(feature_cols: list, target_age: int) -> list:
    """Restrict feature columns to those measured strictly before target_age.

    Matches lipids_raine's 'recent' cutoff: age 28 additionally excludes age 27
    (see _recent_feature_cutoff) since the yr27/yr28 waves are merged into one
    target and yr27 is effectively the same wave, not a genuine prior observation.
    """
    high_suffixes = _high_age_suffixes(_recent_feature_cutoff(target_age))
    return [c for c in feature_cols
            if not any(suf in c for suf in high_suffixes) and not c.startswith("SUM_PGS")]


def prepare_base_dataset():
    """Build the merged dataset with explicit rolling target columns for all 4
    lipid targets at all 5 ages, plus a shared, cleaned/imputed feature set.

    Returns:
        merged (DataFrame): child_id + age{A}_{target} target columns + features.
        feature_cols (list): base feature column names (shared across targets;
                              callers exclude a given target's own raw history
                              via _target_raw_cols before using it for that chain).
    """
    cache_path = _CACHE_DIR / "base_dataset.csv"
    if cache_path.exists():
        print(f"\n=== Loading cached base dataset from {cache_path} ===")
        merged = pd.read_csv(cache_path, low_memory=False)
        feature_cols = [c for c in merged.columns
                        if c != "child_id" and not _is_target_col(c) and not c.endswith("_pred")]
        return merged, feature_cols

    print("\n=== Building merged dataset from lipids_raine's RAINE pipeline ===")
    merged = _build_merged()

    print("\n=== Attaching rolling target columns (4 targets x 5 ages) ===")
    for target in TARGETS:
        for age in AGES:
            merged[target_col(target, age)] = _lipid_target_series(merged, target, age)

    n_before = len(merged)
    merged = merged.dropna(subset=_ALL_TARGET_COLS, how="all")
    print(f"  Dropped {n_before - len(merged)} rows with no lipid target observed at any age "
          f"({len(merged)} remaining)")
    merged = _dedup_by_child_id(merged)

    feature_cols = [c for c in merged.columns if c != "child_id" and c not in _ALL_TARGET_COLS]
    # Bound to columns any rolling step could ever use (max cutoff across AGES is
    # age28's 26) so _clean_and_impute isn't wasted cleaning/imputing columns from
    # waves (27, 28) that no step's cutoff ever admits as a feature.
    feature_cols = get_age_filtered_feature_cols(feature_cols, max(AGES))

    print(f"\n=== Cleaning + imputing {len(feature_cols)} feature columns ===")
    X = _clean_and_impute(merged[feature_cols].copy())
    feature_cols = list(X.columns)

    merged = pd.concat([
        merged[["child_id"] + _ALL_TARGET_COLS].reset_index(drop=True),
        X.reset_index(drop=True),
    ], axis=1)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path, index=False)
    print(f"\n=== Cached base dataset to {cache_path} ===")

    return merged, feature_cols


def load_forecast_data_for_model(merged_df, target, age, prior_target_cols, feature_cols,
                                  model_type='deeppysr'):
    """Prepare (ids, X, y) for predicting `target` at `age`, using model-specific
    prior-target prediction columns where available (falling back to the raw prior
    target column). Only rows with an observed target at `age` are returned.

    Generalizes bmiforecast_utils.load_forecast_data_for_model to a named target
    column instead of the fixed 'y{year}bmi' pattern.
    """
    tcol = target_col(target, age)
    if tcol not in merged_df.columns:
        return None, None, None

    prior_pred_cols = []
    for col in prior_target_cols:
        pred_col = f'{col}_{model_type}_pred'
        if pred_col in merged_df.columns:
            prior_pred_cols.append(pred_col)
        elif col in merged_df.columns:
            prior_pred_cols.append(col)

    fcols = [c for c in feature_cols if c in merged_df.columns] + prior_pred_cols

    sub = merged_df[['child_id'] + fcols + [tcol]].copy()
    sub = sub.dropna(subset=[tcol])
    if len(sub) == 0:
        return None, None, None

    ids = sub['child_id'].values
    X = sub[fcols]
    y = sub[tcol].values
    return ids, X, y


def save_rolling_dataset_with_predictions(merged_df, target, age, results_by_family, rolling_csv,
                                          full_models_dir=None, feature_cols=None,
                                          prior_target_col_names=None, formula_feature_cols=None):
    """Save the rolling dataset with model-specific predictions as new target columns.

    Uses full (non-CV) models to predict missing target values so the next age's
    step can use a complete (real-where-observed, model-filled-where-missing)
    prior-target feature -- generalizes
    bmiforecast_utils.save_rolling_dataset_with_predictions to a named target
    column instead of the fixed 'y{year}bmi' pattern.
    """
    merged_df = merged_df.copy()
    tcol = target_col(target, age)

    if tcol not in merged_df.columns:
        _tmp = rolling_csv + '.tmp'
        merged_df.to_csv(_tmp, index=False)
        os.replace(_tmp, rolling_csv)
        print(f'  Saved rolling dataset to {rolling_csv}')
        return merged_df

    print(f'  Adding model-specific prediction columns for {tcol}...')
    real_mask = merged_df[tcol].notna()
    missing_mask = ~real_mask
    n_missing = int(missing_mask.sum())
    t_median = merged_df.loc[real_mask, tcol].median() if real_mask.any() else np.nan

    def _family_fcols(family):
        if formula_feature_cols is None or prior_target_col_names is None:
            return formula_feature_cols
        prior_set = set(prior_target_col_names)
        result = []
        for col in formula_feature_cols:
            if col in prior_set:
                specific = f'{col}_{family}_pred'
                result.append(specific if specific in merged_df.columns else col)
            else:
                result.append(col)
        return result

    for family in ('deeppysr', 'pysr', 'kan'):
        if family not in results_by_family:
            continue
        for _mname, (formula, _preds, _r2) in results_by_family[family].items():
            pred_col = f'{tcol}_{family}_pred'
            involved_str = ','.join(_extract_variables(formula))
            all_preds = apply_formula(merged_df, formula, involved_str,
                                      feature_cols=_family_fcols(family))
            merged_df[pred_col] = merged_df[tcol].copy()
            merged_df.loc[missing_mask, pred_col] = all_preds[missing_mask]
            still_nan = merged_df[pred_col].isna()
            if still_nan.any():
                merged_df.loc[still_nan, pred_col] = t_median
            n_filled = int(merged_df.loc[missing_mask, pred_col].notna().sum()) if n_missing else 0
            print(f'  {family}: {n_filled}/{n_missing} missing {tcol} filled via full formula')

    if 'baseline' in results_by_family:
        fitted_models = _load_full_baseline_models(full_models_dir) if full_models_dir else {}
        for model_name, (_label, _preds, _r2) in results_by_family['baseline'].items():
            pred_col = f'{tcol}_{model_name}_pred'
            merged_df[pred_col] = merged_df[tcol].copy()

            model = fitted_models.get(model_name)
            if model is not None and feature_cols is not None and n_missing > 0:
                feat_cols = [c for c in (feature_cols or []) if c in merged_df.columns]
                for pcol in (prior_target_col_names or []):
                    specific = f'{pcol}_{model_name}_pred'
                    feat_cols.append(specific if specific in merged_df.columns else pcol)
                feat_cols = [c for c in feat_cols if c in merged_df.columns]
                if feat_cols:
                    X_missing = merged_df.loc[missing_mask, feat_cols]
                    try:
                        preds_arr = model.predict(X_missing.values)
                        merged_df.loc[missing_mask, pred_col] = preds_arr
                    except Exception as ex:
                        print(f'  WARNING: {model_name}.predict failed: {ex}')

            still_nan = merged_df[pred_col].isna()
            if still_nan.any():
                merged_df.loc[still_nan, pred_col] = t_median
            n_filled = int(merged_df.loc[missing_mask, pred_col].notna().sum()) if n_missing else 0
            print(f'  {model_name}: {n_filled}/{n_missing} missing {tcol} filled via full model')

    _tmp = rolling_csv + '.tmp'
    merged_df.to_csv(_tmp, index=False)
    os.replace(_tmp, rolling_csv)
    print(f'  Saved rolling dataset to {rolling_csv}')
    return merged_df