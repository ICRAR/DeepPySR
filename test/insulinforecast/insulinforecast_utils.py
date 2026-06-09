"""Utilities for rolling insulin/glucose forecast pipeline."""
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import r2_score

_current_dir = os.path.dirname(os.path.abspath(__file__))

_CACHE_DIR = os.path.abspath(
    os.path.join(_current_dir, '../../test_data/Health/raine/insulin_glucose_keepto14'))
_INSULIN_RESULTS_DIR = os.path.abspath(
    os.path.join(_current_dir, '../insulin/results_insulin'))
_BASE_DATASET_CACHE = os.path.join(_CACHE_DIR, 'base_dataset.csv')

AGES = [17, 20, 22, 27, 28]
TARGETS = ['insulin', 'glucose']

# Actual column names in the keepto14 cache files
TARGET_COLS = {
    17: {'insulin': 'g2_insulin_orig_yr17', 'glucose': 'g2_glucose_yr17'},
    20: {'insulin': 'g2_insulin_orig_yr20', 'glucose': 'g2_glucose_yr20'},
    22: {'insulin': 'g2_insulin_yr22',      'glucose': 'g2_glucose_yr22'},
    27: {'insulin': 'g2_insulin_yr27',      'glucose': 'g2_glucose_yr27'},
    28: {'insulin': 'g2_insulin_yr28',      'glucose': 'g2_glucose_yr28'},
}

_ALL_TARGET_COLS = set(col for tc in TARGET_COLS.values() for col in tc.values())

# Match x0, x1, ... style variables in formulas
_XI_RE = re.compile(r'^x(\d+)$')
_KNOWN_FUNCTIONS = {'exp', 'cos', 'sin', 'log', 'sqrt', 'abs', 'cond'}
_KNOWN_CONSTANTS = {'pi', 'e'}


def get_target_col(age, target):
    return TARGET_COLS[age][target]


def _is_target_col(c):
    """True if c is an actual target column or a _pred variant of one."""
    if c in _ALL_TARGET_COLS:
        return True
    return any(c.startswith(tc + '_') for tc in _ALL_TARGET_COLS)


def cond(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.where(x > 0, y, 0.0)


def _extract_variables(formula):
    tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', formula)
    return sorted(set(t for t in tokens
                      if t not in _KNOWN_FUNCTIONS and t not in _KNOWN_CONSTANTS))


def apply_formula(df, formula, involved_str, feature_cols=None):
    """Evaluate a symbolic formula on df rows. Returns Series of predictions."""
    involved_raw = [c.strip() for c in involved_str.split(',')]

    xi_to_col = {}
    if feature_cols is not None:
        for name in involved_raw:
            m = _XI_RE.match(name)
            if m:
                i = int(m.group(1))
                if i < len(feature_cols):
                    xi_to_col[name] = feature_cols[i]

    if xi_to_col:
        real_cols = list(dict.fromkeys(
            xi_to_col[n] for n in involved_raw
            if n in xi_to_col and xi_to_col[n] in df.columns
        ))
    else:
        real_cols = [c for c in involved_raw if c in df.columns]

    if not real_cols:
        return pd.Series(np.nan, index=df.index)

    mask = df[real_cols].notna().all(axis=1)
    result = pd.Series(np.nan, index=df.index)
    if mask.sum() == 0:
        return result

    sub = df.loc[mask, real_cols].copy()
    if xi_to_col:
        local_ns = {xi: sub[col].values for xi, col in xi_to_col.items() if col in sub.columns}
    else:
        local_ns = {col: sub[col].values for col in real_cols}

    local_ns.update({
        'exp': np.exp, 'cos': np.cos, 'sin': np.sin, 'log': np.log,
        'sqrt': np.sqrt, 'abs': np.abs, 'pi': math.pi, 'e': math.e,
        'cond': cond,
    })
    try:
        vals = eval(formula, {"__builtins__": {}}, local_ns)
        result.loc[mask] = vals
    except Exception as ex:
        print(f"  [formula eval error] {ex}")
    return result


def _get_best_formula_from_results(run_dir, X, y, model_type='deeppysr'):
    """Return best formula from existing CV results directory."""
    if model_type == 'deeppysr':
        pattern = os.path.join(run_dir, 'deeppysr', '**', 'relationships_fold*.csv')
    elif model_type == 'pysr':
        pattern = os.path.join(run_dir, 'pysr', '**', 'formulas_fold*.csv')
    else:
        return None, None

    files = glob.glob(pattern, recursive=True)
    if not files:
        return None, None

    best_r2 = -float('inf')
    best_formula = None
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'formula' not in df.columns:
                continue
            for _, row in df.iterrows():
                formula = str(row['formula'])
                try:
                    _analysis_dir = os.path.abspath(os.path.join(_current_dir, '..'))
                    if _analysis_dir not in sys.path:
                        sys.path.insert(0, _analysis_dir)
                    from analysis_utils import evaluate_formula
                    y_pred = evaluate_formula(formula, X, model_type=model_type)
                    mask = ~np.isnan(y_pred)
                    if mask.sum() < 2:
                        continue
                    r2_val = r2_score(np.asarray(y)[mask], y_pred[mask])
                    if r2_val > best_r2:
                        best_r2 = r2_val
                        best_formula = formula
                except Exception:
                    continue
        except Exception:
            continue

    if best_formula is None:
        return None, None
    involved = ','.join(_extract_variables(best_formula))
    return best_formula, involved


def prepare_base_dataset():
    """Load and merge all age cache files into a single rolling dataset.

    Uses the age-17 cache as the background features source. Outer-merges
    target columns from subsequent ages. Imputes missing background features
    for children only observed at later ages.

    Creates model-specific prediction columns for insulin/glucose at age 17
    using existing insulin model results (if available).

    Returns:
        merged_df: DataFrame with background features + target columns for all ages.
        non_target_cols: list of non-target, non-id feature column names.
    """
    print('\n=== Loading insulin/glucose data ===')

    if os.path.exists(_BASE_DATASET_CACHE):
        print(f'  Loading base dataset from cache: {_BASE_DATASET_CACHE}')
        merged = pd.read_csv(_BASE_DATASET_CACHE, low_memory=False)
        non_target_cols = [c for c in merged.columns
                           if c != 'child_id' and not _is_target_col(c)]
        print(f'  Loaded: {len(merged)} rows, {len(non_target_cols)} feature cols')

    else:
        # Load age 17 cache as the base (background features + age-17 targets)
        cache_17 = os.path.join(_CACHE_DIR, 'insulin_glucose_17.csv')
        if not os.path.exists(cache_17):
            raise FileNotFoundError(f'Cache file not found: {cache_17}')
        merged = pd.read_csv(cache_17, low_memory=False)
        print(f'  Base (age 17): {len(merged)} rows, {len(merged.columns)} cols')

        # Merge target columns from subsequent ages, keeping only common background columns
        for age in AGES[1:]:
            cache_path = os.path.join(_CACHE_DIR, f'insulin_glucose_{age}.csv')
            if not os.path.exists(cache_path):
                print(f'  WARNING: Cache not found for age {age}, skipping.')
                continue
            df_age = pd.read_csv(cache_path, low_memory=False)

            # Restrict both sides to their common background feature columns + child_id + targets
            target_yr_cols = [c for c in TARGET_COLS[age].values() if c in df_age.columns]
            common_bg = [c for c in merged.columns
                         if c != 'child_id' and not _is_target_col(c) and c in df_age.columns]
            merged = merged[['child_id'] + common_bg +
                            [c for c in merged.columns if _is_target_col(c)]]
            df_age = df_age[['child_id'] + common_bg + target_yr_cols]

            merged = pd.merge(merged, df_age, on='child_id', how='outer', suffixes=('', '_new'))
            # Coalesce duplicated background columns (take left value, fall back to right)
            for col in common_bg:
                new_col = col + '_new'
                if new_col in merged.columns:
                    merged[col] = merged[col].combine_first(merged[new_col])
                    merged.drop(columns=[new_col], inplace=True)

            print(f'  After merging age {age}: {len(merged)} rows, '
                  f'{len(common_bg)} common background cols')

        # Save base dataset before further processing
        merged.to_csv(_BASE_DATASET_CACHE, index=False)
        print(f'  Base dataset saved to: {_BASE_DATASET_CACHE}')

        # Recompute non_target_cols after merge loop (common_bg may have dropped columns)
        non_target_cols = [c for c in merged.columns
                           if c != 'child_id' and not _is_target_col(c)]

        print(f'\nTotal dataset: {len(merged)} rows, {len(merged.columns)} cols')
        print(f'Non-target feature cols: {len(non_target_cols)}')

    print('\n=== Creating model-specific prediction columns for age 17 ===')
    merged = _create_age17_pred_cols(merged, non_target_cols)

    merged = merged.copy()
    return merged, non_target_cols


def _smart_impute(df, cols):
    """Impute categorical cols (mode) and continuous cols (IterativeImputer)."""
    if not cols:
        return df
    result = df.copy()
    cat_cols = [c for c in cols if c in result.columns and result[c].nunique() < 13]
    cont_cols = [c for c in cols if c in result.columns and c not in cat_cols]
    if cat_cols:
        result[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(result[cat_cols])
    if cont_cols:
        result[cont_cols] = IterativeImputer(max_iter=10, random_state=42).fit_transform(result[cont_cols])
    return result


def _create_age17_pred_cols(merged, non_target_cols):
    """Create model-specific prediction columns for age-17 insulin and glucose.

    Tries to load best formulas from existing insulin/results_insulin/ runs.
    Falls back to observed values where models are unavailable.
    """
    for target in TARGETS:
        target_col = TARGET_COLS[17][target]
        if target_col not in merged.columns:
            print(f'  WARNING: {target_col} not in dataset, skipping.')
            continue

        real_mask = merged[target_col].notna()
        missing_mask = ~real_mask
        n_missing = int(missing_mask.sum())
        target_median = merged.loc[real_mask, target_col].median() if real_mask.any() else np.nan

        print(f'\n  {target_col}: {int(real_mask.sum())} observed, {n_missing} missing')

        feature_cols = [c for c in non_target_cols if c in merged.columns]
        X_known = merged.loc[real_mask, feature_cols].reset_index(drop=True)
        y_known = merged.loc[real_mask, target_col].values

        for model_type in ['deeppysr', 'pysr']:
            pred_col = f'{target_col}_{model_type}_pred'
            merged[pred_col] = merged[target_col].copy()

            if n_missing == 0:
                continue

            run_dir = os.path.join(_INSULIN_RESULTS_DIR, f'age_17_{target}')
            if not os.path.exists(run_dir):
                print(f'    No results dir for {model_type} age-17 {target}, using median fallback.')
                merged.loc[missing_mask, pred_col] = target_median
                continue

            formula, involved = _get_best_formula_from_results(run_dir, X_known, y_known, model_type)
            if formula is None:
                print(f'    No formula found for {model_type} age-17 {target}, using median fallback.')
                merged.loc[missing_mask, pred_col] = target_median
                continue

            preds = apply_formula(merged, formula, involved, feature_cols=feature_cols)
            merged.loc[missing_mask, pred_col] = preds[missing_mask]
            still_nan = merged[pred_col].isna()
            if still_nan.any():
                merged.loc[still_nan, pred_col] = target_median
            n_filled = int(merged.loc[missing_mask, pred_col].notna().sum())
            print(f'    {model_type}: {n_filled}/{n_missing} missing filled via formula')

        # Baseline models
        _bl_utils_dir = os.path.abspath(os.path.join(_current_dir, '..'))
        if _bl_utils_dir not in sys.path:
            sys.path.insert(0, _bl_utils_dir)
        try:
            from model_utils import get_baseline_models as _get_bl
            import joblib as _jl
        except Exception:
            continue

        bl_models = _get_bl(task='regression', input_dim=len(feature_cols))
        X_train = merged.loc[real_mask, feature_cols].values
        X_miss = merged.loc[missing_mask, feature_cols].values if n_missing > 0 else None

        for model_name in bl_models:
            if model_name == 'KAN':
                continue
            pred_col = f'{target_col}_{model_name}_pred'
            merged[pred_col] = merged[target_col].copy()
            if n_missing == 0:
                continue
            try:
                import sklearn.base as _sb
                m = _sb.clone(bl_models[model_name])
                m.fit(X_train, y_known)
                preds = m.predict(X_miss)
                merged.loc[missing_mask, pred_col] = preds
            except Exception as ex:
                print(f'    WARNING: {model_name} failed: {ex}')
            still_nan = merged[pred_col].isna()
            if still_nan.any():
                merged.loc[still_nan, pred_col] = target_median

    return merged


def load_forecast_data_for_model(merged_df, age, target, prior_target_cols,
                                  non_target_cols, model_type='deeppysr'):
    """Prepare (ids, X, y) for forecasting a target at a given age.

    Uses model-specific prediction columns for prior target values where available.

    Args:
        merged_df: rolling dataset
        age: age to forecast (int from AGES[1:])
        target: 'insulin' or 'glucose'
        prior_target_cols: list of target column names for prior ages
                           (e.g. ['g2_insulin_orig_yr17', 'g2_glucose_yr17'])
        non_target_cols: list of background feature column names
        model_type: 'deeppysr', 'pysr', or baseline model name

    Returns:
        (ids, X, y) or (None, None, None)
    """
    target_col = TARGET_COLS[age][target]
    if target_col not in merged_df.columns:
        return None, None, None

    # Resolve prior target columns: prefer model-specific pred cols
    prior_pred_cols = []
    for col in prior_target_cols:
        pred_col = f'{col}_{model_type}_pred'
        if pred_col in merged_df.columns:
            prior_pred_cols.append(pred_col)
        elif col in merged_df.columns:
            prior_pred_cols.append(col)

    feature_cols = ([c for c in non_target_cols if c in merged_df.columns] +
                    prior_pred_cols)

    sub = merged_df[['child_id'] + feature_cols + [target_col]].copy()
    sub = sub.dropna(subset=[target_col])
    if len(sub) == 0:
        return None, None, None

    ids = sub['child_id'].values
    X = sub[feature_cols]
    y = sub[target_col].values
    return ids, X, y


def _load_full_baseline_models(full_models_dir):
    """Load full (non-CV) baseline models from full_models_dir/_models/."""
    import joblib as _jl
    models = {}
    models_dir = os.path.join(full_models_dir, '_models')
    if not os.path.exists(models_dir):
        return models
    for fname in os.listdir(models_dir):
        if not fname.endswith('.joblib'):
            continue
        name = fname[:-len('.joblib')]
        try:
            models[name] = _jl.load(os.path.join(models_dir, fname))
        except Exception as ex:
            print(f'  WARNING: Could not load full model {name}: {ex}')
    return models


def save_rolling_dataset_with_predictions(merged_df, age, target,
                                           results_by_family, rolling_csv,
                                           full_models_dir=None,
                                           non_target_cols=None,
                                           prior_target_col_names=None,
                                           formula_feature_cols=None):
    """Fill missing target values and save the updated rolling dataset.

    Args:
        merged_df: current rolling dataset
        age: age being processed
        target: 'insulin' or 'glucose'
        results_by_family: {family: {model_name: (formula_or_label, preds, r2)}}
        rolling_csv: path to save the updated dataset
        full_models_dir: directory with full-trained baseline models
        non_target_cols: background feature column names
        prior_target_col_names: prior-age target column names
        formula_feature_cols: maps x0, x1, ... to actual column names

    Returns:
        updated merged_df
    """
    merged_df = merged_df.copy()
    target_col = TARGET_COLS[age][target]

    if target_col not in merged_df.columns:
        _atomic_save(merged_df, rolling_csv)
        return merged_df

    real_mask = merged_df[target_col].notna()
    missing_mask = ~real_mask
    n_missing = int(missing_mask.sum())
    target_median = merged_df.loc[real_mask, target_col].median() if real_mask.any() else np.nan

    print(f'  Adding prediction columns for {target_col} '
          f'({int(real_mask.sum())} observed, {n_missing} missing)...')

    # Formula families: deeppysr, pysr, kan
    for family in ('deeppysr', 'pysr', 'kan'):
        if family not in results_by_family:
            continue
        for _mname, (formula, _preds, _r2) in results_by_family[family].items():
            pred_col = f'{target_col}_{family}_pred'
            involved_str = ','.join(_extract_variables(formula))
            all_preds = apply_formula(merged_df, formula, involved_str,
                                      feature_cols=formula_feature_cols)
            merged_df[pred_col] = merged_df[target_col].copy()
            merged_df.loc[missing_mask, pred_col] = all_preds[missing_mask]
            still_nan = merged_df[pred_col].isna()
            if still_nan.any():
                merged_df.loc[still_nan, pred_col] = target_median
            n_filled = int(merged_df.loc[missing_mask, pred_col].notna().sum()) if n_missing else 0
            print(f'    {family}: {n_filled}/{n_missing} missing filled via full formula')

    # Baseline family
    if 'baseline' in results_by_family:
        fitted_models = _load_full_baseline_models(full_models_dir) if full_models_dir else {}
        for model_name, (_label, _preds, _r2) in results_by_family['baseline'].items():
            pred_col = f'{target_col}_{model_name}_pred'
            merged_df[pred_col] = merged_df[target_col].copy()

            model = fitted_models.get(model_name)
            if model is not None and non_target_cols is not None and n_missing > 0:
                feat_cols = [c for c in (non_target_cols or []) if c in merged_df.columns]
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
                        print(f'    WARNING: {model_name}.predict failed: {ex}')

            still_nan = merged_df[pred_col].isna()
            if still_nan.any():
                merged_df.loc[still_nan, pred_col] = target_median
            n_filled = int(merged_df.loc[missing_mask, pred_col].notna().sum()) if n_missing else 0
            print(f'    {model_name}: {n_filled}/{n_missing} missing filled via full model')

    # Fill base target_col with first available pred column
    if n_missing > 0:
        pred_cols = [c for c in merged_df.columns
                     if c.startswith(f'{target_col}_') and c.endswith('_pred')]
        for pc in pred_cols:
            still_missing = merged_df[target_col].isna()
            if not still_missing.any():
                break
            merged_df.loc[still_missing, target_col] = merged_df.loc[still_missing, pc]
        still_missing = merged_df[target_col].isna().sum()
        if still_missing:
            merged_df[target_col].fillna(target_median, inplace=True)
            print(f'    Median fallback for {still_missing} remaining rows of {target_col}.')

    _atomic_save(merged_df, rolling_csv)
    return merged_df


def _atomic_save(df, path):
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    print(f'  Rolling dataset saved to {path}')
