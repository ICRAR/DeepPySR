import math
import os
import re

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_current_dir = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(os.path.join(_current_dir, '../../test_data/Health/bmi'))
REL_AGE8 = os.path.abspath(os.path.join(
    _current_dir, '../bmi/results_bmi_all/bmi_best_models_metrics.csv'))

YEARS = [8, 10, 13, 16, 20, 23, 26]
AGE_MAPPING = {13: 14, 16: 17, 26: 27}


def actual_age(year):
    return AGE_MAPPING.get(year, year)


def load_year_df(year):
    path = os.path.join(DATA_DIR, f'rawdata_yr{year}.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# cond(x, y) = y if x > 0 else 0  (matches the PySR/DeepPySR definition in model_utils.py)
def cond(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.where(x > 0, y, 0.0)


_KNOWN_FUNCTIONS = {'exp', 'cos', 'sin', 'log', 'sqrt', 'abs', 'cond'}
_KNOWN_CONSTANTS = {'pi', 'e'}

_XI_RE = re.compile(r'^x(\d+)$')
_bmi_age8_feature_cols_cache = None


def _get_bmi_age8_feature_cols():
    """Return the feature column list used for bmi age_8 model training."""
    global _bmi_age8_feature_cols_cache
    if _bmi_age8_feature_cols_cache is None:
        import sys as _sys
        bmi_dir = os.path.abspath(os.path.join(_current_dir, '..', 'bmi'))
        if bmi_dir not in _sys.path:
            _sys.path.insert(0, bmi_dir)
        try:
            from bmi_utils import load_bmi_agg_data as _load
            _, X, _ = _load(age=8)
            X = X.drop(columns=['age'])
            _bmi_age8_feature_cols_cache = list(X.columns)
        except Exception as e:
            print(f"  WARNING: Could not load bmi age_8 feature cols: {e}")
            _bmi_age8_feature_cols_cache = []
    return _bmi_age8_feature_cols_cache


def _extract_variables(formula):
    """Extract variable names from a formula string by removing known functions/constants."""
    tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', formula)
    return sorted(set(t for t in tokens if t not in _KNOWN_FUNCTIONS and t not in _KNOWN_CONSTANTS))


def get_best_formula_age8_for_model(model_type='deeppysr', df_for_eval=None):
    """Get the best formula for a specific model from age8 results.

    For deeppysr, pysr, kansym: looks up the formula from bmi_best_models_metrics.csv
    (type=age-specific, age=8) using the matching display_model label.

    Args:
        model_type: 'deeppysr', 'pysr', or 'kansym'
        df_for_eval: unused, kept for API compatibility

    Returns:
        (formula, involved_str) or (None, None)
    """
    _display_model_map = {
        'deeppysr': 'Best DeepPySR',
        'pysr': 'Best PySR',
        'kansym': 'KANSym',
    }

    display_model = _display_model_map.get(model_type)
    if display_model is None:
        print(f"  WARNING: model_type '{model_type}' not supported by get_best_formula_age8_for_model")
        return None, None

    if not os.path.exists(REL_AGE8):
        print(f"  WARNING: metrics CSV not found at {REL_AGE8}")
        return None, None

    try:
        df_metrics = pd.read_csv(REL_AGE8)
    except Exception as e:
        print(f"  WARNING: failed to read metrics CSV: {e}")
        return None, None

    mask = (
        (df_metrics['age'] == 8) &
        (df_metrics['type'] == 'age-specific') &
        (df_metrics['display_model'] == display_model)
    )
    rows = df_metrics[mask]
    if rows.empty:
        print(f"  WARNING: No age-8 age-specific entry for display_model='{display_model}'")
        return None, None

    best_row = rows.sort_values('r2', ascending=False).iloc[0]
    best_formula = str(best_row['formula'])
    best_r2 = float(best_row['r2'])

    involved = ','.join(_extract_variables(best_formula))
    print(f"  [{model_type} best formula r2={best_r2:.4f}]: {best_formula}")
    print(f"  [involved]: {involved}")
    return best_formula, involved


def get_baseline_predictions_age8(model_name):
    """Load baseline model predictions from age8 results.

    Args:
        model_name: name of baseline model (e.g., 'RandomForest', 'XGBoost')

    Returns:
        dict: {child_id: prediction} or empty dict if not found
    """
    age_specific_dir = os.path.abspath(os.path.join(
        _current_dir, '../bmi/results_bmi_all/age_specific'))
    predictions_file = os.path.join(
        age_specific_dir, 'age_8', 'baselines', model_name, 'predictions.csv')

    if not os.path.exists(predictions_file):
        print(f"  WARNING: Predictions file not found at {predictions_file}")
        return {}

    try:
        df = pd.read_csv(predictions_file)
        # predictions.csv has columns: y_true, y_pred, id, age (and possibly y_pred_kansym)
        if 'id' in df.columns and 'y_pred' in df.columns:
            return dict(zip(df['id'], df['y_pred']))
        elif 'child_id' in df.columns and 'y_pred' in df.columns:
            return dict(zip(df['child_id'], df['y_pred']))
        elif 'child_id' in df.columns and 'prediction' in df.columns:
            return dict(zip(df['child_id'], df['prediction']))
        else:
            print(f"  WARNING: Unexpected columns in {predictions_file}: {list(df.columns)}")
            return {}
    except Exception as e:
        print(f"  WARNING: Error reading {predictions_file}: {e}")
        return {}


def get_age8_baseline_model_names():
    """Return sorted list of baseline model names from age_8 results directory."""
    age_specific_dir = os.path.abspath(os.path.join(
        _current_dir, '../bmi/results_bmi_all/age_specific'))
    baselines_dir = os.path.join(age_specific_dir, 'age_8', 'baselines')
    if not os.path.exists(baselines_dir):
        return []
    return sorted([d for d in os.listdir(baselines_dir)
                   if os.path.isdir(os.path.join(baselines_dir, d))])


_baseline_models_age8_cache = None


def _train_baseline_models_age8():
    """Train baseline models on age-8 BMI data and cache the fitted models.

    Models are also saved to disk under results_bmi_all/age_specific/age_8/baselines/_fitted_models/
    so they can be inspected later.  Returns dict {model_name: fitted_model}.
    """
    global _baseline_models_age8_cache
    if _baseline_models_age8_cache is not None:
        return _baseline_models_age8_cache

    import sys as _sys
    import joblib

    bmi_dir = os.path.abspath(os.path.join(_current_dir, '..', 'bmi'))
    if bmi_dir not in _sys.path:
        _sys.path.insert(0, bmi_dir)

    model_utils_dir = os.path.abspath(os.path.join(_current_dir, '..'))
    if model_utils_dir not in _sys.path:
        _sys.path.insert(0, model_utils_dir)

    try:
        from bmi_utils import load_bmi_agg_data as _load
        from model_utils import get_baseline_models as _get_models
    except Exception as e:
        print(f'  WARNING: Could not import dependencies for baseline training: {e}')
        _baseline_models_age8_cache = {}
        return _baseline_models_age8_cache

    try:
        ids, X, y = _load(age=8)
        X = X.drop(columns=['age'])
    except Exception as e:
        print(f'  WARNING: load_bmi_agg_data(age=8) failed: {e}')
        _baseline_models_age8_cache = {}
        return _baseline_models_age8_cache

    models = _get_models(task='regression', input_dim=X.shape[1])
    save_dir = os.path.abspath(os.path.join(
        _current_dir, '../bmi/results_bmi_all/age_specific/age_8/baselines/_fitted_models'))
    os.makedirs(save_dir, exist_ok=True)

    X_arr = X.values
    fitted = {}
    for name, model in models.items():
        # Try loading from disk first
        ckpt_path = os.path.join(save_dir, 'KAN')
        path = os.path.join(save_dir, f'{name}.joblib')
        if name == 'KAN':
            if os.path.exists(f'{ckpt_path}_config.yml'):
                try:
                    from kan import KAN as _KAN
                    model.model = _KAN.loadckpt(ckpt_path)
                    fitted[name] = model
                    print(f'  Loaded saved baseline model: {name} (checkpoint)')
                    continue
                except Exception as load_e:
                    print(f'  Could not load {name} checkpoint, retraining: {load_e}')
        else:
            path = os.path.join(save_dir, f'{name}.joblib')
            if os.path.exists(path):
                try:
                    fitted[name] = joblib.load(path)
                    print(f'  Loaded saved baseline model: {name}')
                    continue
                except Exception as load_e:
                    print(f'  Could not load {name}, retraining: {load_e}')

        # Train if not loaded
        try:
            model.fit(X_arr, y)
            fitted[name] = model
            if name == 'KAN' and hasattr(model, 'model') and hasattr(model.model, 'saveckpt'):
                try:
                    model.model.saveckpt(ckpt_path)
                    print(f'  Trained and saved baseline model: {name} (checkpoint)')
                except Exception as save_e:
                    print(f'  Trained {name} (skipping disk save: {save_e})')
            else:
                try:
                    joblib.dump(model, path)
                    print(f'  Trained and saved baseline model: {name}')
                except Exception as save_e:
                    print(f'  Trained {name} (skipping disk save: {save_e})')
        except Exception as e:
            print(f'  WARNING: Failed to train {name}: {e}')
            fitted[name] = None

    _baseline_models_age8_cache = fitted
    return fitted


def get_best_pysr_formula_for_year(run_out_dir, X, y):
    """Extract the best PySR formula from a completed year's results directory.

    Reads all formulas_fold*.csv files under pysr subdirs, evaluates each formula
    against the provided X and y using evaluate_formula, and picks the one with
    the highest r2.
    Returns (formula, involved_str) or (None, None).
    """
    import glob
    import sys
    # Import evaluate_formula from analysis_utils (one level up)
    _analysis_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _analysis_dir not in sys.path:
        sys.path.insert(0, _analysis_dir)
    from analysis_utils import evaluate_formula

    best_r2 = -float('inf')
    best_formula = None

    pysr_dir = os.path.join(run_out_dir, 'pysr')
    if not os.path.exists(pysr_dir):
        return None, None

    pattern = os.path.join(pysr_dir, '**', 'formulas_fold*.csv')
    formula_files = glob.glob(pattern, recursive=True)

    for ff in formula_files:
        try:
            fdf = pd.read_csv(ff)
            if 'formula' not in fdf.columns:
                continue
            for _, row in fdf.iterrows():
                formula = str(row['formula'])
                try:
                    y_pred = evaluate_formula(formula, X, model_type='pysr')
                    from sklearn.metrics import r2_score as _r2
                    mask = ~np.isnan(y_pred)
                    if mask.sum() < 2:
                        continue
                    r2_val = _r2(np.asarray(y)[mask], y_pred[mask])
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
    print(f"  [PySR best formula r2={best_r2:.4f}]: {best_formula}")
    print(f"  [involved]: {involved}")
    return best_formula, involved


def save_baseline_models(run_out_dir, models_dict):
    """Save fitted baseline models to disk.

    KAN models are saved via saveckpt; all others use joblib.

    Args:
        run_out_dir: directory for this year's results (e.g. results_bmiforecast/age_10)
        models_dict: dict of {model_name: fitted_model}
    """
    import joblib
    models_save_dir = os.path.join(run_out_dir, 'baselines', '_fitted_models')
    os.makedirs(models_save_dir, exist_ok=True)
    for name, model in models_dict.items():
        if name == 'KAN' and hasattr(model, 'model') and hasattr(model.model, 'saveckpt'):
            ckpt_path = os.path.join(models_save_dir, 'KAN')
            try:
                model.model.saveckpt(ckpt_path)
                print(f"  Saved fitted model: KAN (checkpoint)")
            except Exception as ex:
                print(f"  Failed to save KAN checkpoint: {ex}")
        else:
            path = os.path.join(models_save_dir, f'{name}.joblib')
            try:
                joblib.dump(model, path)
                print(f"  Saved fitted model: {path}")
            except Exception as ex:
                print(f"  Failed to save {name}: {ex}")


def load_baseline_models(run_out_dir):
    """Load all fitted baseline models saved by save_baseline_models.

    KAN models are loaded via loadckpt; all others use joblib.
    Returns dict of {model_name: fitted_model}, empty dict if none found.
    """
    import joblib
    import glob
    import sys as _sys
    models_save_dir = os.path.join(run_out_dir, 'baselines', '_fitted_models')
    if not os.path.exists(models_save_dir):
        return {}
    models = {}

    # Load KAN via checkpoint if present
    kan_ckpt = os.path.join(models_save_dir, 'KAN')
    if os.path.exists(f'{kan_ckpt}_config.yml'):
        try:
            model_utils_dir = os.path.abspath(os.path.join(_current_dir, '..'))
            if model_utils_dir not in _sys.path:
                _sys.path.insert(0, model_utils_dir)
            from kan import KAN as _KAN
            from model_utils import KANWrapper as _KANWrapper
            wrapper = _KANWrapper.__new__(_KANWrapper)
            wrapper.model = _KAN.loadckpt(kan_ckpt)
            models['KAN'] = wrapper
            print("  Loaded fitted model: KAN (checkpoint)")
        except Exception as ex:
            print(f"  Failed to load KAN checkpoint: {ex}")

    # Load all joblib models
    for path in glob.glob(os.path.join(models_save_dir, '*.joblib')):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            models[name] = joblib.load(path)
            print(f"  Loaded fitted model: {name}")
        except Exception as ex:
            print(f"  Failed to load {path}: {ex}")

    return models


def fill_missing_bmi_with_model(df, bmi_col, model, feature_cols):
    """Fill missing values in bmi_col using a fitted sklearn model.

    Args:
        df: DataFrame with feature_cols and bmi_col
        bmi_col: target column name
        model: fitted sklearn estimator with .predict()
        feature_cols: list of feature column names used during training
    Returns updated DataFrame.
    """
    df = df.copy()
    missing_mask = df[bmi_col].isna()
    n_missing = missing_mask.sum()
    if n_missing == 0:
        return df
    print(f"  Filling {n_missing} missing {bmi_col} values via baseline model.")
    avail_cols = [c for c in feature_cols if c in df.columns]
    can_predict = df.loc[missing_mask, avail_cols].notna().all(axis=1)
    idx = missing_mask[missing_mask].index[can_predict]
    if len(idx) > 0:
        X_pred = df.loc[idx, avail_cols]
        try:
            preds = model.predict(X_pred)
            df.loc[idx, bmi_col] = preds
        except Exception as ex:
            print(f"  [model predict error] {ex}")
    still_missing = df[bmi_col].isna().sum()
    if still_missing > 0:
        median_val = df[bmi_col].median()
        df[bmi_col].fillna(median_val, inplace=True)
        print(f"  Fallback median imputation for {still_missing} remaining rows.")
    return df


def get_best_baseline_model(run_out_dir):
    """Return the name of the baseline model with the highest mean r2.

    Reads overall_metrics.csv files under baselines subdirs (excluding _fitted_models).
    Returns model_name or None.
    """
    import glob
    best_r2 = -float('inf')
    best_name = None
    baselines_dir = os.path.join(run_out_dir, 'baselines')
    if not os.path.exists(baselines_dir):
        return None
    pattern = os.path.join(baselines_dir, '*', 'overall_metrics.csv')
    for mf in glob.glob(pattern):
        name = os.path.basename(os.path.dirname(mf))
        if name == '_fitted_models':
            continue
        try:
            df = pd.read_csv(mf)
            r2_val = df['r2'].mean() if 'r2' in df.columns else -float('inf')
            if r2_val > best_r2:
                best_r2 = r2_val
                best_name = name
        except Exception:
            continue
    if best_name:
        print(f"  [best baseline model r2={best_r2:.4f}]: {best_name}")
    return best_name


def get_best_formula_for_year(run_out_dir):
    """Extract the best DeepPySR formula from a completed year's results directory.

    Reads all relationships_fold*.csv files under deeppysr subdirs, picks the
    formula with the highest r2 (stored in the CSV), and returns (formula, involved_str).
    Returns (None, None) if no formula files are found.
    """
    import glob
    best_r2 = -float('inf')
    best_formula = None

    deeppysr_dir = os.path.join(run_out_dir, 'deeppysr')
    if not os.path.exists(deeppysr_dir):
        return None, None

    pattern = os.path.join(deeppysr_dir, '**', 'relationships_fold*.csv')
    files = glob.glob(pattern, recursive=True)
    # Also check non-fold variant
    pattern2 = os.path.join(deeppysr_dir, '**', 'relationships.csv')
    files += glob.glob(pattern2, recursive=True)

    for f in files:
        try:
            df = pd.read_csv(f)
            if 'formula' not in df.columns:
                continue
            r2_col = 'r2' if 'r2' in df.columns else None
            for _, row in df.iterrows():
                formula = str(row['formula'])
                r2 = float(row[r2_col]) if r2_col and pd.notna(row[r2_col]) else -float('inf')
                if r2 > best_r2:
                    best_r2 = r2
                    best_formula = formula
        except Exception:
            continue

    if best_formula is None:
        return None, None

    involved = ','.join(_extract_variables(best_formula))
    print(f"  [best formula r2={best_r2:.4f}]: {best_formula}")
    print(f"  [involved]: {involved}")
    return best_formula, involved


def apply_formula(df, formula, involved_str, feature_cols=None):
    """Evaluate the symbolic formula on df rows.
    Returns a Series of predicted values (NaN where required columns are missing).

    feature_cols: if provided, maps x0, x1, ... in the formula to these column names.
                  Required when the formula uses index-based variables (x0, x1, ...)
                  as produced by DeepPySR/PySR rather than real column names.
    """
    involved_raw = [c.strip() for c in involved_str.split(',')]

    # Build xi→col mapping when feature_cols is provided and vars look like x<int>
    xi_to_col = {}
    if feature_cols is not None:
        for name in involved_raw:
            m = _XI_RE.match(name)
            if m:
                i = int(m.group(1))
                if i < len(feature_cols):
                    xi_to_col[name] = feature_cols[i]

    # Real column names for building the not-null mask
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


def fill_missing_bmi_with_formula(df, bmi_col, formula, involved_str, feature_cols=None):
    """Fill missing values in bmi_col using the symbolic formula.
    Rows where the formula cannot be evaluated fall back to the column median.
    Returns the updated DataFrame.
    """
    df = df.copy()
    missing_mask = df[bmi_col].isna()
    n_missing = missing_mask.sum()
    if n_missing == 0:
        return df
    print(f"  Filling {n_missing} missing {bmi_col} values via formula.")
    preds = apply_formula(df, formula, involved_str, feature_cols=feature_cols)
    df.loc[missing_mask, bmi_col] = preds[missing_mask]
    still_missing = df[bmi_col].isna().sum()
    if still_missing > 0:
        median_val = df[bmi_col].median()
        df[bmi_col].fillna(median_val, inplace=True)
        print(f"  Fallback median imputation for {still_missing} remaining rows.")
    return df


_CATEGORICAL_PREFIXES = (
    'cohab', 'occupcode', 'edu', 'cob', 'asthma', 'prepreg_psych', 'ppd',
    'smk_t2', 'prepreg_alc', 'preg_plan', 'mode_delivery', 'plac_abrup',
    'occup', 'ethn', 'death_p', 'neo_unit', 'sex', 'con_anomalies',
    'breastfed_ever', 'smk_exp', 'hhincome', 'fam_splitup', 'cats_preg',
    'dogs_preg', 'dogs_2', 'cats_2', 'prepreg_smk', 'preg_smk', 'smk_t3',
    'region', 'miggen_child', 'smk_p', 'childcare_2', 'childcarerel',
    'childcareprof', 'childcarecentre', 'childcare_1', 'alc_t', 'preg_alc',
    'ivf', 'abroad', 'm_vege', 'm_educ', 'm_schyr', 'm_schlev',
)
_CATEGORICAL_EXACT = {
    'preg_dia', 'preg_thyroid', 'preg_fever', 'preeclam', 'preg_ht',
}


def _is_categorical_col(col):
    """Return True if the column name matches the manually defined categorical list."""
    if col in _CATEGORICAL_EXACT:
        return True
    for prefix in _CATEGORICAL_PREFIXES:
        if col == prefix or col.startswith(prefix + '_') or col.startswith(prefix):
            return True
    return False


def _detect_categorical(cols, max_unique=None):
    """Return (cat_cols, cont_cols) split based on the manually defined categorical variable list."""
    cat_cols = [c for c in cols if _is_categorical_col(c)]
    cont_cols = [c for c in cols if not _is_categorical_col(c)]
    return cat_cols, cont_cols


def replace_99_with_nan(df, cols=None):
    """Replace sentinel value 99 with NaN in the specified columns (or all columns if None)."""
    if cols is None:
        cols = df.columns.tolist()
    result = df.copy()
    for c in cols:
        if c in result.columns:
            result[c] = result[c].replace(99, np.nan)
    return result


def smart_impute(df, cols):
    """Impute categorical cols with mode and continuous cols with IterativeImputer.
    Returns a DataFrame with the same columns, imputed in-place copy.
    """
    if not cols:
        return df
    cat_cols, cont_cols = _detect_categorical(cols)
    result = df.copy()
    if cat_cols:
        result[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(result[cat_cols])
    if cont_cols:
        # Drop all-NaN columns before imputation; IterativeImputer skips them,
        # causing a column-count mismatch when assigning results back.
        valid_cont = [c for c in cont_cols if result[c].notna().any()]
        iter_imp = IterativeImputer(max_iter=10, random_state=42)
        result[valid_cont] = iter_imp.fit_transform(result[valid_cont])
    return result


def drop_low_variance_and_extra(df):
    """Drop constant columns and known irrelevant columns."""
    drop_always = ['mother_id', 'preg_no', 'cohab_0']
    df = df.drop(columns=[c for c in drop_always if c in df.columns])
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    return df.drop(columns=const_cols)


def evaluate_predictions(y_true, y_pred, label=''):
    """Compute and print regression metrics; return dict."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {}
    yt, yp = y_true[mask], y_pred[mask]
    metrics = {
        'r2': r2_score(yt, yp),
        'mae': mean_absolute_error(yt, yp),
        'rmse': np.sqrt(mean_squared_error(yt, yp)),
        'n': int(mask.sum()),
    }
    print(f"  [{label}] r2={metrics['r2']:.4f}  mae={metrics['mae']:.4f}  "
          f"rmse={metrics['rmse']:.4f}  n={metrics['n']}")
    return metrics


def _is_bmi_col(c):
    return c.startswith('y') and c.endswith('bmi') and c[1:-3].isdigit()


def prepare_base_dataset():
    """Load and merge all year files, impute non-BMI variables, create model-specific y8bmi pred columns.

    For each model type (deeppysr, pysr, and each baseline), creates a column
    y8bmi_{model}_pred that contains the real y8bmi where observed and the
    model-specific prediction where y8bmi is missing.  The base y8bmi column
    is also filled (using the deeppysr prediction as primary, with median fallback)
    so the dataset has no NaN in features or y8bmi.

    Returns:
        merged (DataFrame): fully imputed dataset with model-specific y8bmi pred columns.
        non_bmi_cols (list): non-BMI, non-id feature column names.
    """
    # Load and merge all year files
    merged = None
    for year in YEARS:
        df = load_year_df(year)
        if df is None:
            continue
        if merged is None:
            merged = df
        else:
            common = [c for c in merged.columns
                      if c in df.columns and not _is_bmi_col(c)]
            merged = pd.merge(merged, df, on=common, how='outer')

    if merged is None:
        raise RuntimeError('No data files found.')

    merged = drop_low_variance_and_extra(merged)
    merged = replace_99_with_nan(merged)

    # Identify non-BMI feature columns (exclude child_id and all bmi cols)
    non_bmi_cols = [c for c in merged.columns
                    if c != 'child_id' and not _is_bmi_col(c)]

    # Remove duplicate rows before imputation
    n_before = len(merged)
    merged = merged.drop_duplicates()
    n_dropped = n_before - len(merged)
    if n_dropped:
        print(f'  Dropped {n_dropped} duplicate rows (before imputation).')

    # Impute non-BMI variables (do NOT touch any bmi columns)
    print('\n=== Imputing non-BMI variables ===')
    merged = smart_impute(merged, non_bmi_cols)

    bmi8_col = 'y8bmi'
    if bmi8_col not in merged.columns:
        return merged, non_bmi_cols

    print('\n=== Creating model-specific y8bmi prediction columns ===')
    real_mask = merged[bmi8_col].notna()
    missing_mask = ~real_mask
    n_missing = int(missing_mask.sum())
    print(f'  y8bmi: {int(real_mask.sum())} observed, {n_missing} missing')

    bmi8_median = merged.loc[real_mask, bmi8_col].median()

    # --- Formula models: deeppysr, pysr ---
    df_for_eval = merged.loc[real_mask].copy() if real_mask.any() else None
    age8_fcols = _get_bmi_age8_feature_cols()
    for model_type in ['deeppysr', 'pysr']:
        formula, involved_str = get_best_formula_age8_for_model(model_type, df_for_eval)
        pred_col = f'{bmi8_col}_{model_type}_pred'
        # Start with real values
        merged[pred_col] = merged[bmi8_col].copy()
        if formula is not None and n_missing > 0:
            preds = apply_formula(merged, formula, involved_str, feature_cols=age8_fcols)
            merged.loc[missing_mask, pred_col] = preds[missing_mask]
        # Median fallback for any remaining NaN (formula could not evaluate)
        still_nan = merged[pred_col].isna()
        if still_nan.any():
            raise ValueError(f'Formula {model_type} failed to fill all missing values for {bmi8_col}')
        n_filled = int(merged.loc[missing_mask, pred_col].notna().sum()) if n_missing else 0
        print(f'  {model_type}: {n_filled}/{n_missing} missing filled via formula')

    # --- Baseline models: fit on observed y8bmi rows, predict on missing rows ---
    # Feature columns: non-BMI features present in merged (no prior BMI for age-8)
    _bl_save_dir = os.path.abspath(os.path.join(
        _current_dir, '../bmi/results_bmi_all/age_specific/age_8/baselines/_fitted_models'))
    os.makedirs(_bl_save_dir, exist_ok=True)

    import sys as _sys
    import joblib as _joblib
    _model_utils_dir = os.path.abspath(os.path.join(_current_dir, '..'))
    if _model_utils_dir not in _sys.path:
        _sys.path.insert(0, _model_utils_dir)
    try:
        from model_utils import get_baseline_models as _get_bl_models
    except Exception as _ie:
        print(f'  WARNING: Could not import get_baseline_models: {_ie}')
        _get_bl_models = None

    if _get_bl_models is not None and age8_fcols:
        X_train8 = merged.loc[real_mask, age8_fcols].values
        y_train8 = merged.loc[real_mask, bmi8_col].values
        X_miss8 = merged.loc[missing_mask, age8_fcols].values if n_missing > 0 else None

        _bl_model_instances = _get_bl_models(task='regression', input_dim=len(age8_fcols))
        for model_name, model_template in _bl_model_instances.items():
            pred_col = f'{bmi8_col}_{model_name}_pred'
            merged[pred_col] = merged[bmi8_col].copy()

            # Try loading saved model first
            fitted_model = None
            _ckpt_path = os.path.join(_bl_save_dir, model_name)
            _jl_path = os.path.join(_bl_save_dir, f'{model_name}.joblib')
            if model_name == 'KAN' and os.path.exists(f'{_ckpt_path}_config.yml'):
                try:
                    from kan import KAN as _KAN
                    from model_utils import KANWrapper as _KANWrapper
                    _w = _KANWrapper.__new__(_KANWrapper)
                    _w.model = _KAN.loadckpt(_ckpt_path)
                    fitted_model = _w
                    print(f'  Loaded saved model: {model_name} (checkpoint)')
                except Exception as _le:
                    print(f'  Could not load {model_name} checkpoint, retraining: {_le}')
            elif model_name != 'KAN' and os.path.exists(_jl_path):
                try:
                    fitted_model = _joblib.load(_jl_path)
                    print(f'  Loaded saved model: {model_name}')
                except Exception as _le:
                    print(f'  Could not load {model_name}, retraining: {_le}')

            # Train if not loaded
            if fitted_model is None:
                try:
                    model_template.fit(X_train8, y_train8)
                    fitted_model = model_template
                    if model_name == 'KAN' and hasattr(model_template, 'model') and hasattr(model_template.model, 'saveckpt'):
                        try:
                            model_template.model.saveckpt(_ckpt_path)
                            print(f'  Trained and saved model: {model_name} (checkpoint)')
                        except Exception as _se:
                            print(f'  Trained {model_name} (skipping disk save: {_se})')
                    else:
                        try:
                            _joblib.dump(model_template, _jl_path)
                            print(f'  Trained and saved model: {model_name}')
                        except Exception as _se:
                            print(f'  Trained {model_name} (skipping disk save: {_se})')
                except Exception as _te:
                    print(f'  WARNING: Failed to train {model_name}: {_te}')

            if fitted_model is not None and X_miss8 is not None:
                try:
                    preds = fitted_model.predict(X_miss8)
                    merged.loc[missing_mask, pred_col] = preds
                except Exception as _pe:
                    print(f'  WARNING: {model_name} predict failed: {_pe}')

            still_nan = merged[pred_col].isna()
            if still_nan.any():
                merged.loc[still_nan, pred_col] = bmi8_median
            n_filled = int(merged.loc[missing_mask, pred_col].notna().sum()) if n_missing else 0
            print(f'  {model_name}: {n_filled}/{n_missing} missing filled via trained model')

    # --- Fill base y8bmi (for training rows that require a non-NaN target) ---
    if n_missing > 0:
        print(f'  Filling {n_missing} missing base y8bmi values...')
        primary = f'{bmi8_col}_deeppysr_pred'
        if primary in merged.columns:
            merged.loc[missing_mask, bmi8_col] = merged.loc[missing_mask, primary]
        else:
            pred_cols = [c for c in merged.columns
                         if c.startswith(f'{bmi8_col}_') and c.endswith('_pred')]
            for pc in pred_cols:
                still_missing = merged[bmi8_col].isna()
                if not still_missing.any():
                    break
                merged.loc[still_missing, bmi8_col] = merged.loc[still_missing, pc]
        # Final median fallback
        still_missing = merged[bmi8_col].isna().sum()
        if still_missing:
            merged[bmi8_col].fillna(bmi8_median, inplace=True)
            print(f'  Median fallback for {still_missing} remaining rows.')

    # Defragment after adding many columns
    merged = merged.copy()
    return merged, non_bmi_cols


def _load_full_baseline_models(full_models_dir):
    """Load full (non-CV) baseline models from full_models_dir/_models/.

    Returns dict {model_name: fitted_model}.
    """
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


def save_rolling_dataset_with_predictions(merged_df, target_year, results_by_family, rolling_csv,
                                          full_models_dir=None, non_bmi_cols=None,
                                          prior_bmi_col_names=None, formula_feature_cols=None,
                                          run_out=None):
    """Save rolling dataset with model-specific predictions as new BMI columns.

    Uses full (non-CV) models to predict missing BMI values:
      - Formula models (deeppysr, pysr, kan): applies the symbolic formula to ALL rows.
      - Baseline models: loads fitted models from full_models_dir/_models/ and predicts.

    Creates columns y{year}bmi_{family}_pred and y{year}bmi_{model_name}_pred.
    Real observed BMI values are preserved; only missing values are filled.

    Args:
        merged_df: current rolling dataset
        target_year: year being processed (int)
        results_by_family: dict {family: {model_name: (formula_or_label, preds, r2)}}
                           from full-model evaluation (not CV).
        rolling_csv: path to save the updated rolling dataset
        full_models_dir: directory containing full-trained models
                         (full_models/_models/ for baselines)
        non_bmi_cols: list of non-BMI feature column names
        prior_bmi_col_names: list of prior BMI column names (e.g. ['y8bmi', 'y10bmi'])
        formula_feature_cols: maps x0, x1, ... in formulas to actual column names
        run_out: deprecated, kept for backward compatibility (ignored)

    Returns:
        updated merged_df
    """
    merged_df = merged_df.copy()
    bmi_col = f'y{target_year}bmi'

    if bmi_col not in merged_df.columns:
        _tmp = rolling_csv + '.tmp'
        merged_df.to_csv(_tmp, index=False)
        os.replace(_tmp, rolling_csv)
        print(f'  Saved rolling dataset to {rolling_csv}')
        return merged_df

    print(f'  Adding model-specific prediction columns for {bmi_col}...')
    real_mask = merged_df[bmi_col].notna()
    missing_mask = ~real_mask
    n_missing = int(missing_mask.sum())
    bmi_median = merged_df.loc[real_mask, bmi_col].median() if real_mask.any() else np.nan

    # Build per-family feature cols: replace raw prior BMI cols with family-specific pred cols.
    # This ensures each family's formula is applied with the same features it was trained on.
    def _family_fcols(family):
        if formula_feature_cols is None or prior_bmi_col_names is None:
            return formula_feature_cols
        prior_set = set(prior_bmi_col_names)
        result = []
        for col in formula_feature_cols:
            if col in prior_set:
                specific = f'{col}_{family}_pred'
                result.append(specific if specific in merged_df.columns else col)
            else:
                result.append(col)
        return result

    # --- Formula families: deeppysr, pysr, kan ---
    for family in ('deeppysr', 'pysr', 'kan'):
        if family not in results_by_family:
            continue
        for _mname, (formula, _preds, _r2) in results_by_family[family].items():
            pred_col = f'{bmi_col}_{family}_pred'
            involved_str = ','.join(_extract_variables(formula))
            all_preds = apply_formula(merged_df, formula, involved_str,
                                      feature_cols=_family_fcols(family))
            merged_df[pred_col] = merged_df[bmi_col].copy()
            merged_df.loc[missing_mask, pred_col] = all_preds[missing_mask]
            still_nan = merged_df[pred_col].isna()
            if still_nan.any():
                merged_df.loc[still_nan, pred_col] = bmi_median
            n_filled = int(merged_df.loc[missing_mask, pred_col].notna().sum()) if n_missing else 0
            print(f'  {family}: {n_filled}/{n_missing} missing {bmi_col} filled via full formula')

    # --- Baseline family: load full models from full_models_dir/_models/ ---
    if 'baseline' in results_by_family:
        fitted_models = _load_full_baseline_models(full_models_dir) if full_models_dir else {}
        for model_name, (_label, _preds, _r2) in results_by_family['baseline'].items():
            pred_col = f'{bmi_col}_{model_name}_pred'
            merged_df[pred_col] = merged_df[bmi_col].copy()

            model = fitted_models.get(model_name)
            if model is not None and non_bmi_cols is not None and n_missing > 0:
                feat_cols = [c for c in (non_bmi_cols or []) if c in merged_df.columns]
                for pcol in (prior_bmi_col_names or []):
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
                merged_df.loc[still_nan, pred_col] = bmi_median
            n_filled = int(merged_df.loc[missing_mask, pred_col].notna().sum()) if n_missing else 0
            print(f'  {model_name}: {n_filled}/{n_missing} missing {bmi_col} filled via full model')

    _tmp = rolling_csv + '.tmp'
    merged_df.to_csv(_tmp, index=False)
    os.replace(_tmp, rolling_csv)
    print(f'  Saved rolling dataset to {rolling_csv}')
    return merged_df


def load_forecast_data(merged_df, target_year, prior_bmi_cols, non_bmi_cols):
    """Prepare (ids, X, y) for predicting target_year BMI using actual prior BMI values.

    Args:
        merged_df: rolling dataset
        target_year: year to predict (int)
        prior_bmi_cols: list of prior bmi column names (e.g. ['y8bmi', 'y10bmi'])
        non_bmi_cols: list of non-BMI feature column names

    Returns:
        (ids, X, y) or (None, None, None)
    """
    bmi_col = f'y{target_year}bmi'
    if bmi_col not in merged_df.columns:
        return None, None, None

    feature_cols = ([c for c in non_bmi_cols if c in merged_df.columns] +
                    [c for c in prior_bmi_cols if c in merged_df.columns])

    sub = merged_df[['child_id'] + feature_cols + [bmi_col]].copy()
    sub = sub.dropna(subset=[bmi_col])

    if len(sub) == 0:
        return None, None, None

    ids = sub['child_id'].values
    X = sub[feature_cols]
    y = sub[bmi_col].values
    return ids, X, y


def load_forecast_data_for_model(merged_df, target_year, prior_bmi_cols, non_bmi_cols, model_type='deeppysr'):
    """Prepare (ids, X, y) using model-specific prior BMI pred columns.

    Uses y{prior_year}bmi_{model_type}_pred columns where available, falling back
    to the real y{prior_year}bmi column.  Only rows with observed target BMI are
    returned (for training/evaluation).

    Args:
        merged_df: rolling dataset with model-specific prediction columns
        target_year: year to predict (int)
        prior_bmi_cols: list of prior bmi column names (e.g. ['y8bmi', 'y10bmi'])
        non_bmi_cols: list of non-BMI feature column names
        model_type: 'deeppysr', 'pysr', or a baseline model name

    Returns:
        (ids, X, y) or (None, None, None)
    """
    bmi_col = f'y{target_year}bmi'
    if bmi_col not in merged_df.columns:
        return None, None, None

    prior_pred_cols = []
    for col in prior_bmi_cols:
        # col is a column name like 'y8bmi'
        pred_col = f'{col}_{model_type}_pred'
        if pred_col in merged_df.columns:
            prior_pred_cols.append(pred_col)
        elif col in merged_df.columns:
            prior_pred_cols.append(col)

    feature_cols = ([c for c in non_bmi_cols if c in merged_df.columns] +
                    prior_pred_cols)

    sub = merged_df[['child_id'] + feature_cols + [bmi_col]].copy()
    sub = sub.dropna(subset=[bmi_col])

    if len(sub) == 0:
        return None, None, None

    ids = sub['child_id'].values
    X = sub[feature_cols]
    y = sub[bmi_col].values
    return ids, X, y
