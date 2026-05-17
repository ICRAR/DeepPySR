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


def _extract_variables(formula):
    """Extract variable names from a formula string by removing known functions/constants."""
    tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', formula)
    return sorted(set(t for t in tokens if t not in _KNOWN_FUNCTIONS and t not in _KNOWN_CONSTANTS))


def get_best_formula_age8():
    """Return the formula string with the highest r2 for age8 in the best models metrics CSV."""
    rel = pd.read_csv(REL_AGE8)
    best = rel[(rel['type'] == 'age-specific') & (rel['display_model'] == 'Best DeepPySR') & (rel['age'] == 8)]
    best = best.iloc[0]
    formula = best['formula']
    if 'involved' in best.index and pd.notna(best['involved']):
        involved = best['involved']
    else:
        involved = ','.join(_extract_variables(formula))
    print(f"[age8] Best formula (r2={best['r2']:.4f}): {formula}")
    print(f"[age8] Involved variables: {involved}")
    return formula, involved


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
    """Save fitted baseline models to disk using joblib.

    Args:
        run_out_dir: directory for this year's results (e.g. results_bmiforecast/age_10)
        models_dict: dict of {model_name: fitted_model}
    """
    import joblib
    models_save_dir = os.path.join(run_out_dir, 'baselines', '_fitted_models')
    os.makedirs(models_save_dir, exist_ok=True)
    for name, model in models_dict.items():
        path = os.path.join(models_save_dir, f'{name}.joblib')
        joblib.dump(model, path)
        print(f"  Saved fitted model: {path}")


def load_baseline_models(run_out_dir):
    """Load all fitted baseline models saved by save_baseline_models.

    Returns dict of {model_name: fitted_model}, empty dict if none found.
    """
    import joblib
    import glob
    models_save_dir = os.path.join(run_out_dir, 'baselines', '_fitted_models')
    if not os.path.exists(models_save_dir):
        return {}
    models = {}
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


def apply_formula(df, formula, involved_str):
    """Evaluate the symbolic formula on df rows.
    Returns a Series of predicted values (NaN where required columns are missing).
    """
    involved = [c.strip() for c in involved_str.split(',')]
    # Only use columns that actually exist in df
    involved = [c for c in involved if c in df.columns]
    if not involved:
        return pd.Series(np.nan, index=df.index)
    mask = df[involved].notna().all(axis=1)
    result = pd.Series(np.nan, index=df.index)
    if mask.sum() == 0:
        return result

    sub = df.loc[mask, involved].copy()
    local_ns = {col: sub[col].values for col in involved}
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


def fill_missing_bmi_with_formula(df, bmi_col, formula, involved_str):
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
    preds = apply_formula(df, formula, involved_str)
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
        iter_imp = IterativeImputer(max_iter=10, random_state=42)
        result[cont_cols] = iter_imp.fit_transform(result[cont_cols])
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
    """Load and merge all year files, impute non-BMI variables, fill y8bmi via formula.

    Returns:
        merged (DataFrame): all subjects with non-BMI vars fully imputed and y8bmi filled.
        non_bmi_cols (list): list of non-BMI, non-id feature column names.
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

    # Impute non-BMI variables (do NOT touch any bmi columns)
    print('\n=== Imputing non-BMI variables ===')
    merged = smart_impute(merged, non_bmi_cols)

    # Fill y8bmi using the best symbolic formula from age8 results
    bmi8_col = 'y8bmi'
    if bmi8_col in merged.columns:
        formula, involved_str = get_best_formula_age8()
        merged = fill_missing_bmi_with_formula(merged, bmi8_col, formula, involved_str)

    return merged, non_bmi_cols


def load_forecast_data(merged_df, target_year, prior_bmi_cols, non_bmi_cols):
    """Prepare (ids, X, y) for predicting target_year BMI.

    Features = non_bmi_cols + prior_bmi_cols (already filled/imputed).
    Only rows with observed target BMI are used for training/evaluation.
    Non-BMI vars are already imputed in merged_df; no further imputation needed here.
    """
    bmi_col = f'y{target_year}bmi'
    if bmi_col not in merged_df.columns:
        return None, None, None

    feature_cols = [c for c in non_bmi_cols if c in merged_df.columns] + \
                   [c for c in prior_bmi_cols if c in merged_df.columns]

    sub = merged_df[['child_id'] + feature_cols + [bmi_col]].copy()
    sub = sub.dropna(subset=[bmi_col])

    ids = sub['child_id'].values
    X = sub[feature_cols]
    y = sub[bmi_col].values
    return ids, X, y
