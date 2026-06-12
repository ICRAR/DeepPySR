"""Production BMI forecast rolling pipeline.

Runs the full pipeline (DeepPySR + PySR grid search + baselines + rolling step)
year by year over all forecast years.

Usage:
    cd /path/to/DeepPySR
    python test/bmiforecast/test_bmiforecast.py
"""
import os
import sys
import glob

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score
from pysr import PySRRegressor
from deeppysr import DeepPySR
from model_utils import (
    get_deeppysr_configs, get_pysr_configs,
    get_pysr_base_kwargs, get_baseline_models, KANWrapper,
)
from eval_utils import run_cv, run_nocv, aggregate_results
from analysis_utils import evaluate_formula
from bmiforecast_utils import (
    YEARS, actual_age,
    prepare_base_dataset,
    _is_bmi_col,
    save_rolling_dataset_with_predictions,
    load_forecast_data_for_model,
)

OUT_ROOT = os.path.join(current_dir, 'results_bmiforecast')
os.makedirs(OUT_ROOT, exist_ok=True)

N_SPLITS = 5

# Populated by main() before each pipeline year so run_rolling_step can access configs
_rolling_step_kwargs = {}


_COND_EXPLICIT = "cond(x,y) = x > 0 ? y : 0f0"

def _pysr_kwargs(base_kwargs, overrides=None):
    """Build PySR kwargs: replace bare 'cond' with an inline Julia definition."""
    kwargs = base_kwargs.copy()
    if overrides:
        kwargs.update(overrides)
    kwargs['binary_operators'] = [
        _COND_EXPLICIT if op == 'cond' else op
        for op in kwargs.get('binary_operators', [])
    ]
    return kwargs


# ── Helpers ───────────────────────────────────────────────────────────────────

def evaluate_all_deeppysr_formulas(run_out_dir, X, y):
    """Evaluate all DeepPySR formulas on the real data.

    Returns:
        dict: {formula: (r2, predictions)} or empty dict if no formulas found.
    """
    results = {}
    deeppysr_dir = os.path.join(run_out_dir, 'deeppysr')

    if not os.path.exists(deeppysr_dir):
        return results

    pattern = os.path.join(deeppysr_dir, '**', 'relationships_fold*.csv')
    files = glob.glob(pattern, recursive=True)
    if not files:
        files = glob.glob(os.path.join(deeppysr_dir, '**', 'relationships.csv'), recursive=True)

    for f in files:
        try:
            df = pd.read_csv(f)
            if 'formula' not in df.columns:
                continue
            for _, row in df.iterrows():
                formula = str(row['formula'])
                if formula in results:
                    continue
                try:
                    y_pred = evaluate_formula(formula, X, model_type='deeppysr')
                    mask = ~np.isnan(y_pred)
                    if mask.sum() < 2:
                        continue
                    r2_val = r2_score(np.asarray(y)[mask], y_pred[mask])
                    results[formula] = (r2_val, y_pred)
                except Exception:
                    continue
        except Exception:
            continue
    return results


def evaluate_all_pysr_formulas(run_out_dir, X, y):
    """Evaluate all PySR formulas on the real data.

    Returns:
        dict: {formula: (r2, predictions)} or empty dict if no formulas found.
    """
    results = {}
    pysr_dir = os.path.join(run_out_dir, 'pysr')

    if not os.path.exists(pysr_dir):
        return results

    for f in glob.glob(os.path.join(pysr_dir, '**', 'formulas_fold*.csv'), recursive=True):
        try:
            df = pd.read_csv(f)
            if 'formula' not in df.columns:
                continue
            for _, row in df.iterrows():
                formula = str(row['formula'])
                if formula in results:
                    continue
                try:
                    y_pred = evaluate_formula(formula, X, model_type='pysr')
                    mask = ~np.isnan(y_pred)
                    if mask.sum() < 2:
                        continue
                    r2_val = r2_score(np.asarray(y)[mask], y_pred[mask])
                    results[formula] = (r2_val, y_pred)
                except Exception:
                    continue
        except Exception:
            continue

    return results


def evaluate_all_kan_formulas(run_out_dir, X, y):
    """Evaluate all KAN formulas on the real data.

    Returns:
        dict: {formula: (r2, predictions)} or empty dict if no formulas found.
    """
    results = {}
    kan_dir = os.path.join(run_out_dir, 'baselines', 'KAN')

    if not os.path.exists(kan_dir):
        return results

    pattern = os.path.join(kan_dir, '**', 'formulas_fold*.csv')
    files = glob.glob(pattern, recursive=True)
    if not files:
        files = glob.glob(os.path.join(kan_dir, '**', 'formulas.csv'), recursive=True)

    for f in files:
        try:
            df = pd.read_csv(f)
            if 'formula' not in df.columns:
                continue
            for _, row in df.iterrows():
                formula = str(row['formula'])
                if formula in results:
                    continue
                try:
                    y_pred = evaluate_formula(formula, X, model_type='kan')
                    mask = ~np.isnan(y_pred)
                    if mask.sum() < 2:
                        continue
                    r2_val = r2_score(np.asarray(y)[mask], y_pred[mask])
                    results[formula] = (r2_val, y_pred)
                except Exception:
                    continue
        except Exception:
            continue

    return results


def evaluate_all_baseline_models(run_out_dir, X, y, feature_cols):
    """Read CV out-of-fold baseline predictions saved by run_cv.

    Returns:
        dict: {model_name: (r2, cv_predictions)} or empty dict if none found.
    """
    results = {}
    baselines_dir = os.path.join(run_out_dir, 'baselines')
    if not os.path.exists(baselines_dir):
        return results

    for name in sorted(os.listdir(baselines_dir)):
        if name.startswith('_') or not os.path.isdir(os.path.join(baselines_dir, name)):
            continue
        pred_file = os.path.join(baselines_dir, name, 'predictions.csv')
        if not os.path.exists(pred_file):
            continue
        try:
            df_pred = pd.read_csv(pred_file)
            if 'y_true' not in df_pred.columns or 'y_pred' not in df_pred.columns:
                continue
            y_true_cv = df_pred['y_true'].values
            y_pred_cv = df_pred['y_pred'].values
            mask = ~np.isnan(y_pred_cv) & ~np.isnan(y_true_cv)
            if mask.sum() < 2:
                continue
            r2_val = r2_score(y_true_cv[mask], y_pred_cv[mask])
            results[name] = (r2_val, y_pred_cv)
        except Exception:
            continue

    return results


def evaluate_full_baseline_models(full_models_dir, X_eval, y_eval, feature_cols):
    """Evaluate full (non-CV) baseline models on known data.

    Loads models from full_models_dir/_models/ and predicts on X_eval.

    Returns:
        dict: {model_name: (r2, predictions)} or empty dict if none found.
    """
    import joblib as _jl
    results = {}
    models_dir = os.path.join(full_models_dir, '_models')
    if not os.path.exists(models_dir):
        return results

    # Load and evaluate each saved model
    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith('.joblib'):
            continue
        model_name = fname.replace('.joblib', '')
        try:
            m = _jl.load(os.path.join(models_dir, fname))
            y_pred = m.predict(X_eval.values)
            mask = ~np.isnan(y_pred)
            if mask.sum() < 2:
                continue
            r2_val = r2_score(np.asarray(y_eval)[mask], y_pred[mask])
            results[model_name] = (r2_val, y_pred)
        except Exception:
            continue
    return results


def evaluate_full_formula_models(full_models_dir, X_eval, y_eval):
    """Read formulas_nocv.csv / relationships_nocv.csv and evaluate on X_eval."""
    results = {}

    # DeepPySR
    deeppysr_dir = os.path.join(full_models_dir, 'deeppysr')
    if os.path.exists(deeppysr_dir):
        dsr_res = {}
        for f in glob.glob(os.path.join(deeppysr_dir, '**', 'relationships_nocv.csv'), recursive=True):
            try:
                df = pd.read_csv(f)
                if 'formula' not in df.columns:
                    continue
                for _, row in df.iterrows():
                    formula = str(row['formula'])
                    if formula in dsr_res:
                        continue
                    try:
                        y_pred = evaluate_formula(formula, X_eval, model_type='deeppysr')
                        mask = ~np.isnan(y_pred)
                        if mask.sum() < 2:
                            continue
                        r2_val = r2_score(np.asarray(y_eval)[mask], y_pred[mask])
                        dsr_res[formula] = (r2_val, y_pred)
                    except Exception:
                        continue
            except Exception:
                continue
        if dsr_res:
            results['deeppysr'] = dsr_res

    # PySR
    pysr_dir = os.path.join(full_models_dir, 'pysr')
    if os.path.exists(pysr_dir):
        psr_res = {}
        for f in glob.glob(os.path.join(pysr_dir, '**', 'formulas_foldnocv.csv'), recursive=True):
            try:
                df = pd.read_csv(f)
                if 'formula' not in df.columns:
                    continue
                for _, row in df.iterrows():
                    formula = str(row['formula'])
                    if formula in psr_res:
                        continue
                    try:
                        y_pred = evaluate_formula(formula, X_eval, model_type='pysr')
                        mask = ~np.isnan(y_pred)
                        if mask.sum() < 2:
                            continue
                        r2_val = r2_score(np.asarray(y_eval)[mask], y_pred[mask])
                        psr_res[formula] = (r2_val, y_pred)
                    except Exception:
                        continue
            except Exception:
                continue
        if psr_res:
            results['pysr'] = psr_res

    # KAN
    kan_dir = os.path.join(full_models_dir, 'kan')
    if os.path.exists(kan_dir):
        kan_res = {}
        for f in glob.glob(os.path.join(kan_dir, '**', 'formulas_foldnocv.csv'), recursive=True):
            try:
                df = pd.read_csv(f)
                if 'formula' not in df.columns:
                    continue
                for _, row in df.iterrows():
                    formula = str(row['formula'])
                    if formula in kan_res:
                        continue
                    try:
                        y_pred = evaluate_formula(formula, X_eval, model_type='kan')
                        mask = ~np.isnan(y_pred)
                        if mask.sum() < 2:
                            continue
                        r2_val = r2_score(np.asarray(y_eval)[mask], y_pred[mask])
                        kan_res[formula] = (r2_val, y_pred)
                    except Exception:
                        continue
            except Exception:
                continue
        if kan_res:
            results['kan'] = kan_res

    return results


def select_best_cv_model_and_save_metrics(run_out, all_results_dict, y_eval=None):
    """Select the best CV result per family and save cv_metrics_summary.csv."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    def _extra_metrics(y_true, y_pred):
        mask = ~np.isnan(y_pred) & ~np.isnan(np.asarray(y_true, dtype=float))
        yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
        if len(yt) < 2:
            return np.nan, np.nan
        return mean_absolute_error(yt, yp), np.sqrt(mean_squared_error(yt, yp))

    results_by_family = {}

    for family, label in [('deeppysr', 'DeepPySR'), ('pysr', 'PySR'), ('kan', 'KAN')]:
        best_r2 = -float('inf')
        best_name = best_preds = best_formula = None
        for formula, (r2_val, preds) in all_results_dict.get(family, {}).items():
            print(f"    {label} formula r2={r2_val:.4f}: {formula[:60]}...")
            if r2_val > best_r2:
                best_r2 = r2_val
                best_name = label
                best_preds = preds
                best_formula = formula
        if best_name:
            print(f"  [{label} best CV] r2={best_r2:.4f}")
            results_by_family[family] = {best_name: (best_formula, best_preds, best_r2)}

    results_by_family['baseline'] = {}
    for model_name, (r2_val, preds) in all_results_dict.get('baseline', {}).items():
        print(f"    Baseline {model_name} CV r2={r2_val:.4f}")
        results_by_family['baseline'][model_name] = (model_name, preds, r2_val)

    rows = []
    for family, models_dict in results_by_family.items():
        for model_name, (formula_or_label, preds, r2_val) in models_dict.items():
            row = {'family': family, 'model': model_name, 'cv_r2': r2_val}
            if family == 'baseline':
                pred_file = os.path.join(run_out, 'baselines', model_name, 'predictions.csv')
                try:
                    df_pred = pd.read_csv(pred_file)
                    mae, rmse = _extra_metrics(df_pred['y_true'].values, df_pred['y_pred'].values)
                    row['cv_mae'] = mae
                    row['cv_rmse'] = rmse
                except Exception:
                    row['cv_mae'] = np.nan
                    row['cv_rmse'] = np.nan
            elif y_eval is not None and preds is not None:
                mae, rmse = _extra_metrics(y_eval, preds)
                row['cv_mae'] = mae
                row['cv_rmse'] = rmse
            rows.append(row)

    if rows:
        metrics_csv = os.path.join(run_out, 'cv_metrics_summary.csv')
        pd.DataFrame(rows).to_csv(metrics_csv, index=False)
        print(f"  Saved CV metrics to {metrics_csv}\n")

    return results_by_family


def select_best_full_model(all_results_dict):
    """Select best result per family from full-training evaluation results."""
    results_by_family = {}

    for family, label in [('deeppysr', 'DeepPySR'), ('pysr', 'PySR'), ('kan', 'KAN')]:
        best_r2 = -float('inf')
        best_name = best_preds = best_formula = None
        for formula, (r2_val, preds) in all_results_dict.get(family, {}).items():
            print(f"    [Full] {label} formula r2={r2_val:.4f}: {formula[:60]}...")
            if r2_val > best_r2:
                best_r2 = r2_val
                best_name = label
                best_preds = preds
                best_formula = formula
        if best_name:
            print(f"  [{label} best full] r2={best_r2:.4f}")
            results_by_family[family] = {best_name: (best_formula, best_preds, best_r2)}

    results_by_family['baseline'] = {}
    for model_name, (r2_val, preds) in all_results_dict.get('baseline', {}).items():
        print(f"    [Full] Baseline {model_name} r2={r2_val:.4f}")
        results_by_family['baseline'][model_name] = (model_name, preds, r2_val)

    return results_by_family


# ── Step 1: DeepPySR grid search ──────────────────────────────────────────────

def run_deeppysr_for_year(merged_df, target_year, prior_bmi_cols, non_bmi_cols,
                          pysr_base_kwargs, deeppysr_configs, r2w_list, lambda_list):
    ids, X, y = load_forecast_data_for_model(
        merged_df, target_year, prior_bmi_cols, non_bmi_cols, model_type='deeppysr')
    if ids is None or len(y) < 10:
        print(f"  Skipping yr{target_year}: insufficient data.")
        return None

    age_label = actual_age(target_year)
    run_out = os.path.join(OUT_ROOT, f'age_{age_label}')
    os.makedirs(run_out, exist_ok=True)

    print(f"\n{'='*20}\n[DeepPySR] Forecasting y{target_year}bmi (age {age_label})  n={len(y)}\n{'='*20}")

    nit = pysr_base_kwargs.get('niterations', 500)
    pop = pysr_base_kwargs.get('populations', 100)
    sz  = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f'nit{nit}_pop{pop}_sz{sz}'

    cv_kwargs = {'ids': ids, 'task': 'regression', 'n_splits': N_SPLITS, 'random_state': 42}

    for cfg_name, cfg_overrides in deeppysr_configs.items():
        parts = cfg_name.split('_', 1)
        full_cfg_name = f'{parts[0]}_{param_suffix}_{parts[1] if len(parts) > 1 else ""}_grid'
        deeppysr_out = os.path.join(run_out, 'deeppysr', full_cfg_name)

        if os.path.exists(os.path.join(deeppysr_out, 'overall_metrics.csv')):
            print(f'  Skipping {full_cfg_name} (results exist)')
            continue

        print(f'  Running {full_cfg_name}...')

        def deeppysr_factory(co=cfg_overrides, dout=deeppysr_out):
            kwargs = pysr_base_kwargs.copy()
            kwargs.update(co)
            return DeepPySR(max_layers=1, output_dir=dout,
                            pareto_r2_weight=r2w_list, pareto_lambda=lambda_list, **kwargs)

        run_cv(deeppysr_factory, X, y, outdir=deeppysr_out, scaler=False, **cv_kwargs)

    print(f'  Aggregating DeepPySR results for age_{age_label}...')
    aggregate_results(run_out, task='regression')
    return run_out


# ── Step 2: Baselines + PySR grid search ──────────────────────────────────────

def run_baselines_for_year(merged_df, target_year, prior_bmi_cols, non_bmi_cols,
                           pysr_base_kwargs, pysr_configs):
    age_label = actual_age(target_year)
    run_out = os.path.join(OUT_ROOT, f'age_{age_label}')
    os.makedirs(run_out, exist_ok=True)

    print(f"\n{'='*20}\n[Baselines/PySR] Forecasting y{target_year}bmi (age {age_label})\n{'='*20}")

    nit = pysr_base_kwargs.get('niterations', 500)
    pop = pysr_base_kwargs.get('populations', 100)
    sz  = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f'nit{nit}_pop{pop}_sz{sz}'

    print('  Evaluating Baseline Models...')
    baseline_model_instances = get_baseline_models(task='regression', input_dim=1)
    any_ran = False

    for name in baseline_model_instances:
        ids, X, y = load_forecast_data_for_model(
            merged_df, target_year, prior_bmi_cols, non_bmi_cols, model_type=name)
        if ids is None or len(y) < 10:
            print(f'    Skipping {name}: insufficient data.')
            continue
        any_ran = True
        model_out = os.path.join(run_out, 'baselines', name)

        if os.path.exists(os.path.join(model_out, 'overall_metrics.csv')):
            print(f'    Skipping {name} (CV results exist)')
            continue

        print(f'    {name}...')
        if name == 'KAN':
            m = KANWrapper(input_dim=X.shape[1], output_dim=1,
                           hidden_dim=5, steps=200, update_grid=False, task='regression')
        else:
            m = clone(baseline_model_instances[name])

        if os.path.exists(os.path.join(model_out, 'overall_metrics.csv')):
            print(f'    Skipping {name} (CV results exist)')
            continue

        print(f'    {name}...')

        def baseline_factory(inst=m):
            return clone(inst) if hasattr(inst, 'get_params') else inst

        cv_kwargs = {'ids': ids, 'task': 'regression', 'n_splits': N_SPLITS, 'random_state': 42}
        run_cv(baseline_factory, X, y, outdir=model_out, **cv_kwargs)

    if not any_ran:
        print(f'  Skipping yr{target_year}: insufficient data for all baseline models.')
        return None

    print('  Evaluating PySR Models...')
    ids_pysr, X_pysr, y_pysr = load_forecast_data_for_model(
        merged_df, target_year, prior_bmi_cols, non_bmi_cols, model_type='pysr')
    if ids_pysr is not None and len(y_pysr) >= 10:
        cv_kwargs_pysr = {'ids': ids_pysr, 'task': 'regression', 'n_splits': N_SPLITS, 'random_state': 42}
        for cfg_name, cfg_overrides in pysr_configs.items():
            aps = cfg_overrides.get('adaptive_parsimony_scaling', 50.0)
            full_name = f'pysr_{param_suffix}_aps{aps}_grid'
            pysr_out = os.path.join(run_out, 'pysr', full_name)
            if os.path.exists(os.path.join(pysr_out, 'overall_metrics.csv')):
                print(f'    Skipping {full_name} (results exist)')
                continue
            print(f'    {full_name}...')

            def pysr_factory(co=cfg_overrides):
                return PySRRegressor(**_pysr_kwargs(pysr_base_kwargs, co))

            run_cv(pysr_factory, X_pysr, y_pysr, outdir=pysr_out, scaler=False, **cv_kwargs_pysr)

    print(f'  Aggregating Baselines/PySR results for age_{age_label}...')
    aggregate_results(run_out, task='regression')
    return run_out


# ── Step 3: Train all models on full data ─────────────────────────────────────

def train_full_models_for_year(merged_df, target_year, prior_bmi_cols, non_bmi_cols,
                               pysr_base_kwargs, deeppysr_configs, pysr_configs,
                               r2w_list, lambda_list, run_out):
    """Train DeepPySR, PySR, KAN, and baseline models on ALL available data (no CV).

    Saves results under {run_out}/full_models/:
      - DeepPySR: full_models/deeppysr/<cfg>/relationships_nocv.csv  (via run_nocv)
      - PySR:     full_models/pysr/<cfg>/formulas_nocv.csv          (via run_nocv)
      - KAN:      full_models/kan/full/formulas_nocv.csv            (via run_nocv)
      - Baselines: full_models/_models/<name>.joblib
    """
    import joblib as _jl
    full_out = os.path.join(run_out, 'full_models')

    nit = pysr_base_kwargs.get('niterations', 500)
    pop = pysr_base_kwargs.get('populations', 100)
    sz  = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f'nit{nit}_pop{pop}_sz{sz}'

    def _all_full_models_exist():
        for cfg_name in deeppysr_configs:
            parts = cfg_name.split('_', 1)
            full_cfg_name = f'{parts[0]}_{param_suffix}_{parts[1] if len(parts) > 1 else ""}_full'
            p = os.path.join(full_out, 'deeppysr', full_cfg_name, 'overall_metrics.csv')
            if not os.path.exists(p):
                return False
        for cfg_name, cfg_overrides in pysr_configs.items():
            aps = cfg_overrides.get('adaptive_parsimony_scaling', 50.0)
            full_name = f'pysr_{param_suffix}_aps{aps}_full'
            p = os.path.join(full_out, 'pysr', full_name, 'overall_metrics.csv')
            if not os.path.exists(p):
                return False
        if not os.path.exists(os.path.join(full_out, 'kan', 'full', 'formulas_foldnocv.csv')):
            return False
        baseline_model_instances = get_baseline_models(task='regression', input_dim=1)
        for name in baseline_model_instances:
            if name == 'KAN':
                if not os.path.exists(os.path.join(full_out,'_models','KAN_state')):
                    return False
            else:
                if not os.path.exists(os.path.join(full_out, '_models', f'{name}.joblib')):
                    return False
        return True

    if _all_full_models_exist():
        print('  Skipping train_full_models_for_year (all full models exist)')
        return full_out

    # os.makedirs(full_out, exist_ok=True)

    # DeepPySR — train on all data
    ids_dsr, X_dsr, y_dsr = load_forecast_data_for_model(
        merged_df, target_year, prior_bmi_cols, non_bmi_cols, model_type='deeppysr')
    if ids_dsr is not None and len(y_dsr) >= 10:
        for cfg_name, cfg_overrides in deeppysr_configs.items():
            parts = cfg_name.split('_', 1)
            full_cfg_name = f'{parts[0]}_{param_suffix}_{parts[1] if len(parts) > 1 else ""}_full'
            dsr_full_out = os.path.join(full_out, 'deeppysr', full_cfg_name)
            if os.path.exists(os.path.join(dsr_full_out, 'overall_metrics.csv')):
                print(f'  Skipping DeepPySR full training: {full_cfg_name} (exists)')
                continue
            print(f'  Training DeepPySR (full) {full_cfg_name}...')

            def deeppysr_factory(co=cfg_overrides, dout=dsr_full_out):
                kwargs = pysr_base_kwargs.copy()
                kwargs.update(co)
                return DeepPySR(max_layers=1, output_dir=dout,
                                pareto_r2_weight=r2w_list, pareto_lambda=lambda_list, **kwargs)

            run_nocv(deeppysr_factory, X_dsr, y_dsr, ids=ids_dsr,
                     task='regression', outdir=dsr_full_out, scaler=False)

    # PySR — train on all data
    ids_psr, X_psr, y_psr = load_forecast_data_for_model(
        merged_df, target_year, prior_bmi_cols, non_bmi_cols, model_type='pysr')
    if ids_psr is not None and len(y_psr) >= 10:
        for cfg_name, cfg_overrides in pysr_configs.items():
            aps = cfg_overrides.get('adaptive_parsimony_scaling', 50.0)
            full_name = f'pysr_{param_suffix}_aps{aps}_full'
            psr_full_out = os.path.join(full_out, 'pysr', full_name)
            if os.path.exists(os.path.join(psr_full_out, 'overall_metrics.csv')):
                print(f'  Skipping PySR full training: {full_name} (exists)')
                continue
            print(f'  Training PySR (full) {full_name}...')

            def pysr_factory(co=cfg_overrides):
                return PySRRegressor(**_pysr_kwargs(pysr_base_kwargs, co))

            run_nocv(pysr_factory, X_psr, y_psr, ids=ids_psr,
                     task='regression', outdir=psr_full_out, scaler=False)

    # KAN — train on all data via run_nocv so symbolize()+formulas_nocv.csv are handled automatically
    ids_kan, X_kan, y_kan = load_forecast_data_for_model(
        merged_df, target_year, prior_bmi_cols, non_bmi_cols, model_type='KAN')
    if ids_kan is not None and len(y_kan) >= 10:
        kan_full_out = os.path.join(full_out, 'kan', 'full')
        if os.path.exists(os.path.join(kan_full_out, 'formulas_nocv.csv')):
            print('  Skipping KAN full training (exists)')
        else:
            print('  Training KAN (full)...')
            def kan_factory():
                return KANWrapper(input_dim=X_kan.shape[1], output_dim=1,
                                  hidden_dim=5, steps=200, update_grid=False, task='regression')
            run_nocv(kan_factory, X_kan, y_kan, ids=ids_kan,
                     task='regression', outdir=kan_full_out, scaler=False)

    # Baseline models (non-KAN) — train on all data
    models_save_dir = os.path.join(full_out, '_models')
    os.makedirs(models_save_dir, exist_ok=True)
    baseline_model_instances = get_baseline_models(task='regression', input_dim=1)

    for name in baseline_model_instances:
        if name == 'KAN':
            continue
        save_path = os.path.join(models_save_dir, f'{name}.joblib')
        if os.path.exists(save_path):
            print(f'  Skipping {name} full training (exists)')
            continue
        ids_bl, X_bl, y_bl = load_forecast_data_for_model(
            merged_df, target_year, prior_bmi_cols, non_bmi_cols, model_type=name)
        if ids_bl is None or len(y_bl) < 10:
            print(f'    Skipping {name}: insufficient data.')
            continue
        m = clone(baseline_model_instances[name])
        print(f'  Training {name} (full)...')
        try:
            m.fit(X_bl.values, y_bl)
            _jl.dump(m, save_path)
            print(f'  Saved full model: {name}')
        except Exception as ex:
            print(f'  WARNING: Failed to train {name} (full): {ex}')

    return full_out


# ── Step 4: Rolling step for one year ─────────────────────────────────────────

def _year_done_flag(age_label):
    return os.path.join(OUT_ROOT, f'age_{age_label}', 'done.flag')


def run_rolling_step(merged_df, non_bmi_cols, target_year, rolling_csv):
    bmi_col = f'y{target_year}bmi'
    age_label = actual_age(target_year)
    run_out = os.path.join(OUT_ROOT, f'age_{age_label}')

    print(f'\n{"#"*50}')
    print(f'# Rolling step: y{target_year}bmi (age {age_label})')
    print(f'{"#"*50}')

    if not os.path.exists(run_out):
        print(f'  No results directory at {run_out}, skipping.')
        return merged_df

    if bmi_col not in merged_df.columns:
        print(f'  {bmi_col} not in dataset, skipping.')
        return merged_df

    year_idx = YEARS.index(target_year)
    prior_bmi_cols = [f'y{y}bmi' for y in YEARS[:year_idx]
                      if f'y{y}bmi' in merged_df.columns]
    feature_cols = [c for c in _rolling_step_kwargs.get('non_bmi_cols_ref', non_bmi_cols)
                    if c in merged_df.columns] + \
                   [c for c in prior_bmi_cols if c in merged_df.columns]

    known_mask = merged_df[bmi_col].notna()
    X_eval = merged_df.loc[known_mask, feature_cols].reset_index(drop=True)
    y_eval = merged_df.loc[known_mask, bmi_col].values

    print(f'  Evaluating CV results on {len(X_eval)} known samples...\n')

    # ── CV evaluation (metrics only, no model saving) ─────────────────────────
    print('  Evaluating DeepPySR CV formulas...')
    cv_deeppysr = evaluate_all_deeppysr_formulas(run_out, X_eval, y_eval)

    print('  Evaluating PySR CV formulas...')
    cv_pysr = evaluate_all_pysr_formulas(run_out, X_eval, y_eval)

    print('  Evaluating KAN CV formulas...')
    cv_kan = evaluate_all_kan_formulas(run_out, X_eval, y_eval)

    print('  Reading Baseline CV predictions...')
    cv_baseline = evaluate_all_baseline_models(run_out, X_eval, y_eval, feature_cols)

    cv_results = {
        'deeppysr': cv_deeppysr,
        'pysr': cv_pysr,
        'kan': cv_kan,
        'baseline': cv_baseline,
    }

    if any(cv_results.values()):
        print('\n  --- CV metrics ---')
        select_best_cv_model_and_save_metrics(run_out, cv_results, y_eval=y_eval)
    else:
        print('  No CV results found.')

    # ── Full model training (all available data, no CV) ───────────────────────
    # Retrieve configs from main() — passed via run_rolling_step kwargs
    pysr_base_kwargs = _rolling_step_kwargs.get('pysr_base_kwargs', {})
    deeppysr_configs = _rolling_step_kwargs.get('deeppysr_configs', {})
    pysr_configs = _rolling_step_kwargs.get('pysr_configs', {})
    r2w_list = _rolling_step_kwargs.get('r2w_list', [1.5])
    lambda_list = _rolling_step_kwargs.get('lambda_list', [0.001])

    print('\n  --- Training full models on all available data ---')
    full_out = train_full_models_for_year(
        merged_df, target_year, prior_bmi_cols, non_bmi_cols,
        pysr_base_kwargs, deeppysr_configs, pysr_configs,
        r2w_list, lambda_list, run_out)

    # ── Evaluate full models ───────────────────────────────────────────────────
    print('\n  --- Evaluating full models ---')
    full_formula_results = evaluate_full_formula_models(full_out, X_eval, y_eval)
    full_baseline_results = evaluate_full_baseline_models(full_out, X_eval, y_eval, feature_cols)

    full_results = {**full_formula_results, 'baseline': full_baseline_results}

    if not any(full_results.values()):
        print('  No full-model results found for NaN prediction.')
        _tmp = rolling_csv + '.tmp'
        merged_df.to_csv(_tmp, index=False)
        os.replace(_tmp, rolling_csv)
        print(f'\n  Rolling dataset saved to {rolling_csv}')
        return merged_df

    full_results_by_family = select_best_full_model(full_results)

    print(f'\n  Full-model results (used for NaN prediction):')
    for family, models_dict in full_results_by_family.items():
        for model_name, (_, _, r2) in models_dict.items():
            print(f'    {family}_{model_name}: r2={r2:.4f}')

    # ── Save rolling dataset using full models ─────────────────────────────────
    merged_df = save_rolling_dataset_with_predictions(
        merged_df, target_year, full_results_by_family, rolling_csv,
        full_models_dir=full_out, non_bmi_cols=non_bmi_cols,
        prior_bmi_col_names=prior_bmi_cols,
        formula_feature_cols=feature_cols)
    print(f'\n  Rolling dataset saved with full-model predictions')

    open(_year_done_flag(age_label), 'w').close()
    return merged_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rolling_csv = os.path.join(OUT_ROOT, 'rolling_dataset.csv')

    if os.path.exists(rolling_csv):
        print(f'\n=== Loading existing rolling dataset from {rolling_csv} ===')
        merged_df = pd.read_csv(rolling_csv)
        non_bmi_cols = [c for c in merged_df.columns
                        if c != 'child_id' and not _is_bmi_col(c) and not c.endswith('_pred')]
    else:
        print('\n=== Preparing base dataset ===')
        merged_df, non_bmi_cols = prepare_base_dataset()
        _tmp = os.path.join(OUT_ROOT, 'base_dataset.csv.tmp')
        merged_df.to_csv(_tmp, index=False)
        os.replace(_tmp, os.path.join(OUT_ROOT, 'base_dataset.csv'))
        print(f'Base dataset: {len(merged_df)} rows, {len(merged_df.columns)} cols')

    pysr_base_kwargs = get_pysr_base_kwargs()
    deeppysr_configs = get_deeppysr_configs()
    pysr_configs = get_pysr_configs()

    r2w_list = [1, 1.5, 2]
    lambda_list = [0.001, 0.005, 0.01]

    print(f'\nDeepPySR configs: {len(deeppysr_configs)} combinations')
    print(f'PySR configs: {len(pysr_configs)} combinations')

    global _rolling_step_kwargs
    _rolling_step_kwargs = {
        'pysr_base_kwargs': pysr_base_kwargs,
        'deeppysr_configs': deeppysr_configs,
        'pysr_configs': pysr_configs,
        'r2w_list': r2w_list,
        'lambda_list': lambda_list,
        'non_bmi_cols_ref': non_bmi_cols,
    }

    for year in YEARS[1:]:
        year_idx = YEARS.index(year)
        prior_bmi_cols = [f'y{y}bmi' for y in YEARS[:year_idx]
                          if f'y{y}bmi' in merged_df.columns]

        age_label = actual_age(year)
        if os.path.exists(_year_done_flag(age_label)):
            print(f'\n[SKIP] y{year}bmi (age {age_label}) — done.flag found, skipping.')
            continue

        print(f'\n{"="*60}')
        print(f'PIPELINE: predicting y{year}bmi')
        print(f'  prior BMI cols: {prior_bmi_cols}')
        print(f'{"="*60}')

        # Step 1: DeepPySR CV
        run_deeppysr_for_year(
            merged_df, year, prior_bmi_cols, non_bmi_cols,
            pysr_base_kwargs, deeppysr_configs, r2w_list, lambda_list,
        )

        # Step 2: Baselines + PySR CV
        run_baselines_for_year(
            merged_df, year, prior_bmi_cols, non_bmi_cols,
            pysr_base_kwargs, pysr_configs,
        )

        # Step 3: Evaluate CV, train full models, predict NaNs, save rolling dataset
        merged_df = run_rolling_step(merged_df, non_bmi_cols, year, rolling_csv)

    print('\n=== Pipeline complete ===')
    print(f'Results saved under: {OUT_ROOT}')


if __name__ == '__main__':
    main()
