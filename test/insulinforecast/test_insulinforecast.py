"""Production insulin/glucose forecast rolling pipeline.

Runs the full pipeline (DeepPySR + PySR grid search + baselines + rolling step)
age by age, for both insulin and glucose, over all forecast ages.

Starting age is 17 (used as the base with no prior insulin/glucose features).
Rolling forecast ages: 20, 22, 27, 28.

Usage:
    cd /path/to/DeepPySR
    python test/insulinforecast/test_insulinforecast.py
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score
from pysr import PySRRegressor
from deeppysr import DeepPySR

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from model_utils import (
    get_deeppysr_configs, get_pysr_configs,
    get_pysr_base_kwargs, get_baseline_models, KANWrapper,
)
from eval_utils import run_cv, run_nocv, aggregate_results
from analysis_utils import evaluate_formula
from insulinforecast_utils import (
    AGES, TARGETS,
    TARGET_COLS, get_target_col,
    prepare_base_dataset,
    _is_target_col,
    load_forecast_data_for_model,
    save_rolling_dataset_with_predictions,
    apply_formula, _extract_variables,
)

OUT_ROOT = os.path.join(current_dir, 'results_insulinforecast')
os.makedirs(OUT_ROOT, exist_ok=True)

N_SPLITS = 5

_COND_EXPLICIT = "cond(x,y) = x > 0 ? y : 0f0"

# Populated by main() before each rolling step
_rolling_step_kwargs = {}


def _pysr_kwargs(base_kwargs, overrides=None):
    kwargs = base_kwargs.copy()
    if overrides:
        kwargs.update(overrides)
    kwargs['binary_operators'] = [
        _COND_EXPLICIT if op == 'cond' else op
        for op in kwargs.get('binary_operators', [])
    ]
    return kwargs


# ── Formula / model evaluation helpers ────────────────────────────────────────

def evaluate_all_deeppysr_formulas(run_out_dir, X, y):
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
    results = {}
    kan_dir = os.path.join(run_out_dir, 'kan')
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


def evaluate_all_baseline_models(run_out_dir, X, y):
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


def evaluate_full_baseline_models(full_models_dir, X_eval, y_eval):
    import joblib as _jl
    results = {}
    models_dir = os.path.join(full_models_dir, '_models')
    if not os.path.exists(models_dir):
        return results
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
    results = {}
    for family, fname_pattern in [
        ('deeppysr', 'relationships_nocv.csv'),
        ('pysr', 'formulas_nocv.csv'),
        ('kan', 'formulas_nocv.csv'),
    ]:
        family_dir = os.path.join(full_models_dir, family)
        if not os.path.exists(family_dir):
            continue
        family_res = {}
        for f in glob.glob(os.path.join(family_dir, '**', fname_pattern), recursive=True):
            try:
                df = pd.read_csv(f)
                if 'formula' not in df.columns:
                    continue
                for _, row in df.iterrows():
                    formula = str(row['formula'])
                    if formula in family_res:
                        continue
                    try:
                        y_pred = evaluate_formula(formula, X_eval, model_type=family)
                        mask = ~np.isnan(y_pred)
                        if mask.sum() < 2:
                            continue
                        r2_val = r2_score(np.asarray(y_eval)[mask], y_pred[mask])
                        family_res[formula] = (r2_val, y_pred)
                    except Exception:
                        continue
            except Exception:
                continue
        if family_res:
            results[family] = family_res
    return results


def select_best_cv_model_and_save_metrics(run_out, all_results_dict, y_eval=None):
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
                best_r2, best_name, best_preds, best_formula = r2_val, label, preds, formula
        if best_name:
            print(f"  [{label} best CV] r2={best_r2:.4f}")
            results_by_family[family] = {best_name: (best_formula, best_preds, best_r2)}

    results_by_family['baseline'] = {}
    for model_name, (r2_val, preds) in all_results_dict.get('baseline', {}).items():
        print(f"    Baseline {model_name} CV r2={r2_val:.4f}")
        results_by_family['baseline'][model_name] = (model_name, preds, r2_val)

    rows = []
    for family, models_dict in results_by_family.items():
        for model_name, (_, preds, r2_val) in models_dict.items():
            row = {'family': family, 'model': model_name, 'cv_r2': r2_val}
            if y_eval is not None and preds is not None:
                mae_val, rmse_val = _extra_metrics(y_eval, preds)
                row['cv_mae'] = mae_val
                row['cv_rmse'] = rmse_val
            rows.append(row)
    if rows:
        metrics_csv = os.path.join(run_out, 'cv_metrics_summary.csv')
        pd.DataFrame(rows).to_csv(metrics_csv, index=False)
        print(f"  Saved CV metrics to {metrics_csv}")

    return results_by_family


def select_best_full_model(all_results_dict):
    results_by_family = {}
    for family, label in [('deeppysr', 'DeepPySR'), ('pysr', 'PySR'), ('kan', 'KAN')]:
        best_r2 = -float('inf')
        best_name = best_preds = best_formula = None
        for formula, (r2_val, preds) in all_results_dict.get(family, {}).items():
            print(f"    [Full] {label} formula r2={r2_val:.4f}: {formula[:60]}...")
            if r2_val > best_r2:
                best_r2, best_name, best_preds, best_formula = r2_val, label, preds, formula
        if best_name:
            print(f"  [{label} best full] r2={best_r2:.4f}")
            results_by_family[family] = {best_name: (best_formula, best_preds, best_r2)}

    results_by_family['baseline'] = {}
    for model_name, (r2_val, preds) in all_results_dict.get('baseline', {}).items():
        print(f"    [Full] Baseline {model_name} r2={r2_val:.4f}")
        results_by_family['baseline'][model_name] = (model_name, preds, r2_val)

    return results_by_family


# ── Step 1: DeepPySR CV ────────────────────────────────────────────────────────

def run_deeppysr_for_age(merged_df, age, target, prior_target_cols, non_target_cols,
                          pysr_base_kwargs, deeppysr_configs, r2w_list, lambda_list):
    target_col = get_target_col(age, target)
    ids, X, y = load_forecast_data_for_model(
        merged_df, age, target, prior_target_cols, non_target_cols, model_type='deeppysr')
    if ids is None or len(y) < 10:
        print(f"  Skipping age{age} {target}: insufficient data.")
        return None

    run_out = os.path.join(OUT_ROOT, f'age_{age}_{target}')
    os.makedirs(run_out, exist_ok=True)

    nit = pysr_base_kwargs.get('niterations', 500)
    pop = pysr_base_kwargs.get('populations', 100)
    sz  = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f'nit{nit}_pop{pop}_sz{sz}'

    print(f"\n{'='*20}\n[DeepPySR] age{age} {target} ({target_col})  n={len(y)}\n{'='*20}")

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

    print(f'  Aggregating DeepPySR results for age_{age}_{target}...')
    aggregate_results(run_out, task='regression')
    return run_out


# ── Step 2: Baselines + PySR CV ───────────────────────────────────────────────

def run_baselines_for_age(merged_df, age, target, prior_target_cols, non_target_cols,
                           pysr_base_kwargs, pysr_configs):
    run_out = os.path.join(OUT_ROOT, f'age_{age}_{target}')
    os.makedirs(run_out, exist_ok=True)

    nit = pysr_base_kwargs.get('niterations', 500)
    pop = pysr_base_kwargs.get('populations', 100)
    sz  = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f'nit{nit}_pop{pop}_sz{sz}'

    print(f"\n{'='*20}\n[Baselines/PySR] age{age} {target}\n{'='*20}")

    print('  Evaluating Baseline Models...')
    baseline_model_instances = get_baseline_models(task='regression', input_dim=1)
    any_ran = False

    for name in baseline_model_instances:
        ids, X, y = load_forecast_data_for_model(
            merged_df, age, target, prior_target_cols, non_target_cols, model_type=name)
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

        def baseline_factory(inst=m):
            return clone(inst) if hasattr(inst, 'get_params') else inst

        cv_kwargs = {'ids': ids, 'task': 'regression', 'n_splits': N_SPLITS, 'random_state': 42}
        run_cv(baseline_factory, X, y, outdir=model_out, **cv_kwargs)

    if not any_ran:
        print(f'  Skipping age{age} {target}: no baseline ran.')
        return None

    print('  Evaluating PySR Models...')
    ids_pysr, X_pysr, y_pysr = load_forecast_data_for_model(
        merged_df, age, target, prior_target_cols, non_target_cols, model_type='pysr')
    if ids_pysr is not None and len(y_pysr) >= 10:
        cv_kwargs_pysr = {'ids': ids_pysr, 'task': 'regression',
                          'n_splits': N_SPLITS, 'random_state': 42}
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

    print(f'  Aggregating Baselines/PySR results for age_{age}_{target}...')
    aggregate_results(run_out, task='regression')
    return run_out


# ── Step 3: Train on full data ─────────────────────────────────────────────────

def train_full_models_for_age(merged_df, age, target, prior_target_cols, non_target_cols,
                               pysr_base_kwargs, deeppysr_configs, pysr_configs,
                               r2w_list, lambda_list, run_out):
    import joblib as _jl
    full_out = os.path.join(run_out, 'full_models')

    nit = pysr_base_kwargs.get('niterations', 500)
    pop = pysr_base_kwargs.get('populations', 100)
    sz  = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f'nit{nit}_pop{pop}_sz{sz}'

    # DeepPySR
    ids_dsr, X_dsr, y_dsr = load_forecast_data_for_model(
        merged_df, age, target, prior_target_cols, non_target_cols, model_type='deeppysr')
    if ids_dsr is not None and len(y_dsr) >= 10:
        for cfg_name, cfg_overrides in deeppysr_configs.items():
            parts = cfg_name.split('_', 1)
            full_cfg_name = f'{parts[0]}_{param_suffix}_{parts[1] if len(parts) > 1 else ""}_full'
            dsr_full_out = os.path.join(full_out, 'deeppysr', full_cfg_name)
            if os.path.exists(os.path.join(dsr_full_out, 'overall_metrics.csv')):
                print(f'  Skipping DeepPySR full: {full_cfg_name} (exists)')
                continue
            print(f'  Training DeepPySR (full) {full_cfg_name}...')

            def deeppysr_factory(co=cfg_overrides, dout=dsr_full_out):
                kwargs = pysr_base_kwargs.copy()
                kwargs.update(co)
                return DeepPySR(max_layers=1, output_dir=dout,
                                pareto_r2_weight=r2w_list, pareto_lambda=lambda_list, **kwargs)

            run_nocv(deeppysr_factory, X_dsr, y_dsr, ids=ids_dsr,
                     task='regression', outdir=dsr_full_out, scaler=False)

    # PySR
    ids_psr, X_psr, y_psr = load_forecast_data_for_model(
        merged_df, age, target, prior_target_cols, non_target_cols, model_type='pysr')
    if ids_psr is not None and len(y_psr) >= 10:
        for cfg_name, cfg_overrides in pysr_configs.items():
            aps = cfg_overrides.get('adaptive_parsimony_scaling', 50.0)
            full_name = f'pysr_{param_suffix}_aps{aps}_full'
            psr_full_out = os.path.join(full_out, 'pysr', full_name)
            if os.path.exists(os.path.join(psr_full_out, 'overall_metrics.csv')):
                print(f'  Skipping PySR full: {full_name} (exists)')
                continue
            print(f'  Training PySR (full) {full_name}...')

            def pysr_factory(co=cfg_overrides):
                return PySRRegressor(**_pysr_kwargs(pysr_base_kwargs, co))

            run_nocv(pysr_factory, X_psr, y_psr, ids=ids_psr,
                     task='regression', outdir=psr_full_out, scaler=False)

    # Baseline models
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
            merged_df, age, target, prior_target_cols, non_target_cols, model_type=name)
        if ids_bl is None or len(y_bl) < 10:
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


# ── Step 4: Rolling step for one age ──────────────────────────────────────────

def _age_done_flag(age, target):
    return os.path.join(OUT_ROOT, f'age_{age}_{target}', 'done.flag')


def run_rolling_step(merged_df, non_target_cols, age, rolling_csv):
    """Run rolling step for one age, handling both insulin and glucose."""
    age_idx = AGES.index(age)
    prior_ages = AGES[:age_idx]
    prior_target_cols = [TARGET_COLS[pa][t] for pa in prior_ages for t in TARGETS
                         if TARGET_COLS[pa][t] in merged_df.columns]

    feature_cols = ([c for c in non_target_cols if c in merged_df.columns] +
                    [c for c in prior_target_cols if c in merged_df.columns])

    pysr_base_kwargs = _rolling_step_kwargs.get('pysr_base_kwargs', {})
    deeppysr_configs = _rolling_step_kwargs.get('deeppysr_configs', {})
    pysr_configs = _rolling_step_kwargs.get('pysr_configs', {})
    r2w_list = _rolling_step_kwargs.get('r2w_list', [1.5])
    lambda_list = _rolling_step_kwargs.get('lambda_list', [0.001])

    for target in TARGETS:
        target_col = get_target_col(age, target)
        run_out = os.path.join(OUT_ROOT, f'age_{age}_{target}')

        print(f'\n{"#"*50}')
        print(f'# Rolling step: age {age} {target} ({target_col})')
        print(f'{"#"*50}')

        if not os.path.exists(run_out):
            print(f'  No results directory at {run_out}, skipping.')
            continue

        if target_col not in merged_df.columns:
            print(f'  {target_col} not in dataset, skipping.')
            continue

        known_mask = merged_df[target_col].notna()
        X_eval = merged_df.loc[known_mask, feature_cols].reset_index(drop=True)
        y_eval = merged_df.loc[known_mask, target_col].values

        print(f'  Evaluating CV results on {len(X_eval)} known samples...\n')

        cv_deeppysr = evaluate_all_deeppysr_formulas(run_out, X_eval, y_eval)
        cv_pysr = evaluate_all_pysr_formulas(run_out, X_eval, y_eval)
        cv_kan = evaluate_all_kan_formulas(run_out, X_eval, y_eval)
        cv_baseline = evaluate_all_baseline_models(run_out, X_eval, y_eval)

        cv_results = {'deeppysr': cv_deeppysr, 'pysr': cv_pysr,
                      'kan': cv_kan, 'baseline': cv_baseline}

        if any(cv_results.values()):
            print('\n  --- CV metrics ---')
            select_best_cv_model_and_save_metrics(run_out, cv_results, y_eval=y_eval)
        else:
            print('  No CV results found.')

        print('\n  --- Training full models on all available data ---')
        full_out = train_full_models_for_age(
            merged_df, age, target, prior_target_cols, non_target_cols,
            pysr_base_kwargs, deeppysr_configs, pysr_configs,
            r2w_list, lambda_list, run_out)

        print('\n  --- Evaluating full models ---')
        full_formula_results = evaluate_full_formula_models(full_out, X_eval, y_eval)
        full_baseline_results = evaluate_full_baseline_models(full_out, X_eval, y_eval)

        full_results = {**full_formula_results, 'baseline': full_baseline_results}

        if not any(full_results.values()):
            print('  No full-model results; saving rolling dataset as-is.')
            merged_df = _ensure_saved(merged_df, rolling_csv)
            continue

        full_results_by_family = select_best_full_model(full_results)

        print(f'\n  Full-model results (used for NaN prediction):')
        for family, models_dict in full_results_by_family.items():
            for model_name, (_, _, r2) in models_dict.items():
                print(f'    {family}_{model_name}: r2={r2:.4f}')

        merged_df = save_rolling_dataset_with_predictions(
            merged_df, age, target, full_results_by_family, rolling_csv,
            full_models_dir=full_out, non_target_cols=non_target_cols,
            prior_target_col_names=prior_target_cols,
            formula_feature_cols=feature_cols)

        open(_age_done_flag(age, target), 'w').close()

    return merged_df


def _ensure_saved(df, path):
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rolling_csv = os.path.join(OUT_ROOT, 'rolling_dataset.csv')

    if os.path.exists(rolling_csv):
        print(f'\n=== Loading existing rolling dataset from {rolling_csv} ===')
        merged_df = pd.read_csv(rolling_csv)
        non_target_cols = [c for c in merged_df.columns
                           if c != 'child_id' and not _is_target_col(c)
                           and not c.endswith('_pred')]
    else:
        print('\n=== Preparing base dataset ===')
        merged_df, non_target_cols = prepare_base_dataset()
        tmp = os.path.join(OUT_ROOT, 'base_dataset.csv.tmp')
        merged_df.to_csv(tmp, index=False)
        os.replace(tmp, os.path.join(OUT_ROOT, 'base_dataset.csv'))
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
        'non_target_cols_ref': non_target_cols,
    }

    # Rolling forecast: skip age 17 (base), process 20, 22, 27, 28
    for age in AGES[1:]:
        age_idx = AGES.index(age)
        prior_ages = AGES[:age_idx]
        prior_target_cols = [TARGET_COLS[pa][t] for pa in prior_ages for t in TARGETS
                             if TARGET_COLS[pa][t] in merged_df.columns]

        # Check if ALL targets for this age are already done
        all_done = all(os.path.exists(_age_done_flag(age, t)) for t in TARGETS)
        if all_done:
            print(f'\n[SKIP] age {age} — done.flag found for all targets, skipping.')
            continue

        print(f'\n{"="*60}')
        print(f'PIPELINE: age {age}')
        print(f'  prior target cols: {prior_target_cols}')
        print(f'{"="*60}')

        for target in TARGETS:
            if os.path.exists(_age_done_flag(age, target)):
                print(f'\n[SKIP] age {age} {target} — done.flag found, skipping.')
                continue

            # Step 1: DeepPySR CV
            run_deeppysr_for_age(
                merged_df, age, target, prior_target_cols, non_target_cols,
                pysr_base_kwargs, deeppysr_configs, r2w_list, lambda_list,
            )

            # Step 2: Baselines + PySR CV
            run_baselines_for_age(
                merged_df, age, target, prior_target_cols, non_target_cols,
                pysr_base_kwargs, pysr_configs,
            )

        # Step 3: Rolling step (evaluate CV, train full, predict NaNs, save)
        merged_df = run_rolling_step(merged_df, non_target_cols, age, rolling_csv)

    print('\n=== Pipeline complete ===')
    print(f'Results saved under: {OUT_ROOT}')


if __name__ == '__main__':
    main()
