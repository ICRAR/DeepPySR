"""Lipids forecast rolling pipeline.

Runs the full pipeline (DeepPySR + PySR grid search + baselines + rolling step)
age by age, for each of the 4 lipid targets independently. Each target's chain
is built from scratch starting at age 14 (no external seed model) and rolled
forward through 17, 20, 22, 28 -- mirroring test/newbmiforecast/test_newbmiforecast.py's
single-script rolling design, generalized from one target (BMI) to four (lipids).

Usage:
    cd /path/to/DeepPySR
    python test/lipidsforecast/test_lipidsforecast.py
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
from scipy.stats import pearsonr
from pysr import PySRRegressor
from deeppysr import DeepPySR
from model_utils import (
    get_deeppysr_configs, get_pysr_configs,
    get_pysr_base_kwargs, get_baseline_models, KANWrapper,
)
from eval_utils import run_cv, run_nocv, aggregate_results
from analysis_utils import evaluate_formula
from data_utils import (
    AGES, TARGETS, actual_age, target_col,
    prepare_base_dataset,
    _target_raw_cols,
    save_rolling_dataset_with_predictions,
    load_forecast_data_for_model,
    get_age_filtered_feature_cols,
)

OUT_ROOT = os.path.join(current_dir, 'results_lipidsforecast')
os.makedirs(OUT_ROOT, exist_ok=True)

N_SPLITS = 5

_rolling_step_kwargs = {}

_COND_EXPLICIT = "cond(x,y) = x > 0 ? y : 0f0"


def _pysr_kwargs(base_kwargs, overrides=None):
    kwargs = base_kwargs.copy()
    if overrides:
        kwargs.update(overrides)
    kwargs['binary_operators'] = [
        _COND_EXPLICIT if op == 'cond' else op
        for op in kwargs.get('binary_operators', [])
    ]
    return kwargs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pearson_from_predictions(pred_path, pred_col='y_pred'):
    """Pearson r computed fresh from a predictions.csv (genuine out-of-fold
    predictions from run_cv) -- overall_metrics.csv doesn't carry pearson_r
    (eval_utils.calculate_metrics never computed it), so it's derived here
    instead of retraining."""
    if not os.path.exists(pred_path):
        return np.nan
    try:
        df = pd.read_csv(pred_path)
        y_true = df['y_true'].values.astype(float)
        y_pred = df[pred_col].values.astype(float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 2 or np.std(y_true[mask]) == 0 or np.std(y_pred[mask]) == 0:
            return np.nan
        return float(pearsonr(y_true[mask], y_pred[mask])[0])
    except Exception:
        return np.nan


def _read_overall_metrics(path, pred_path=None, pred_col='y_pred'):
    """Read r2/mae/rmse from an overall_metrics.csv, plus pearson_r derived
    from the sibling predictions.csv if given.

    These are genuine out-of-fold CV metrics already computed by run_cv (each
    fold's model/formula is only ever applied to the rows it did NOT train on),
    so no re-evaluation against the full dataset is needed or safe.
    """
    if not os.path.exists(path):
        return None
    try:
        row = pd.read_csv(path).iloc[0]
        metrics = {'r2': float(row['r2']), 'mae': float(row['mae']), 'rmse': float(row['rmse'])}
        metrics['pearson_r'] = _pearson_from_predictions(pred_path, pred_col=pred_col) \
            if pred_path is not None else np.nan
        return metrics
    except Exception:
        return None


def collect_cv_grid_metrics(run_out_dir, family_dirname):
    """Genuine out-of-fold CV metrics for each grid config of a formula family."""
    results = {}
    family_dir = os.path.join(run_out_dir, family_dirname)
    if not os.path.exists(family_dir):
        return results
    for cfg in sorted(os.listdir(family_dir)):
        cfg_path = os.path.join(family_dir, cfg)
        metrics = _read_overall_metrics(os.path.join(cfg_path, 'overall_metrics.csv'),
                                        pred_path=os.path.join(cfg_path, 'predictions.csv'))
        if metrics is not None:
            results[cfg] = metrics
    return results


def collect_cv_baseline_metrics(run_out_dir):
    """Genuine out-of-fold CV metrics for each baseline model."""
    results = {}
    baselines_dir = os.path.join(run_out_dir, 'baselines')
    if not os.path.exists(baselines_dir):
        return results
    for name in sorted(os.listdir(baselines_dir)):
        model_dir = os.path.join(baselines_dir, name)
        if name.startswith('_') or not os.path.isdir(model_dir):
            continue
        metrics = _read_overall_metrics(os.path.join(model_dir, 'overall_metrics.csv'),
                                        pred_path=os.path.join(model_dir, 'predictions.csv'))
        if metrics is not None:
            results[name] = metrics
    return results


def collect_cv_kan_symbolic_metrics(run_out_dir):
    """Genuine out-of-fold CV metrics for KAN's extracted symbolic formula."""
    kan_dir = os.path.join(run_out_dir, 'baselines', 'KAN')
    metrics = _read_overall_metrics(
        os.path.join(kan_dir, 'overall_metrics_sym.csv'),
        pred_path=os.path.join(kan_dir, 'predictions.csv'), pred_col='y_pred_kansym')
    return {'KAN_symbolic': metrics} if metrics is not None else {}


def evaluate_full_baseline_models(full_models_dir, get_x_eval, y_eval):
    """Evaluate full baseline models.

    get_x_eval: callable(model_name) -> X_eval DataFrame, matching the features
                each model was trained with (i.e. using age{prior}_{target}_{name}_pred cols).
    """
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
            X_eval = get_x_eval(model_name)
            y_pred = m.predict(X_eval.values)
            mask = ~np.isnan(y_pred)
            if mask.sum() < 2:
                continue
            r2_val = r2_score(np.asarray(y_eval)[mask], y_pred[mask])
            results[model_name] = (r2_val, y_pred)
        except Exception:
            continue
    return results


def evaluate_full_formula_models(full_models_dir, get_x_eval, y_eval):
    """Evaluate full formula models (deeppysr / pysr / kan).

    get_x_eval: callable(model_type) -> X_eval DataFrame, matching the features
                each model was trained with.
    """
    results = {}

    deeppysr_dir = os.path.join(full_models_dir, 'deeppysr')
    if os.path.exists(deeppysr_dir):
        X_eval = get_x_eval('deeppysr')
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

    pysr_dir = os.path.join(full_models_dir, 'pysr')
    if os.path.exists(pysr_dir):
        X_eval = get_x_eval('pysr')
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

    kan_dir = os.path.join(full_models_dir, 'kan')
    if os.path.exists(kan_dir):
        X_eval = get_x_eval('kan')
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


def select_best_cv_model_and_save_metrics(run_out):
    """Pick the best grid config per family from genuine out-of-fold CV metrics
    and save cv_metrics_summary.csv."""
    rows = []

    for family, label in [('deeppysr', 'DeepPySR'), ('pysr', 'PySR')]:
        grid_metrics = collect_cv_grid_metrics(run_out, family)
        for cfg, m in grid_metrics.items():
            print(f"    {label} [{cfg}] cv_r2={m['r2']:.4f}")
        if grid_metrics:
            best_cfg = max(grid_metrics, key=lambda k: grid_metrics[k]['r2'])
            m = grid_metrics[best_cfg]
            print(f"  [{label} best CV config: {best_cfg}] r2={m['r2']:.4f}")
            rows.append({'family': family, 'model': label,
                         'cv_r2': m['r2'], 'cv_mae': m['mae'], 'cv_rmse': m['rmse'],
                         'cv_pearson_r': m['pearson_r']})

    for name, m in collect_cv_kan_symbolic_metrics(run_out).items():
        print(f"    KAN symbolic cv_r2={m['r2']:.4f}")
        rows.append({'family': 'kan', 'model': name,
                     'cv_r2': m['r2'], 'cv_mae': m['mae'], 'cv_rmse': m['rmse'],
                     'cv_pearson_r': m['pearson_r']})

    for name, m in collect_cv_baseline_metrics(run_out).items():
        print(f"    Baseline {name} CV r2={m['r2']:.4f}")
        rows.append({'family': 'baseline', 'model': name,
                     'cv_r2': m['r2'], 'cv_mae': m['mae'], 'cv_rmse': m['rmse'],
                     'cv_pearson_r': m['pearson_r']})

    if rows:
        metrics_csv = os.path.join(run_out, 'cv_metrics_summary.csv')
        pd.DataFrame(rows).to_csv(metrics_csv, index=False)
        print(f"  Saved CV metrics to {metrics_csv}\n")

    return rows


def select_best_full_model(all_results_dict):
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

def run_deeppysr_for_year(merged_df, target, age, prior_target_cols, feature_cols,
                          pysr_base_kwargs, deeppysr_configs, r2w_list, lambda_list, out_root):
    feature_cols = get_age_filtered_feature_cols(feature_cols, age)
    ids, X, y = load_forecast_data_for_model(
        merged_df, target, age, prior_target_cols, feature_cols, model_type='deeppysr')
    if ids is None or len(y) < 10:
        print(f"  Skipping {target} age {age}: insufficient data.")
        return None

    run_out = os.path.join(out_root, f'age_{age}')
    os.makedirs(run_out, exist_ok=True)

    print(f"\n{'='*20}\n[DeepPySR] Forecasting {target} age {age}  n={len(y)}\n{'='*20}")

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

    print(f'  Aggregating DeepPySR results for {target} age {age}...')
    aggregate_results(run_out, task='regression')
    return run_out


# ── Step 2: Baselines + PySR grid search ──────────────────────────────────────

def run_baselines_for_year(merged_df, target, age, prior_target_cols, feature_cols,
                           pysr_base_kwargs, pysr_configs, out_root):
    feature_cols = get_age_filtered_feature_cols(feature_cols, age)
    run_out = os.path.join(out_root, f'age_{age}')
    os.makedirs(run_out, exist_ok=True)

    print(f"\n{'='*20}\n[Baselines/PySR] Forecasting {target} age {age}\n{'='*20}")

    nit = pysr_base_kwargs.get('niterations', 500)
    pop = pysr_base_kwargs.get('populations', 100)
    sz  = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f'nit{nit}_pop{pop}_sz{sz}'

    print('  Evaluating Baseline Models...')
    baseline_model_instances = get_baseline_models(task='regression', input_dim=1)
    any_ran = False

    for name in baseline_model_instances:
        ids, X, y = load_forecast_data_for_model(
            merged_df, target, age, prior_target_cols, feature_cols, model_type=name)
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
        print(f'  Skipping {target} age {age}: insufficient data for all baseline models.')
        return None

    print('  Evaluating PySR Models...')
    ids_pysr, X_pysr, y_pysr = load_forecast_data_for_model(
        merged_df, target, age, prior_target_cols, feature_cols, model_type='pysr')
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

    print(f'  Aggregating Baselines/PySR results for {target} age {age}...')
    aggregate_results(run_out, task='regression')
    return run_out


# ── Step 3: Train all models on full data ─────────────────────────────────────

def train_full_models_for_year(merged_df, target, age, prior_target_cols, feature_cols,
                               pysr_base_kwargs, deeppysr_configs, pysr_configs,
                               r2w_list, lambda_list, run_out):
    import joblib as _jl
    feature_cols = get_age_filtered_feature_cols(feature_cols, age)
    full_out = os.path.join(run_out, 'full_models')

    nit = pysr_base_kwargs.get('niterations', 500)
    pop = pysr_base_kwargs.get('populations', 100)
    sz  = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f'nit{nit}_pop{pop}_sz{sz}'

    # DeepPySR — train on all data
    ids_dsr, X_dsr, y_dsr = load_forecast_data_for_model(
        merged_df, target, age, prior_target_cols, feature_cols, model_type='deeppysr')
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
        merged_df, target, age, prior_target_cols, feature_cols, model_type='pysr')
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

    # KAN — train on all data
    ids_kan, X_kan, y_kan = load_forecast_data_for_model(
        merged_df, target, age, prior_target_cols, feature_cols, model_type='KAN')
    if ids_kan is not None and len(y_kan) >= 10:
        kan_full_out = os.path.join(full_out, 'kan', 'full')
        if os.path.exists(os.path.join(kan_full_out, 'formulas_foldnocv.csv')):
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
            merged_df, target, age, prior_target_cols, feature_cols, model_type=name)
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


# ── Step 4: Rolling step for one age ──────────────────────────────────────────

def _age_done_flag(run_out):
    return os.path.join(run_out, 'done.flag')


def run_rolling_step(merged_df, target, age, feature_cols, rolling_csv, run_out):
    tcol = target_col(target, age)

    print(f'\n{"#"*50}')
    print(f'# Rolling step: {tcol}')
    print(f'{"#"*50}')

    if not os.path.exists(run_out):
        print(f'  No results directory at {run_out}, skipping.')
        return merged_df

    if tcol not in merged_df.columns:
        print(f'  {tcol} not in dataset, skipping.')
        return merged_df

    age_idx = AGES.index(age)
    prior_target_cols = [target_col(target, a) for a in AGES[:age_idx]
                         if target_col(target, a) in merged_df.columns]
    # Reassign to the age-filtered set so every use below (CV eval, full-model
    # training, and the baseline feature set inside save_rolling_dataset_with_predictions)
    # matches exactly what each model was trained on -- mirrors newbmiforecast's
    # run_rolling_step, which reassigns non_bmi_cols the same way.
    feature_cols = get_age_filtered_feature_cols(feature_cols, age)

    def _model_feature_cols(model_type):
        prior_pred = []
        for col in prior_target_cols:
            pred_col = f'{col}_{model_type}_pred'
            prior_pred.append(pred_col if pred_col in merged_df.columns else col)
        return [c for c in feature_cols if c in merged_df.columns] + prior_pred

    feature_cols_raw = [c for c in feature_cols if c in merged_df.columns] + \
                       [c for c in prior_target_cols if c in merged_df.columns]

    known_mask = merged_df[tcol].notna()
    y_eval = merged_df.loc[known_mask, tcol].values

    def _get_x_eval(model_type):
        fcols = _model_feature_cols(model_type)
        return merged_df.loc[known_mask, fcols].reset_index(drop=True)

    print(f'  Evaluating CV results on {known_mask.sum()} known samples...\n')

    print('\n  --- CV metrics (out-of-fold, no leakage) ---')
    select_best_cv_model_and_save_metrics(run_out)

    pysr_base_kwargs = _rolling_step_kwargs.get('pysr_base_kwargs', {})
    deeppysr_configs = _rolling_step_kwargs.get('deeppysr_configs', {})
    pysr_configs = _rolling_step_kwargs.get('pysr_configs', {})
    r2w_list = _rolling_step_kwargs.get('r2w_list', [1.5])
    lambda_list = _rolling_step_kwargs.get('lambda_list', [0.001])

    print('\n  --- Training full models on all available data ---')
    full_out = train_full_models_for_year(
        merged_df, target, age, prior_target_cols, feature_cols,
        pysr_base_kwargs, deeppysr_configs, pysr_configs,
        r2w_list, lambda_list, run_out)

    print('\n  --- Evaluating full models ---')
    full_formula_results = evaluate_full_formula_models(full_out, _get_x_eval, y_eval)
    full_baseline_results = evaluate_full_baseline_models(full_out, _get_x_eval, y_eval)

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

    merged_df = save_rolling_dataset_with_predictions(
        merged_df, target, age, full_results_by_family, rolling_csv,
        full_models_dir=full_out, feature_cols=feature_cols,
        prior_target_col_names=prior_target_cols,
        formula_feature_cols=feature_cols_raw)
    print(f'\n  Rolling dataset saved with full-model predictions')

    open(_age_done_flag(run_out), 'w').close()
    return merged_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
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
    }

    print('\n=== Preparing base dataset ===')
    merged_base, base_feature_cols = prepare_base_dataset()
    print(f'Base dataset: {len(merged_base)} rows, {len(merged_base.columns)} cols')

    for target in TARGETS:
        print(f'\n{"#"*60}')
        print(f'# TARGET: {target}')
        print(f'{"#"*60}')

        target_out_root = os.path.join(OUT_ROOT, target)
        os.makedirs(target_out_root, exist_ok=True)
        rolling_csv = os.path.join(target_out_root, 'rolling_dataset.csv')

        if os.path.exists(rolling_csv):
            print(f'\n=== Loading existing rolling dataset from {rolling_csv} ===')
            merged_df = pd.read_csv(rolling_csv)
        else:
            merged_df = merged_base.copy()

        raw_cols = _target_raw_cols(merged_df, target)
        feature_cols = [c for c in base_feature_cols if c not in raw_cols]

        for age in AGES:
            age_idx = AGES.index(age)
            prior_target_cols = [target_col(target, a) for a in AGES[:age_idx]
                                 if target_col(target, a) in merged_df.columns]

            run_out = os.path.join(target_out_root, f'age_{age}')
            if os.path.exists(_age_done_flag(run_out)):
                print(f'\n[SKIP] {target} age {age} — done.flag found, skipping.')
                continue

            print(f'\n{"="*60}')
            print(f'PIPELINE: predicting {target} at age {age}')
            print(f'  prior target cols: {prior_target_cols}')
            print(f'{"="*60}')

            # Step 1: DeepPySR CV
            run_deeppysr_for_year(
                merged_df, target, age, prior_target_cols, feature_cols,
                pysr_base_kwargs, deeppysr_configs, r2w_list, lambda_list, target_out_root,
            )

            # Step 2: Baselines + PySR CV
            run_baselines_for_year(
                merged_df, target, age, prior_target_cols, feature_cols,
                pysr_base_kwargs, pysr_configs, target_out_root,
            )

            # Step 3: Evaluate CV, train full models, predict NaNs, save rolling dataset
            merged_df = run_rolling_step(merged_df, target, age, feature_cols, rolling_csv, run_out)

    print('\n=== Pipeline complete ===')
    print(f'Results saved under: {OUT_ROOT}')


if __name__ == '__main__':
    main()