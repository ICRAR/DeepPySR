"""Rolling pipeline step for BMI forecasting.

Run this for a single year AFTER all parallel grid-search jobs (DeepPySR,
PySR, KAN, and baselines) for that year have finished. It:
  1. Evaluates all formulas (DeepPySR, PySR, KAN) on real data and selects the best by r2.
  2. Evaluates all baseline models on real data and selects the best by r2.
  3. Selects the overall best formula/model across all families.
  4. Saves the predictions to CSV by model name.
  5. Fills missing BMI values in the rolling dataset using the best result.
  6. Saves the updated rolling dataset for the next year's grid search to use.
  7. Checks if models for the next age are already trained; if not, trains them.

Usage:
    python run_rolling_bmiforecast.py --year 10
    python run_rolling_bmiforecast.py --year 13
    ...
"""
import os
import sys
import argparse
import glob
import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from bmiforecast_utils import (
    YEARS, actual_age,
    prepare_base_dataset,
    _is_bmi_col,
    save_rolling_dataset_with_predictions,
    load_forecast_data_for_model,
)
from analysis_utils import evaluate_formula, calculate_metrics
from eval_utils import aggregate_results


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
        pattern2 = os.path.join(deeppysr_dir, '**', 'relationships.csv')
        files = glob.glob(pattern2, recursive=True)
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'formula' not in df.columns:
                continue
            for _, row in df.iterrows():
                formula = str(row['formula'])
                if formula in results:
                    continue  # Skip duplicates
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
    
    pattern = os.path.join(pysr_dir, '**', 'formulas_fold*.csv')
    files = glob.glob(pattern, recursive=True)
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'formula' not in df.columns:
                continue
            for _, row in df.iterrows():
                formula = str(row['formula'])
                if formula in results:
                    continue  # Skip duplicates
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
    kan_dir = os.path.join(run_out_dir, 'kan')
    
    if not os.path.exists(kan_dir):
        return results
    
    pattern = os.path.join(kan_dir, '**', 'formulas_fold*.csv')
    files = glob.glob(pattern, recursive=True)
    if not files:
        pattern2 = os.path.join(kan_dir, '**', 'formulas.csv')
        files = glob.glob(pattern2, recursive=True)
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'formula' not in df.columns:
                continue
            for _, row in df.iterrows():
                formula = str(row['formula'])
                if formula in results:
                    continue  # Skip duplicates
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
    """Evaluate full (non-CV) baseline models from full_models_dir/_models/.

    Returns:
        dict: {model_name: (r2, predictions)} or empty dict if none found.
    """
    results = {}
    models_dir = os.path.join(full_models_dir, '_models')
    if not os.path.exists(models_dir):
        return results

    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith('.joblib'):
            continue
        name = fname[:-len('.joblib')]
        try:
            model = joblib.load(os.path.join(models_dir, fname))
        except Exception:
            continue
        try:
            avail_cols = [c for c in feature_cols if c in X_eval.columns]
            can_predict = X_eval[avail_cols].notna().all(axis=1)
            y_pred_full = np.full(len(X_eval), np.nan)
            X_sub = X_eval.loc[can_predict, avail_cols]
            if len(X_sub) > 0:
                y_pred_full[can_predict.values] = model.predict(X_sub)
            mask = ~np.isnan(y_pred_full)
            if mask.sum() < 2:
                continue
            r2_val = r2_score(np.asarray(y_eval)[mask], y_pred_full[mask])
            results[name] = (r2_val, y_pred_full)
        except Exception:
            continue

    return results


def evaluate_full_formula_models(full_models_dir, X_eval, y_eval):
    """Evaluate full (non-CV) formula models from full_models_dir.

    Returns:
        dict: {family: {formula: (r2, predictions)}}
    """
    results = {}

    for family, fname_pattern in [
        ('deeppysr', 'relationships_nocv.csv'),
        ('pysr', 'formulas_foldnocv.csv'),
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


def select_best_cv_metrics(run_out, all_results_dict):
    """Select best CV result per family and save CV metrics. No model loading/saving."""
    results_by_family = {}

    for family, label in [('deeppysr', 'DeepPySR'), ('pysr', 'PySR'), ('kan', 'KAN')]:
        best_r2 = -float('inf')
        best_name = None
        best_preds = None
        best_formula = None
        for formula, (r2_val, preds) in all_results_dict.get(family, {}).items():
            print(f"    {label} formula CV r2={r2_val:.4f}: {formula[:60]}...")
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
        for model_name, (_, _, r2_val) in models_dict.items():
            rows.append({'family': family, 'model': model_name, 'cv_r2': r2_val})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(run_out, 'cv_metrics_summary.csv'), index=False)

    return results_by_family


def select_best_full_results(all_results_dict):
    """Select best full-training result per family."""
    results_by_family = {}

    for family, label in [('deeppysr', 'DeepPySR'), ('pysr', 'PySR'), ('kan', 'KAN')]:
        best_r2 = -float('inf')
        best_name = None
        best_preds = None
        best_formula = None
        for formula, (r2_val, preds) in all_results_dict.get(family, {}).items():
            print(f"    [Full] {label} r2={r2_val:.4f}: {formula[:60]}...")
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


def train_full_models_for_year(merged_df, year, prior_bmi_cols, non_bmi_cols,
                               run_out, pysr_base_kwargs, deeppysr_configs, pysr_configs,
                               r2w_list, lambda_list):
    """Train all models on ALL available data (no CV); save to run_out/full_models/."""
    from sklearn.base import clone
    from pysr import PySRRegressor
    from deeppysr import DeepPySR
    from model_utils import get_baseline_models, KANWrapper
    from eval_utils import run_nocv

    full_out = os.path.join(run_out, 'full_models')
    os.makedirs(full_out, exist_ok=True)

    nit = pysr_base_kwargs.get('niterations', 100)
    pop = pysr_base_kwargs.get('populations', 30)
    sz  = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f'nit{nit}_pop{pop}_sz{sz}'

    # DeepPySR
    ids_dsr, X_dsr, y_dsr = load_forecast_data_for_model(
        merged_df, year, prior_bmi_cols, non_bmi_cols, model_type='deeppysr')
    if ids_dsr is not None and len(y_dsr) >= 20:
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
        merged_df, year, prior_bmi_cols, non_bmi_cols, model_type='pysr')
    if ids_psr is not None and len(y_psr) >= 20:
        for cfg_name, cfg_overrides in pysr_configs.items():
            aps = cfg_overrides.get('adaptive_parsimony_scaling', 50.0)
            full_name = f'pysr_{param_suffix}_aps{aps}_full'
            psr_full_out = os.path.join(full_out, 'pysr', full_name)
            if os.path.exists(os.path.join(psr_full_out, 'overall_metrics.csv')):
                print(f'  Skipping PySR full: {full_name} (exists)')
                continue
            print(f'  Training PySR (full) {full_name}...')

            def pysr_factory(co=cfg_overrides):
                kwargs = pysr_base_kwargs.copy()
                kwargs.update(co)
                return PySRRegressor(**kwargs)

            run_nocv(pysr_factory, X_psr, y_psr, ids=ids_psr,
                     task='regression', outdir=psr_full_out, scaler=False)

    # Baseline models
    models_save_dir = os.path.join(full_out, '_models')
    os.makedirs(models_save_dir, exist_ok=True)
    baseline_model_instances = get_baseline_models(task='regression', input_dim=1)

    for name in baseline_model_instances:
        save_path = os.path.join(models_save_dir, f'{name}.joblib')
        kan_ckpt = os.path.join(models_save_dir, 'KAN')
        if name == 'KAN' and os.path.exists(f'{kan_ckpt}_config.yml'):
            print(f'  Skipping {name} full training (exists)')
            continue
        if name != 'KAN' and os.path.exists(save_path):
            print(f'  Skipping {name} full training (exists)')
            continue

        ids_bl, X_bl, y_bl = load_forecast_data_for_model(
            merged_df, year, prior_bmi_cols, non_bmi_cols, model_type=name)
        if ids_bl is None or len(y_bl) < 20:
            continue

        if name == 'KAN':
            m = KANWrapper(input_dim=X_bl.shape[1], output_dim=1,
                           hidden_dim=5, steps=200, update_grid=False, task='regression')
        else:
            m = clone(baseline_model_instances[name])

        print(f'  Training {name} (full)...')
        try:
            m.fit(X_bl.values, y_bl)
            if name == 'KAN' and hasattr(m, 'model') and hasattr(m.model, 'saveckpt'):
                try:
                    m.model.saveckpt(kan_ckpt)
                    print(f'  Saved full model: {name} (checkpoint)')
                except Exception as ex:
                    print(f'  Failed to save KAN checkpoint: {ex}')
            else:
                joblib.dump(m, save_path)
                print(f'  Saved full model: {name}')
        except Exception as ex:
            print(f'  WARNING: Failed to train {name} (full): {ex}')

    return full_out


def check_and_train_next_age_models(next_year, out_root, merged_df, feature_cols):
    """Check if models for next age exist; if not, train them.
    
    This ensures consistency: if year N predictions are used for year N+1 training,
    we need to ensure year N+1 models are available for the next rolling step.
    """
    if next_year not in YEARS:
        return
    
    next_age_label = actual_age(next_year)
    next_run_out = os.path.join(out_root, f'age_{next_age_label}')
    
    # Check if results exist for next age
    if os.path.exists(os.path.join(next_run_out, 'age_{next_age_label}', 'deeppysr')):
        print(f"  Models for year {next_year} already exist.")
        return
    
    print(f"  Note: Models for year {next_year} (age {next_age_label}) not yet trained.")
    print(f"        They will be generated by the corresponding grid-search job.")


def main():
    parser = argparse.ArgumentParser(description='BMI Forecast: rolling pipeline step for one year')
    parser.add_argument('--year', type=int, required=True,
                        help='Forecast year to process (e.g. 10, 13, 16, 20, 23, 26).')
    args = parser.parse_args()

    if args.year not in YEARS:
        raise ValueError(f'--year must be one of {YEARS}, got {args.year}')

    out_root = os.path.join(current_dir, 'results_bmiforecast')
    rolling_csv = os.path.join(out_root, 'rolling_dataset.csv')

    if os.path.exists(rolling_csv):
        print(f'Loading rolling dataset from {rolling_csv}')
        merged_df = pd.read_csv(rolling_csv)
        non_bmi_cols = [c for c in merged_df.columns
                        if c != 'child_id' and not _is_bmi_col(c) and not c.endswith('_pred')]
    else:
        print('No rolling dataset found; loading base dataset.')
        merged_df, non_bmi_cols = prepare_base_dataset()

    bmi_col = f'y{args.year}bmi'
    age_label = actual_age(args.year)
    run_out = os.path.join(out_root, f'age_{age_label}')

    print(f'\n{"#"*50}')
    print(f'# Rolling step: y{args.year}bmi (age {age_label})')
    print(f'{"#"*50}')

    year_idx = YEARS.index(args.year)
    prior_bmi_cols = [f'y{y}bmi' for y in YEARS[:year_idx]
                      if f'y{y}bmi' in merged_df.columns]

    saved_rolling = False

    if not os.path.exists(run_out):
        print(f'  No results directory found at {run_out}, skipping.')
    elif bmi_col not in merged_df.columns:
        print(f'  Column {bmi_col} not in dataset, skipping.')
    else:
        feature_cols = ([c for c in non_bmi_cols if c in merged_df.columns] +
                        [c for c in prior_bmi_cols if c in merged_df.columns])
        known_mask = merged_df[bmi_col].notna()
        X_eval = merged_df.loc[known_mask, feature_cols].reset_index(drop=True)
        y_eval = merged_df.loc[known_mask, bmi_col].values

        # ── CV evaluation (metrics only) ──────────────────────────────────────
        print(f'  Evaluating CV results on {len(X_eval)} known samples...\n')
        cv_all = {
            'deeppysr': evaluate_all_deeppysr_formulas(run_out, X_eval, y_eval),
            'pysr':     evaluate_all_pysr_formulas(run_out, X_eval, y_eval),
            'kan':      evaluate_all_kan_formulas(run_out, X_eval, y_eval),
            'baseline': evaluate_all_baseline_models(run_out, X_eval, y_eval, feature_cols),
        }
        if any(cv_all.values()):
            print('\n  --- CV metrics ---')
            select_best_cv_metrics(run_out, cv_all)

        # ── Full model training ───────────────────────────────────────────────
        from model_utils import get_deeppysr_configs, get_pysr_configs, get_pysr_base_kwargs
        pysr_base_kwargs = get_pysr_base_kwargs()
        deeppysr_configs = get_deeppysr_configs()
        pysr_configs = get_pysr_configs()
        r2w_list = [1.5]
        lambda_list = [0.001]

        print('\n  --- Training full models on all available data ---')
        full_out = train_full_models_for_year(
            merged_df, args.year, prior_bmi_cols, non_bmi_cols,
            run_out, pysr_base_kwargs, deeppysr_configs, pysr_configs,
            r2w_list, lambda_list)

        # ── Evaluate full models ──────────────────────────────────────────────
        print('\n  --- Evaluating full models ---')
        full_all = {
            **evaluate_full_formula_models(full_out, X_eval, y_eval),
            'baseline': evaluate_full_baseline_models(full_out, X_eval, y_eval, feature_cols),
        }

        if any(full_all.values()):
            full_results_by_family = select_best_full_results(full_all)
            merged_df = save_rolling_dataset_with_predictions(
                merged_df, args.year, full_results_by_family, rolling_csv,
                full_models_dir=full_out, non_bmi_cols=non_bmi_cols,
                prior_bmi_col_names=prior_bmi_cols,
                formula_feature_cols=feature_cols)
            print(f'\nRolling dataset saved with full-model predictions')
            saved_rolling = True
        else:
            print(f'  No full-model results found; skipping NaN prediction.')

    if not saved_rolling:
        os.makedirs(out_root, exist_ok=True)
        merged_df.to_csv(rolling_csv, index=False)
        print(f'\nRolling dataset saved to {rolling_csv}')

    if os.path.exists(run_out):
        print(f'\n=== Aggregating results for age_{age_label} ===')
        aggregate_results(run_out, task='regression')

    next_year_idx = year_idx + 1
    if next_year_idx < len(YEARS):
        next_year = YEARS[next_year_idx]
        check_and_train_next_age_models(next_year, out_root, merged_df,
                                        [c for c in non_bmi_cols if c in merged_df.columns] +
                                        [f'y{YEARS[i]}bmi' for i in range(next_year_idx)
                                         if f'y{YEARS[i]}bmi' in merged_df.columns])


if __name__ == '__main__':
    main()
