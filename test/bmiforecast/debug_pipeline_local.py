"""Local debug script for the BMI forecast rolling pipeline.

Runs the full pipeline (DeepPySR + baselines/PySR + rolling step) year by year
for a small subset of years, with reduced iterations so it finishes quickly.

Usage:
    cd /path/to/DeepPySR
    python test/bmiforecast/debug_pipeline_local.py

Tunable constants at the top of the file:
    DEBUG_YEARS   - which forecast years to run (subset of YEARS[1:])
    N_ITERATIONS  - niterations for both DeepPySR and PySR (keep small, e.g. 5)
    VPS, VPR, APS - single values for DeepPySR grid-search filtering
    PYSR_APS      - single aps value for PySR filtering
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

# ── Debug parameters ──────────────────────────────────────────────────────────
DEBUG_YEARS  = [10, 13]   # subset of YEARS[1:] to run
N_ITERATIONS = 5          # niterations for DeepPySR and PySR
VPS          = 25         # variable_prune_start filter
VPR          = 50         # variable_prune_ramp filter
APS          = 1.0        # adaptive_parsimony_scaling filter (DeepPySR)
PYSR_APS     = 1.0        # adaptive_parsimony_scaling filter (PySR)
N_SPLITS     = 2          # CV folds (keep small for speed)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from sklearn.base import clone
from pysr import PySRRegressor
from deeppysr import DeepPySR
from model_utils import (
    get_deeppysr_configs, get_pysr_configs,
    get_pysr_base_kwargs, get_baseline_models, KANWrapper,
)
from eval_utils import run_cv, aggregate_results
from bmiforecast_utils import (
    YEARS, actual_age,
    prepare_base_dataset, load_forecast_data,
    save_baseline_models, load_baseline_models,
    get_best_formula_for_year, get_best_pysr_formula_for_year,
    get_best_baseline_model,
    fill_missing_bmi_with_formula, fill_missing_bmi_with_model,
    _is_bmi_col,
)

OUT_ROOT = os.path.join(current_dir, 'results_bmiforecast_debug')
os.makedirs(OUT_ROOT, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_best_r2_from_dir(run_out_dir, subdir):
    import glob
    best = -float('inf')
    pattern = os.path.join(run_out_dir, subdir, '**', 'overall_metrics.csv')
    for mf in glob.glob(pattern, recursive=True):
        try:
            df = pd.read_csv(mf)
            if 'r2' in df.columns:
                best = max(best, df['r2'].mean())
        except Exception:
            pass
    return best


# ── Step 1: DeepPySR grid search for one year ─────────────────────────────────

def run_deeppysr_for_year(merged_df, target_year, prior_bmi_cols, non_bmi_cols,
                          pysr_base_kwargs, deeppysr_configs, r2w_list, lambda_list):
    ids, X, y = load_forecast_data(merged_df, target_year, prior_bmi_cols, non_bmi_cols)
    if ids is None or len(y) < 10:
        print(f"  Skipping yr{target_year}: insufficient data.")
        return None

    age_label = actual_age(target_year)
    run_out = os.path.join(OUT_ROOT, f'age_{age_label}')
    os.makedirs(run_out, exist_ok=True)

    print(f"\n{'='*20}\n[DeepPySR] Forecasting y{target_year}bmi (age {age_label})  n={len(y)}\n{'='*20}")

    cv_kwargs = {
        'ids': ids,
        'task': 'regression',
        'n_splits': N_SPLITS,
        'random_state': 42,
    }

    nit = pysr_base_kwargs.get('niterations', N_ITERATIONS)
    pop = pysr_base_kwargs.get('populations', 30)
    sz  = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f'nit{nit}_pop{pop}_sz{sz}'

    for cfg_name, cfg_overrides in deeppysr_configs.items():
        parts = cfg_name.split('_', 1)
        setting_prefix = parts[0]
        params_part = parts[1] if len(parts) > 1 else ''
        full_cfg_name = f'{setting_prefix}_{param_suffix}_{params_part}_grid'
        deeppysr_out = os.path.join(run_out, 'deeppysr', full_cfg_name)

        if os.path.exists(os.path.join(deeppysr_out, 'overall_metrics.csv')):
            print(f'  Skipping {full_cfg_name} (results exist)')
            continue

        print(f'  Running {full_cfg_name}...')

        def deeppysr_factory(co=cfg_overrides, dout=deeppysr_out):
            kwargs = pysr_base_kwargs.copy()
            kwargs.update(co)
            return DeepPySR(
                max_layers=1,
                output_dir=dout,
                pareto_r2_weight=r2w_list,
                pareto_lambda=lambda_list,
                **kwargs
            )

        run_cv(deeppysr_factory, X, y, outdir=deeppysr_out, scaler=False, **cv_kwargs)

    print(f'  Aggregating DeepPySR results for age_{age_label}...')
    aggregate_results(run_out, task='regression')
    return run_out


# ── Step 2: Baselines + PySR for one year ─────────────────────────────────────

def run_baselines_for_year(merged_df, target_year, prior_bmi_cols, non_bmi_cols,
                           pysr_base_kwargs, pysr_configs):
    ids, X, y = load_forecast_data(merged_df, target_year, prior_bmi_cols, non_bmi_cols)
    if ids is None or len(y) < 10:
        print(f"  Skipping yr{target_year}: insufficient data.")
        return None

    age_label = actual_age(target_year)
    run_out = os.path.join(OUT_ROOT, f'age_{age_label}')
    os.makedirs(run_out, exist_ok=True)

    print(f"\n{'='*20}\n[Baselines/PySR] Forecasting y{target_year}bmi (age {age_label})  n={len(y)}\n{'='*20}")

    cv_kwargs = {
        'ids': ids,
        'task': 'regression',
        'n_splits': N_SPLITS,
        'random_state': 42,
    }

    nit = pysr_base_kwargs.get('niterations', N_ITERATIONS)
    pop = pysr_base_kwargs.get('populations', 30)
    sz  = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f'nit{nit}_pop{pop}_sz{sz}'

    # Baseline models
    print('  Evaluating Baseline Models...')
    baseline_models = get_baseline_models(task='regression', input_dim=X.shape[1])
    fitted_models = {}
    for name, model_instance in baseline_models.items():
        model_out = os.path.join(run_out, 'baselines', name)
        if os.path.exists(os.path.join(model_out, 'overall_metrics.csv')):
            print(f'    Skipping {name} (results exist)')
            existing = load_baseline_models(run_out)
            if name not in existing:
                if name == 'KAN':
                    m = KANWrapper(input_dim=X.shape[1], output_dim=1,
                                   hidden_dim=5, steps=200, update_grid=False,
                                   task='regression')
                else:
                    m = clone(model_instance)
                m.fit(X.values, y)
                fitted_models[name] = m
            continue
        print(f'    {name}...')
        if name == 'KAN':
            m = KANWrapper(input_dim=X.shape[1], output_dim=1,
                           hidden_dim=5, steps=200, update_grid=False,
                           task='regression')
        else:
            m = clone(model_instance)

        def baseline_factory(inst=m):
            return clone(inst) if hasattr(inst, 'get_params') else inst

        run_cv(baseline_factory, X, y, outdir=model_out, **cv_kwargs)
        m_full = clone(m) if hasattr(m, 'get_params') else m
        m_full.fit(X.values, y)
        fitted_models[name] = m_full
        # Save immediately after each model completes so aborted jobs don't lose progress
        save_baseline_models(run_out, {name: m_full})

    # PySR models
    print('  Evaluating PySR Models...')
    for cfg_name, cfg_overrides in pysr_configs.items():
        aps = cfg_overrides.get('adaptive_parsimony_scaling', 50.0)
        full_name = f'pysr_{param_suffix}_aps{aps}_grid'
        pysr_out = os.path.join(run_out, 'pysr', full_name)
        if os.path.exists(os.path.join(pysr_out, 'overall_metrics.csv')):
            print(f'    Skipping {full_name} (results exist)')
            continue
        print(f'    {full_name}...')

        def pysr_factory(co=cfg_overrides):
            kwargs = pysr_base_kwargs.copy()
            kwargs.update(co)
            return PySRRegressor(**kwargs)

        run_cv(pysr_factory, X, y, outdir=pysr_out, scaler=False, **cv_kwargs)

    print(f'  Aggregating Baselines/PySR results for age_{age_label}...')
    aggregate_results(run_out, task='regression')
    return run_out


# ── Step 3: Rolling step for one year ─────────────────────────────────────────

def run_rolling_step(merged_df, non_bmi_cols, target_year, rolling_csv):
    bmi_col = f'y{target_year}bmi'
    age_label = actual_age(target_year)
    run_out = os.path.join(OUT_ROOT, f'age_{age_label}')

    print(f'\n{"#"*40}')
    print(f'# Rolling step: y{target_year}bmi')
    print(f'{"#"*40}')

    if not os.path.exists(run_out):
        print(f'  No results directory at {run_out}, skipping.')
        return merged_df

    if bmi_col not in merged_df.columns:
        print(f'  Column {bmi_col} not in dataset, skipping.')
        return merged_df

    year_idx = YEARS.index(target_year)
    prior_bmi_cols = [f'y{y}bmi' for y in YEARS[:year_idx]
                      if f'y{y}bmi' in merged_df.columns]
    feature_cols = [c for c in non_bmi_cols if c in merged_df.columns] + \
                   [c for c in prior_bmi_cols if c in merged_df.columns]

    known_mask = merged_df[bmi_col].notna()
    X_eval = merged_df.loc[known_mask, feature_cols].reset_index(drop=True)
    y_eval = merged_df.loc[known_mask, bmi_col].values

    r2_deeppysr = _get_best_r2_from_dir(run_out, 'deeppysr')
    r2_pysr     = _get_best_r2_from_dir(run_out, 'pysr')
    r2_baseline = _get_best_r2_from_dir(run_out, 'baselines')

    print(f'  Best r2 — DeepPySR: {r2_deeppysr:.4f}  PySR: {r2_pysr:.4f}  Baseline: {r2_baseline:.4f}')

    best_family = max(
        [('deeppysr', r2_deeppysr), ('pysr', r2_pysr), ('baseline', r2_baseline)],
        key=lambda x: x[1]
    )[0]
    print(f'  Selected family for imputation: {best_family}')

    # Each family uses only its own best result — no cross-family fallback.
    if best_family == 'deeppysr':
        formula, involved_str = get_best_formula_for_year(run_out)
        if formula is not None:
            merged_df = fill_missing_bmi_with_formula(merged_df, bmi_col, formula, involved_str)
        else:
            print(f'  No DeepPySR formula found; missing {bmi_col} values remain NaN.')

    elif best_family == 'pysr':
        formula, involved_str = get_best_pysr_formula_for_year(run_out, X_eval, y_eval)
        if formula is not None:
            merged_df = fill_missing_bmi_with_formula(merged_df, bmi_col, formula, involved_str)
        else:
            print(f'  No PySR formula found; missing {bmi_col} values remain NaN.')

    elif best_family == 'baseline':
        best_name = get_best_baseline_model(run_out)
        if best_name is not None:
            models = load_baseline_models(run_out)
            if best_name in models:
                merged_df = fill_missing_bmi_with_model(
                    merged_df, bmi_col, models[best_name], feature_cols)
            else:
                print(f'  Fitted model {best_name} not found on disk; missing values remain NaN.')
        else:
            print(f'  No baseline model found; missing {bmi_col} values remain NaN.')

    # Save updated rolling dataset
    merged_df.to_csv(rolling_csv, index=False)
    print(f'  Rolling dataset saved to {rolling_csv}')
    return merged_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rolling_csv = os.path.join(OUT_ROOT, 'rolling_dataset.csv')

    # Prepare base dataset (or load existing)
    if os.path.exists(rolling_csv):
        print(f'\n=== Loading existing rolling dataset from {rolling_csv} ===')
        merged_df = pd.read_csv(rolling_csv)
        non_bmi_cols = [c for c in merged_df.columns
                        if c != 'child_id' and not _is_bmi_col(c)]
    else:
        print('\n=== Preparing base dataset ===')
        merged_df, non_bmi_cols = prepare_base_dataset()
        merged_df.to_csv(os.path.join(OUT_ROOT, 'base_dataset.csv'), index=False)
        print(f'Base dataset: {len(merged_df)} rows, {len(merged_df.columns)} cols')

    # Build shared configs with reduced iterations
    pysr_base_kwargs = get_pysr_base_kwargs()
    pysr_base_kwargs['niterations'] = N_ITERATIONS  # override for debug

    r2w_list    = [1.5]
    lambda_list = [0.001]

    # Filter DeepPySR configs to a single combination
    deeppysr_configs = get_deeppysr_configs()
    aps_str = str(APS) if '.' in str(APS) else f'{APS}.0'
    deeppysr_configs = {k: v for k, v in deeppysr_configs.items()
                        if f'vps{VPS}_' in k and f'vpr{VPR}_' in k and f'aps{aps_str}' in k}
    print(f'\nDeepPySR configs selected: {list(deeppysr_configs.keys())}')

    # Filter PySR configs to a single aps
    pysr_configs = get_pysr_configs()
    pysr_aps_str = str(PYSR_APS) if '.' in str(PYSR_APS) else f'{PYSR_APS}.0'
    pysr_configs = {k: v for k, v in pysr_configs.items()
                    if f'aps{pysr_aps_str}' in k}
    print(f'PySR configs selected: {list(pysr_configs.keys())}')

    # Year-by-year rolling pipeline
    for year in DEBUG_YEARS:
        if year not in YEARS:
            print(f'WARNING: year {year} not in YEARS={YEARS}, skipping.')
            continue

        year_idx = YEARS.index(year)
        prior_bmi_cols = [f'y{y}bmi' for y in YEARS[:year_idx]
                          if f'y{y}bmi' in merged_df.columns]

        print(f'\n{"="*60}')
        print(f'PIPELINE: predicting y{year}bmi')
        print(f'  prior BMI cols: {prior_bmi_cols}')
        print(f'{"="*60}')

        # Step 1: DeepPySR
        run_deeppysr_for_year(
            merged_df, year, prior_bmi_cols, non_bmi_cols,
            pysr_base_kwargs, deeppysr_configs, r2w_list, lambda_list,
        )

        # Step 2: Baselines + PySR
        run_baselines_for_year(
            merged_df, year, prior_bmi_cols, non_bmi_cols,
            pysr_base_kwargs, pysr_configs,
        )

        # Step 3: Rolling step — select best, fill missing, save
        merged_df = run_rolling_step(merged_df, non_bmi_cols, year, rolling_csv)

    print('\n=== Debug pipeline complete ===')
    print(f'Results saved under: {OUT_ROOT}')


if __name__ == '__main__':
    main()
