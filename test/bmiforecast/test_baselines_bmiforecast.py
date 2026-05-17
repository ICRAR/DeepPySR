import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
from sklearn.base import clone
from pysr import PySRRegressor
from model_utils import (
    get_pysr_configs, get_baseline_models,
    get_pysr_base_kwargs, KANWrapper,
)
from eval_utils import run_cv, aggregate_results
from bmiforecast_utils import (
    YEARS, actual_age,
    prepare_base_dataset, load_forecast_data,
    save_baseline_models,
)


def run_baselines_for_year(merged_df, target_year, prior_bmi_cols, non_bmi_cols,
                           out_root, pysr_base_kwargs, pysr_configs):
    ids, X, y = load_forecast_data(merged_df, target_year, prior_bmi_cols, non_bmi_cols)
    if ids is None or len(y) < 20:
        print(f"  Skipping yr{target_year}: insufficient data.")
        return None
    age_label = actual_age(target_year)
    run_name = f'age_{age_label}'
    run_out = os.path.join(out_root, run_name)
    os.makedirs(run_out, exist_ok=True)
    print(f"\n{'='*20}\nForecasting y{target_year}bmi (age {age_label})  n={len(y)}\n{'='*20}")
    cv_kwargs = {
        'ids': ids,
        'task': 'regression',
        'n_splits': 5,
        'random_state': 42,
    }
    nit = pysr_base_kwargs.get('niterations', 100)
    pop = pysr_base_kwargs.get('populations', 30)
    sz  = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f'nit{nit}_pop{pop}_sz{sz}'

    feature_cols = list(X.columns)

    # 1. Baseline models — run CV and also fit on full data for imputation
    print('Evaluating Baseline Models...')
    baseline_models = get_baseline_models(task='regression', input_dim=X.shape[1])
    fitted_models = {}
    for name, model_instance in baseline_models.items():
        model_out = os.path.join(run_out, 'baselines', name)
        if os.path.exists(os.path.join(model_out, 'overall_metrics.csv')):
            print(f'  Skipping {name} (results exist)')
            # Still fit on full data for imputation if not already saved
            from bmiforecast_utils import load_baseline_models
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
        print(f'  {name}...')
        if name == 'KAN':
            m = KANWrapper(input_dim=X.shape[1], output_dim=1,
                           hidden_dim=5, steps=200, update_grid=False,
                           task='regression')
        else:
            m = clone(model_instance)

        def baseline_factory(inst=m):
            return clone(inst) if hasattr(inst, 'get_params') else inst

        run_cv(baseline_factory, X, y, outdir=model_out, **cv_kwargs)
        # Fit on full data for imputation
        m_full = clone(m) if hasattr(m, 'get_params') else m
        m_full.fit(X.values, y)
        fitted_models[name] = m_full
        # Save immediately after each model so aborted jobs don't lose progress
        save_baseline_models(run_out, {name: m_full})

    # 2. PySR models
    print('Evaluating PySR Models...')
    for cfg_name, cfg_overrides in pysr_configs.items():
        aps = cfg_overrides.get('adaptive_parsimony_scaling', 50.0)
        full_name = f'pysr_{param_suffix}_aps{aps}_grid'
        pysr_out = os.path.join(run_out, 'pysr', full_name)
        if os.path.exists(os.path.join(pysr_out, 'overall_metrics.csv')):
            continue
        print(f'  {full_name}...')
        def pysr_factory(co=cfg_overrides):
            kwargs = pysr_base_kwargs.copy()
            kwargs.update(co)
            return PySRRegressor(**kwargs)
        run_cv(pysr_factory, X, y, outdir=pysr_out, scaler=False, **cv_kwargs)

    print(f'Aggregating results for {run_name}...')
    aggregate_results(run_out, task='regression')
    return run_out


def main():
    import argparse
    parser = argparse.ArgumentParser(description='BMI Forecast: Baselines + PySR pipeline')
    parser.add_argument('--year', type=int, default=None,
                        help='Single forecast year to run (e.g. 10, 13). If omitted, runs all years.')
    parser.add_argument('--aps', type=float, default=None,
                        help='Filter PySR configs by aps value.')
    args = parser.parse_args()

    if args.year is not None and args.year not in YEARS:
        raise ValueError(f'--year must be one of {YEARS}, got {args.year}')

    out_root = os.path.join(current_dir, 'results_bmiforecast')
    os.makedirs(out_root, exist_ok=True)

    rolling_csv = os.path.join(out_root, 'rolling_dataset.csv')
    import pandas as pd
    if os.path.exists(rolling_csv):
        print(f'\n=== Loading rolling dataset from {rolling_csv} ===')
        merged_df = pd.read_csv(rolling_csv)
        # Reconstruct non_bmi_cols from the dataset
        from bmiforecast_utils import _is_bmi_col
        non_bmi_cols = [c for c in merged_df.columns
                        if c != 'child_id' and not _is_bmi_col(c)]
    else:
        print('\n=== Preparing base dataset ===')
        merged_df, non_bmi_cols = prepare_base_dataset()
        merged_df.to_csv(os.path.join(out_root, 'base_dataset.csv'), index=False)
        print(f'Base dataset saved ({len(merged_df)} rows, {len(merged_df.columns)} cols)')

    pysr_configs = get_pysr_configs()
    if args.aps is not None:
        aps_str = str(args.aps) if '.' in str(args.aps) else f'{args.aps}.0'
        pysr_configs = {k: v for k, v in pysr_configs.items()
                        if f'aps{aps_str}' in k}
    pysr_base_kwargs = get_pysr_base_kwargs()

    if args.year is not None:
        # Single-year mode: use prior BMI cols already present in rolling dataset
        from bmiforecast_utils import _is_bmi_col
        year_idx = YEARS.index(args.year)
        prior_bmi_cols = [f'y{y}bmi' for y in YEARS[:year_idx]
                          if f'y{y}bmi' in merged_df.columns]
        print(f'\n{"#"*40}')
        print(f'# Baselines/PySR: predicting y{args.year}bmi')
        print(f'# Features: non-BMI vars + {prior_bmi_cols}')
        print(f'{"#"*40}')
        run_baselines_for_year(
            merged_df, args.year, prior_bmi_cols, non_bmi_cols,
            out_root, pysr_base_kwargs, pysr_configs,
        )
    else:
        # All-years rolling mode
        forecast_years = YEARS[1:]
        prior_bmi_cols = ['y8bmi'] if 'y8bmi' in merged_df.columns else []
        for year in forecast_years:
            print(f'\n{"#"*40}')
            print(f'# Rolling step: predicting y{year}bmi')
            print(f'# Features: non-BMI vars + {prior_bmi_cols}')
            print(f'{"#"*40}')
            run_baselines_for_year(
                merged_df, year, prior_bmi_cols, non_bmi_cols,
                out_root, pysr_base_kwargs, pysr_configs,
            )
            bmi_col = f'y{year}bmi'
            if bmi_col not in prior_bmi_cols and bmi_col in merged_df.columns:
                prior_bmi_cols.append(bmi_col)

    print('\n=== Aggregating all forecast results ===')
    aggregate_results(out_root, task='regression')


if __name__ == '__main__':
    main()
