import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from deeppysr import DeepPySR
from model_utils import (
    get_deeppysr_configs, get_pysr_base_kwargs,
)
from eval_utils import run_cv, aggregate_results
from bmiforecast_utils import (
    YEARS, actual_age,
    prepare_base_dataset, load_forecast_data,
)


def run_deeppysr_for_year(merged_df, target_year, prior_bmi_cols, non_bmi_cols,
                          out_root, pysr_base_kwargs, deeppysr_configs,
                          r2w_list, lambda_list):
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

    # DeepPySR models
    print('Evaluating DeepPySR Models...')
    for cfg_name, cfg_overrides in deeppysr_configs.items():
        parts = cfg_name.split('_', 1)
        setting_prefix = parts[0]
        params_part = parts[1] if len(parts) > 1 else ''
        full_cfg_name = f'{setting_prefix}_{param_suffix}_{params_part}_grid'
        deeppysr_out = os.path.join(run_out, 'deeppysr', full_cfg_name)
        if os.path.exists(os.path.join(deeppysr_out, 'overall_metrics.csv')):
            continue
        print(f'  {full_cfg_name}...')

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

    print(f'Aggregating results for {run_name}...')
    aggregate_results(run_out, task='regression')

    return run_out


def main():
    import argparse
    parser = argparse.ArgumentParser(description='BMI Forecast: DeepPySR grid search for one year')
    parser.add_argument('--year', type=int, required=True,
                        help='Forecast year to run grid search for (e.g. 10, 13, 16, 20, 23, 26).')
    parser.add_argument('--vps', type=int, default=None,
                        help='Filter DeepPySR configs by vps value.')
    parser.add_argument('--vpr', type=int, default=None,
                        help='Filter DeepPySR configs by vpr value.')
    parser.add_argument('--aps', type=float, default=None,
                        help='Filter DeepPySR configs by aps value.')
    args = parser.parse_args()

    if args.year not in YEARS:
        raise ValueError(f'--year must be one of {YEARS}, got {args.year}')

    out_root = os.path.join(current_dir, 'results_bmiforecast')
    os.makedirs(out_root, exist_ok=True)

    print('\n=== Preparing base dataset ===')
    merged_df, non_bmi_cols = prepare_base_dataset()
    merged_df.to_csv(os.path.join(out_root, 'base_dataset.csv'), index=False)
    print(f'Base dataset saved ({len(merged_df)} rows, {len(merged_df.columns)} cols)')

    r2w_list = [1, 1.5, 2]
    lambda_list = [0.001, 0.005, 0.01]
    deeppysr_configs = get_deeppysr_configs()
    if args.vps is not None:
        deeppysr_configs = {k: v for k, v in deeppysr_configs.items()
                            if f'vps{args.vps}_' in k}
    if args.vpr is not None:
        deeppysr_configs = {k: v for k, v in deeppysr_configs.items()
                            if f'vpr{args.vpr}_' in k}
    if args.aps is not None:
        aps_str = str(args.aps) if '.' in str(args.aps) else f'{args.aps}.0'
        deeppysr_configs = {k: v for k, v in deeppysr_configs.items()
                            if f'aps{aps_str}' in k}
    pysr_base_kwargs = get_pysr_base_kwargs()

    # Build prior BMI cols from real data only (no formula-filling here).
    # Include all BMI years that come before the target year.
    prior_bmi_cols = []
    for y in YEARS:
        if y >= args.year:
            break
        col = f'y{y}bmi'
        if col in merged_df.columns:
            prior_bmi_cols.append(col)

    print(f'\n{"#"*40}')
    print(f'# Grid search: predicting y{args.year}bmi')
    print(f'# Features: non-BMI vars + {prior_bmi_cols}')
    print(f'{"#"*40}')

    run_deeppysr_for_year(
        merged_df, args.year, prior_bmi_cols, non_bmi_cols,
        out_root, pysr_base_kwargs, deeppysr_configs,
        r2w_list, lambda_list,
    )


if __name__ == '__main__':
    main()
