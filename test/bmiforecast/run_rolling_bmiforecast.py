"""Rolling pipeline step for BMI forecasting.

Run this for a single year AFTER all parallel grid-search jobs (DeepPySR,
PySR, and baselines) for that year have finished. It:
  1. Selects the best model/formula across ALL configs for that year:
       - DeepPySR: best symbolic formula by r2
       - PySR: best symbolic formula by r2
       - Baselines: best fitted model by r2
  2. Fills missing BMI values in the rolling dataset using the best result.
  3. Saves the updated rolling dataset for the next year's grid search to use.

Usage:
    python run_rolling_bmiforecast.py --year 10
    python run_rolling_bmiforecast.py --year 13
    ...
"""
import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from bmiforecast_utils import (
    YEARS, actual_age,
    prepare_base_dataset,
    get_best_formula_for_year,
    get_best_pysr_formula_for_year,
    get_best_baseline_model,
    load_baseline_models,
    fill_missing_bmi_with_formula,
    fill_missing_bmi_with_model,
    load_forecast_data, _is_bmi_col,
)
from eval_utils import aggregate_results


def _get_best_r2_from_dir(run_out_dir, subdir):
    """Return the best mean r2 from overall_metrics.csv files under run_out_dir/subdir."""
    import glob
    import pandas as pd
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


def main():
    parser = argparse.ArgumentParser(description='BMI Forecast: rolling pipeline step for one year')
    parser.add_argument('--year', type=int, required=True,
                        help='Forecast year to process (e.g. 10, 13, 16, 20, 23, 26).')
    args = parser.parse_args()

    if args.year not in YEARS:
        raise ValueError(f'--year must be one of {YEARS}, got {args.year}')

    out_root = os.path.join(current_dir, 'results_bmiforecast')
    rolling_csv = os.path.join(out_root, 'rolling_dataset.csv')

    import pandas as pd
    if os.path.exists(rolling_csv):
        print(f'Loading rolling dataset from {rolling_csv}')
        merged_df = pd.read_csv(rolling_csv)
        non_bmi_cols = [c for c in merged_df.columns
                        if c != 'child_id' and not _is_bmi_col(c)]
    else:
        print('No rolling dataset found; loading base dataset.')
        merged_df, non_bmi_cols = prepare_base_dataset()

    bmi_col = f'y{args.year}bmi'
    age_label = actual_age(args.year)
    run_out = os.path.join(out_root, f'age_{age_label}')

    print(f'\n{"#"*40}')
    print(f'# Rolling step: y{args.year}bmi')
    print(f'{"#"*40}')

    if not os.path.exists(run_out):
        print(f'  No results directory found at {run_out}, skipping formula selection.')
    elif bmi_col not in merged_df.columns:
        print(f'  Column {bmi_col} not in dataset, skipping.')
    else:
        # Build X and y for formula evaluation (rows where bmi_col is known)
        year_idx = YEARS.index(args.year)
        prior_bmi_cols = [f'y{y}bmi' for y in YEARS[:year_idx]
                          if f'y{y}bmi' in merged_df.columns]
        feature_cols = [c for c in non_bmi_cols if c in merged_df.columns] + \
                       [c for c in prior_bmi_cols if c in merged_df.columns]
        known_mask = merged_df[bmi_col].notna()
        X_eval = merged_df.loc[known_mask, feature_cols].reset_index(drop=True)
        y_eval = merged_df.loc[known_mask, bmi_col].values

        # Gather best r2 from each model family
        r2_deeppysr = _get_best_r2_from_dir(run_out, 'deeppysr')
        r2_pysr = _get_best_r2_from_dir(run_out, 'pysr')
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
    os.makedirs(out_root, exist_ok=True)
    merged_df.to_csv(rolling_csv, index=False)
    print(f'\nRolling dataset updated and saved to {rolling_csv}')

    if os.path.exists(run_out):
        print(f'\n=== Aggregating results for age_{age_label} ===')
        aggregate_results(run_out, task='regression')


if __name__ == '__main__':
    main()
