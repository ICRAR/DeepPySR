import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

import pandas as pd
from deeppysr import DeepPySR
from model_utils import get_deeppysr_configs, get_pysr_base_kwargs
from eval_utils import run_cv, aggregate_results
from data_utils import load_data, load_data_longitudinal

import argparse


def compute_homa_ir(y: pd.DataFrame) -> pd.Series:
    """HOMA-IR = insulin (mU/L) * glucose (mmol/L) / 22.5"""
    insulin_col = [c for c in y.columns if "insulin" in c][0]
    glucose_col = [c for c in y.columns if "glucose" in c][0]
    return (y[insulin_col] * y[glucose_col] / 22.5).rename("homa_ir")


def resolve_target(y_df: pd.DataFrame, target: str) -> pd.Series:
    if target == 'homa_ir':
        return compute_homa_ir(y_df)
    elif target == 'insulin':
        return y_df[[c for c in y_df.columns if "insulin" in c][0]]
    else:
        return y_df[[c for c in y_df.columns if "glucose" in c][0]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default='age_specific',
                        choices=['longitudinal', 'age_specific'])
    parser.add_argument('--age', type=int, default=17)
    parser.add_argument('--target', type=str, default='homa_ir',
                        choices=['homa_ir', 'insulin', 'glucose'])
    parser.add_argument('--vps', type=int, default=25)
    args = parser.parse_args()

    out_root = os.path.join(current_dir, "results_insulin")
    os.makedirs(out_root, exist_ok=True)

    deeppysr_configs = get_deeppysr_configs()
    if args.vps is not None:
        deeppysr_configs = {k: v for k, v in deeppysr_configs.items()
                            if f"vps{args.vps}_" in k}

    pysr_base_kwargs = get_pysr_base_kwargs()
    nit = pysr_base_kwargs.get('niterations', 100)
    pop = pysr_base_kwargs.get('populations', 30)
    sz = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f"nit{nit}_pop{pop}_sz{sz}"

    r2w_list = [1, 1.5, 2]
    lambda_list = [0.001, 0.005, 0.01]

    if args.setting == 'longitudinal':
        print("\nLoading longitudinal data...")
        ids, X, y_df = load_data_longitudinal(["insulin", "glucose"])
        y = resolve_target(y_df, args.target)
        runs = [(f"longitudinal_{args.target}", ids, X, y)]
    else:
        print(f"\nLoading data for age={args.age}...")
        ids, X, y_df = load_data(["insulin", "glucose"], args.age)
        y = resolve_target(y_df, args.target)
        runs = [(f"age_{args.age}_{args.target}", ids, X, y)]

    for run_name, ids, X, y in runs:
        print(f"\n--- Run: {run_name} ---")
        print(f"  n={len(X)}, features={X.shape[1]}, target={args.target}")

        run_out = os.path.join(out_root, run_name)
        os.makedirs(run_out, exist_ok=True)

        cv_kwargs = {
            'ids': ids,
            'task': 'regression',
            'n_splits': 5,
            'random_state': 42,
            'extra_data': X[['age']] if 'age' in X.columns else None,
        }
        if args.setting == 'longitudinal':
            cv_kwargs['groups'] = ids
            cv_kwargs['stratify_by'] = X['age'] if 'age' in X.columns else None

        print("\nEvaluating DeepPySR Models...")
        for cfg_name, cfg_overrides in deeppysr_configs.items():
            parts = cfg_name.split('_', 1)
            setting_prefix = parts[0]
            params_part = parts[1] if len(parts) > 1 else ""
            full_name = f"{setting_prefix}_{param_suffix}_{params_part}_grid"
            deeppysr_out = os.path.join(run_out, "deeppysr", full_name)
            if os.path.exists(os.path.join(deeppysr_out, "overall_metrics.csv")):
                print(f"  Skipping {full_name} (results exist)")
                continue
            print(f"  {full_name}...")

            def deeppysr_factory(co=cfg_overrides, out=deeppysr_out):
                kwargs = pysr_base_kwargs.copy()
                kwargs.update(co)
                return DeepPySR(
                    max_layers=1,
                    output_dir=out,
                    pareto_r2_weight=r2w_list,
                    pareto_lambda=lambda_list,
                    **kwargs,
                )

            run_cv(deeppysr_factory, X, y, outdir=deeppysr_out, scaler=False, **cv_kwargs)

        print(f"\nAggregating results for {run_name}...")
        aggregate_results(run_out, task='regression')


if __name__ == "__main__":
    main()
