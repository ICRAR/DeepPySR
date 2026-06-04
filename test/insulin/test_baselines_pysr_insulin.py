import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

import pandas as pd
from pysr import PySRRegressor
from model_utils import get_pysr_configs, get_baseline_models, get_pysr_base_kwargs, KANWrapper
from sklearn.base import clone
from eval_utils import run_cv, aggregate_results
from data_utils import load_data

import argparse


def compute_homa_ir(y: pd.DataFrame) -> pd.Series:
    """HOMA-IR = insulin (mU/L) * glucose (mmol/L) / 22.5"""
    insulin_col = [c for c in y.columns if "insulin" in c][0]
    glucose_col = [c for c in y.columns if "glucose" in c][0]
    return (y[insulin_col] * y[glucose_col] / 22.5).rename("homa_ir")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--age', type=int, default=17)
    parser.add_argument('--target', type=str, default='homa_ir',
                        choices=['homa_ir', 'insulin', 'glucose'])
    args = parser.parse_args()

    out_root = os.path.join(current_dir, "results_insulin")
    os.makedirs(out_root, exist_ok=True)

    print(f"\nLoading data for age={args.age}...")
    ids, X, y_df = load_data(["insulin", "glucose"], args.age)

    if args.target == 'homa_ir':
        y = compute_homa_ir(y_df)
    elif args.target == 'insulin':
        y = y_df[[c for c in y_df.columns if "insulin" in c][0]]
    else:
        y = y_df[[c for c in y_df.columns if "glucose" in c][0]]

    print(f"  n={len(X)}, features={X.shape[1]}, target={args.target}")

    run_name = f"age_{args.age}_{args.target}"
    run_out = os.path.join(out_root, run_name)
    os.makedirs(run_out, exist_ok=True)

    cv_kwargs = {
        'ids': ids,
        'task': 'regression',
        'n_splits': 5,
        'random_state': 42,
        'extra_data': None,
    }

    pysr_configs = get_pysr_configs()
    pysr_base_kwargs = get_pysr_base_kwargs()
    nit = pysr_base_kwargs.get('niterations', 100)
    pop = pysr_base_kwargs.get('populations', 30)
    sz = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f"nit{nit}_pop{pop}_sz{sz}"

    # 1. Baseline models
    print("\nEvaluating Baseline Models...")
    baseline_models = get_baseline_models(task='regression', input_dim=X.shape[1])
    for name, model_instance in baseline_models.items():
        model_out = os.path.join(run_out, "baselines", name)
        if os.path.exists(os.path.join(model_out, "overall_metrics.csv")):
            print(f"  Skipping {name} (results exist)")
            continue
        print(f"  {name}...")

        def baseline_factory(m=model_instance, n=name):
            if n == 'KAN':
                return KANWrapper(input_dim=X.shape[1], output_dim=1,
                                  hidden_dim=5, steps=200, update_grid=False,
                                  task='regression')
            return clone(m)

        run_cv(baseline_factory, X, y, outdir=model_out, **cv_kwargs)

    # 2. PySR models
    print("\nEvaluating PySR Models...")
    for cfg_name, cfg_overrides in pysr_configs.items():
        aps = cfg_overrides.get("adaptive_parsimony_scaling", 50.0)
        full_name = f"pysr_{param_suffix}_aps{aps}_grid"
        pysr_out = os.path.join(run_out, "pysr", full_name)
        if os.path.exists(os.path.join(pysr_out, "overall_metrics.csv")):
            print(f"  Skipping {full_name} (results exist)")
            continue
        print(f"  {full_name}...")

        def pysr_factory(co=cfg_overrides):
            kwargs = pysr_base_kwargs.copy()
            kwargs.update(co)
            return PySRRegressor(**kwargs)

        run_cv(pysr_factory, X, y, outdir=pysr_out, scaler=False, **cv_kwargs)

    print(f"\nAggregating results for {run_name}...")
    aggregate_results(run_out, task='regression')


if __name__ == "__main__":
    main()
