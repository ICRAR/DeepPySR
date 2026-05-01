import os
import sys

# Add parent directory to sys.path to import from test/
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

import numpy as np
import pandas as pd
from pysr import PySRRegressor
from model_utils import (
    get_pysr_configs, get_baseline_models, 
    get_pysr_base_kwargs, KANWrapper
)

from sklearn.base import clone
from eval_utils import run_cv, aggregate_results
from bmi_utils import load_bmi_agg_data

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default=None, choices=['longitudinal', 'age_specific'])
    parser.add_argument('--age', type=int, default=None)
    args = parser.parse_args()

    out_root = os.path.join(current_dir, "results_bmi_all")
    os.makedirs(out_root, exist_ok=True)
    
    settings = [args.setting] if args.setting else ['longitudinal', 'age_specific']
    ages = [args.age] if args.age else [8, 10, 14, 17, 20, 23, 27]
    
    # DeepPySR grid search params
    r2w_list = [1, 1.5, 2]
    lambda_list = [0.001, 0.005, 0.01]
    
    pysr_configs = get_pysr_configs()
    pysr_base_kwargs = get_pysr_base_kwargs()
    
    # Extract parameters for folder naming
    nit = pysr_base_kwargs.get('niterations', 100)
    pop = pysr_base_kwargs.get('populations', 30)
    sz = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f"nit{nit}_pop{pop}_sz{sz}"
    
    for setting in settings:
        print(f"\n{'='*20}\nSetting: {setting}\n{'='*20}")
        if setting == 'longitudinal':
            ids, X, y = load_bmi_agg_data()
            runs = [("longitudinal", ids, X, y)]
            run_out_root_base = out_root
        else:
            runs = []
            for age in ages:
                ids, X, y = load_bmi_agg_data(age=age)
                if not X.empty:
                    runs.append((f"age_{age}", ids, X, y))
            run_out_root_base = os.path.join(out_root, setting)
        
        for run_name, ids, X, y in runs:
            print(f"\n--- Run: {run_name} ---")
            run_out_root = os.path.join(run_out_root_base, run_name)
            os.makedirs(run_out_root, exist_ok=True)
            
            # Prepare cv arguments
            cv_kwargs = {
                'ids': ids,
                'task': 'regression',
                'n_splits': 5,
                'random_state': 42,
                'extra_data': X[['age']] if 'age' in X.columns else None
            }
            if setting == 'longitudinal':
                cv_kwargs['groups'] = ids
                cv_kwargs['stratify_by'] = X['age'] if 'age' in X.columns else None

            # 1. Baseline Models
            print(f"Evaluating Baseline Models...")
            baseline_models = get_baseline_models(task='regression', input_dim=X.shape[1])
            for name, model_instance in baseline_models.items():
                model_out = os.path.join(run_out_root, "baselines", name)
                if os.path.exists(os.path.join(model_out, "overall_metrics.csv")):
                    print(f"  Skipping {name} (results exist)")
                    continue
                print(f"  {name}...")
                
                def baseline_factory(m=model_instance, n=name):
                    if n == 'KAN':
                        return KANWrapper(input_dim=X.shape[1], output_dim=1, hidden_dim=5, steps=200, update_grid=False, task='regression')
                    return clone(m)
                
                run_cv(baseline_factory, X, y, outdir=model_out, **cv_kwargs)

            # 2. DeepPySR (pysr) Model
            print(f"Evaluating DeepPySR (pysr) Model...")
            for cfg_name, cfg_overrides in pysr_configs.items():
                aps = cfg_overrides.get("adaptive_parsimony_scaling", 50.0)
                full_name = f"pysr_{param_suffix}_aps{aps}_grid"
                deeppysr_pysr_out = os.path.join(run_out_root, "pysr", full_name)
                if os.path.exists(os.path.join(deeppysr_pysr_out, "overall_metrics.csv")):
                    continue

                print(f"  {full_name}...")
                def deeppysr_pysr_factory(co=cfg_overrides):
                    kwargs = pysr_base_kwargs.copy()
                    kwargs.update(co)
                    return PySRRegressor(
                        **kwargs
                    )

                run_cv(deeppysr_pysr_factory, X, y, outdir=deeppysr_pysr_out, scaler=False, **cv_kwargs)

            print(f"Aggregating results for {run_name}...")
            aggregate_results(run_out_root, task='regression')

    if args.setting is None:
        print(f"\nAggregating all results across all settings...")
        aggregate_results(out_root, task='regression')

if __name__ == "__main__":
    main()
