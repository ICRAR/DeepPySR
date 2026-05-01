import os
import sys

# Add parent directory to sys.path to import from test/
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

import numpy as np
import pandas as pd
from deeppysr import DeepPySR
from model_utils import (
    get_deeppysr_configs, get_pysr_configs, get_baseline_models, 
    get_pysr_base_kwargs, KANWrapper,
)

from sklearn.base import clone
from eval_utils import run_cv, run_nocv, aggregate_results
from bmi_utils import load_bmi_agg_data

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default=None, choices=['longitudinal', 'age_specific'])
    parser.add_argument('--age', type=int, default=None)
    parser.add_argument('--vps', type=int, default=None)
    args = parser.parse_args()

    out_root = os.path.join(current_dir, "results_bmi_all")
    os.makedirs(out_root, exist_ok=True)
    
    settings = [args.setting] if args.setting else ['longitudinal', 'age_specific']
    ages = [args.age] if args.age else [8, 10, 14, 17, 20, 23, 27]
    
    # DeepPySR grid search params
    r2w_list = [1, 1.5, 2]
    lambda_list = [0.001, 0.005, 0.01]
    
    deeppysr_configs = get_deeppysr_configs()
    # Filter by vps if specified
    if args.vps is not None:
        deeppysr_configs = {k: v for k, v in deeppysr_configs.items() if f"vps{args.vps}_" in k}

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

    # 2. DeepPySR Models
            print(f"Evaluating DeepPySR (pypysr) Models...")
            for cfg_name, cfg_overrides in deeppysr_configs.items():
                parts = cfg_name.split('_', 1)
                setting_prefix = parts[0]
                params_part = parts[1] if len(parts) > 1 else ""
                
                full_cfg_name = f"{setting_prefix}_{param_suffix}_{params_part}_grid"
                deeppysr_out = os.path.join(run_out_root, "deeppysr", full_cfg_name)
                if os.path.exists(os.path.join(deeppysr_out, "overall_metrics.csv")):
                    continue
                
                print(f"  {full_cfg_name}...")
                def deeppysr_factory():
                    kwargs = pysr_base_kwargs.copy()
                    kwargs.update(cfg_overrides)
                    
                    return DeepPySR(
                        max_layers=1,
                        output_dir=deeppysr_out,
                        pareto_r2_weight=r2w_list,
                        pareto_lambda=lambda_list,
                        **kwargs
                    )
                
                run_cv(deeppysr_factory, X, y, outdir=deeppysr_out, scaler=False, **cv_kwargs)

            print(f"Aggregating results for {run_name}...")
            aggregate_results(run_out_root, task='regression')

    if args.setting is None and args.vps is None:
        print(f"\nAggregating all results across all settings...")
        aggregate_results(out_root, task='regression')

if __name__ == "__main__":
    main()
