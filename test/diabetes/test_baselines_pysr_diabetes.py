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
from diabetes_utils import load_diabetes_brfss_data

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_group', type=str, choices=['baselines', 'pysr', 'all'], default='all',
                        help='Which group of models to run: baselines, pysr, or all')
    parser.add_argument('--aps', type=float, default=None,
                        help='Adaptive parsimony scaling for PySR (if model_group is pysr)')
    args = parser.parse_args()

    out_root = os.path.join(current_dir, "results_diabetes_all")
    os.makedirs(out_root, exist_ok=True)
    
    print(f"\n" + "="*50)
    print(f"Processing Diabetes BRFSS 2015 ({args.model_group.capitalize()})")
    print("="*50)
    
    X, y = load_diabetes_brfss_data()
    
    task = 'classification'
    
    pysr_configs = get_pysr_configs()
    pysr_base_kwargs = get_pysr_base_kwargs()
    pysr_base_kwargs['niterations'] = 100
    
    # Extract parameters for folder naming
    nit = pysr_base_kwargs.get('niterations', 100)
    pop = pysr_base_kwargs.get('populations', 30)
    sz = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f"nit{nit}_pop{pop}_sz{sz}"
    
    # Prepare cv arguments
    cv_kwargs = {
        'task': task,
        'n_splits': 5,
        'random_state': 42,
        'stratify_by': y,
        'feature_selection': False 
    }
    
    # 1. Baseline Models
    if args.model_group in ['baselines', 'all']:
        print(f"Evaluating Baseline Models...")
        baseline_models = get_baseline_models(task=task, input_dim=X.shape[1])
        for name, model_instance in baseline_models.items():
            def baseline_factory(m=model_instance, n=name):
                if n == 'KAN':
                    return KANWrapper(input_dim=X.shape[1], output_dim=1, hidden_dim=5, steps=200, update_grid=False, task=task)
                return clone(m)
                
            model_out = os.path.join(out_root, "baselines", name)
            if os.path.exists(os.path.join(model_out, "overall_metrics.csv")):
                print(f"  Skipping {name} (results exist)")
            else:
                print(f"  {name}...")
                run_cv(baseline_factory, X, y, outdir=model_out, **cv_kwargs)

    # 2. PySR Comparison
    if args.model_group in ['pysr', 'all']:
        print(f"\nEvaluating PySR Comparison...")
        
        # If aps is provided, only run configs that match it
        target_configs = {}
        if args.aps is not None:
            # Look for a config that matches the aps value
            # In model_utils, configs are usually named like 'aps1.0', 'aps10.0' etc.
            for cfg_name, cfg in pysr_configs.items():
                if f"aps{args.aps}" in cfg_name:
                    target_configs[cfg_name] = cfg
            if not target_configs:
                print(f"Warning: No PySR config found matching aps={args.aps}")
        else:
            target_configs = pysr_configs

        for cfg_name, cfg in target_configs.items():
            print(f"  Config: {cfg_name}...")
            pysr_out = os.path.join(out_root, "pysr", f"{cfg_name}_{param_suffix}")

            def deeppysr_pysr_factory(co=cfg):
                return PySRRegressor(
                    **pysr_base_kwargs,
                    **co,
                    batching=True,
                    batch_size=500
                )

            if os.path.exists(os.path.join(pysr_out, "overall_metrics.csv")):
                print(f"    Skipping (results exist)")
            else:
                run_cv(deeppysr_pysr_factory, X, y, outdir=pysr_out, scaler=False, **cv_kwargs)

    # Final Aggregation (only if all or specifically requested)
    # Usually better to run aggregation separately or at the end of all jobs
    # For now, let's keep it if we are doing 'all' or 'baselines' (arbitrary choice)
    if args.model_group == 'all':
        print(f"\nAggregating results...")
        aggregate_results(out_root, task=task)

if __name__ == "__main__":
    main()
