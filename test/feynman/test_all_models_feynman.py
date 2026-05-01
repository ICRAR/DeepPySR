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
    get_pysr_base_kwargs, KANWrapper
)

from sklearn.base import clone
from eval_utils import run_cv, run_nocv, aggregate_results
from feynman_utils import load_feynman_data, equations

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eq_name', type=str, default=None)
    parser.add_argument('--vps', type=int, default=None)
    args = parser.parse_args()

    eq_names = [args.eq_name] if args.eq_name else list(equations.keys())
    
    for eq_name in eq_names:
        print(f"\n" + "="*50)
        print(f"Processing Feynman equation: {eq_name}")
        print("="*50)
        
        out_root = os.path.join(current_dir, f"results_{eq_name.replace('.', '_')}_all")
        os.makedirs(out_root, exist_ok=True)
        
        X, y = load_feynman_data(eq_name, n_samples=1000)
        
        task = 'regression'
        print(f"Task: {task}, X shape: {X.shape}, y range: [{y.min():.3f}, {y.max():.3f}]")
        
        # DeepPySR grid search params
        r2w_list = [1, 1.5, 2]
        lambda_list = [0.001, 0.005, 0.01]
        
        deeppysr_configs = get_deeppysr_configs()
        if args.vps is not None:
            deeppysr_configs = {k: v for k, v in deeppysr_configs.items() if f"vps{args.vps}_" in k}

        pysr_configs = get_pysr_configs()
        pysr_base_kwargs = get_pysr_base_kwargs()
        
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
            'stratify_by': None,  # Regression, no stratification
            'feature_selection': False
        }
        
        # 2. DeepPySR Grid Search
        print("Evaluating DeepPySR...")
        for cfg_name, cfg in deeppysr_configs.items():
            print(f"  Config: {cfg_name}...")
            grid_out = os.path.join(out_root, "deeppysr", f"{cfg_name}_{param_suffix}_grid")

            def deeppysr_factory(co=cfg, gout=grid_out):
                kwargs = pysr_base_kwargs.copy()
                kwargs.update(co)
                return DeepPySR(
                    **kwargs,
                    max_layers=1,
                    output_dir=gout,
                    pareto_r2_weight=r2w_list,
                    pareto_lambda=lambda_list,
                    stopping_score=0.01,
                )

            if os.path.exists(os.path.join(grid_out, "overall_metrics.csv")):
                print("    Skipping grid (results exist)")
            else:
                run_cv(deeppysr_factory, X, y, outdir=grid_out, scaler=False, **cv_kwargs)
        
        # Final Aggregation
        print(f"Aggregating results for {eq_name}...")
        aggregate_results(out_root, task=task)

if __name__ == "__main__":
    main()