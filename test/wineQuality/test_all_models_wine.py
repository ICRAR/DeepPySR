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
from wine_utils import load_wine_data

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wine_type', type=str, default=None, choices=['red', 'white'])
    parser.add_argument('--vps', type=int, default=None)
    args = parser.parse_args()

    wine_types = [args.wine_type] if args.wine_type else ['red', 'white']
    
    for wine_type in wine_types:
        print(f"\n" + "="*50)
        print(f"Processing {wine_type.upper()} wine quality")
        print("="*50)
        
        out_root = os.path.join(current_dir, f"results_{wine_type}_all")
        # out_root_nocv = os.path.join(current_dir, f"results_{wine_type}_nocv")
        # No feature selection as requested
        os.makedirs(out_root, exist_ok=True)
        # os.makedirs(out_root_nocv, exist_ok=True)
        
        df = load_wine_data(wine_type)
        
        X = df.drop(columns=['quality'])
        y = df['quality']

        task = 'regression'
        print(f"Task: {task}, target range: [{y.min()}, {y.max()}]")
        
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
        # Using stratified CV on 'quality' as requested
        cv_kwargs = {
            'task': task,
            'n_splits': 5,
            'random_state': 42,
            'stratify_by': y, # Stratification on quality
            'feature_selection': False # No feature selection as requested
        }
        
        # 2. DeepPySR Grid Search
        print(f"\nEvaluating DeepPySR...")
        for cfg_name, cfg in deeppysr_configs.items():
            print(f"  Config: {cfg_name}...")
            grid_out = os.path.join(out_root, "deeppysr", f"{cfg_name}_{param_suffix}_grid")

            def deeppysr_factory(co=cfg, gout=grid_out):
                kwargs = pysr_base_kwargs.copy()
                kwargs.update(co)
                provider = kwargs.pop('model_provider', 'pypysr')
                return DeepPySR(
                    **kwargs,
                    max_layers=1,
                    output_dir=gout,
                    # model_provider=provider,
                    pareto_r2_weight=r2w_list,
                    pareto_lambda=lambda_list,
                    stopping_score = 0.01,
                )

            if os.path.exists(os.path.join(grid_out, "overall_metrics.csv")):
                print(f"    Skipping grid (results exist)")
            else:
                run_cv(deeppysr_factory, X, y, outdir=grid_out, scaler=False, **cv_kwargs)
            
        # Final Aggregation
        print(f"\nAggregating results for {wine_type}...")
        aggregate_results(out_root, task=task)

if __name__ == "__main__":
    main()
