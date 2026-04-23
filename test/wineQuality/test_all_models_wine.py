import os
import sys
import numpy as np
import pandas as pd
from DeepPySR.regressor import DeepPySRRegressor

# Add parent directory to sys.path to import from test/
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from sklearn.base import clone
from model_utils import get_deeppysr_configs, get_pysr_configs, get_baseline_models, get_pysr_base_kwargs, KANWrapper
from eval_utils import run_cv, run_nocv, aggregate_results
from wine_utils import load_wine_data

def main():
    wine_types = ['red', 'white']
    
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
        
        # Prepare nocv arguments
        nocv_kwargs = {
            'task': task,
            'random_state': 42,
            'feature_selection': False
        }
        # 1. Baseline Models
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

            # # Baseline No CV
            # model_out_nocv = os.path.join(out_root_nocv, "baselines", name)
            # if os.path.exists(os.path.join(model_out_nocv, "overall_metrics.csv")):
            #     print(f"  Skipping {name} nocv (results exist)")
            # else:
            #     print(f"  {name} nocv...")
            #     run_nocv(baseline_factory, X, y, outdir=model_out_nocv, **nocv_kwargs)

        # 2. DeepPySR Grid Search
        print(f"\nEvaluating DeepPySR...")
        for cfg_name, cfg in deeppysr_configs.items():
            print(f"  Config: {cfg_name}...")
            grid_out = os.path.join(out_root, "deeppysr", f"{cfg_name}_{param_suffix}_grid_warm")

            def deeppysr_factory(co=cfg, gout=grid_out):
                kwargs = pysr_base_kwargs.copy()
                kwargs.update(co)
                provider = kwargs.pop('model_provider', 'pypysr')
                return DeepPySRRegressor(
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
            
            # # DeepPySR No CV
            # grid_out_nocv = os.path.join(out_root_nocv, "deeppysr", f"{cfg_name}_{param_suffix}_grid")
            # def deeppysr_factory_nocv(co=cfg, gout=grid_out_nocv):
            #     kwargs = pysr_base_kwargs.copy()
            #     kwargs.update(co)
            #     provider = kwargs.pop('model_provider', 'pypysr')
            #     return DeepPySRRegressor(
            #         **kwargs,
            #         max_layers=1,
            #         output_dir=gout,
            #         model_provider=provider,
            #         pareto_r2_weight=r2w_list,
            #         pareto_lambda=lambda_list,
            #         stopping_score = 0.01,
            #     )
            #
            # if os.path.exists(os.path.join(grid_out_nocv, "overall_metrics.csv")):
            #     print(f"    Skipping grid nocv (results exist)")
            # else:
            #     run_nocv(deeppysr_factory_nocv, X, y, outdir=grid_out_nocv, scaler=False, **nocv_kwargs)

        # 3. PySR Comparison
        print(f"\nEvaluating PySR Comparison...")
        for cfg_name, cfg in pysr_configs.items():
            print(f"  Config: {cfg_name}...")
            pysr_out = os.path.join(out_root, "pysr", f"{cfg_name}_{param_suffix}")

            def deeppysr_pysr_factory(co=cfg):
                return DeepPySRRegressor(
                    **pysr_base_kwargs,
                    **co,
                    max_layers=1,
                    output_dir=pysr_out,
                    # model_provider='pysr',
                    pareto_r2_weight=r2w_list,
                    pareto_lambda=lambda_list,
                    stopping_score = 0.01,
                )


            if os.path.exists(os.path.join(pysr_out, "overall_metrics.csv")):
                print(f"    Skipping (results exist)")
            else:
                run_cv(deeppysr_pysr_factory, X, y, outdir=pysr_out, scaler=False, **cv_kwargs)

            # # PySR No CV
            # pysr_out_nocv = os.path.join(out_root_nocv, "pysr", f"{cfg_name}_{param_suffix}")
            # if os.path.exists(os.path.join(pysr_out_nocv, "overall_metrics.csv")):
            #     print(f"    Skipping nocv (results exist)")
            # else:
            #     run_nocv(deeppysr_pysr_factory, X, y, outdir=pysr_out_nocv, scaler=False, **nocv_kwargs)
        # Final Aggregation
        print(f"\nAggregating results for {wine_type}...")
        aggregate_results(out_root, task=task)
        # aggregate_results(out_root_nocv, task=task)

if __name__ == "__main__":
    main()
