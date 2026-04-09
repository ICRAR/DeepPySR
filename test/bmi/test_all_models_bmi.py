import os
import sys
import numpy as np
import pandas as pd
from DeepPySR.regressor import DeepPySRRegressor

# Add parent directory to sys.path to import from test/
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from sklearn.base import clone
from model_utils import get_deeppysr_configs, get_baseline_models, get_pysr_base_kwargs, KANWrapper
from eval_utils import run_cv, run_nocv, aggregate_results
from bmi_utils import load_bmi_agg_data

def main():
    out_root = os.path.join(current_dir, "results_bmi_all")
    # out_root_nocv = os.path.join(current_dir, "results_bmi_nocv")
    os.makedirs(out_root, exist_ok=True)
    # os.makedirs(out_root_nocv, exist_ok=True)
    
    settings = ['longitudinal', 'age_specific']
    ages = [8, 10, 14, 17, 20, 23, 27]
    
    # DeepPySR grid search params
    r2w_list = [1, 1.5, 2]
    lambda_list = [0.001, 0.005, 0.01]
    
    deeppysr_configs = get_deeppysr_configs()
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
            # Use flattened root for longitudinal
            run_out_root_base = out_root
            # run_out_root_nocv_base = out_root_nocv
        else:
            runs = []
            for age in ages:
                ids, X, y = load_bmi_agg_data(age=age)
                if not X.empty:
                    runs.append((f"age_{age}", ids, X, y))
            run_out_root_base = os.path.join(out_root, setting)
            # run_out_root_nocv_base = os.path.join(out_root_nocv, setting)
        
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
                        # Explicitly create a new KANWrapper for each fold
                        return KANWrapper(input_dim=X.shape[1], output_dim=1, hidden_dim=5, steps=200, update_grid=False, task='regression')
                    # Clone standard sklearn models
                    return clone(m)
                
                run_cv(baseline_factory, X, y, outdir=model_out, **cv_kwargs)

            # 2. DeepPySR Models (pypysr - Grid Search)
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
                    
                    # If model_provider is not in cfg_overrides, default to 'pypysr'
                    provider = cfg_overrides.get('model_provider', 'pypysr')
                    
                    return DeepPySRRegressor(
                        max_layers=1,
                        output_dir=deeppysr_out,
                        model_provider=provider,
                        pareto_r2_weight=r2w_list,
                        pareto_lambda=lambda_list,
                        **kwargs
                    )
                
                run_cv(deeppysr_factory, X, y, outdir=deeppysr_out, scaler=False, **cv_kwargs)

            # # 3. No-CV runs for DeepPySR (3 settings)
            # print(f"Evaluating No-CV Models...")
            # run_out_root_nocv = os.path.join(run_out_root_nocv_base, run_name)
            # os.makedirs(run_out_root_nocv, exist_ok=True)
            #
            # nocv_kwargs = {
            #     'ids': ids,
            #     'task': 'regression',
            #     'extra_data': X[['age']] if 'age' in X.columns else None
            # }
            #
            # # 3.1 DeepPySR No-CV
            # for cfg_name, cfg_overrides in deeppysr_configs.items():
            #     full_cfg_name = f"{cfg_name}_{param_suffix}_grid_nocv"
            #     deeppysr_out = os.path.join(run_out_root_nocv, "deeppysr", full_cfg_name)
            #     if not os.path.exists(os.path.join(deeppysr_out, "overall_metrics.csv")):
            #         print(f"  {full_cfg_name}...")
            #         def deeppysr_factory_nocv(co=cfg_overrides, d_out=deeppysr_out):
            #             kwargs = pysr_base_kwargs.copy()
            #             kwargs.update(co)
            #
            #             # If model_provider is not in co, default to 'pypysr'
            #             provider = co.get('model_provider', 'pypysr')
            #
            #             return DeepPySRRegressor(
            #                 max_layers=1,
            #                 output_dir=d_out,
            #                 model_provider=provider,
            #                 pareto_r2_weight=r2w_list,
            #                 pareto_lambda=lambda_list,
            #                 **kwargs
            #             )
            #         run_nocv(deeppysr_factory_nocv, X, y, outdir=deeppysr_out, scaler=False, **nocv_kwargs)
            #
            # Aggregate results for this run
            print(f"Aggregating results for {run_name}...")
            aggregate_results(run_out_root, task='regression')
            # aggregate_results(run_out_root_nocv, task='regression')

    # # Finally, aggregate EVERYTHING across settings
    print(f"\nAggregating all results across all settings...")
    aggregate_results(out_root, task='regression')
    # aggregate_results(out_root_nocv, task='regression')

if __name__ == "__main__":
    main()
