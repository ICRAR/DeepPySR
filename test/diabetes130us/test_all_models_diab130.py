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
from diab130_utils import load_and_clean_data

def main():
    out_root = os.path.join(current_dir, "results_diab130_all")
    out_root_nocv = os.path.join(current_dir, "results_diab130_nocv")
    out_root_ftsl = os.path.join(out_root, "ftsl")
    out_root_noftsl = os.path.join(out_root, "noftsl")
    out_root_nocv_ftsl = os.path.join(out_root_nocv, "ftsl")
    out_root_nocv_noftsl = os.path.join(out_root_nocv, "noftsl")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(out_root_nocv, exist_ok=True)
    os.makedirs(out_root_ftsl, exist_ok=True)
    os.makedirs(out_root_noftsl, exist_ok=True)
    os.makedirs(out_root_nocv_ftsl, exist_ok=True)
    os.makedirs(out_root_nocv_noftsl, exist_ok=True)
    
    file_path = '/home/00101787/Projects/DeepPySR/test_data/Health/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv'
    df = load_and_clean_data(file_path)
    
    # Extract encounter_id as ids for tracking
    ids = df['encounter_id'] if 'encounter_id' in df.columns else None
    
    # Drop IDs and target from features
    cols_to_drop = ['encounter_id', 'patient_nbr', 'readmitted']
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = df['readmitted']
    
    # Extra data to save: target as well
    extra_data = {'readmitted': y}
    
    # For classification decision: 
    # The target 'readmitted' has values 0, 1. 
    # Determine task based on target uniqueness
    unique_y = np.unique(y)
    if len(unique_y) <= 10 and np.all(np.equal(unique_y, unique_y.astype(int))):
        task = 'classification'
    else:
        task = 'regression'
    
    print(f"Detected task: {task} (unique values: {len(unique_y)})")
    
    # DeepPySR grid search params
    r2w_list = [1, 1.5, 2]
    lambda_list = [0.001, 0.005, 0.01]
    
    n_features_to_select = 20
    
    deeppysr_configs = get_deeppysr_configs()
    pysr_configs = get_pysr_configs()
    pysr_base_kwargs = get_pysr_base_kwargs()
    
    # Extract parameters for folder naming
    nit = pysr_base_kwargs.get('niterations', 10)
    pop = pysr_base_kwargs.get('populations', 30)
    sz = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f"nit{nit}_pop{pop}_sz{sz}"
    
    print(f"\n--- Run: diabetes130 ---")
    
    # Prepare cv arguments
    cv_kwargs = {
        'task': task,
        'n_splits': 5,
        'random_state': 42,
        'stratify_by': y,
        'ids': ids,
        'extra_data': extra_data,
        'use_smote': True
    }
    
    cv_ftsl_kwargs = cv_kwargs.copy()
    cv_ftsl_kwargs.update({'feature_selection': True, 'n_features_to_select': n_features_to_select})

    # 1. Baseline Models
    print(f"Evaluating Baseline Models...")
    baseline_models = get_baseline_models(task=task, input_dim=X.shape[1])
    for name, model_instance in baseline_models.items():
        model_out = os.path.join(out_root_noftsl, "baselines", name)
        if os.path.exists(os.path.join(model_out, "overall_metrics.csv")):
            print(f"  Skipping {name} (results exist)")
        else:
            print(f"  {name}...")
            
            def baseline_factory(m=model_instance, n=name):
                if n == 'KAN':
                    return KANWrapper(input_dim=X.shape[1], output_dim=1, hidden_dim=5, steps=200, update_grid=False, task=task)
                return clone(m)
            
            run_cv(baseline_factory, X, y, outdir=model_out, **cv_kwargs)

        # Baseline with Feature Selection
        model_out_ftsl = os.path.join(out_root_ftsl, "baselines", name)
        if os.path.exists(os.path.join(model_out_ftsl, "overall_metrics.csv")):
            print(f"  Skipping {name} ftsl (results exist)")
        else:
            print(f"  {name} with ftsl...")
            def baseline_factory_ftsl(m=model_instance, n=name):
                if n == 'KAN':
                    return KANWrapper(input_dim=n_features_to_select, output_dim=1, hidden_dim=5, steps=200, update_grid=False, task=task)
                return clone(m)
            run_cv(baseline_factory_ftsl, X, y, outdir=model_out_ftsl, **cv_ftsl_kwargs)

    # 2. DeepPySR Models (pypysr - Grid Search)
    print(f"Evaluating DeepPySR (pypysr) Models...")
    for cfg_name, cfg_overrides in deeppysr_configs.items():
        full_cfg_name = f"{cfg_name}_{param_suffix}_grid"
        deeppysr_out = os.path.join(out_root_noftsl, "deeppysr", full_cfg_name)
        if os.path.exists(os.path.join(deeppysr_out, "overall_metrics.csv")):
            continue
        
        print(f"  {full_cfg_name}...")
        def deeppysr_factory():
            kwargs = pysr_base_kwargs.copy()
            kwargs.update(cfg_overrides)
            # Default to pypysr if model_provider is not in the config
            provider = kwargs.pop('model_provider', 'pypysr')
            return DeepPySRRegressor(
                max_layers=1,
                output_dir=deeppysr_out,
                model_provider=provider,
                pareto_r2_weight=r2w_list,
                pareto_lambda=lambda_list,
                stopping_score = 0.01,
                **kwargs
            )
        
        run_cv(deeppysr_factory, X, y, outdir=deeppysr_out, scaler=False, **cv_kwargs)

    # 3. DeepPySR (pysr) Model
    print(f"Evaluating DeepPySR (pysr) Model...")
    for cfg_name, cfg_overrides in pysr_configs.items():
        aps = cfg_overrides.get("adaptive_parsimony_scaling", 50.0)
        full_name = f"pysr_{param_suffix}_aps{aps}_grid"
        deeppysr_pysr_out = os.path.join(out_root_noftsl, "pysr", full_name)
        if os.path.exists(os.path.join(deeppysr_pysr_out, "overall_metrics.csv")):
            continue
        
        print(f"  {full_name}...")
        def deeppysr_pysr_factory(co=cfg_overrides):
            kwargs = pysr_base_kwargs.copy()
            kwargs.update(co)
            return DeepPySRRegressor(
                max_layers=1,
                output_dir=deeppysr_pysr_out,
                model_provider='pysr',
                pareto_r2_weight=r2w_list,
                pareto_lambda=lambda_list,
                stopping_score = 0.01,
                **kwargs
            )
        
        run_cv(deeppysr_pysr_factory, X, y, outdir=deeppysr_pysr_out, scaler=False, **cv_kwargs)

    # 4. No-CV runs
    print(f"Evaluating No-CV Models...")
    nocv_kwargs = {
        'task': task,
        'ids': ids,
        'extra_data': extra_data,
        'use_smote': True
    }
    
    nocv_ftsl_kwargs = nocv_kwargs.copy()
    nocv_ftsl_kwargs.update({'feature_selection': True, 'n_features_to_select': n_features_to_select})

    # 4.1 DeepPySR No-CV
    for cfg_name, cfg_overrides in deeppysr_configs.items():
        full_cfg_name = f"{cfg_name}_{param_suffix}_grid_nocv"
        deeppysr_out = os.path.join(out_root_nocv_noftsl, "deeppysr", full_cfg_name)
        if not os.path.exists(os.path.join(deeppysr_out, "overall_metrics.csv")):
            print(f"  {full_cfg_name}...")
            def deeppysr_factory_nocv(co=cfg_overrides, d_out=deeppysr_out):
                kwargs = pysr_base_kwargs.copy()
                kwargs.update(co)
                # Default to pypysr if model_provider is not in the config
                provider = kwargs.pop('model_provider', 'pypysr')
                return DeepPySRRegressor(
                    max_layers=1,
                    output_dir=d_out,
                    model_provider=provider,
                    pareto_r2_weight=r2w_list,
                    pareto_lambda=lambda_list,
                    stopping_score = 0.01,
                    **kwargs
                )
            run_nocv(deeppysr_factory_nocv, X, y, outdir=deeppysr_out, scaler=False, **nocv_kwargs)

    # 4.2 DeepPySR (pysr) No-CV
    for cfg_name, cfg_overrides in pysr_configs.items():
        aps = cfg_overrides.get("adaptive_parsimony_scaling", 50.0)
        full_name = f"pysr_{param_suffix}_aps{aps}_grid_nocv"
        deeppysr_pysr_out = os.path.join(out_root_nocv_noftsl, "pysr", full_name)
        if not os.path.exists(os.path.join(deeppysr_pysr_out, "overall_metrics.csv")):
            print(f"  {full_name}...")
            def deeppysr_pysr_factory_nocv(co=cfg_overrides, d_out=deeppysr_pysr_out):
                kwargs = pysr_base_kwargs.copy()
                kwargs.update(co)
                return DeepPySRRegressor(
                    max_layers=1,
                    output_dir=d_out,
                    model_provider='pysr',
                    pareto_r2_weight=r2w_list,
                    pareto_lambda=lambda_list,
                    stopping_score = 0.01,
                    **kwargs
                )
            run_nocv(deeppysr_pysr_factory_nocv, X, y, outdir=deeppysr_pysr_out, scaler=False, **nocv_kwargs)

    # 5. Feature Selection runs
    print(f"Evaluating Models with Feature Selection...")
    
    # 5.1 DeepPySR (pypysr) with Feature Selection
    for cfg_name, cfg_overrides in deeppysr_configs.items():
        full_cfg_name = f"{cfg_name}_{param_suffix}_ftsl"
        deeppysr_out = os.path.join(out_root_ftsl, "deeppysr", full_cfg_name)
        if not os.path.exists(os.path.join(deeppysr_out, "overall_metrics.csv")):
            print(f"  {full_cfg_name}...")
            def deeppysr_factory_ftsl():
                kwargs = pysr_base_kwargs.copy()
                kwargs.update(cfg_overrides)
                # Default to pypysr if model_provider is not in the config
                provider = kwargs.pop('model_provider', 'pypysr')
                return DeepPySRRegressor(
                    max_layers=1,
                    output_dir=deeppysr_out,
                    model_provider=provider,
                    pareto_r2_weight=r2w_list,
                    pareto_lambda=lambda_list,
                    stopping_score = 0.01,
                    **kwargs
                )
            run_cv(deeppysr_factory_ftsl, X, y, outdir=deeppysr_out, scaler=False, **cv_ftsl_kwargs)

    # 5.2 DeepPySR (pypysr) No-CV with Feature Selection
    for cfg_name, cfg_overrides in deeppysr_configs.items():
        full_cfg_name = f"{cfg_name}_{param_suffix}_ftsl_nocv"
        deeppysr_out = os.path.join(out_root_nocv_ftsl, "deeppysr", full_cfg_name)
        if not os.path.exists(os.path.join(deeppysr_out, "overall_metrics.csv")):
            print(f"  {full_cfg_name}...")
            def deeppysr_factory_ftsl_nocv(co=cfg_overrides, d_out=deeppysr_out):
                kwargs = pysr_base_kwargs.copy()
                kwargs.update(co)
                # Default to pypysr if model_provider is not in the config
                provider = kwargs.pop('model_provider', 'pypysr')
                return DeepPySRRegressor(
                    max_layers=1,
                    output_dir=d_out,
                    model_provider=provider,
                    pareto_r2_weight=r2w_list,
                    pareto_lambda=lambda_list,
                    stopping_score = 0.01,
                    **kwargs
                )
            run_nocv(deeppysr_factory_ftsl_nocv, X, y, outdir=deeppysr_out, scaler=False, **nocv_ftsl_kwargs)

    # 5.3 DeepPySR (pysr) with Feature Selection
    for cfg_name, cfg_overrides in pysr_configs.items():
        aps = cfg_overrides.get("adaptive_parsimony_scaling", 50.0)
        full_name = f"pysr_{param_suffix}_aps{aps}_ftsl"
        deeppysr_pysr_out = os.path.join(out_root_ftsl, "pysr", full_name)
        if not os.path.exists(os.path.join(deeppysr_pysr_out, "overall_metrics.csv")):
            print(f"  {full_name}...")
            def deeppysr_pysr_factory_ftsl(co=cfg_overrides):
                kwargs = pysr_base_kwargs.copy()
                kwargs.update(co)
                return DeepPySRRegressor(
                    max_layers=1,
                    output_dir=deeppysr_pysr_out,
                    model_provider='pysr',
                    pareto_r2_weight=r2w_list,
                    pareto_lambda=lambda_list,
                    stopping_score = 0.01,
                    **kwargs
                )
            run_cv(deeppysr_pysr_factory_ftsl, X, y, outdir=deeppysr_pysr_out, scaler=False, **cv_ftsl_kwargs)

    # 5.4 DeepPySR (pysr) No-CV with Feature Selection
    for cfg_name, cfg_overrides in pysr_configs.items():
        aps = cfg_overrides.get("adaptive_parsimony_scaling", 50.0)
        full_name = f"pysr_{param_suffix}_aps{aps}_ftsl_nocv"
        deeppysr_pysr_out = os.path.join(out_root_nocv_ftsl, "pysr", full_name)
        if not os.path.exists(os.path.join(deeppysr_pysr_out, "overall_metrics.csv")):
            print(f"  {full_name}...")
            def deeppysr_pysr_factory_ftsl_nocv(co=cfg_overrides, d_out=deeppysr_pysr_out):
                kwargs = pysr_base_kwargs.copy()
                kwargs.update(co)
                return DeepPySRRegressor(
                    max_layers=1,
                    output_dir=d_out,
                    model_provider='pysr',
                    pareto_r2_weight=r2w_list,
                    pareto_lambda=lambda_list,
                    stopping_score = 0.01,
                    **kwargs
                )
            run_nocv(deeppysr_pysr_factory_ftsl_nocv, X, y, outdir=deeppysr_pysr_out, scaler=False, **nocv_ftsl_kwargs)

    # Aggregate results
    print(f"Aggregating results...")
    aggregate_results(out_root_noftsl, task=task)
    aggregate_results(out_root_nocv_noftsl, task=task)
    aggregate_results(out_root_ftsl, task=task)
    aggregate_results(out_root_nocv_ftsl, task=task)

if __name__ == "__main__":
    main()
