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
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(out_root_nocv, exist_ok=True)
    
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
    
    # For classification/regression decision: 
    # The target 'readmitted' has values 0, 1, 2. 
    # DeepPySRRegressor is currently imported as Regressor, but let's see if we should use classification.
    # The issue says "stratify on the target of readmitted", which strongly implies classification.
    task = 'classification'
    
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
    
    print(f"\n--- Run: diabetes130 ---")
    
    # Prepare cv arguments
    cv_kwargs = {
        'task': task,
        'n_splits': 5,
        'random_state': 42,
        'stratify_by': y,
        'ids': ids,
        'extra_data': extra_data
    }

    # 1. Baseline Models
    print(f"Evaluating Baseline Models...")
    baseline_models = get_baseline_models(task=task, input_dim=X.shape[1])
    for name, model_instance in baseline_models.items():
        model_out = os.path.join(out_root, "baselines", name)
        if os.path.exists(os.path.join(model_out, "overall_metrics.csv")):
            print(f"  Skipping {name} (results exist)")
            continue
        print(f"  {name}...")
        
        def baseline_factory(m=model_instance, n=name):
            if n == 'KAN':
                return KANWrapper(input_dim=X.shape[1], output_dim=1, hidden_dim=5, steps=200, update_grid=False, task=task)
            return clone(m)
        
        run_cv(baseline_factory, X, y, outdir=model_out, **cv_kwargs)

    # 2. DeepPySR Models (pypysr - Grid Search)
    print(f"Evaluating DeepPySR (pypysr) Models...")
    for cfg_name, cfg_overrides in deeppysr_configs.items():
        full_cfg_name = f"{cfg_name}_{param_suffix}_grid"
        deeppysr_out = os.path.join(out_root, "deeppysr", full_cfg_name)
        if os.path.exists(os.path.join(deeppysr_out, "overall_metrics.csv")):
            continue
        
        print(f"  {full_cfg_name}...")
        def deeppysr_factory():
            kwargs = pysr_base_kwargs.copy()
            kwargs.update(cfg_overrides)
            return DeepPySRRegressor(
                max_layers=1,
                output_dir=deeppysr_out,
                model_provider='pypysr',
                pareto_r2_weight=r2w_list,
                pareto_lambda=lambda_list,
                **kwargs
            )
        
        run_cv(deeppysr_factory, X, y, outdir=deeppysr_out, scaler=False, **cv_kwargs)

    # 3. DeepPySR (pysr) Model
    print(f"Evaluating DeepPySR (pysr) Model...")
    for cfg_name, cfg_overrides in pysr_configs.items():
        aps = cfg_overrides.get("adaptive_parsimony_scaling", 50.0)
        full_name = f"pysr_{param_suffix}_aps{aps}_grid"
        deeppysr_pysr_out = os.path.join(out_root, "pysr", full_name)
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
                **kwargs
            )
        
        run_cv(deeppysr_pysr_factory, X, y, outdir=deeppysr_pysr_out, scaler=False, **cv_kwargs)

    # 4. No-CV runs
    print(f"Evaluating No-CV Models...")
    nocv_kwargs = {
        'task': task,
        'ids': ids,
        'extra_data': extra_data
    }

    # 4.1 DeepPySR No-CV
    for cfg_name, cfg_overrides in deeppysr_configs.items():
        full_cfg_name = f"{cfg_name}_{param_suffix}_grid_nocv"
        deeppysr_out = os.path.join(out_root_nocv, "deeppysr", full_cfg_name)
        if not os.path.exists(os.path.join(deeppysr_out, "overall_metrics.csv")):
            print(f"  {full_cfg_name}...")
            def deeppysr_factory_nocv(co=cfg_overrides, d_out=deeppysr_out):
                kwargs = pysr_base_kwargs.copy()
                kwargs.update(co)
                return DeepPySRRegressor(
                    max_layers=1,
                    output_dir=d_out,
                    model_provider='pypysr',
                    pareto_r2_weight=r2w_list,
                    pareto_lambda=lambda_list,
                    **kwargs
                )
            run_nocv(deeppysr_factory_nocv, X, y, outdir=deeppysr_out, scaler=False, **nocv_kwargs)

    # 4.2 DeepPySR (pysr) No-CV
    for cfg_name, cfg_overrides in pysr_configs.items():
        aps = cfg_overrides.get("adaptive_parsimony_scaling", 50.0)
        full_name = f"pysr_{param_suffix}_aps{aps}_grid_nocv"
        deeppysr_pysr_out = os.path.join(out_root_nocv, "pysr", full_name)
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
                    **kwargs
                )
            run_nocv(deeppysr_pysr_factory_nocv, X, y, outdir=deeppysr_pysr_out, scaler=False, **nocv_kwargs)

    # Aggregate results
    print(f"Aggregating results...")
    aggregate_results(out_root, task=task)
    aggregate_results(out_root_nocv, task=task)

if __name__ == "__main__":
    main()
