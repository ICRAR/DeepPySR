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
from bodyfat_utils import load_bodyfat_data
import argparse

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    out_root = os.path.join(current_dir, 'results_bodyfat_all')
    os.makedirs(out_root, exist_ok=True)

    print('\n' + '='*50)
    print('Processing BodyFat dataset (Baselines & PySR)')
    print('='*50)

    X, y = load_bodyfat_data()
    task = 'regression'
    
    # Bin the Age feature into 5 bins by percentile for stratified CV
    age_bins = pd.qcut(X['Age'], q=5, labels=False, duplicates='drop')

    pysr_configs = get_pysr_configs()
    pysr_base_kwargs = get_pysr_base_kwargs()

    nit = pysr_base_kwargs.get('niterations', 100)
    pop = pysr_base_kwargs.get('populations', 30)
    sz = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f'nit{nit}_pop{pop}_sz{sz}'

    cv_kwargs = {
        'task': task,
        'n_splits': 5,
        'random_state': 42,
        'feature_selection': False,
        'stratify_by': age_bins,
    }

    print('Evaluating Baseline Models...')
    baseline_models = get_baseline_models(task=task, input_dim=X.shape[1])
    for name, model_instance in baseline_models.items():
        def baseline_factory(m=model_instance, n=name):
            if n == 'KAN':
                return KANWrapper(input_dim=X.shape[1], output_dim=1, hidden_dim=5, steps=200, update_grid=False, task=task)
            return clone(m)

        model_out = os.path.join(out_root, 'baselines', name)
        if os.path.exists(os.path.join(model_out, 'overall_metrics.csv')):
            print(f'  Skipping {name} (results exist)')
        else:
            print(f'  {name}...')
            run_cv(baseline_factory, X, y, outdir=model_out, **cv_kwargs)

    print('Evaluating PySR Comparison...')
    for cfg_name, cfg in pysr_configs.items():
        print(f'  Config: {cfg_name}...')
        pysr_out = os.path.join(out_root, 'pysr', f'{cfg_name}_{param_suffix}')

        def pysr_factory(co=cfg):
            return PySRRegressor(
                **pysr_base_kwargs,
                **co,
            )

        if os.path.exists(os.path.join(pysr_out, 'overall_metrics.csv')):
            print('    Skipping (results exist)')
        else:
            run_cv(pysr_factory, X, y, outdir=pysr_out, scaler=False, **cv_kwargs)

    print('Aggregating results...')
    aggregate_results(out_root, task=task)

if __name__ == '__main__':
    main()
