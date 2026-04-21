import os
import sys
import numpy as np
import pandas as pd
import time
import shutil

# Add project root and test directories to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'feynman'))
sys.path.append(os.path.join(current_dir, 'heart'))
sys.path.append(os.path.join(current_dir, 'wineQuality'))
sys.path.append(os.path.join(current_dir, 'bmi'))
sys.path.append(os.path.join(current_dir, 'bodyfat'))
sys.path.append(os.path.join(current_dir, 'diabetes'))
sys.path.append(os.path.join(current_dir, 'diabetes130us'))
sys.path.append(os.path.join(current_dir, 'stroke'))

# Import loaders
from model_utils import get_pysr_base_kwargs
from feynman.feynman_utils import load_feynman_data, equations as feynman_equations
from heart.heart_utils import load_heart_cleveland_data
from wineQuality.wine_utils import load_wine_data
from bmi.bmi_utils import load_bmi_agg_data
from bodyfat.bodyfat_utils import load_bodyfat_data
from diabetes.diabetes_utils import load_diabetes_brfss_data
from diabetes130us.diab130_utils import load_and_clean_data as load_diab130_data
from stroke.stroke_utils import load_stroke_data

def get_best_loss(model):
    """Extracts the best loss from a PySRRegressor model's hall of fame."""
    try:
        if hasattr(model, 'equations_'):
            if isinstance(model.equations_, list):
                return model.equations_[0]['loss'].min()
            else:
                return model.equations_['loss'].min()
        return np.inf
    except Exception:
        return np.inf

def train_iteratively(model_provider, X, y, n_iterations, output_dir, pysr_kwargs):
    # Environment switching
    if "pysr" in sys.modules or "pypysr" in sys.modules:
        for mod in list(sys.modules.keys()):
            if mod == 'pysr' or mod.startswith('pysr.') or mod == 'pypysr' or mod.startswith('pypysr.'):
                del sys.modules[mod]

    if model_provider == "deeppysr":
        from pypysr import PySRRegressor
    else:
        if "juliacall" in sys.modules:
            try:
                from juliacall import Main as jl
                import juliapkg
                jl.seval("using Pkg")
                pysr_env = juliapkg.project()
                jl.Pkg.activate(pysr_env)
            except Exception: pass
        from pysr import PySRRegressor

    model_output_dir = os.path.join(output_dir, model_provider)
    os.makedirs(model_output_dir, exist_ok=True)

    model = PySRRegressor(
        **pysr_kwargs,
        niterations=1,
        warm_start=True,
        output_directory=model_output_dir
    )

    history = []
    start_time = time.time()
    for i in range(1, n_iterations + 1):
        try:
            model.fit(X, y)
            loss = get_best_loss(model)
            elapsed = time.time() - start_time
            history.append({'Iteration': i, 'Loss': loss, 'Time': elapsed})
        except Exception as e:
            print(f"      Error in iteration {i}: {e}")
            break
    return pd.DataFrame(history)

def run_cv_convergence(test_name, X, y, task='regression', n_iterations=500):
    print(f"\n{'='*60}\nDataset: {test_name} ({task})\n{'='*60}")
    
    X_values = X.values if hasattr(X, 'values') else X
    y_values = y.values if hasattr(y, 'values') else y

    pysr_kwargs = get_pysr_base_kwargs()
    pysr_kwargs.update({"adaptive_parsimony_scaling":10.0})
    pysr_kwargs.pop("niterations", None)
    deeppysr_kwargs = pysr_kwargs.copy()
    deeppysr_kwargs.update({"variable_prune_start": 25, "variable_prune_ramp": 50, "variable_prune_max": 0.7})

    fold_dir = os.path.join(current_dir, test_name.replace(".", "_"))
    os.makedirs(fold_dir, exist_ok=True)

    # DeepPySR
    dp_hist = train_iteratively("deeppysr", X_values, y_values, n_iterations, fold_dir, deeppysr_kwargs)
    dp_hist['Model'] = 'DeepPySR'

    # PySR
    sr_hist = train_iteratively("pysr", X_values, y_values, n_iterations, fold_dir, pysr_kwargs)
    sr_hist['Model'] = 'PySR'

    combined = pd.concat([dp_hist, sr_hist], ignore_index=True)
    combined.to_csv(os.path.join(fold_dir, "convergence_history.csv"), index=False)
    print(f"    Saved {test_name} history.")

def main():
    # Define test cases
    test_cases = []

    # 1. Feynman (Regression)
    for eq in feynman_equations.keys():
        test_cases.append({'name': f"feynman/results_{eq}_all/convergence", 'loader': lambda e=eq: load_feynman_data(e, n_samples=500), 'task': 'regression'})

    # 2. Heart (Classification)
    test_cases.append({'name': 'heart/results_heart_all/convergence', 'loader': load_heart_cleveland_data, 'task': 'classification'})

    # 3. Wine (Regression)
    test_cases.append({'name': 'wineQuality/results_red_all/convergence', 'loader': lambda: (load_wine_data('red').drop(columns=['quality']), load_wine_data('red')['quality']), 'task': 'regression'})
    test_cases.append({'name': 'wineQuality/results_white_all/convergence', 'loader': lambda: (load_wine_data('white').drop(columns=['quality']), load_wine_data('white')['quality']), 'task': 'regression'})

    # 4. Bodyfat (Regression)
    test_cases.append({'name': 'bodyfat/results_bodyfat_all/convergence', 'loader': load_bodyfat_data, 'task': 'regression'})

    # 5. Stroke (Classification)
    test_cases.append({'name': 'stroke/results_stroke_all/convergence', 'loader': load_stroke_data, 'task': 'classification','nit':100})

    # 6. Diabetes BRFSS (Classification)
    test_cases.append({'name': 'diabetes/results_diabetes_all/convergence', 'loader': lambda: (load_diabetes_brfss_data()[0], load_diabetes_brfss_data()[1]), 'task': 'classification'})

    # 7. Diabetes 130US (Classification)
    test_cases.append({'name': 'diabetes130us/results_diabetes130us_all/convergence', 'loader': lambda: (load_diab130_data()[1], load_diab130_data()[2]), 'task': 'classification','nit':100})

    # 8. BMI (Regression)
    test_cases.append({'name': 'bmi/results_bmi_all/longitudinal/convergence', 'loader': lambda: (load_bmi_agg_data()[1], load_bmi_agg_data()[2]), 'task': 'regression'})
    for age in [8, 10, 13, 16, 20, 23, 26]:
        test_cases.append({'name': f"bmi/results_bmi_all/age_specific/age_{age}/convergence", 'loader': lambda a=age: (load_bmi_agg_data(age)[1], load_bmi_agg_data(age)[2]), 'task': 'regression'})

    for tc in test_cases:
        try:
            X, y = tc['loader']()
            n_iterations = tc.get('nit',500)
            run_cv_convergence(tc['name'], X, y, task=tc['task'], n_iterations=n_iterations)
        except Exception as e:
            print(f"Failed {tc['name']}: {e}")

if __name__ == "__main__":
    main()
