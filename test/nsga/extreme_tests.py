import os
import sys
import time
import json
import gc
import numpy as np
import pandas as pd
import sympy

# Add parent directory to sys.path to import from test/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'test'))

from model_utils import get_pysr_base_kwargs
from DeepPySR.regressor import DeepPySRRegressor

def generate_extreme_data(n_samples=500, noise=0.01, random_state=42):
    np.random.seed(random_state)
    X = np.random.uniform(-5, 5, (n_samples, 5))
    
    # Ground truth: y = 2.5 * sin(x0) + 3.14 * x1**2 + 0.5 * exp(x2) + x3 - 7.2
    # We use some constants that are not integers to test constant discovery accuracy.
    y = 2.5 * np.sin(X[:, 0]) + 3.14 * (X[:, 1]**2) + 0.5 * np.exp(X[:, 2]) + X[:, 3] - 7.2
    
    if noise > 0:
        y += noise * np.random.normal(size=y.shape)
        
    return X, y, "2.5 * sin(x0) + 3.14 * x1**2 + 0.5 * exp(x2) + x3 - 7.2"


def generate_nsga2_sch(n_samples=500, noise=0, random_state=42):
    """Schaffer's study (SCH) - f1(x) = x^2, f2(x) = (x-2)^2"""
    np.random.seed(random_state)
    X = np.random.uniform(-10, 10, (n_samples, 1))
    y1 = X[:, 0]**2
    y2 = (X[:, 0] - 2)**2
    # Multiple targets
    y = np.column_stack([y1, y2])
    return X, y, ["x0**2", "(x0 - 2)**2"]

def generate_nsga2_fon(n_samples=500, noise=0, random_state=42):
    """Fonseca and Fleming's study (FON)"""
    np.random.seed(random_state)
    n = 3
    X = np.random.uniform(-4, 4, (n_samples, n))
    s1 = np.sum((X - 1/np.sqrt(n))**2, axis=1)
    s2 = np.sum((X + 1/np.sqrt(n))**2, axis=1)
    y1 = 1 - np.exp(-s1)
    y2 = 1 - np.exp(-s2)
    y = np.column_stack([y1, y2])
    return X, y, ["1 - exp(-sum((xi - 1/sqrt(3))**2))", "1 - exp(-sum((xi + 1/sqrt(3))**2))"]

def generate_nsga2_pol(n_samples=500, noise=0, random_state=42):
    """Poloni's study (POL)"""
    np.random.seed(random_state)
    X = np.random.uniform(-np.pi, np.pi, (n_samples, 2))
    x1, x2 = X[:, 0], X[:, 1]
    A1 = 0.5*np.sin(1) - 2*np.cos(1) + np.sin(2) - 1.5*np.cos(2)
    A2 = 1.5*np.sin(1) - np.cos(1) + 2*np.sin(2) - 0.5*np.cos(2)
    B1 = 0.5*np.sin(x1) - 2*np.cos(x1) + np.sin(x2) - 1.5*np.cos(x2)
    B2 = 1.5*np.sin(x1) - np.cos(x1) + 2*np.sin(x2) - 0.5*np.cos(x2)
    y1 = 1 + (A1 - B1)**2 + (A2 - B2)**2
    y2 = (x1 + 3)**2 + (x2 + 1)**2
    y = np.column_stack([y1, y2])
    return X, y, ["1 + (A1 - B1)**2 + (A2 - B2)**2", "(x1 + 3)**2 + (x2 + 1)**2"]

def generate_nsga2_kur(n_samples=500, noise=0, random_state=42):
    """Kursawe's study (KUR)"""
    np.random.seed(random_state)
    X = np.random.uniform(-5, 5, (n_samples, 3))
    y1 = np.sum(-10 * np.exp(-0.2 * np.sqrt(X[:, :-1]**2 + X[:, 1:]**2)), axis=1)
    y2 = np.sum(np.abs(X)**0.8 + 5 * np.sin(X**3), axis=1)
    y = np.column_stack([y1, y2])
    return X, y, ["sum(-10 * exp(-0.2 * sqrt(xi**2 + xi+1**2)))", "sum(abs(xi)**0.8 + 5*sin(xi**3))"]

def generate_zdt1(n_samples=500, noise=0, random_state=42):
    np.random.seed(random_state)
    n = 30
    X = np.random.uniform(0, 1, (n_samples, n))
    f1 = X[:, 0]
    g = 1 + 9 * np.sum(X[:, 1:], axis=1) / (n - 1)
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    y = np.column_stack([f1, f2])
    return X, y, ["x0", "g * (1 - sqrt(f1/g)) where g = 1 + 9 * mean(x[1:])"]

def generate_zdt2(n_samples=500, noise=0, random_state=42):
    np.random.seed(random_state)
    n = 30
    X = np.random.uniform(0, 1, (n_samples, n))
    f1 = X[:, 0]
    g = 1 + 9 * np.sum(X[:, 1:], axis=1) / (n - 1)
    h = 1 - (f1 / g)**2
    f2 = g * h
    y = np.column_stack([f1, f2])
    return X, y, ["x0", "g * (1 - (f1/g)**2)"]

def generate_zdt3(n_samples=500, noise=0, random_state=42):
    np.random.seed(random_state)
    n = 30
    X = np.random.uniform(0, 1, (n_samples, n))
    f1 = X[:, 0]
    g = 1 + 9 * np.sum(X[:, 1:], axis=1) / (n - 1)
    h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
    f2 = g * h
    y = np.column_stack([f1, f2])
    return X, y, ["x0", "g * (1 - sqrt(f1/g) - (f1/g) * sin(10*pi*f1))"]

def generate_zdt4(n_samples=500, noise=0, random_state=42):
    np.random.seed(random_state)
    n = 10
    X1 = np.random.uniform(0, 1, (n_samples, 1))
    Xrest = np.random.uniform(-5, 5, (n_samples, n-1))
    X = np.hstack([X1, Xrest])
    f1 = X[:, 0]
    g = 1 + 10 * (n - 1) + np.sum(X[:, 1:]**2 - 10 * np.cos(4 * np.pi * X[:, 1:]), axis=1)
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    y = np.column_stack([f1, f2])
    return X, y, ["x1", "g * (1 - sqrt(f1/g)) with Rastrigin-like g"]

def generate_zdt6(n_samples=500, noise=0, random_state=42):
    np.random.seed(random_state)
    n = 10
    X = np.random.uniform(0, 1, (n_samples, n))
    f1 = 1 - np.exp(-4 * X[:, 0]) * (np.sin(6 * np.pi * X[:, 0]))**6
    g = 1 + 9 * (np.sum(X[:, 1:], axis=1) / (n - 1))**0.25
    h = 1 - (f1 / g)**2
    f2 = g * h
    y = np.column_stack([f1, f2])
    return X, y, ["f1(x1)", "g * (1 - (f1/g)**2) with non-uniform density"]


def run_model_directly(name, config, X, y, out_dir):
    """Run a single model directly in the same process and return result."""
    print(f"Starting model {name}...")
    
    # Re-add non-serializable params
    sympy_cond = lambda x, y: sympy.Piecewise((y, x > 0), (0, True))
    if "params" in config:
        config["params"]["extra_sympy_mappings"] = {'cond': sympy_cond}
        # Enable warm start and set iterations to 1 initially if we want to track
        total_iterations = config["params"].get("niterations", 10)
        config["params"]["warm_start"] = True
    else:
        total_iterations = 10

    convergence_history = []
    start_time = time.time()
    
    try:
        model = DeepPySRRegressor(**config["params"])
        model.fit(X, y)
        
        # Get loss history
        if hasattr(model, "loss_history_") and model.loss_history_:
            # Combine loss history from all targets if multiple
            for target_history in model.loss_history_:
                target_name = target_history["target"]
                for i, iter_num in enumerate(target_history["iterations"]):
                    loss_val = target_history["losses"][i]
                    convergence_history.append({
                        "iteration": iter_num,
                        "loss": float(loss_val),
                        "target": target_name
                    })

        end_time = time.time()
        convergence_time = end_time - start_time
        
        # Get best formula(s)
        try:
            sym = model.sympy()
            if isinstance(sym, list):
                best_formula = [str(s) for s in sym]
            else:
                best_formula = str(sym)
        except Exception:
            best_formula = "Error getting formula"
        
        # Final Accuracy (R2 on training data)
        try:
            y_pred = model.predict(X)
            if y.ndim == 1:
                r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            else:
                r2s = []
                for j in range(y.shape[1]):
                    y_target = y[:, j]
                    y_pred_target = y_pred[:, j]
                    var_y = np.sum((y_target - np.mean(y_target))**2)
                    if var_y == 0:
                        r2_val = 1.0 if np.allclose(y_target, y_pred_target) else 0.0
                    else:
                        r2_val = 1 - np.sum((y_target - y_pred_target)**2) / var_y
                    r2_val = np.clip(r2_val, 0.0, 1.0)
                    r2s.append(r2_val)
                r2 = np.mean(r2s)
        except ValueError:
            r2 = 0.0
        
        result = {
            "model": name,
            "time": convergence_time,
            "r2": float(r2),
            "formula": best_formula,
            "convergence_history": convergence_history,
            "status": "success"
        }
    except Exception as e:
        import traceback
        result = {
            "model": name,
            "time": 0,
            "r2": 0,
            "formula": "ERROR: " + str(e),
            "traceback": traceback.format_exc(),
            "convergence_history": [],
            "status": "error"
        }
        print(f"ERROR in {name}: {result.get('formula')}")
        print(result["traceback"])

    # Release the model and trigger garbage collection
    try:
        del model
    except NameError:
        pass
    gc.collect()
    
    # We may also want to clear any Julia related state if possible,
    # but since DeepPySRRegressor internally manages sys.modules for providers,
    # this might be sufficient.
    
    return result

def run_extreme_test():
    out_dir = os.path.join(current_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    
    test_problems = {
        "Extreme": generate_extreme_data,
        "SCH": generate_nsga2_sch,
        "FON": generate_nsga2_fon,
        "POL": generate_nsga2_pol,
        "KUR": generate_nsga2_kur,
        "ZDT1": generate_zdt1,
        "ZDT2": generate_zdt2,
        "ZDT3": generate_zdt3,
        "ZDT4": generate_zdt4,
        "ZDT6": generate_zdt6
    }
    
    # Common parameters
    aps = 1.0
    vps = 25
    vpr = 50
    vpm = 0.7
    
    base_kwargs = get_pysr_base_kwargs()
    base_kwargs['niterations'] = 100
    base_kwargs['populations'] = 10
    base_kwargs['population_size'] = 100
    base_kwargs['verbosity'] = 1
    
    all_results = []
    
    for prob_name, prob_func in test_problems.items():
        print(f"\n{'#'*60}")
        print(f"### PROBLEM: {prob_name}")
        print(f"{'#'*60}")
        
        X, y, gt_formula = prob_func()
        print(f"Ground Truth: {gt_formula}")
        
        configs = {
            # REMEMBER TO RUN STDSR AND FULLSR AND V2FULLSR IN 2 SEPARATE PROCESSES
            "stdsr": {
                "type": "deeppysr",
                "params": {
                    "model_provider": "pysr",
                    "adaptive_parsimony_scaling": aps,
                    **base_kwargs
                }
            },
            # "fullsr": {
            #     "type": "deeppysr",
            #     "params": {
            #         "model_provider": "pypysrdev1",
            #         "adaptive_parsimony_scaling": aps,
            #         "variable_prune_start": vps,
            #         "variable_prune_ramp": vpr,
            #         "variable_prune_max": vpm,
            #         "use_mdl": False,
            #         "use_nsga2": False,
            #         "use_lexicase": False,
            #         "use_hotspot_protection": False,
            #         **base_kwargs
            #     }
            # },
            # "v2fullsr": {
            #     "type": "deeppysr",
            #     "params": {
            #         "model_provider": "pypysrdev1",
            #         "adaptive_parsimony_scaling": aps,
            #         "variable_prune_start": vps,
            #         "variable_prune_ramp": vpr,
            #         "variable_prune_max": vpm,
            #         "use_mdl": True,
            #         "use_nsga2": True,
            #         "use_lexicase": True,
            #         "use_hotspot_protection": True,
            #         **base_kwargs
            #     }
            # }
        }
        
        for name, config in configs.items():
            full_name = f"{prob_name}_{name}"
            print(f"\n--- Testing {full_name} ---")
            
            result = run_model_directly(full_name, config, X, y, out_dir)
            result["problem"] = prob_name
            result["config_name"] = name
            
            if result["status"] == "success":
                print(f"Time: {result['time']:.2f}s | R2: {result['r2']:.4f}")
                print(f"Best Formula: {result['formula']}")
            
            all_results.append(result)
            
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(out_dir, "all_extreme_results1.csv"), index=False)
    

if __name__ == "__main__":
    run_extreme_test()
