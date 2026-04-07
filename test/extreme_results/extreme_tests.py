import os
import sys
import time
import json
import subprocess
import numpy as np
import pandas as pd
import sympy

# Add parent directory to sys.path to import from test/
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..')))

from model_utils import get_pysr_base_kwargs

def generate_extreme_data(n_samples=500, noise=0.01, random_state=42):
    np.random.seed(random_state)
    X = np.random.uniform(-5, 5, (n_samples, 5))
    
    # Ground truth: y = 2.5 * sin(x0) + 3.14 * x1**2 + 0.5 * exp(x2) + x3 - 7.2
    # We use some constants that are not integers to test constant discovery accuracy.
    y = 2.5 * np.sin(X[:, 0]) + 3.14 * (X[:, 1]**2) + 0.5 * np.exp(X[:, 2]) + X[:, 3] - 7.2
    
    if noise > 0:
        y += noise * np.random.normal(size=y.shape)
        
    return X, y, "2.5 * sin(x0) + 3.14 * x1**2 + 0.5 * exp(x2) + x3 - 7.2"


def run_model_subprocess(name, config, X, y, out_dir):
    """Run a single model in a separate process to avoid Julia environment conflicts."""
    # Save data to temporary files for the subprocess
    X_path = os.path.join(out_dir, f"X_{name}.npy")
    y_path = os.path.join(out_dir, f"y_{name}.npy")
    config_path = os.path.join(out_dir, f"config_{name}.json")
    
    np.save(X_path, X)
    np.save(y_path, y)
    
    # Filter out non-serializable parts of the config
    serializable_config = config.copy()
    if "params" in serializable_config:
        params = serializable_config["params"].copy()
        # Remove extra_sympy_mappings as it contains lambdas/functions
        params.pop("extra_sympy_mappings", None)
        serializable_config["params"] = params
        
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f)
        
    python_script = f"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import sympy

# Setup path
project_root = "{os.path.abspath(os.path.join(current_dir, '../..'))}"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'test'))

# Load data
X = np.load("{X_path}")
y = np.load("{y_path}")
with open("{config_path}", 'r') as f:
    config = json.load(f)

# Re-add non-serializable params
sympy_cond = lambda x, y: sympy.Piecewise((y, x > 0), (0, True))
config["params"]["extra_sympy_mappings"] = {{'cond': sympy_cond}}

start_time = time.time()
try:
    if config["type"] == "pysr":
        from pysr import PySRRegressor
        model = PySRRegressor(**config["params"])
        model.fit(X, y)
    else:
        from DeepPySR.regressor import DeepPySRRegressor
        model = DeepPySRRegressor(**config["params"])
        model.fit(X, y)
        
    end_time = time.time()
    convergence_time = end_time - start_time
    
    # Get best formula
    best_formula = str(model.sympy())
    
    # Simple Accuracy (R2 on training data)
    y_pred = model.predict(X)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    
    result = {{
        "model": "{name}",
        "time": convergence_time,
        "r2": float(r2),
        "formula": best_formula,
        "status": "success"
    }}
except Exception as e:
    import traceback
    result = {{
        "model": "{name}",
        "time": 0,
        "r2": 0,
        "formula": "ERROR: " + str(e),
        "traceback": traceback.format_exc(),
        "status": "error"
    }}

# Output result as JSON
print("JSON_RESULT_START")
print(json.dumps(result))
print("JSON_RESULT_END")
"""
    
    # Execute subprocess
    print(f"Starting subprocess for {name}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.path.abspath(os.path.join(current_dir, '../..'))}:{os.path.abspath(current_dir)}"
    
    process = subprocess.Popen(
        [sys.executable, "-c", python_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    stdout, stderr = process.communicate()
    
    # Parse result
    result = None
    if "JSON_RESULT_START" in stdout:
        try:
            json_part = stdout.split("JSON_RESULT_START")[1].split("JSON_RESULT_END")[0].strip()
            result = json.loads(json_part)
        except Exception as e:
            print(f"Error parsing JSON for {name}: {e}")
            
    if result is None:
        result = {
            "model": name,
            "time": 0,
            "r2": 0,
            "formula": "ERROR: Subprocess failed",
            "stderr": stderr,
            "stdout": stdout,
            "status": "error"
        }
        
    if result["status"] == "error":
        print(f"ERROR in {name}: {result.get('formula')}")
        if "traceback" in result:
            print(result["traceback"])
        if "stderr" in result:
            print(result["stderr"])
            
    # Cleanup temp files
    for p in [X_path, y_path, config_path]:
        if os.path.exists(p): os.remove(p)
        
    return result

def run_extreme_test():
    out_dir = os.path.join(current_dir, "")
    os.makedirs(out_dir, exist_ok=True)
    
    X, y, gt_formula = generate_extreme_data()
    print(f"Ground Truth Formula: {gt_formula}")
    
    # Common parameters
    aps = 1.0
    vps = 25
    vpr = 50
    vpm = 0.7
    
    base_kwargs = get_pysr_base_kwargs()
    # Adjust for "extreme" test
    base_kwargs['niterations'] = 30
    base_kwargs['populations'] = 50
    base_kwargs['population_size'] = 200
    
    configs = {
        "pysr_pure": {
            "type": "pysr",
            "params": {
                **base_kwargs
            }
        },
        "deeppysr_pypysr": {
            "type": "deeppysr",
            "params": {
                "model_provider": "pypysr",
                "adaptive_parsimony_scaling": aps,
                "variable_prune_start": vps,
                "variable_prune_ramp": vpr,
                "variable_prune_max": vpm,
                **base_kwargs
            }
        },
        "deeppysr_pypysrdev1": {
            "type": "deeppysr",
            "params": {
                "model_provider": "pypysrdev1",
                "adaptive_parsimony_scaling": aps,
                "variable_prune_start": vps,
                "variable_prune_ramp": vpr,
                "variable_prune_max": vpm,
                "use_mdl": True,
                "use_nsga2": True,
                "use_lexicase": True,
                "use_hotspot_protection": True,
                **base_kwargs
            }
        }
    }
    
    all_model_results = []
    
    for name, config in configs.items():
        print(f"\n{'='*40}")
        print(f"--- Testing {name} ---")
        print(f"{'='*40}")
        
        result = run_model_subprocess(name, config, X, y, out_dir)
        
        if result["status"] == "success":
            print(f"Time: {result['time']:.2f}s")
            print(f"Best Formula: {result['formula']}")
        
        all_model_results.append(result)
        
    df_results = pd.DataFrame(all_model_results)
    # Filter for saving
    cols_to_save = ["model", "time", "r2", "formula", "status"]
    df_results[cols_to_save].to_csv(os.path.join(out_dir, "extreme_test_results.csv"), index=False)
    
    print("\n" + "="*40)
    print("Results Summary:")
    print(df_results[cols_to_save])
    print("="*40)

if __name__ == "__main__":
    run_extreme_test()
