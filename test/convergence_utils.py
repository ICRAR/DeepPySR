import os
import sys
import numpy as np
import pandas as pd
import time
import re
import matplotlib.pyplot as plt
import sympy as sp
from model_utils import get_pysr_base_kwargs
from eval_utils import calculate_metrics
from analysis_utils import evaluate_formula

def get_loss_from_equations(model, X, y, task='regression',model_type='deeppysr'):
    """
    Extract the best equation based on highest R² score (regression) or F1 score (classification)
    from the model's equations_ DataFrame.
    Calculates metrics manually to ensure consistency and completeness.
    Returns metrics for the single best equation.
    """
    try:
        if not hasattr(model, 'equations_') or model.equations_ is None:
            return []
        
        eqs = model.equations_
        if isinstance(eqs, list):
            if len(eqs) == 0:
                return []
            eqs = eqs[0]
        
        if len(eqs) == 0:
            return []
        
        all_results = []
        for idx, row in eqs.iterrows():
            equation_str = str(row.get('equation', ''))
            complexity = int(row.get('complexity', 1))
            
            # Predict using this specific equation manually using analysis_utils.evaluate_formula
            try:
                y_pred = evaluate_formula(equation_str, X, model_type=model_type)
                
                # calculate_metrics handles rounding for classification if needed.
                metrics = calculate_metrics(y, y_pred, task=task)
            except Exception as e:
                print(f"Error calculating metrics for equation {idx}: {e}")
                continue
            
            res = {
                'complexity': complexity,
                'equation': equation_str,
                'equation_index': int(idx),
                'score': float(row.get('score', 0.0))
            }
            res.update(metrics)
            
            # Ensure regression specific metrics are present if task is regression
            if task == 'regression':
                res['loss'] = res.get('mse', metrics.get('rmse', 0.0)**2) # MSE is our loss
                if 'mse' not in res:
                    res['mse'] = res['loss']
            
            all_results.append(res)
            
        if not all_results:
            return []
            
        # Select the best equation
        if task == 'regression':
            # Use R2 for regression
            best_idx = np.argmax([r.get('r2', -np.inf) for r in all_results])
        else:
            # Use F1 for classification
            best_idx = np.argmax([r.get('f1', -np.inf) for r in all_results])
            
        return [all_results[best_idx]]
    
    except Exception as e:
        print(f"Error extracting losses: {e}")
        import traceback
        traceback.print_exc()
        return []

def _clear_pysr_modules():
    """Clears pysr-related modules from sys.modules to allow switching between DeepPySR and PySR."""
    pysr_related = [mod for mod in sys.modules.keys() 
                   if mod in ("pysr", "pypysr") or mod.startswith(("pysr.", "pypysr."))]
    for mod in pysr_related:
        del sys.modules[mod]

def _setup_julia_environment():
    """Sets up the Julia environment for PySR."""
    try:
        import juliapkg
        import juliacall
        from juliacall import Main as jl

        # Initialize Julia if not already initialized
        jl.seval("using Pkg")
        pysr_env = juliapkg.project()
        jl.Pkg.activate(pysr_env)
        print(f"Julia environment activated: {pysr_env}")
    except Exception as e:
        # Don't fail if Julia/juliapkg is not available,
        # as PySR handles its own setup if needed
        pass

def parse_model_string(model_str):
    """
    Parses hyperparameters from model string like:
    fullsr_nit100_pop30_sz200_vps25_vpr100_aps50.0_grid_r2w1.5_L0.001
    """
    params = {}
    
    # Extract vps (variable_prune_start)
    vps_match = re.search(r'_vps(\d+)', model_str)
    if vps_match:
        params['variable_prune_start'] = int(vps_match.group(1))
        
    # Extract vpr (variable_prune_ramp)
    vpr_match = re.search(r'_vpr(\d+)', model_str)
    if vpr_match:
        params['variable_prune_ramp'] = int(vpr_match.group(1))
        
    # Extract aps (adaptive_parsimony_scaling)
    aps_match = re.search(r'_aps([\d.]+)', model_str)
    if aps_match:
        params['adaptive_parsimony_scaling'] = float(aps_match.group(1))
        
    # Extract r2w (r2 weight)
    r2w_match = re.search(r'_r2w([\d.]+)', model_str)
    if r2w_match:
        params['r2_weight'] = float(r2w_match.group(1))
        
    # Extract L (lambda_complexity)
    l_match = re.search(r'_L([\d.]+)', model_str)
    if l_match:
        params['lambda'] = float(l_match.group(1))
        
    # Default vpm (variable_prune_max) if not in string, usually 0.7
    params['variable_prune_max'] = 0.7
    
    return params

def train_model(model_provider, X, y, n_iterations=10, output_dir="./convergence_results", params=None, task='regression'):
    """
    Train model iteratively and record loss for each iteration.
    Uses DeepPySRRegressor with model_provider parameter for clean switching.
    """
    if params is None:
        params = {}
        
    # Clear modules and setup environment to avoid conflicts when switching providers
    _clear_pysr_modules()
    _setup_julia_environment()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use base kwargs from model_utils.py
    pysr_kwargs = get_pysr_base_kwargs()
    
    # Apply requested overrides from params
    if "adaptive_parsimony_scaling" in params:
        pysr_kwargs["adaptive_parsimony_scaling"] = params["adaptive_parsimony_scaling"]
    else:
        pysr_kwargs["adaptive_parsimony_scaling"] = 50.0
    
    if model_provider == "deeppysr":
        # Specific parameters for DeepPySR (pypysr)
        pysr_kwargs["variable_prune_start"] = params.get("variable_prune_start", 25)
        pysr_kwargs["variable_prune_ramp"] = params.get("variable_prune_ramp", 100)
        pysr_kwargs["variable_prune_max"] = params.get("variable_prune_max", 0.7)
    
    model_output_dir = os.path.join(output_dir, f"{model_provider}_run")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Set niterations to 1 for tracking convergence per iteration
    pysr_kwargs["niterations"] = 1
    pysr_kwargs["output_directory"] = model_output_dir
    
    r2w = params.get("r2_weight", 1.5)
    lambda_val = params.get("lambda", 0.001)
    
    r2w_list = [r2w]
    lambda_list = [lambda_val]
    
    if model_provider == "deeppysr":
        from model_utils import DeepPySRRegressor
        model = DeepPySRRegressor(
            max_layers=1,
            pareto_r2_weight=r2w_list,
            pareto_lambda=lambda_list,  # Single layer for regression comparison
            warm_start=True,  # Enable warm start for iterative training
            **pysr_kwargs
        )
    else:
        from model_utils import PySRRegressor
        model = PySRRegressor(
            warm_start=True,
            **pysr_kwargs
        )
    
    history = []
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"Training {model_provider.upper()} for {n_iterations} iterations (Task: {task})")

    for outer_iter in range(1, n_iterations + 1):
        try:
            print(f"[{model_provider}] Iteration {outer_iter}/{n_iterations}...", end=' ', flush=True)
            
            model.fit(X, y)
            elapsed = time.time() - start_time
            
            equations_data = get_loss_from_equations(model, X, y, task=task, model_type=model_provider)
            print(
                f"Elapsed: {elapsed:.2f}s, Equations: {len(equations_data)}"
            )
            
            if equations_data:
                # get_loss_from_equations now returns only the best equation
                best_result = equations_data[0]
                
                equation_str = best_result['equation']
                if len(equation_str) > 100:
                    equation_display = equation_str[:97] + "..."
                else:
                    equation_display = equation_str
                
                row = {
                    'Iteration': outer_iter,
                    'Time': elapsed,
                    'Model': model_provider,
                    'Complexity': best_result['complexity'],
                    'Equation': equation_str,
                    'Equation_Index': best_result.get('equation_index', -1),
                    'Score': best_result.get('score', 0),
                    'N_Equations': 1
                }
                
                if task == 'regression':
                    row.update({
                        'Loss': best_result['loss'],
                        'RMSE': best_result.get('rmse', np.nan),
                        'R2': best_result['r2']
                    })
                    print(f"MSE: {best_result['loss']:.6f}, R2: {best_result['r2']:.4f}, C: {best_result['complexity']}")
                else:
                    row.update({
                        'Accuracy': best_result.get('accuracy', np.nan),
                        'F1': best_result.get('f1', np.nan),
                        'Precision': best_result.get('precision', np.nan),
                        'Recall': best_result.get('recall', np.nan),
                        'AUC': best_result.get('auc', np.nan),
                        # Use 1-F1 as a "loss" for plotting if needed
                        'Loss': 1.0 - best_result.get('f1', 0.0) 
                    })
                    print(f"F1: {best_result.get('f1', 0.0):.4f}, Acc: {best_result.get('accuracy', 0.0):.4f}, C: {best_result['complexity']}")
                
                history.append(row)
                print(f"       Equation: {equation_display}")
            else:
                print("No valid equations found")
        
        except Exception as e:
            print(f"ERROR in iteration {outer_iter}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    return pd.DataFrame(history)

def plot_convergence(combined_df, output_dir, title="Loss Convergence Comparison", filename="loss_convergence.png", params=None, task='regression'):
    """
    Creates a line plot for the loss across iterations and saves it.
    Specifies model parameters in the text.
    """
    if params is None:
        params = {}
        
    plt.figure(figsize=(12, 8))
    
    for model_name in combined_df['Model'].unique():
        model_data = combined_df[combined_df['Model'] == model_name]
        plt.plot(model_data['Iteration'], model_data['Loss'], label=f"{model_name.upper()}")
    
    if task == 'regression':
        plt.yscale('log')
        plt.ylabel('Loss (MSE)')
    else:
        plt.ylabel('Loss (1 - F1)')
        
    plt.xlabel('Iteration')
    plt.title(title)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # Model parameters to specify in text
    params_text = "Model Parameters:\n"
    params_text += f"adaptive_parsimony_scaling: {params.get('adaptive_parsimony_scaling', 'N/A')}\n"
    params_text += f"variable_prune_start: {params.get('variable_prune_start', 'N/A')}\n"
    params_text += f"ramp: {params.get('variable_prune_ramp', 'N/A')}\n"
    params_text += f"max: {params.get('variable_prune_max', 'N/A')}\n"
    params_text += f"r2_weight: {params.get('r2_weight', 'N/A')}\n"
    params_text += f"lambda_complexity: {params.get('lambda', 'N/A')}"
    
    plt.figtext(0.75, 0.5, params_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path)
    print(f"Convergence plot saved to: {plot_path}")
    plt.close()

def run_convergence_comparison(X, y, model_params_dict, output_root, name, n_iterations=100, task='regression'):
    """
    Runs convergence test for Best DeepPySR and Best PySR for a given type and age.
    """
    os.makedirs(output_root, exist_ok=True)
    
    results_list = []
    
    # Run DeepPySR (pypysr)
    if 'Best DeepPySR' in model_params_dict:
        print(f"\n--- Running DeepPySR convergence ---")
        pypysr_params = model_params_dict['Best DeepPySR']
        pypysr_hist = train_model("deeppysr", X, y, n_iterations=n_iterations,
                                  output_dir=output_root, params=pypysr_params, task=task)
        results_list.append(pypysr_hist)
    
    # Run PySR (pysr)
    if 'Best PySR' in model_params_dict:
        print(f"\n--- Running PySR convergence ---")
        pysr_params = model_params_dict['Best PySR']
        pysr_hist = train_model("pysr", X, y, n_iterations=n_iterations, 
                                output_dir=output_root, params=pysr_params, task=task)
        results_list.append(pysr_hist)
        
    if results_list:
        combined = pd.concat(results_list, ignore_index=True)
        csv_file = os.path.join(output_root, f"convergence_{name}.csv")
        combined.to_csv(csv_file, index=False)
        
        title = f"Loss Convergence - {name}"
        filename = f"loss_convergence_{name}.png"
        
        # Use pypysr params for display text if available, otherwise pysr
        display_params = model_params_dict.get('Best DeepPySR', model_params_dict.get('Best PySR', {}))
        plot_convergence(combined, output_root, title=title, filename=filename, params=display_params, task=task)
        
        return combined
    return None
