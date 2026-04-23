import os
import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(current_dir, ""))

from sklearn.metrics import r2_score
from model_utils import get_pysr_base_kwargs
from bmi_utils import load_bmi_agg_data

# Import DeepPySRRegressor which handles provider switching
sys.path.insert(0, os.path.join(project_root, "../.."))
from DeepPySR.regressor import DeepPySRRegressor

try:
    import sympy
    from sympy import sympify, symbols
    from sympy.utilities.lambdify import lambdify
except ImportError:
    sympy = None


def get_loss_from_equations(model):
    """
    Extract the best equation based on highest R² score from the model's equations_ DataFrame.
    Uses the pre-calculated metrics from the regressor to follow PySRRegressor definition.
    Returns metrics for the single best equation by R².
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
        
        # Find the equation with the highest R² from the DataFrame
        best_idx = eqs['r2'].idxmax()
        best_row = eqs.loc[best_idx]
        
        # Extract pre-calculated metrics from the DataFrame
        equation_str = str(best_row.get('equation', ''))
        loss = float(best_row.get('loss', 0.0))
        rmse = np.sqrt(loss)
        r2 = float(best_row.get('r2', 0.0))
        score = float(best_row.get('score', 0.0))
        complexity = int(best_row.get('complexity', 1))
        
        return [{
            'complexity': complexity,
            'loss': loss,
            'rmse': rmse,
            'r2': r2,
            'score': score,
            'equation': equation_str,
            'equation_index': int(best_idx)
        }]
    
    except Exception as e:
        print(f"Error extracting losses: {e}")
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


def train_model(model_provider, X, y, n_iterations=10, output_dir="./convergence_results"):
    """
    Train model iteratively and record loss for each iteration.
    Uses DeepPySRRegressor with model_provider parameter for clean switching.
    """
    # Clear modules and setup environment to avoid conflicts when switching providers
    _clear_pysr_modules()
    _setup_julia_environment()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use base kwargs from model_utils.py
    pysr_kwargs = get_pysr_base_kwargs()
    
    # Apply requested overrides
    pysr_kwargs["adaptive_parsimony_scaling"] = 50.0
    
    if model_provider == "pypysr":
        # Specific parameters for DeepPySR (pypysr)
        pysr_kwargs["variable_prune_start"] = 25
        pysr_kwargs["variable_prune_ramp"] = 100
        pysr_kwargs["variable_prune_max"] = 0.7
    
    model_output_dir = os.path.join(output_dir, f"{model_provider}_run")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Set niterations to 1 for tracking convergence per iteration
    pysr_kwargs["niterations"] = 1
    pysr_kwargs["output_directory"] = model_output_dir
    
    r2w_list = [1.5]
    lambda_list = [0.001]
    # Create DeepPySRRegressor with the specified provider
    model = DeepPySRRegressor(
        model_provider=model_provider,
        max_layers=1,
        pareto_r2_weight=r2w_list,
        pareto_lambda=lambda_list,  # Single layer for regression comparison
        warm_start=True,  # Enable warm start for iterative training
        **pysr_kwargs
    )
    
    history = []
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"Training {model_provider.upper()} for {n_iterations} iterations")

    variable_names = list(X.columns)
    
    for outer_iter in range(1, n_iterations + 1):
        try:
            print(f"[{model_provider}] Iteration {outer_iter}/{n_iterations}...", end=' ', flush=True)
            
            model.fit(X, y)
            elapsed = time.time() - start_time
            
            equations_data = get_loss_from_equations(model)
            
            if equations_data:
                # get_loss_from_equations now returns only the best equation by R²
                best_result = equations_data[0]
                
                equation_str = best_result['equation']
                if len(equation_str) > 100:
                    equation_display = equation_str[:97] + "..."
                else:
                    equation_display = equation_str
                
                history.append({
                    'Iteration': outer_iter,
                    'Loss': best_result['loss'],
                    'RMSE': best_result.get('rmse', np.nan),
                    'R2': best_result['r2'],
                    'Score': best_result.get('score', 0),
                    'Time': elapsed,
                    'Model': model_provider,
                    'Complexity': best_result['complexity'],
                    'Equation': equation_str,
                    'Equation_Index': best_result.get('equation_index', -1),
                    'N_Equations': 1  # get_loss_from_equations now returns only the best equation
                })
                
                print(f"MSE: {best_result['loss']:.6f}, R2: {best_result['r2']:.4f}, C: {best_result['complexity']}, Eq_idx: {best_result.get('equation_index', -1)}")
                print(f"       Equation: {equation_display}")
            else:
                print("No valid equations found")
        
        except Exception as e:
            print(f"ERROR in iteration {outer_iter}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    return pd.DataFrame(history)


def main():
    print("\n" + "="*70)
    print("BMI LONGITUDINAL CONVERGENCE TEST ")
    print("="*70)
    
    # Load bmi longitudinal data
    print("Generating BMI longitudinal data...")
    id, X, y = load_bmi_agg_data()
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    
    output_root = os.path.join(current_dir, './convergence_results')
    os.makedirs(output_root, exist_ok=True)
    
    results_list = []
    
    # Test pypysr
    try:
        pypysr_hist = train_model("pypysr", X, y, n_iterations=500, output_dir=output_root)
        results_list.append(pypysr_hist)
    except Exception as e:
        print(f"ERROR training pypysr: {e}")
        import traceback
        traceback.print_exc()
    
    # Test pysr
    try:
        pysr_hist = train_model("pysr", X, y, n_iterations=500, output_dir=output_root)
        results_list.append(pysr_hist)
    except Exception as e:
        print(f"ERROR training pysr: {e}")
        import traceback
        traceback.print_exc()
    
    if results_list:
        combined = pd.concat(results_list, ignore_index=True)
        output_file = os.path.join(output_root, "convergence_comparison.csv")
        combined.to_csv(output_file, index=False)
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*70}\n")
        
        # Print summary statistics
        print("CONVERGENCE SUMMARY STATISTICS\n")
        
        for model in combined['Model'].unique():
            model_data = combined[combined['Model'] == model]
            print(f"\n{model.upper()}:")
            print(f"  Iterations: {len(model_data)}")
            print(f"  Final MSE: {model_data.iloc[-1]['Loss']:.6f}")
            print(f"  Final R2: {model_data.iloc[-1]['R2']:.4f}")
            print(f"  Final Complexity: {int(model_data.iloc[-1]['Complexity'])}")
            print(f"  Best MSE (across all iterations): {model_data['Loss'].min():.6f}")
            print(f"  Best R2 (across all iterations): {model_data['R2'].max():.4f}")
            print(f"  Average Equation Length: {model_data['Complexity'].mean():.1f}")
            print(f"  Num Equations per Iteration: {model_data['N_Equations'].mean():.0f}")
            
            # Show convergence trajectory
            print(f"\n  Convergence (every 5 iterations):")
            for idx in model_data.index:
                if (idx + 1) % 5 == 0 or idx == len(model_data) - 1:
                    row = model_data.loc[idx]
                    print(f"    Iter {int(row['Iteration']):2d}: MSE={row['Loss']:.6f}, R2={row['R2']:.4f}, C={int(row['Complexity']):3d}")
        
        print(f"\n{'='*70}")
        print("Full convergence table:")
        print(f"{'='*70}\n")
        print(combined.to_string(index=False))

if __name__ == "__main__":
    main()
