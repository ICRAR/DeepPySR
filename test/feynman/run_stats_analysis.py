import os
import sys
import pandas as pd
import statsmodels.api as sm
from sympy import sympify

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, ".."))

from feynman_utils import load_feynman_data, equations

def get_features_from_formula(formula_str):
    """Extract variable names from a sympy-compatible formula string."""
    operators = {'exp', 'log', 'sin', 'cos', 'tan', 'sqrt', 'abs', 'cond', 'sign'}
    try:
        # Handle some common non-sympy formats
        f_str = formula_str.replace('np.', '')
        expr = sympify(f_str, locals={'cond': lambda a, b: a})
        symbols = [str(s) for s in expr.free_symbols]
        return symbols
    except Exception as e:
        print(f"Error parsing formula: {formula_str}. Error: {e}")
        import re
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', formula_str)
        features = set()
        for t in tokens:
            if t not in operators and not t.lower() in ['true', 'false', 'np', 'pi']:
                features.add(t)
        return list(features)

def run_stats(X, y, features, title, formula=None, metrics=None, output_file=None):
    print(f"\n{'='*70}")
    print(f"STATS ANALYSIS: {title}")
    if formula:
        print(f"Formula: {formula}")
    if metrics:
        metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        print(f"Metrics: {metrics_str}")
    print(f"{'='*70}")
    
    available_features = [f for f in features if f in X.columns]
    missing_features = [f for f in features if f not in X.columns]
    
    if missing_features:
        print(f"Warning: Missing features in data: {missing_features}")
    
    if not available_features:
        print("No available features for analysis.")
        return
    
    X_subset = X[available_features]
    X_subset = sm.add_constant(X_subset)
    
    try:
        model = sm.OLS(y, X_subset).fit()
        summary = model.summary()
        print(summary)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"STATS ANALYSIS: {title}\n")
                if formula:
                    f.write(f"Formula: {formula}\n")
                if metrics:
                    metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
                    f.write(f"Metrics: {metrics_str}\n")
                f.write(f"Features: {available_features}\n")
                if missing_features:
                    f.write(f"Missing Features: {missing_features}\n")
                f.write("="*70 + "\n")
                f.write(summary.as_text())
            print(f"Summary saved to {output_file}")
            
        return model
    except Exception as e:
        print(f"Error fitting OLS: {e}")
        return None

def main():
    eq_names = list(equations.keys())
    output_dir = os.path.join(current_dir, 'results_feynman_stats')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for eq_name in eq_names:
        print(f"\nProcessing Feynman equation: {eq_name}")
        
        deep_results_dir = os.path.join(current_dir, f'results_{eq_name.replace(".", "_")}_deep')
        if not os.path.exists(deep_results_dir):
            print(f"WARNING: Deep results directory not found at {deep_results_dir}")
            continue

        rel_file = os.path.join(deep_results_dir, 'relationships.csv')
        if not os.path.exists(rel_file):
            print(f"WARNING: Relationships file not found at {rel_file}")
            continue

        rel_df = pd.read_csv(rel_file)
        # Filter for layer 1 and target 'y', and sort by r2 and f1 descending
        layer1 = rel_df[rel_df['layer'] == 1]
        
        X, y = load_feynman_data(eq_name, n_samples=1000)
        
        if not layer1.empty:
            # Sort by r2 and f1 to get the best one
            layer1 = layer1.sort_values(by=['r2', 'f1'], ascending=False)
            best_row = layer1.iloc[0]
            formula = best_row['formula']
            metrics = {'r2': best_row['r2'], 'f1': best_row['f1']}
            features = get_features_from_formula(formula)
            print(f"Extracted features: {features}")
            print(f"Metrics: {metrics}")
            
            output_file = os.path.join(output_dir, f'stats_{eq_name.replace(".", "_")}.txt')
            run_stats(X, y, features, f"Feynman {eq_name}", formula=formula, metrics=metrics, output_file=output_file)
        else:
            print(f"No layer 1 formula found in relationships.csv for {eq_name}")

if __name__ == "__main__":
    main()
