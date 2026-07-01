import os
import sys
import pandas as pd

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, ".."))

from feynman_utils import load_feynman_data, equations

# Import shared convergence utilities
sys.path.append(os.path.join(project_root, "test"))
from convergence_utils import (
    parse_model_string,
    run_convergence_comparison
)

def main():
    eq_names = list(equations.keys())
    
    output_root = os.path.join(current_dir, './convergence_results')
    
    # Load aggregated results once
    aggregated_file = os.path.join(current_dir, 'aggregated_results.csv')
    if not os.path.exists(aggregated_file):
        print(f"ERROR: Aggregated results file not found at {aggregated_file}")
        return
        
    aggregated_df = pd.read_csv(aggregated_file)
    
    for eq_name in [eq_names[3]]:
        print("\n" + "="*70)
        print(f"CONVERGENCE TESTS FOR FEYNMAN EQUATION: {eq_name}")
        print("="*70)

        # Filter for this equation
        eq_df = aggregated_df[aggregated_df['equation'] == eq_name]
        
        if eq_df.empty:
            print(f"WARNING: No results found for {eq_name} in aggregated results, skipping")
            continue
            
        # Get unique best models
        models = {}
        for _, row in eq_df.iterrows():
            if row['model'] == 'DeepPySR':
                # Use model_path to get configuration string
                model_str = os.path.basename(row['model_path'])
                models['Best DeepPySR'] = parse_model_string(model_str)
            elif row['model'] == 'PySR':
                # Use model_path to get configuration string
                model_str = os.path.basename(row['model_path'])
                models['Best PySR'] = parse_model_string(model_str)
        
        if not models:
            print(f"No best models found for {eq_name}")
            continue
        for model_name in models:
            if models[model_name] is None:
                models[model_name] = {}
            models[model_name]["extra_constants"] = ["pi", "c", "G", "e"]

        # Load data
        X, y = load_feynman_data(eq_name, n_samples=1000)
        
        eq_output_root = os.path.join(output_root, f"eq_{eq_name.replace('.', '_')}")
        if os.path.exists(os.path.join(eq_output_root,f"convergence_Feynman: {eq_name}.csv")):
            continue
        run_convergence_comparison(X, y, models, eq_output_root, n_iterations=500, name=f'Feynman: {eq_name}')

if __name__ == "__main__":
    main()
