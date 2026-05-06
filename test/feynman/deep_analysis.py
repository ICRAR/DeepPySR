import os
import sys
import pandas as pd

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, ".."))

from feynman_utils import load_feynman_data, equations

# Import shared deep analysis utilities
sys.path.append(os.path.join(project_root, "test"))
from deep_analysis_utils import (
    run_deep_analysis
)

def main():
    eq_names = list(equations.keys())
    
    for eq_name in eq_names:
        print("\n" + "="*70)
        print(f"DEEP ANALYSIS FOR FEYNMAN EQUATION: {eq_name}")
        print("="*70)
        
        # Load model metrics to get best models and their parameters
        # Use aggregated_results.csv as the primary source
        aggregated_file = os.path.join(current_dir, 'aggregated_results.csv')
        
        if not os.path.exists(aggregated_file):
            print(f"WARNING: Aggregated results file not found at {aggregated_file}, skipping {eq_name}")
            continue
            
        metrics_df = pd.read_csv(aggregated_file)
        
        # Filter for the current equation and DeepPySR models
        eq_metrics = metrics_df[(metrics_df['equation'] == eq_name) & (metrics_df['model'] == 'DeepPySR')]
        
        if eq_metrics.empty:
            print(f"No DeepPySR models found for {eq_name} in aggregated results")
            continue
            
        models = {}
        for _, row in eq_metrics.iterrows():
            model_path = row.get('model_path', '')
            if pd.isna(model_path) or not model_path:
                print(f"Warning: No model_path for {eq_name} DeepPySR, using default params")
                models['DeepPySR'] = {}
            else:
                # Extract params from model_path (which contains the model configuration string)
                from convergence_utils import parse_model_string
                model_name = os.path.basename(model_path)
                params = parse_model_string(model_name)
                models[f"DeepPySR_{model_name}"] = params
        
        if not models:
            print(f"No best models found for {eq_name}")
            continue

        # Load data
        X, y = load_feynman_data(eq_name, n_samples=1000)
        
        eq_output_root = os.path.join(current_dir, f"results_{eq_name.replace('.', '_')}_deep")
        
        # Check if results already exist
        if os.path.exists(os.path.join(eq_output_root, 'relationships.csv')):
            print(f"Skipping {eq_name} deep analysis as results already exist in {eq_output_root}")
        else:
            run_deep_analysis(X, y, models, eq_output_root, name=f'Feynman: {eq_name}', n_iterations=500, n_layers=3)

if __name__ == "__main__":
    main()
