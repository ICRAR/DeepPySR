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
    
    for eq_name in eq_names:
        print("\n" + "="*70)
        print(f"CONVERGENCE TESTS FOR FEYNMAN EQUATION: {eq_name}")
        print("="*70)
        
        # Load model metrics to get best models and their parameters
        metrics_file = os.path.join(current_dir,
                                    f'results_{eq_name.replace(".", "_")}_all/feynman_best_models_metrics.csv')
        
        # In some cases the metrics file might have a slightly different name or the prefix might be just "best"
        if not os.path.exists(metrics_file):
            metrics_file = os.path.join(current_dir,
                                        f'results_{eq_name.replace(".", "_")}_all/best_models_metrics.csv')

        if not os.path.exists(metrics_file):
            print(f"WARNING: Metrics file not found at {metrics_file}, skipping {eq_name}")
            continue
            
        metrics_df = pd.read_csv(metrics_file)
        
        # Get unique best models
        models = {}
        for _, row in metrics_df.iterrows():
            if row['display_model'] in ['Best DeepPySR', 'Best PySR']:
                if row['display_model'] not in models:
                    models[row['display_model']] = parse_model_string(row['model'])
        
        if not models:
            print(f"No best models found for {eq_name}")
            continue

        # Load data
        X, y = load_feynman_data(eq_name, n_samples=1000)
        
        eq_output_root = os.path.join(output_root, f"eq_{eq_name.replace('.', '_')}")
        run_convergence_comparison(X, y, models, eq_output_root, name=f'Feynman: {eq_name}')

if __name__ == "__main__":
    main()
