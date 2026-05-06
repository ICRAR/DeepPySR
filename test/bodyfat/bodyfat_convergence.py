import os
import sys
import pandas as pd

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(current_dir, ""))

# Import shared convergence utilities
sys.path.append(os.path.join(project_root, "test"))
from convergence_utils import (
    parse_model_string,
    run_convergence_comparison
)

from bodyfat_utils import load_bodyfat_data

def main():
    # Load model metrics to get best models and their parameters
    metrics_file = os.path.join(current_dir,
                                'bodyfat_best_models_metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"ERROR: Metrics file not found at {metrics_file}")
        return
        
    metrics_df = pd.read_csv(metrics_file)
    output_root = os.path.join(current_dir, './convergence_results')
    
    print("\n" + "="*70)
    print("BODYFAT CONVERGENCE TESTS")
    print("="*70)
    
    # Get unique best models
    best_models = {}
    for _, row in metrics_df.iterrows():
        if row['display_model'] in ['Best DeepPySR', 'Best PySR']:
            if row['display_model'] not in best_models:
                best_models[row['display_model']] = parse_model_string(row['model'])
    
    # Load data
    X, y = load_bodyfat_data()
    run_convergence_comparison(X, y, best_models, output_root, name='BodyFat', task='regression')

if __name__ == "__main__":
    main()
