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

from deep_analysis_utils import (
    run_deep_analysis,
    get_best_interpretable_params
)

from heart_utils import load_heart_cleveland_data

def main():
    # Load model metrics to get best models and their parameters
    metrics_file = os.path.join(current_dir, 'heart_best_models_metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"ERROR: Metrics file not found at {metrics_file}")
        return

    metrics_df = pd.read_csv(metrics_file)
    output_root = os.path.join(current_dir, './results_heart_deep')

    print("\n" + "="*70)
    print("HEART DEEP ANALYSIS")
    print("="*70)

    # Get unique best models
    best_models = get_best_interpretable_params(metrics_df, model_type='Best DeepPySR')

    # Load data
    X, y = load_heart_cleveland_data(binary=True)
    
    # Check if results already exist
    if os.path.exists(os.path.join(output_root, 'relationships.csv')):
        print(f"Skipping heart analysis as results already exist in {output_root}")
    else:
        run_deep_analysis(X, y, best_models, output_root, name='Heart', n_iterations=500, n_layers=3, task='classification')

if __name__ == "__main__":
    main()
