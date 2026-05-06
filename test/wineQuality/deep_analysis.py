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

from wine_utils import load_wine_data

def main():
    # Load model metrics to get best models and their parameters
    metrics_file = os.path.join(current_dir, 'wine_best_models_metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"ERROR: Metrics file not found at {metrics_file}")
        return

    metrics_df = pd.read_csv(metrics_file)
    output_root = os.path.join(current_dir, './results_wine_deep')

    # 1. Red Wine Models
    print("\n" + "="*70)
    print("RED WINE DEEP ANALYSIS")
    print("="*70)

    red_metrics = metrics_df[metrics_df['type'] == 'red']
    red_models = get_best_interpretable_params(red_metrics, model_type='Best DeepPySR')

    df_red = load_wine_data('red')
    X_red = df_red.drop(columns=['quality'])
    y_red = df_red['quality']
    
    red_output = os.path.join(output_root, 'red')
    if os.path.exists(os.path.join(red_output, 'relationships.csv')):
        print(f"Skipping red wine analysis as results already exist in {red_output}")
    else:
        run_deep_analysis(X_red, y_red, red_models, red_output, name='Red Wine', n_iterations=500, n_layers=3)

    # 2. White Wine Models
    print("\n" + "="*70)
    print("WHITE WINE DEEP ANALYSIS")
    print("="*70)

    white_metrics = metrics_df[metrics_df['type'] == 'white']
    white_models = get_best_interpretable_params(white_metrics, model_type='Best DeepPySR')

    df_white = load_wine_data('white')
    X_white = df_white.drop(columns=['quality'])
    y_white = df_white['quality']
    
    white_output = os.path.join(output_root, 'white')
    if os.path.exists(os.path.join(white_output, 'relationships.csv')):
        print(f"Skipping white wine analysis as results already exist in {white_output}")
    else:
        run_deep_analysis(X_white, y_white, white_models, white_output, name='White Wine', n_iterations=500, n_layers=3)

if __name__ == "__main__":
    main()
