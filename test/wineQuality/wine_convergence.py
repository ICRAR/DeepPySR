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

from wine_utils import load_wine_data

def main():
    # Load model metrics to get best models and their parameters
    metrics_file = os.path.join(current_dir, 'wine_best_models_metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"ERROR: Metrics file not found at {metrics_file}")
        return
        
    metrics_df = pd.read_csv(metrics_file)
    output_root = os.path.join(current_dir, './convergence_results')
    
    # 1. Red Wine Models
    print("\n" + "="*70)
    print("RED WINE CONVERGENCE TESTS")
    print("="*70)
    
    red_metrics = metrics_df[metrics_df['type'] == 'red']
    red_models = {}
    for _, row in red_metrics.iterrows():
        if row['display_model'] in ['Best DeepPySR', 'Best PySR']:
            if row['display_model'] not in red_models:
                red_models[row['display_model']] = parse_model_string(row['model'])
    
    df_red = load_wine_data('red')
    X_red = df_red.drop(columns=['quality'])
    y_red = df_red['quality']
    
    red_output = os.path.join(output_root, 'red')
    run_convergence_comparison(X_red, y_red, red_models, red_output, name='Red Wine', task='regression')

    # 2. White Wine Models
    print("\n" + "="*70)
    print("WHITE WINE CONVERGENCE TESTS")
    print("="*70)
    
    white_metrics = metrics_df[metrics_df['type'] == 'white']
    white_models = {}
    for _, row in white_metrics.iterrows():
        if row['display_model'] in ['Best DeepPySR', 'Best PySR']:
            if row['display_model'] not in white_models:
                white_models[row['display_model']] = parse_model_string(row['model'])
    
    df_white = load_wine_data('white')
    X_white = df_white.drop(columns=['quality'])
    y_white = df_white['quality']
    
    white_output = os.path.join(output_root, 'white')
    run_convergence_comparison(X_white, y_white, white_models, white_output, name='White Wine', task='regression')

if __name__ == "__main__":
    main()
