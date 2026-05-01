import os
import sys
import pandas as pd

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(current_dir, ""))

# Import DeepPySRRegressor which handles provider switching
sys.path.insert(0, os.path.join(project_root, "../.."))

from heart_utils import load_heart_cleveland_data

# Import shared convergence utilities
sys.path.append(os.path.join(project_root, "test"))
from convergence_utils import (
    parse_model_string,
    run_convergence_comparison
)

try:
    import sympy
    from sympy import sympify, symbols
    from sympy.utilities.lambdify import lambdify
except ImportError:
    sympy = None


def main():
    # Load model metrics to get best models and their parameters
    metrics_file = os.path.join(current_dir, 'heart_best_models_metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"ERROR: Metrics file not found at {metrics_file}")
        return
        
    metrics_df = pd.read_csv(metrics_file)
    output_root = os.path.join(current_dir, './convergence_results')
    
    # 1. Longitudinal Models
    print("\n" + "="*70)
    print("LONGITUDINAL CONVERGENCE TESTS")
    print("="*70)

    # Get unique best models
    models = {}
    for _, row in metrics_df.iterrows():
        if row['display_model'] in ['Best DeepPySR', 'Best PySR']:
            if row['display_model'] not in models:
                models[row['display_model']] = parse_model_string(row['model'])
    
    # Load longitudinal data
    X, y = load_heart_cleveland_data(binary=True)
    run_convergence_comparison(X, y, models, output_root, name='Heart', task='classification')

if __name__ == "__main__":
    main()
