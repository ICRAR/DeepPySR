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

from bmi_utils import load_bmi_agg_data

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
    metrics_file = os.path.join(current_dir, 'results_bmi_all/bmi_best_models_metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"ERROR: Metrics file not found at {metrics_file}")
        return
        
    metrics_df = pd.read_csv(metrics_file)
    output_root = os.path.join(current_dir, './convergence_results')
    
    # 1. Longitudinal Models
    print("\n" + "="*70)
    print("LONGITUDINAL CONVERGENCE TESTS")
    print("="*70)
    
    long_metrics = metrics_df[metrics_df['type'] == 'longitudinal']
    # Get unique best models for longitudinal
    long_models = {}
    for _, row in long_metrics.iterrows():
        if row['display_model'] in ['Best DeepPySR', 'Best PySR']:
            if row['display_model'] not in long_models:
                long_models[row['display_model']] = parse_model_string(row['model'])
    
    # Load longitudinal data
    id_long, X_long, y_long = load_bmi_agg_data()
    long_output_root = os.path.join(output_root, 'longitudinal')
    run_convergence_comparison(X_long, y_long, long_models, long_output_root, name='Longitudinal')

    # 2. Age-Specific Models
    print("\n" + "="*70)
    print("AGE-SPECIFIC CONVERGENCE TESTS")
    print("="*70)
    
    age_specific_metrics = metrics_df[metrics_df['type'] == 'age-specific']
    unique_ages = sorted(age_specific_metrics['age'].unique())
    
    for age in unique_ages:
        print(f"\nProcessing Age: {age}")
        age_metrics = age_specific_metrics[age_specific_metrics['age'] == age]
        age_models = {}
        for _, row in age_metrics.iterrows():
            if row['display_model'] in ['Best DeepPySR', 'Best PySR']:
                if row['display_model'] not in age_models:
                    age_models[row['display_model']] = parse_model_string(row['model'])
        
        # Filter data for specific age
        if 'age' in X_long.columns:
            mask = X_long['age'] == age
            X_age = X_long[mask].drop(columns=['age'])
            y_age = y_long[mask]
        else:
            # Fallback if 'age' is not in columns, though load_bmi_agg_data usually includes it for longitudinal
            X_age = X_long
            y_age = y_long
        
        if len(X_age) > 0 and age_models:
            age_output_root = os.path.join(output_root, 'age-specific', f'age{age}')
            run_convergence_comparison(X_age, y_age, age_models, age_output_root, name=f'Age: {age} years')
        else:
            print(f"No data or no best models found for age {age}")

if __name__ == "__main__":
    main()
