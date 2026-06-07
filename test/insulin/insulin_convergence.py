import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(current_dir, ""))
sys.path.insert(0, os.path.join(project_root, "../.."))
sys.path.append(os.path.join(project_root, "test"))

from data_utils import load_data, load_data_longitudinal
from convergence_utils import parse_model_string, run_convergence_comparison

try:
    import sympy
except ImportError:
    sympy = None

AGES = [14, 17, 20, 22, 27, 28]
TARGETS = ['insulin', 'glucose']


def _extract_y(y_df, target):
    col = [c for c in y_df.columns if target in c][0]
    return y_df[col].rename(target)


def _load_age(age, target):
    ids, X, y_df = load_data(["insulin", "glucose"], age)
    return ids, X, _extract_y(y_df, target)


def _load_longitudinal(target):
    ids, X, y_df = load_data_longitudinal(["insulin", "glucose"])
    return ids, X, _extract_y(y_df, target)


def _run_for_target(target, output_root):
    metrics_file = os.path.join(current_dir, f'results_insulin/insulin_{target}_best_models_metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"Skipping {target}: metrics file not found at {metrics_file}")
        return

    metrics_df = pd.read_csv(metrics_file)

    consistent_config = {
        'adaptive_parsimony_scaling': 10.0,
        'variable_prune_start': 25,
        'variable_prune_ramp': 150,
        'r2_weight': 1.5,
        'lambda': 0.001
    }

    # Longitudinal
    print(f"\n{'='*70}")
    print(f"LONGITUDINAL CONVERGENCE TESTS — {target.upper()}")
    print('='*70)

    long_metrics = metrics_df[metrics_df['type'] == 'longitudinal']
    long_models = {}
    for _, row in long_metrics.iterrows():
        if row['display_model'] in ['Best DeepPySR', 'Best PySR'] and row['display_model'] not in long_models:
            long_models[row['display_model']] = consistent_config.copy()

    _, X_long, y_long = _load_longitudinal(target)
    long_output_root = os.path.join(output_root, target, 'longitudinal')
    if not os.path.exists(long_output_root):
        run_convergence_comparison(X_long, y_long, long_models, long_output_root, name=f'Longitudinal ({target})')

    # Age-Specific
    print(f"\n{'='*70}")
    print(f"AGE-SPECIFIC CONVERGENCE TESTS — {target.upper()}")
    print('='*70)

    age_specific_metrics = metrics_df[metrics_df['type'] == 'age-specific']
    unique_ages = sorted(age_specific_metrics['age'].unique())

    for age in unique_ages:
        print(f"\nProcessing Age: {age}")
        age_metrics = age_specific_metrics[age_specific_metrics['age'] == age]
        age_models = {}
        for _, row in age_metrics.iterrows():
            if row['display_model'] in ['Best DeepPySR', 'Best PySR'] and row['display_model'] not in age_models:
                age_models[row['display_model']] = consistent_config.copy()

        if 'age' in X_long.columns:
            mask = X_long['age'] == age
            X_age = X_long[mask].drop(columns=['age'])
            y_age = y_long[mask]
        else:
            _, X_age, y_age = _load_age(age, target)

        if len(X_age) > 0 and age_models:
            age_output_root = os.path.join(output_root, target, 'age-specific', f'age{age}')
            if not os.path.exists(age_output_root):
                run_convergence_comparison(X_age, y_age, age_models, age_output_root, name=f'Age: {age} ({target})')
        else:
            print(f"No data or no best models found for age {age}")


def main():
    output_root = os.path.join(current_dir, './convergence_results_insulin')
    for target in TARGETS:
        _run_for_target(target, output_root)


if __name__ == "__main__":
    main()
