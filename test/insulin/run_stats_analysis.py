import os
import sys
import pandas as pd
import statsmodels.api as sm
import sympy
from sympy import sympify

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

from data_utils import load_data, load_data_longitudinal

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


def get_features_from_formula(formula_str):
    operators = {'exp', 'log', 'sin', 'cos', 'tan', 'sqrt', 'abs', 'cond', 'sign'}
    try:
        expr = sympify(formula_str, locals={'cond': lambda a, b: a})
        return [str(s) for s in expr.free_symbols]
    except Exception as e:
        print(f"Error parsing formula: {formula_str}. Error: {e}")
        import re
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', formula_str)
        return list({t for t in tokens if t not in operators and t.lower() not in ['true', 'false']})


def run_stats(X, y, features, title, formula=None, metrics=None, output_file=None):
    print(f"\n{'='*70}")
    print(f"STATS ANALYSIS: {title}")
    if formula:
        print(f"Formula: {formula}")
    if metrics:
        print(f"Metrics: {', '.join(f'{k}: {v}' for k, v in metrics.items())}")
    print('='*70)

    available_features = [f for f in features if f in X.columns]
    missing_features = [f for f in features if f not in X.columns]
    if missing_features:
        print(f"Warning: Missing features in data: {missing_features}")
    if not available_features:
        print("No available features for analysis.")
        return

    X_subset = sm.add_constant(X[available_features])
    try:
        model = sm.OLS(y, X_subset).fit()
        summary = model.summary()
        print(summary)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"STATS ANALYSIS: {title}\n")
                if formula:
                    f.write(f"Formula: {formula}\n")
                if metrics:
                    f.write(f"Metrics: {', '.join(f'{k}: {v}' for k, v in metrics.items())}\n")
                f.write(f"Features: {available_features}\n")
                if missing_features:
                    f.write(f"Missing Features: {missing_features}\n")
                f.write("="*70 + "\n")
                f.write(summary.as_text())
            print(f"Summary saved to {output_file}")
        return model
    except Exception as e:
        print(f"Error fitting OLS: {e}")
        return None


def _run_for_target(target, output_dir, deep_results_dir):
    print(f"\n{'#'*70}")
    print(f"TARGET: {target.upper()}")
    print('#'*70)

    # Longitudinal
    print("\nProcessing Longitudinal Model...")
    long_rel_file = os.path.join(deep_results_dir, target, 'longitudinal', 'relationships.csv')
    if os.path.exists(long_rel_file):
        rel_df = pd.read_csv(long_rel_file)
        layer1 = rel_df[rel_df['layer'] == 1]
        if not layer1.empty:
            layer1 = layer1.sort_values(by=['r2', 'f1'], ascending=False)
            best_row = layer1.iloc[0]
            formula = best_row['formula']
            metrics = {'r2': best_row['r2'], 'f1': best_row['f1']}
            features = get_features_from_formula(formula)
            print(f"Extracted features: {features}")
            _, X_all, y_all = _load_longitudinal(target)
            run_stats(X_all, y_all, features, f"Longitudinal ({target})", formula=formula, metrics=metrics,
                      output_file=os.path.join(output_dir, f'stats_longitudinal_{target}.txt'))
        else:
            print(f"No layer 1 formula found for longitudinal ({target}).")
    else:
        print(f"Longitudinal relationships file not found: {long_rel_file}")

    # Age-Specific
    print(f"\nProcessing Age-Specific Models ({target})...")
    age_specific_dir = os.path.join(deep_results_dir, target, 'age-specific')
    if os.path.exists(age_specific_dir):
        for age_folder in sorted(os.listdir(age_specific_dir)):
            if not age_folder.startswith('age'):
                continue
            try:
                age = int(age_folder.replace('age', ''))
            except ValueError:
                continue

            print(f"\nProcessing Age: {age} ({target})")
            age_rel_file = os.path.join(age_specific_dir, age_folder, 'relationships.csv')
            if os.path.exists(age_rel_file):
                rel_df = pd.read_csv(age_rel_file)
                layer1 = rel_df[rel_df['layer'] == 1]
                if not layer1.empty:
                    layer1 = layer1.sort_values(by=['r2', 'f1'], ascending=False)
                    best_row = layer1.iloc[0]
                    formula = best_row['formula']
                    metrics = {'r2': best_row['r2'], 'f1': best_row['f1']}
                    features = get_features_from_formula(formula)
                    _, X_age, y_age = _load_age(age, target)
                    if len(X_age) > 0:
                        run_stats(X_age, y_age, features, f"Age Specific — Age {age} ({target})",
                                  formula=formula, metrics=metrics,
                                  output_file=os.path.join(output_dir, f'stats_age_{age}_{target}.txt'))
                    else:
                        print(f"No data found for age {age}")
                else:
                    print(f"No layer 1 formula found for age {age}")
            else:
                print(f"Relationships file not found: {age_rel_file}")
    else:
        print(f"Age-specific deep results directory not found: {age_specific_dir}")


def main():
    output_dir = os.path.join(current_dir, 'results_insulin_stats')
    os.makedirs(output_dir, exist_ok=True)

    deep_results_dir = os.path.join(current_dir, 'results_insulin_deep')
    if not os.path.exists(deep_results_dir):
        print(f"ERROR: Deep results directory not found at {deep_results_dir}")
        return

    for target in TARGETS:
        _run_for_target(target, output_dir, deep_results_dir)


if __name__ == "__main__":
    main()
