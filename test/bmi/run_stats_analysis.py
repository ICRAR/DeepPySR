import os
import sys
import pandas as pd
import statsmodels.api as sm
import sympy
from sympy import sympify

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(current_dir)

from bmi_utils import load_bmi_agg_data

def get_features_from_formula(formula_str):
    """Extract variable names from a sympy-compatible formula string."""
    # Common SR operators that are not features
    operators = {'exp', 'log', 'sin', 'cos', 'tan', 'sqrt', 'abs', 'cond', 'sign'}
    
    # Handle 'cond' which might not be in standard sympy
    # For extraction purposes, we can just replace it or let sympify handle it if we define it
    # Actually, sympify might fail on 'cond' if it's not defined.
    # Let's do a simple extraction based on tokens that look like variables
    
    try:
        # Pre-process 'cond(a, b)' to something sympy likes if needed
        # but here we just want symbols.
        expr = sympify(formula_str, locals={'cond': lambda a, b: a}) # dummy cond
        symbols = [str(s) for s in expr.free_symbols]
        return symbols
    except Exception as e:
        print(f"Error parsing formula: {formula_str}. Error: {e}")
        # Fallback: regex or manual tokenization if sympy fails
        import re
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', formula_str)
        features = set()
        for t in tokens:
            if t not in operators and not t.lower() in ['true', 'false']:
                features.add(t)
        return list(features)

def run_stats(X, y, features, title, output_file=None):
    print(f"\n{'='*70}")
    print(f"STATS ANALYSIS: {title}")
    print(f"{'='*70}")
    
    # Filter features that exist in X
    available_features = [f for f in features if f in X.columns]
    missing_features = [f for f in features if f not in X.columns]
    
    if missing_features:
        print(f"Warning: Missing features in data: {missing_features}")
    
    if not available_features:
        print("No available features for analysis.")
        return
    
    X_subset = X[available_features]
    X_subset = sm.add_constant(X_subset)
    
    try:
        model = sm.OLS(y, X_subset).fit()
        summary = model.summary()
        print(summary)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"STATS ANALYSIS: {title}\n")
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

def main():
    output_dir = os.path.join(current_dir, 'results_bmi_stats')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    deep_results_dir = os.path.join(current_dir, 'results_bmi_deep')
    if not os.path.exists(deep_results_dir):
        print(f"ERROR: Deep results directory not found at {deep_results_dir}")
        return

    # 1. Longitudinal Model
    print("Processing Longitudinal Model...")
    long_rel_file = os.path.join(deep_results_dir, 'longitudinal', 'relationships.csv')
    if os.path.exists(long_rel_file):
        rel_df = pd.read_csv(long_rel_file)
        # Filter for layer 1 and target 'y'
        layer1 = rel_df[(rel_df['layer'] == 1)]
        if not layer1.empty:
            formula = layer1.iloc[0]['formula']
            features = get_features_from_formula(formula)
            print(f"Extracted features: {features}")

            _, X_all, y_all = load_bmi_agg_data()
            long_output = os.path.join(output_dir, 'stats_longitudinal.txt')
            run_stats(X_all, y_all, features, "Longitudinal", output_file=long_output)
        else:
            print("No layer 1 formula found for longitudinal.")
    else:
        print(f"Longitudinal relationships file not found: {long_rel_file}")

    # 2. Age-Specific Models
    print("\nProcessing Age-Specific Models...")
    age_specific_dir = os.path.join(deep_results_dir, 'age-specific')
    if os.path.exists(age_specific_dir):
        for age_folder in sorted(os.listdir(age_specific_dir)):
            if not age_folder.startswith('age'):
                continue
            
            try:
                age = int(age_folder.replace('age', ''))
            except ValueError:
                continue

            print(f"\nProcessing Age: {age}")
            age_rel_file = os.path.join(age_specific_dir, age_folder, 'relationships.csv')
            
            if os.path.exists(age_rel_file):
                rel_df = pd.read_csv(age_rel_file)
                layer1 = rel_df[(rel_df['layer'] == 1)]
                if not layer1.empty:
                    formula = layer1.iloc[0]['formula']
                    features = get_features_from_formula(formula)
                    print(f"Age {age}: Extracted features: {features}")

                    # Load/Filter data for specific age
                    _, X_age, y_age = load_bmi_agg_data(age=age)

                    if len(X_age) > 0:
                        age_output = os.path.join(output_dir, f'stats_age_{age}.txt')
                        run_stats(X_age, y_age, features, f"Age Specific - Age {age}", output_file=age_output)
                    else:
                        print(f"No data found for age {age}")
                else:
                    print(f"No layer 1 formula found for age {age}")
            else:
                print(f"Relationships file not found for age {age}: {age_rel_file}")
    else:
        print(f"Age-specific deep results directory not found: {age_specific_dir}")

if __name__ == "__main__":
    main()
