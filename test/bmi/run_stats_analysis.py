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

    interpretable_formulas_file = os.path.join(current_dir, 'results_bmi_all', 'interpretable_deeppysr_formulas.csv')
    if not os.path.exists(interpretable_formulas_file):
        print(f"ERROR: Interpretable formulas file not found at {interpretable_formulas_file}")
        return

    df_formulas = pd.read_csv(interpretable_formulas_file)

    # 1. Longitudinal Model
    print("Processing Longitudinal Model...")
    long_formulas = df_formulas[df_formulas['type'] == 'longitudinal']
    if not long_formulas.empty:
        # All longitudinal entries usually have the same formula, take the first one
        formula = long_formulas.iloc[0]['formula']
        features = get_features_from_formula(formula)
        print(f"Extracted features: {features}")

        _, X_all, y_all = load_bmi_agg_data()
        long_output = os.path.join(output_dir, 'stats_longitudinal.txt')
        run_stats(X_all, y_all, features, "Longitudinal", output_file=long_output)
    else:
        print("No longitudinal formula found in CSV.")

    # 2. Age-Specific Models
    print("\nProcessing Age-Specific Models...")
    age_specific_formulas = df_formulas[df_formulas['type'] == 'age-specific']
    if not age_specific_formulas.empty:
        for _, row in age_specific_formulas.sort_values('age').iterrows():
            age = int(row['age'])
            formula = row['formula']
            
            print(f"\nProcessing Age: {age}")
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
        print("No age-specific formulas found in CSV.")

if __name__ == "__main__":
    main()
