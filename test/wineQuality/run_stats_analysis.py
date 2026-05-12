import os
import sys
import pandas as pd
import statsmodels.api as sm
from sympy import sympify

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(current_dir)

from wine_utils import load_wine_data

def get_features_from_formula(formula_str):
    """Extract variable names from a sympy-compatible formula string."""
    operators = {'exp', 'log', 'sin', 'cos', 'tan', 'sqrt', 'abs', 'cond', 'sign'}
    try:
        expr = sympify(formula_str, locals={'cond': lambda a, b: a})
        symbols = [str(s) for s in expr.free_symbols]
        return symbols
    except Exception as e:
        print(f"Error parsing formula: {formula_str}. Error: {e}")
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
    output_dir = os.path.join(current_dir, 'results_wine_stats')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    interpretable_formulas_file = os.path.join(current_dir, 'interpretable_deeppysr_formulas.csv')
    if not os.path.exists(interpretable_formulas_file):
        print(f"ERROR: Interpretable formulas file not found at {interpretable_formulas_file}")
        return

    df_formulas = pd.read_csv(interpretable_formulas_file)

    for wine_type in ['red', 'white']:
        print(f"\nProcessing {wine_type} wine...")
        
        type_formulas = df_formulas[df_formulas['type'] == wine_type]
        if not type_formulas.empty:
            formula = type_formulas.iloc[0]['formula']
            features = get_features_from_formula(formula)
            print(f"Extracted features: {features}")
            
            df = load_wine_data(wine_type)
            X = df.drop(columns=['quality'])
            y = df['quality']
            
            output_file = os.path.join(output_dir, f'stats_{wine_type}.txt')
            run_stats(X, y, features, f"{wine_type.capitalize()} Wine", output_file=output_file)
        else:
            print(f"No formula found for {wine_type} wine in CSV")

if __name__ == "__main__":
    main()
