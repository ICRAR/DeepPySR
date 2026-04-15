import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import glob
import re
import sympy as sp

def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan
    r2 = r2_score(y_true, y_pred)
    # R2 should be no smaller than 0
    r2 = max(0, r2)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

def calculate_complexity(formula_str):
    """
    Calculate complexity as the number of operands and operators.
    Operands: variables and constants.
    Operators: +, -, *, /, sin, cos, exp, etc.
    """
    if not formula_str or pd.isna(formula_str):
        return 0

    # Tokenize formula
    # Identify variables, operators, function names, and numbers

    # Operators and function names
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|\d+\.?\d*|[\+\-\*\/\^]', str(formula_str))

    # Complexity is just the number of tokens found
    return len(tokens)

def map_variable_names(formula_str, feature_names):
    """
    Map x0, x1... or x_0, x_1... back to original feature names.
    Use regex to avoid partial matches (e.g., x1 matching x10).
    """
    if not formula_str or pd.isna(formula_str):
        return formula_str

    # Sort indices in descending order to avoid x10 being partially replaced by x1
    indices = sorted(range(len(feature_names)), reverse=True)

    mapped_formula = str(formula_str)
    for i in indices:
        name = feature_names[i]
        # Replace x_i and xi
        mapped_formula = re.sub(rf'\bx_{i}\b', name, mapped_formula)
        mapped_formula = re.sub(rf'\bx{i}\b', name, mapped_formula)

    return mapped_formula

def evaluate_formula(formula_str, X):
    """
    Evaluate a symbolic formula (PySR, DeepPySR, or KAN) using SymPy.
    Supports both indexed variables (x0, x1...) and raw feature names.
    """
    if not formula_str or pd.isna(formula_str):
        return np.zeros(len(X))

    # Identify variables in formula
    try:
        # Pre-process some KAN-style or other common functional names to be SymPy compatible if needed
        # For now, let's try standard sp.sympify

        # local_dict for sympify to handle some common functions if they are not standard
        # PySR sometimes uses 'inv(x)', 'neg(x)', 'square(x)', 'cube(x)'
        custom_functions = {
            'log': lambda x: sp.log(x),
            'inv': lambda x: 1/x,
            'neg': lambda x: -x,
            'square': lambda x: x**2,
            'cube': lambda x: x**3,
            'add': lambda x, y: x + y,
            'sub': lambda x, y: x - y,
            'mul': lambda x, y: x * y,
            'div': lambda x, y: x / y,
            'power': lambda x, y: x**y,
            'cond': lambda x, y: sp.Piecewise((y, x > 0), (0, True))
        }

        expr = sp.sympify(str(formula_str), locals=custom_functions)

        # Feature names from X
        feature_names = list(X.columns) if hasattr(X, 'columns') else []

        # Mapping for indexed variables if they exist in formula
        # We replace x0, x1... and x_0, x_1... with the actual column names
        subs_dict = {}
        for i, name in enumerate(feature_names):
            subs_dict[sp.Symbol(f"x{i}")] = sp.Symbol(name)
            subs_dict[sp.Symbol(f"x_{i}")] = sp.Symbol(name)

        expr = expr.xreplace(subs_dict)

        # Now evaluate the expression with X's data
        # We use lambdify for performance
        variables = sorted([str(s) for s in expr.free_symbols])

        # Prepare input data for lambdify
        input_data = []
        for var in variables:
            if var in X.columns:
                input_data.append(X[var].values)
            else:
                # If variable is not in X, it might be an indexed variable that wasn't replaced (shouldn't happen with raw names requirement)
                # or it's a constant that was parsed as a symbol
                input_data.append(np.zeros(len(X)))

        f_lambdified = sp.lambdify(variables, expr, modules=['numpy'])

        if not variables:
            # Constant formula
            y_pred = float(expr)
        else:
            with np.errstate(all='ignore'):
                y_pred = f_lambdified(*input_data)

        # Ensure it's a numpy array of correct length
        if np.isscalar(y_pred):
            y_pred = np.full(len(X), y_pred)

        return np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)

    except Exception as e:
        # print(f"Error evaluating formula {formula_str}: {e}")
        return np.zeros(len(X))

def get_best_formula_from_raw(folder_path, X, y_true, prefix='relationships_fold'):
    """
    Find the best formula by evaluating all fold-specific formulas on raw data.
    Works for DeepPySR, PySR (prefix='relationships_fold') and KAN (prefix='formulas_fold').
    Returns a dictionary of (r2w, lambda): (formula, complexity, metrics) for DeepPySR grid search,
    or just a single (formula, complexity, metrics) if no grid search info is found.
    """
    results = {} # {(r2w, lambda): (best_r2, best_formula, best_complexity, best_metrics)}
    overall_best = (-float('inf'), "", np.nan, (np.nan, np.nan, np.nan))

    feature_names = list(X.columns) if hasattr(X, 'columns') else []

    pattern = os.path.join(folder_path, f"{prefix}*.csv")
    files = glob.glob(pattern)

    # Also check for non-prefixed relationships.csv or formulas.csv if no folds found
    if not files:
        if 'relationships' in prefix:
            alt = os.path.join(folder_path, "relationships.csv")
        else:
            alt = os.path.join(folder_path, "formulas.csv")
        if os.path.exists(alt):
            files = [alt]

    for f in files:
        try:
            df = pd.read_csv(f)
            if 'formula' in df.columns:
                for idx, row in df.iterrows():
                    formula = str(row['formula'])
                    complexity = calculate_complexity(formula)

                    # Identify grid search parameters
                    r2w = row.get('pareto_r2_weight', 1.0)
                    lamb = row.get('pareto_lambda', 0.001)
                    key = (r2w, lamb)

                    y_pred = evaluate_formula(formula, X)
                    r2, rmse, mae = calculate_metrics(y_true, y_pred)

                    if not np.isnan(r2):
                        # Update per-parameter best
                        if key not in results or r2 > results[key][0]:
                            results[key] = (r2, map_variable_names(formula, feature_names), complexity, (r2, rmse, mae))

                        # Update overall best
                        if r2 > overall_best[0]:
                            overall_best = (r2, map_variable_names(formula, feature_names), complexity, (r2, rmse, mae))
        except Exception as e:
            continue

    if not results:
        return overall_best[1], overall_best[2], overall_best[3]

    # If there's only one parameter set and it's default, just return it directly for backward compatibility
    if len(results) == 1:
        key = list(results.keys())[0]
        # check if pareto columns exist in the first file
        first_df = pd.read_csv(files[0])
        if key == (1.0, 0.001) and 'pareto_r2_weight' not in first_df.columns:
            return results[key][1], results[key][2], results[key][3]

    # Return the full results dictionary
    # For each key, we only need (formula, complexity, metrics)
    return {k: (v[1], v[2], v[3]) for k, v in results.items()}
