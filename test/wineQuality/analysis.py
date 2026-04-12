import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import glob
import sys
import re
import sympy as sp

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Add test/ and test/bmi to path to import load_bmi_agg_data
current_dir = os.path.dirname(os.path.abspath(__file__))
if not current_dir:
    current_dir = "."
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

from wine_utils import load_wine_data

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
        if key == (1.0, 0.001) and 'pareto_r2_weight' not in pd.read_csv(files[0]).columns:
            return results[key][1], results[key][2], results[key][3]

    # Return the full results dictionary
    # For each key, we only need (formula, complexity, metrics)
    return {k: (v[1], v[2], v[3]) for k, v in results.items()}

def process_results():
    wine_types = ['red', 'white']
    all_data = []
    for wine_type in wine_types:
        base_dir = os.path.join(current_dir, f"results_{wine_type}_all")

        df = load_wine_data(wine_type)
        X = df.drop(columns=['quality'])
        y = df['quality']
        # Baselines (including KAN/KANSym)
        baselines_dir = os.path.join(base_dir, "baselines")
        if os.path.exists(baselines_dir):
            for model_name in os.listdir(baselines_dir):
                model_path = os.path.join(baselines_dir, model_name)
                if not os.path.isdir(model_path):
                    continue

                pred_file = os.path.join(model_path, "predictions.csv")
                if os.path.exists(pred_file):
                    df_pred = pd.read_csv(pred_file)
                    if model_name.lower() == 'kan':
                        # KAN
                        r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                        all_data.append(['KAN', wine_type, r2, rmse, mae, np.nan, ""])

                        # KANSym
                        if 'y_pred_kansym' in df_pred.columns:
                            # For KANSym, we need formula and complexity
                            # The user wants us to check all formulas and pick the best one
                            formula, complexity, metrics = get_best_formula_from_raw(model_path, X, y, prefix='formulas_fold')
                            r2, rmse, mae = metrics

                            if not formula:
                                r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred_kansym'])

                            all_data.append(['KANSym', wine_type, r2, rmse, mae, complexity, formula])
                    else:
                        # Other baselines
                        r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                        all_data.append([model_name, wine_type, r2, rmse, mae, np.nan, ""])

        # DeepPySR
        deeppysr_dir = os.path.join(base_dir, "deeppysr")
        if os.path.exists(deeppysr_dir):
            for variant in os.listdir(deeppysr_dir):
                v_path = os.path.join(deeppysr_dir, variant)
                if not os.path.isdir(v_path): continue

                res = get_best_formula_from_raw(v_path, X, y)

                if isinstance(res, dict):
                    for (r2w, lamb), (formula, complexity, metrics) in res.items():
                        r2, rmse, mae = metrics
                        model_name = f"{variant}_r2w{r2w}_L{lamb}"
                        all_data.append([model_name, wine_type, r2, rmse, mae, complexity, formula])
                else:
                    formula, complexity, metrics = res
                    r2, rmse, mae = metrics
                    if not formula:
                        pred_file = os.path.join(v_path, "predictions.csv")
                        if os.path.exists(pred_file):
                            df_pred = pd.read_csv(pred_file)
                            r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                    all_data.append([variant, wine_type, r2, rmse, mae, complexity, formula])



    # Create DataFrame and save
    result_df = pd.DataFrame(all_data, columns=['model', 'wine type', 'r2', 'rmse', 'mae', 'complexity', 'formula'])
    # Clip r2 to 0
    result_df['r2'] = result_df['r2'].clip(lower=0)
    result_df.to_csv(os.path.join(current_dir, "aggregated_results.csv"), index=False)
    print(f"Results saved to {os.path.join(current_dir, 'aggregated_results.csv')}")
    return result_df

def save_results(df):
    """
    1. Plot r2, rmse, mae for the models, along the age.
    """
    # Clip r2 to 0 for plotting
    df = df.copy()
    df['r2'] = df['r2'].clip(lower=0)

    metrics = ['r2', 'rmse', 'mae']
    types = ['red', 'white']

    selected_data = []
    interpretable_formulas = []

    for t in types:
        type_df = df[df['wine type'] == t]
        if type_df.empty:
            continue

        # DeepPySR variants
        deeppysr_df = type_df[type_df['model'].str.contains('fullsr|stdsr|v2fullsr', na=False)]
        if not deeppysr_df.empty:
            best_deeppysr = deeppysr_df.loc[deeppysr_df['r2'].idxmax()].copy()
            best_deeppysr['display_model'] = 'Best DeepPySR'
            selected_data.append(best_deeppysr)

            interp_deeppysr_df = deeppysr_df[deeppysr_df['complexity'] < 25]
            if not interp_deeppysr_df.empty:
                interp_deeppysr = interp_deeppysr_df.loc[interp_deeppysr_df['r2'].idxmax()].copy()
                interp_deeppysr['display_model'] = 'Interpretable DeepPySR'
                selected_data.append(interp_deeppysr)
                interpretable_formulas.append({
                    'type': t, 'model': interp_deeppysr['model'],
                    'formula': interp_deeppysr['formula'], 'r2': interp_deeppysr['r2'], 'complexity': interp_deeppysr['complexity']
                })


        # KAN and KANSym
        for m in ['KAN', 'KANSym']:
            m_df = type_df[type_df['model'] == m]
            if not m_df.empty:
                m_row = m_df.iloc[0].copy()
                m_row['display_model'] = m
                selected_data.append(m_row)

        # Other baselines (ElasticNet, ExtraTrees, MLP, RandomForest, XGBoost)
        baselines = ['ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']
        for b in baselines:
            b_df = type_df[type_df['model'] == b]
            if not b_df.empty:
                b_row = b_df.iloc[0].copy()
                b_row['display_model'] = b
                selected_data.append(b_row)

    plot_df = pd.DataFrame(selected_data)

    # Save the plot data for the best models to CSV
    plot_csv_path = os.path.join(current_dir, 'wine_best_models_metrics.csv')
    plot_df.to_csv(plot_csv_path, index=False)
    print(f"Best models plot data saved to {plot_csv_path}")

    # Print interpretable DeepPySR formulas
    print("\n--- Interpretable DeepPySR Formulas (Complexity < 30) ---")
    interp_df = pd.DataFrame(interpretable_formulas)
    print(interp_df.to_string(index=False))
    interp_csv_path = os.path.join(current_dir, 'interpretable_deeppysr_formulas.csv')
    interp_df.to_csv(interp_csv_path, index=False)


def aggregate_feature_importance():
    """
    Aggregate feature importance for ElasticNet, ExtraTrees, RandomForest, XGBoost, KAN.
    Exclude DeepPySR, PySR, MLP.
    Average across folds, percentage it.
    """
    types = ['red', 'white']
    importance_data = []
    for type in types:
        base_dir = os.path.join(current_dir, f"results_{type}_all")


        # Helper to process importance file
        def process_importance(path, model_name, type):
            if os.path.exists(path):
                df_imp = pd.read_csv(path)
                # Ensure it has 'feature' and 'importance' columns
                if 'feature' in df_imp.columns and 'importance' in df_imp.columns:
                    # Percentage it
                    total = df_imp['importance'].sum()
                    if total > 0:
                        df_imp['importance_pct'] = (df_imp['importance'] / total) * 100
                    else:
                        df_imp['importance_pct'] = 0

                    for _, row in df_imp.iterrows():
                        importance_data.append({
                            'model': model_name,
                            'type': type,
                            'variable': row['feature'],
                            'weight': row['importance_pct']
                        })

        if os.path.exists(base_dir):
            baselines_dir = os.path.join(base_dir, "baselines")
            if os.path.exists(baselines_dir):
                for m in os.listdir(baselines_dir):
                    if m in ['ElasticNet', 'ExtraTrees', 'RandomForest', 'XGBoost', 'KAN']:
                        imp_file = os.path.join(baselines_dir, m, "feature_importance.csv")
                        process_importance(imp_file, m, type)

        imp_df = pd.DataFrame(importance_data)
        imp_df.to_csv(os.path.join(base_dir,"feature_importance_aggregated.csv"), index=False)
        print("Feature importance aggregated to feature_importance_aggregated.csv")

        # Grouped bar plot for all models comparison
        if not imp_df.empty:
            # Average importance across ages and types per model/variable
            agg_imp = imp_df.groupby(['model', 'variable'])['weight'].mean().reset_index()

            # Find top 15 features based on average across all models
            top_features = agg_imp.groupby('variable')['weight'].mean().sort_values(ascending=False).head(15).index

            plot_df = agg_imp[agg_imp['variable'].isin(top_features)].copy()
            plot_df['variable'] = pd.Categorical(plot_df['variable'], categories=top_features, ordered=True)

            plt.figure(figsize=(14, 10))
            sns.barplot(data=plot_df, x='weight', y='variable', hue='model', palette="bright")

            plt.title('Feature Importance Comparison across Models', fontsize=22, fontweight='bold', pad=20)
            plt.xlabel('Average Percentage Importance (%)', fontsize=18)
            plt.ylabel('Feature', fontsize=18)
            plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
            plt.tick_params(labelsize=14)

            plt.tight_layout()
            plot_path = os.path.join(base_dir, "feature_importance_by_model.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Combined feature importance plot saved to {plot_path}")

if __name__ == "__main__":
    # process_results: aggregate all the results from the 5 fold cv, select one formula among the 5 which achieves the highest r2.\
    # The r2 is calculated by applying this formula on the entire dataset, not the fold.

    df = process_results()

    # plot_results:
    # longitudinal: for deeppysr, pysr, kansym that provides equations, I handled it specially.
    # I got the equations from the bmi_aggregated_results.csv. Use the equation to predict the entire dataset (7 ages).
    # The best equations would be ploted and saved in the results.
    # For other models, we do it by assessing the metrics on the predictions.csv

    save_results(df)
    aggregate_feature_importance()
