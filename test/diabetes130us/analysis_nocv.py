import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import glob
import sys
import re
import sympy as sp

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Add test/ and test/diabetes130us to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if not current_dir:
    current_dir = "."
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
sys.path.append(current_dir)

from diab130_utils import load_and_clean_data

def calculate_metrics(y_true, y_pred, y_prob=None):
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan
    acc = accuracy_score(y_true, y_pred)
    
    unique_y = np.unique(y_true)
    is_multiclass = len(unique_y) > 2
    avg = 'macro' if is_multiclass else 'binary'
    
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    auc = np.nan
    if y_prob is not None:
        try:
            if is_multiclass:
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            else:
                auc = roc_auc_score(y_true, y_prob)
        except:
            pass
    return acc, f1, auc

def calculate_complexity(formula_str):
    if not formula_str or pd.isna(formula_str):
        return 0
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|\d+\.?\d*|[\+\-\*\/\^]', str(formula_str))
    return len(tokens)

def map_variable_names(formula_str, feature_names):
    if not formula_str or pd.isna(formula_str):
        return formula_str
    indices = sorted(range(len(feature_names)), reverse=True)
    mapped_formula = str(formula_str)
    for i in indices:
        name = feature_names[i]
        mapped_formula = re.sub(rf'\bx_{i}\b', name, mapped_formula)
        mapped_formula = re.sub(rf'\bx{i}\b', name, mapped_formula)
    return mapped_formula

def evaluate_formula(formula_str, X):
    if not formula_str or pd.isna(formula_str):
        return np.zeros(len(X))
    try:
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
        }
        expr = sp.sympify(str(formula_str), locals=custom_functions)
        feature_names = list(X.columns)
        subs_dict = {}
        for i, name in enumerate(feature_names):
            subs_dict[sp.Symbol(f"x{i}")] = sp.Symbol(name)
            subs_dict[sp.Symbol(f"x_{i}")] = sp.Symbol(name)
        expr = expr.xreplace(subs_dict)
        variables = sorted([str(s) for s in expr.free_symbols])
        input_data = []
        for var in variables:
            if var in X.columns:
                input_data.append(X[var].values)
            else:
                input_data.append(np.zeros(len(X)))
        f_lambdified = sp.lambdify(variables, expr, modules=['numpy'])
        if not variables:
            y_pred = float(expr)
        else:
            with np.errstate(all='ignore'):
                y_pred = f_lambdified(*input_data)
        if np.isscalar(y_pred):
            y_pred = np.full(len(X), y_pred)
        return np.nan_to_num(y_pred, nan=0.0)
    except Exception:
        return np.zeros(len(X))

def get_best_formula_from_raw(folder_path, X, y_true, prefix='relationships_fold'):
    """
    Find the best formula by evaluating all fold-specific formulas on raw data.
    Returns a dictionary of (r2w, lambda): (formula, complexity, metrics) for DeepPySR grid search,
    or just a single (formula, complexity, metrics) if no grid search info is found.
    """
    results = {} # {(r2w, lambda): (best_acc, best_formula, best_complexity, best_metrics)}
    overall_best = (-float('inf'), "", np.nan, (np.nan, np.nan, np.nan))
    
    feature_names = list(X.columns)
    pattern = os.path.join(folder_path, f"{prefix}*.csv")
    files = glob.glob(pattern)
    if not files:
        alt = os.path.join(folder_path, "relationships.csv" if 'relationships' in prefix else "formulas.csv")
        if os.path.exists(alt): files = [alt]
        
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'formula' in df.columns:
                for _, row in df.iterrows():
                    formula = str(row['formula'])
                    complexity = calculate_complexity(formula)
                    
                    # Identify grid search parameters
                    r2w = row.get('pareto_r2_weight', 1.0)
                    lamb = row.get('pareto_lambda', 0.001)
                    key = (r2w, lamb)
                    
                    y_pred_raw = evaluate_formula(formula, X)
                    y_pred = np.round(y_pred_raw).astype(int)
                    y_pred = np.clip(y_pred, 0, 2)
                    acc, f1, auc = calculate_metrics(y_true, y_pred)
                    
                    if not np.isnan(acc):
                        # Update per-parameter best
                        if key not in results or acc > results[key][0]:
                            results[key] = (acc, map_variable_names(formula, feature_names), complexity, (acc, f1, auc))
                        
                        # Update overall best
                        if acc > overall_best[0]:
                            overall_best = (acc, map_variable_names(formula, feature_names), complexity, (acc, f1, auc))
        except Exception:
            continue
            
    if not results:
        return overall_best[1], overall_best[2], overall_best[3]
    
    if len(results) == 1:
        key = list(results.keys())[0]
        if key == (1.0, 0.001) and 'pareto_r2_weight' not in pd.read_csv(files[0]).columns:
             return results[key][1], results[key][2], results[key][3]

    return {k: (v[1], v[2], v[3]) for k, v in results.items()}

def process_results():
    base_dir = os.path.join(current_dir, "results_diab130_nocv")
    all_data = []
    
    file_path = '/home/00101787/Projects/DeepPySR/test_data/Health/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv'
    df_full = load_and_clean_data(file_path)
    X_full = df_full.drop(columns=['encounter_id', 'patient_nbr', 'readmitted'], errors='ignore')
    y_full = df_full['readmitted']

    sub_dirs = ['baselines', 'deeppysr', 'pysr']
    for sd in sub_dirs:
        sd_path = os.path.join(base_dir, sd)
        if not os.path.exists(sd_path): continue
        for model_folder in os.listdir(sd_path):
            m_path = os.path.join(sd_path, model_folder)
            if not os.path.isdir(m_path): continue
            pred_file = os.path.join(m_path, "predictions.csv")
            if not os.path.exists(pred_file):
                pred_file = os.path.join(m_path, "predictions_foldnocv.csv")
            if os.path.exists(pred_file):
                df_pred = pd.read_csv(pred_file)
                y_prob = None
                if 'y_prob' in df_pred.columns:
                    y_prob = df_pred['y_prob']
                elif 'y_prob_0' in df_pred.columns:
                    # Collect all y_prob_i columns
                    prob_cols = sorted([c for c in df_pred.columns if c.startswith('y_prob_')])
                    y_prob = df_pred[prob_cols].values

                if sd == 'baselines' and model_folder.lower() == 'kan':
                    acc, f1, auc = calculate_metrics(df_pred['y_true'], df_pred['y_pred'], y_prob)
                    all_data.append([model_folder, acc, f1, auc, np.nan, ""])
                    if 'y_pred_sym' in df_pred.columns:
                        formula, complexity, metrics = get_best_formula_from_raw(m_path, X_full, y_full, prefix='formulas_fold')
                        acc, f1, auc = metrics
                        if not formula:
                            acc, f1, auc = calculate_metrics(df_pred['y_true'], np.clip(np.round(df_pred['y_pred_sym']), 0, 2))
                        all_data.append(['KANSym', acc, f1, auc, complexity, formula])
                elif sd in ['deeppysr', 'pysr']:
                    res = get_best_formula_from_raw(m_path, X_full, y_full)
                    if isinstance(res, dict):
                        for (r2w, lamb), (formula, complexity, metrics) in res.items():
                            acc, f1, auc = metrics
                            model_name = f"{model_folder}_r2w{r2w}_L{lamb}"
                            all_data.append([model_name, acc, f1, auc, complexity, formula])
                    else:
                        formula, complexity, metrics = res
                        acc, f1, auc = metrics
                        if not formula:
                            acc, f1, auc = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                        all_data.append([model_folder, acc, f1, auc, complexity, formula])
                else:
                    acc, f1, auc = calculate_metrics(df_pred['y_true'], df_pred['y_pred'], y_prob)
                    all_data.append([model_folder, acc, f1, auc, np.nan, ""])

    result_df = pd.DataFrame(all_data, columns=['model', 'accuracy', 'f1', 'auc', 'complexity', 'formula'])
    result_df.to_csv(os.path.join(base_dir, "diab130_aggregated_results.csv"), index=False)
    print(f"Results saved to {os.path.join(base_dir, 'diab130_aggregated_results.csv')}")
    return result_df

if __name__ == "__main__":
    df = process_results()
    print(df.sort_values(by='accuracy', ascending=False).head(20))
