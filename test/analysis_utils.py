import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold
import glob
import re
import sympy as sp
from scipy.stats import wilcoxon as scipy_wilcoxon

def calculate_metrics(y_true, y_pred, y_prob=None, task='regression'):
    if len(y_true) == 0:
        if task == 'regression':
            return np.nan, np.nan, np.nan
        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan

    # Clean NaNs and Infs for metrics calculation to avoid ValueError in sklearn
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
    if y_prob is not None:
        y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)

    if task == 'regression':
        r2 = r2_score(y_true, y_pred)
        # R2 should be no smaller than 0
        r2 = max(0, r2)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return r2, rmse, mae
    else:
        # For classification, ensure y_pred is discrete (integer classes)
        # Symbolic outputs are often continuous and need rounding/clipping
        if not np.issubdtype(y_pred.dtype, np.integer):
            # Clip to the range of y_true before/after rounding
            y_min, y_max = np.min(y_true), np.max(y_true)
            y_pred = np.clip(np.round(y_pred), y_min, y_max).astype(int)
        else:
            # Even if it's already integer, ensure it's within y_true's range
            y_min, y_max = np.min(y_true), np.max(y_true)
            y_pred = np.clip(y_pred, y_min, y_max)

        # Check if classification type
        unique_y_true = np.unique(y_true)
        unique_y_pred = np.unique(y_pred)
        all_labels = np.unique(np.concatenate([unique_y_true, unique_y_pred]))

        # Consider it binary only if ALL labels (true and pred) are strictly subset of {0, 1}
        is_binary = set(all_labels).issubset({0, 1})
        avg = 'binary' if is_binary else 'macro'

        # AUC multiclass check
        is_multiclass = len(unique_y_true) > 2

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
        auc = 0.5
        if y_prob is not None:
            try:
                if is_multiclass:
                    # For multiclass, y_prob should be (n_samples, n_classes)
                    auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                else:
                    auc = roc_auc_score(y_true, y_prob)
            except:
                auc = 0.5
        return acc, prec, rec, f1, auc

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

def map_variable_names(formula_str, feature_names, model_type='deeppysr'):
    """
    Map x0, x1... or x_0, x_1... back to original feature names.
    Use regex to avoid partial matches (e.g., x1 matching x10).

    Note on indexing:
    - KAN models: Use 0-based indexing (x0, x1...).
    - PySR models: Use 0-based indexing (x0, x1...).
    - DeepPySR (pypysr) models: Use 1-based indexing (x1, x2...).
    """
    if not formula_str or pd.isna(formula_str):
        return formula_str

    # Sort indices in descending order to avoid x10 being partially replaced by x1
    indices = sorted(range(len(feature_names)), reverse=True)

    mapped_formula = str(formula_str)

    # If it's KAN or PySR, it uses 0-based indexing.
    # If it's DeepPySR (pypysr), it uses 1-based indexing.
    is_0_based = model_type.lower() in ['deeppysr', 'pysr', 'kan']

    for i in indices:
        name = feature_names[i]
        if is_0_based:
            # Replace x_i and xi (PySR, KAN)
            mapped_formula = re.sub(rf'\bx_{i}\b', name, mapped_formula)
            mapped_formula = re.sub(rf'\bx{i}\b', name, mapped_formula)
        else:
            # Assume 1-based (DeepPySR/pypysr): x_{i+1} and x{i+1}
            mapped_formula = re.sub(rf'\bx_{i+1}\b', name, mapped_formula)
            mapped_formula = re.sub(rf'\bx{i+1}\b', name, mapped_formula)

    return mapped_formula

def evaluate_formula(formula_str, X, model_type='deeppysr'):
    """
    Evaluate a symbolic formula (PySR, DeepPySR, or KAN) using SymPy.
    Supports both indexed variables (x0, x1...) and raw feature names.

    Note on indexing:
    - KAN models: Use 0-based indexing (x0, x1...).
    - PySR models: Use 0-based indexing (x0, x1...).
    - DeepPySR (pypysr) models: Use 1-based indexing (x1, x2...).
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

        # Feature names from X
        feature_names = list(X.columns) if hasattr(X, 'columns') else []

        # Some feature names (e.g. "group") collide with SymPy builtin function
        # names, so sympify would bind them to the builtin instead of treating
        # them as free variables. Force them to Symbols by giving them priority
        # in the locals dict used for parsing.
        local_dict = dict(custom_functions)
        local_dict.update({name: sp.Symbol(name) for name in feature_names})

        expr = sp.sympify(str(formula_str), locals=local_dict)

        # Mapping for indexed variables if they exist in formula
        # We replace x0, x1... and x_0, x_1... with the actual column names
        # Models use either 0-based (PySR, KAN) or 1-based (DeepPySR/pypysr) indexing.
        is_0_based = model_type.lower() in ['deeppysr', 'pysr', 'kan']

        subs_dict = {}
        for i, name in enumerate(feature_names):
            if is_0_based:
                subs_dict[sp.Symbol(f"x{i}")] = sp.Symbol(name)
                subs_dict[sp.Symbol(f"x_{i}")] = sp.Symbol(name)
            else:
                subs_dict[sp.Symbol(f"x{i+1}")] = sp.Symbol(name)
                subs_dict[sp.Symbol(f"x_{i+1}")] = sp.Symbol(name)

        expr = expr.xreplace(subs_dict)

        # Now evaluate the expression with X's data
        # We use lambdify for performance
        symbols = [sp.Symbol(str(s)) for s in expr.free_symbols]

        # Prepare input data for lambdify
        input_data = []
        for s in symbols:
            var = str(s)
            if var in X.columns:
                input_data.append(X[var].values)
            else:
                # If variable is not in X, it might be an indexed variable that wasn't replaced (shouldn't happen with raw names requirement)
                # or it's a constant that was parsed as a symbol
                input_data.append(np.zeros(len(X)))

        f_lambdified = sp.lambdify(symbols, expr, modules=['numpy'])

        if not symbols:
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
        print(f"Warning: error evaluating formula {formula_str!r}: {e}")
        return np.zeros(len(X))

def load_fold_metrics(model_dir, task='regression'):
    """Load per-fold metrics from fold_metrics.csv saved by run_cv.  Returns a DataFrame or None."""
    path = os.path.join(model_dir, 'fold_metrics.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def compute_se(values):
    """Standard error of the mean (ddof=1) across folds, ignoring NaN."""
    arr = np.array([v for v in values if not np.isnan(float(v))])
    if len(arr) < 2:
        return np.nan
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def compute_fold_metrics_from_predictions(model_dir, X, y, task='regression', n_splits=5,
                                           random_state=42, stratify=None, groups=None):
    """Re-slice predictions.csv into per-fold metrics using reconstructed CV splits.

    predictions.csv rows are in fold-concatenated order (fold 0 test rows first, etc.),
    matching exactly the order produced by run_cv.  Saves fold_metrics.csv to model_dir
    and returns the list of per-fold metric dicts, or None if the file is missing or
    the total row count doesn't match.
    """
    pred_file = os.path.join(model_dir, 'predictions.csv')
    if not os.path.exists(pred_file):
        return None

    df_pred = pd.read_csv(pred_file)
    if 'y_true' not in df_pred.columns or 'y_pred' not in df_pred.columns:
        return None

    y_values = y.values if hasattr(y, 'values') else np.array(y)
    X_df = X if hasattr(X, 'columns') else pd.DataFrame(
        X, columns=[f'x{i}' for i in range(np.array(X).shape[1])])

    if groups is not None:
        groups_arr = groups.values if hasattr(groups, 'values') else np.array(groups)
        if stratify is not None:
            skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(skf.split(X_df, stratify, groups=groups_arr))
        else:
            skf = GroupKFold(n_splits=n_splits)
            splits = list(skf.split(X_df, y_values, groups=groups_arr))
    elif stratify is not None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(X_df, stratify))
    elif task == 'classification':
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(X_df, y_values))
    else:
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(X_df, y_values))

    total_test = sum(len(test_idx) for _, test_idx in splits)
    if total_test != len(df_pred):
        return None

    fold_metrics_list = []
    row_offset = 0
    for _, test_idx in splits:
        fold_size = len(test_idx)
        fold_rows = df_pred.iloc[row_offset:row_offset + fold_size]
        row_offset += fold_size

        y_true_fold = fold_rows['y_true'].values
        y_pred_fold = fold_rows['y_pred'].values

        y_prob_fold = None
        if 'y_prob' in fold_rows.columns:
            y_prob_fold = fold_rows['y_prob'].values

        m = calculate_metrics(y_true_fold, y_pred_fold, y_prob_fold, task=task)
        if task == 'regression':
            fold_metrics_list.append({'r2': m[0], 'rmse': m[1], 'mae': m[2]})
        else:
            fold_metrics_list.append({'accuracy': m[0], 'precision': m[1],
                                      'recall': m[2], 'f1': m[3]})

    rows = [{'fold': i, **fm} for i, fm in enumerate(fold_metrics_list)]
    pd.DataFrame(rows).to_csv(os.path.join(model_dir, 'fold_metrics.csv'), index=False)

    return fold_metrics_list


def compute_formula_fold_metrics(formula_str, X, y, task='regression', n_splits=5,
                                  random_state=42, stratify=None, model_type='deeppysr'):
    """Evaluate *formula_str* on each CV fold's test set and return a list of metric dicts.

    Uses the same split strategy as run_cv so results are directly comparable.
    """
    y_values = y.values if hasattr(y, 'values') else np.array(y)
    X_df = X if hasattr(X, 'columns') else pd.DataFrame(
        X, columns=[f'x{i}' for i in range(np.array(X).shape[1])])

    if stratify is not None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(X_df, stratify))
    elif task == 'classification':
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(X_df, y_values))
    else:
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(X_df, y_values))

    fold_metrics_list = []
    for _, test_idx in splits:
        X_test = X_df.iloc[test_idx]
        y_test = y_values[test_idx]
        y_pred = evaluate_formula(formula_str, X_test, model_type=model_type)
        m = calculate_metrics(y_test, y_pred, task=task)
        if task == 'regression':
            fold_metrics_list.append({'r2': m[0], 'rmse': m[1], 'mae': m[2]})
        else:
            fold_metrics_list.append({'accuracy': m[0], 'precision': m[1],
                                      'recall': m[2], 'f1': m[3]})
    return fold_metrics_list


def wilcoxon_compare(a_scores, b_scores, alternative='greater'):
    """Wilcoxon signed-rank test: H1 = a > b (DeepPySR > baseline).

    Returns (statistic, p_value).  NaN when insufficient non-tied data.
    """
    a = np.array([float(v) for v in a_scores if not np.isnan(float(v))])
    b = np.array([float(v) for v in b_scores if not np.isnan(float(v))])
    if len(a) != len(b) or len(a) < 2:
        return np.nan, np.nan
    diff = a - b
    if np.all(diff == 0):
        return np.nan, 1.0
    try:
        stat, p = scipy_wilcoxon(diff, alternative=alternative)
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def collect_model_fold_data(model_dir, formula, X, y, task, model_type,
                            n_splits=5, random_state=42, stratify=None, groups=None):
    """Return per-fold metric dicts for a model.

    Preference order:
    1. fold_metrics.csv written by run_cv
    2. predictions.csv sliced by reconstructed CV folds (also saves fold_metrics.csv)
    3. Evaluate formula string on reconstructed CV folds (symbolic models)
    """
    fold_df = load_fold_metrics(model_dir, task)
    if fold_df is not None:
        if task == 'regression':
            return [{'r2': row.get('r2', np.nan), 'rmse': row.get('rmse', np.nan),
                     'mae': row.get('mae', np.nan)} for _, row in fold_df.iterrows()]
        else:
            return [{'accuracy': row.get('accuracy', np.nan), 'precision': row.get('precision', np.nan),
                     'recall': row.get('recall', np.nan), 'f1': row.get('f1', np.nan)}
                    for _, row in fold_df.iterrows()]

    fold_metrics = compute_fold_metrics_from_predictions(
        model_dir, X, y, task=task, n_splits=n_splits, random_state=random_state,
        stratify=stratify, groups=groups)
    if fold_metrics is not None:
        return fold_metrics

    if formula and str(formula) not in ('', 'nan'):
        return compute_formula_fold_metrics(str(formula), X, y, task=task, n_splits=n_splits,
                                            random_state=random_state, stratify=stratify,
                                            model_type=model_type)
    return None


def se_from_fold_data(fold_metrics_list):
    """Compute SE for each metric from a list of per-fold metric dicts."""
    if not fold_metrics_list:
        return {}
    keys = list(fold_metrics_list[0].keys())
    return {f'{k}_se': compute_se([m.get(k, np.nan) for m in fold_metrics_list]) for k in keys}


def run_wilcoxon_analysis(fold_data, deeppysr_key, task, output_file=None):
    """Compare DeepPySR against every other model using Wilcoxon signed-rank test.

    Args:
        fold_data: dict {model_name -> list of per-fold metric dicts, or None}
        deeppysr_key: model_name of the DeepPySR (best) entry
        task: 'regression' or 'classification'
        output_file: path to save CSV; skipped if None

    Returns:
        DataFrame with columns: model, deeppysr_mean, deeppysr_se, other_mean, other_se,
        wilcoxon_stat, wilcoxon_p, significant
    """
    primary = 'r2' if task == 'regression' else 'f1'

    if deeppysr_key not in fold_data or fold_data[deeppysr_key] is None:
        return pd.DataFrame()

    deep_scores = [m.get(primary, np.nan) for m in fold_data[deeppysr_key]]

    rows = []
    for model_name, metrics_list in fold_data.items():
        if model_name == deeppysr_key or metrics_list is None:
            continue
        other_scores = [m.get(primary, np.nan) for m in metrics_list]
        stat, p = wilcoxon_compare(deep_scores, other_scores)
        rows.append({
            'model': model_name,
            f'deeppysr_mean_{primary}': float(np.nanmean(deep_scores)),
            f'deeppysr_se_{primary}': compute_se(deep_scores),
            f'other_mean_{primary}': float(np.nanmean(other_scores)),
            f'other_se_{primary}': compute_se(other_scores),
            'wilcoxon_stat': stat,
            'wilcoxon_p': p,
            'significant_p05': (p < 0.05) if not np.isnan(p) else False,
        })

    result_df = pd.DataFrame(rows)
    if output_file and not result_df.empty:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        result_df.to_csv(output_file, index=False)
        print(f"Wilcoxon results saved to {output_file}")
    return result_df


def get_best_formula_from_raw(folder_path, X, y_true, prefix='relationships_fold', task='regression', model_type='deeppysr'):
    """
    Find the best formula by evaluating all fold-specific formulas on raw data.
    Works for DeepPySR, PySR and KAN.
    Returns a dictionary of (r2w, lambda): (formula, complexity, metrics) for DeepPySR grid search,
    or just a single (formula, complexity, metrics) if no grid search info is found.
    """
    results = {} # {(r2w, lambda): (best_score, best_formula, best_complexity, best_metrics)}
    
    # Use R2 for regression and F1 for classification as the sorting metric
    if task == 'regression':
        overall_best = (-float('inf'), "", np.nan, (np.nan, np.nan, np.nan))
    else:
        overall_best = (-float('inf'), "", np.nan, (np.nan, np.nan, np.nan, np.nan, np.nan))

    feature_names = list(X.columns) if hasattr(X, 'columns') else []

    # If prefix matches KAN common prefixes, adjust model_type if not explicitly set
    if 'formulas' in prefix and model_type == 'kan':
        model_type = 'kan'

    if model_type == 'pysr':
        prefix = prefix.replace('relationships', 'formulas')
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

                    y_pred = evaluate_formula(formula, X, model_type=model_type)
                    metrics = calculate_metrics(y_true, y_pred, task=task)
                    
                    # Sort by R2 for regression, F1 for classification
                    score = metrics[0] if task == 'regression' else metrics[3]

                    if not np.isnan(score):
                        mapped_formula = map_variable_names(formula, feature_names, model_type=model_type)
                        # Update per-parameter best
                        if key not in results or score > results[key][0]:
                            results[key] = (score, mapped_formula, complexity, metrics)

                        # Update overall best
                        if score > overall_best[0]:
                            overall_best = (score, mapped_formula, complexity, metrics)
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
