import os
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold
import glob
import re
import sympy as sp
from scipy.stats import wilcoxon as scipy_wilcoxon
from scipy.stats import pearsonr as scipy_pearsonr

def calculate_metrics(y_true, y_pred, y_prob=None, task='regression'):
    if len(y_true) == 0:
        if task == 'regression':
            return np.nan, np.nan, np.nan, np.nan
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
        # Pearson correlation coefficient (r). Undefined (NaN) when either
        # series is constant -- not clipped, since unlike R2 there's no
        # "no worse than the mean" floor convention for it.
        try:
            if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
                pearson_r = np.nan
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pearson_r = float(scipy_pearsonr(y_true, y_pred)[0])
        except Exception:
            pearson_r = np.nan
        return r2, rmse, mae, pearson_r
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
                                           random_state=42, stratify=None, groups=None,
                                           pred_col='y_pred'):
    """Re-slice predictions.csv into per-fold metrics using reconstructed CV splits.

    predictions.csv rows are in fold-concatenated order (fold 0 test rows first, etc.),
    matching exactly the order produced by run_cv.  Saves fold_metrics.csv to model_dir
    (only when scoring the primary pred_col='y_pred' -- not for an alternate column
    like 'y_pred_kansym', which isn't what that cache file name means) and returns the
    list of per-fold metric dicts, or None if the file is missing or the total row
    count doesn't match.
    """
    pred_file = os.path.join(model_dir, 'predictions.csv')
    if not os.path.exists(pred_file):
        return None

    df_pred = pd.read_csv(pred_file)
    if 'y_true' not in df_pred.columns or pred_col not in df_pred.columns:
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
        y_pred_fold = fold_rows[pred_col].values

        y_prob_fold = None
        if pred_col == 'y_pred' and 'y_prob' in fold_rows.columns:
            y_prob_fold = fold_rows['y_prob'].values

        m = calculate_metrics(y_true_fold, y_pred_fold, y_prob_fold, task=task)
        if task == 'regression':
            fold_metrics_list.append({'r2': m[0], 'rmse': m[1], 'mae': m[2], 'pearson_r': m[3]})
        else:
            fold_metrics_list.append({'accuracy': m[0], 'precision': m[1],
                                      'recall': m[2], 'f1': m[3]})

    if pred_col == 'y_pred':
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
            fold_metrics_list.append({'r2': m[0], 'rmse': m[1], 'mae': m[2], 'pearson_r': m[3]})
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


def _reconstruct_splits(X_df, y_values, task, n_splits, random_state, stratify=None, groups=None):
    """Reconstruct the exact CV split eval_utils.run_cv used to train the
    models being re-scored. Mirrors run_cv's own branching exactly:
    groups+stratify -> StratifiedGroupKFold, groups only -> GroupKFold,
    stratify (explicit or the classification target) -> StratifiedKFold,
    else plain KFold. Passing the wrong `stratify`/`groups` here silently
    reconstructs the wrong folds -- check the dataset's training script's
    run_cv/cv_kwargs call for what it actually passed.
    """
    if groups is not None:
        groups_arr = groups.values if hasattr(groups, 'values') else np.array(groups)
        if stratify is not None:
            stratify_values = stratify.values if hasattr(stratify, 'values') else np.array(stratify)
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            return list(splitter.split(X_df, stratify_values, groups=groups_arr))
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(X_df, y_values, groups=groups_arr))
    if stratify is not None:
        stratify_values = stratify.values if hasattr(stratify, 'values') else np.array(stratify)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(X_df, stratify_values))
    if task == 'classification':
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(X_df, y_values))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(X_df, y_values))




def _cv_pooled_oof_predictions(folder_path, X, y_true, prefix='relationships_fold', task='regression',
                                model_type='deeppysr', n_splits=5, random_state=42, max_complexity=None,
                                stratify=None, groups=None, pareto_point=None):
    """
    Core leak-free CV logic shared by get_best_formula_from_raw and
    get_oof_predictions.

    Reconstructs the exact CV split used by eval_utils.run_cv (see
    _reconstruct_splits; same X/y row order required), and for each fold:
    (1) picks that fold's own best formula using only information the SR
    search already computed while fitting that fold's training split (the
    stored 'r2' fitness column) -- never anything from the held-out rows,
    then (2) evaluates that formula ONLY on the fold's held-out test rows.

    If max_complexity is given, each fold's candidates are restricted to
    formulas with complexity < max_complexity *before* ranking by r2 (a fold
    is skipped if none of its candidates satisfy this). Filtering by
    complexity after picking each fold's unconstrained best (as the original
    leaky code does, post-hoc across all folds/rows) doesn't work here since
    each fold only ever contributes one evaluated candidate -- the
    constraint has to be applied at selection time.

    If pareto_point=(r2w, lam) is given, each fold's candidates are instead
    restricted to rows whose own pareto_r2_weight/pareto_lambda match that
    exact grid point (defaulting to 1.0/0.001 for rows missing those columns,
    matching the SR search's own default) -- for reconstructing a genuine
    held-out r2w/lambda ablation. Mutually exclusive with max_complexity.

    Returns (y_pred_oof, valid_mask, fold_candidates) where fold_candidates
    is a list of (fold_idx, mapped_formula, complexity, fold_score).
    """
    feature_names = list(X.columns) if hasattr(X, 'columns') else []
    X_df = X if hasattr(X, 'columns') else pd.DataFrame(
        X, columns=[f'x{i}' for i in range(np.array(X).shape[1])])
    y_values = y_true.values if hasattr(y_true, 'values') else np.array(y_true)

    # If prefix matches KAN common prefixes, adjust model_type if not explicitly set
    if 'formulas' in prefix and model_type == 'kan':
        model_type = 'kan'
    if model_type == 'pysr':
        prefix = prefix.replace('relationships', 'formulas')

    splits = _reconstruct_splits(X_df, y_values, task, n_splits, random_state,
                                  stratify=stratify, groups=groups)

    y_pred_oof = np.full(len(X_df), np.nan)
    fold_candidates = []  # (fold_idx, mapped_formula, complexity, fold_score)

    for fold_idx, (_, test_idx) in enumerate(splits):
        fold_file = os.path.join(folder_path, f"{prefix}{fold_idx}.csv")
        if not os.path.exists(fold_file):
            continue
        try:
            df = pd.read_csv(fold_file)
        except Exception:
            continue
        if 'formula' not in df.columns or df.empty:
            continue

        if max_complexity is not None:
            df = df.copy()
            df['_complexity'] = df['formula'].astype(str).map(calculate_complexity)
            df = df[df['_complexity'] < max_complexity]
            if df.empty:
                continue

        if pareto_point is not None:
            r2w, lam = pareto_point
            r2w_col = df['pareto_r2_weight'] if 'pareto_r2_weight' in df.columns else pd.Series(1.0, index=df.index)
            lam_col = df['pareto_lambda'] if 'pareto_lambda' in df.columns else pd.Series(0.001, index=df.index)
            df = df[np.isclose(r2w_col.fillna(1.0), r2w) & np.isclose(lam_col.fillna(0.001), lam)]
            if df.empty:
                continue

        # Best candidate *within this fold's own search results* -- ranked
        # by the fold's own in-search fitness, not by anything evaluated on
        # held-out data.
        if 'r2' in df.columns and df['r2'].notna().any():
            best_row = df.loc[df['r2'].idxmax()]
        else:
            best_row = df.iloc[0]

        formula = str(best_row['formula'])
        complexity = calculate_complexity(formula)

        X_test = X_df.iloc[test_idx]
        try:
            y_pred_fold = evaluate_formula(formula, X_test, model_type=model_type)
        except Exception:
            continue
        y_pred_oof[test_idx] = y_pred_fold

        fold_metrics = calculate_metrics(y_values[test_idx], y_pred_fold, task=task)
        fold_score = fold_metrics[0] if task == 'regression' else fold_metrics[3]
        mapped_formula = map_variable_names(formula, feature_names, model_type=model_type)
        fold_candidates.append((fold_idx, mapped_formula, complexity, fold_score, fold_metrics))

    valid = ~np.isnan(y_pred_oof)
    return y_pred_oof, valid, fold_candidates


def get_best_formula_from_raw(folder_path, X, y_true, prefix='relationships_fold', task='regression',
                               model_type='deeppysr', n_splits=5, random_state=42, max_complexity=None,
                               stratify=None, groups=None, pareto_point=None):
    """
    Leak-free counterpart of analysis_utils.get_best_formula_from_raw.

    The original version evaluates every fold's formula on the *entire*
    dataset (including the rows that formula was fit on) and reports that
    in-sample score as the model's R2/F1 -- this inflates DeepPySR/PySR/KAN
    metrics relative to the baselines, whose R2 comes from genuine held-out
    CV predictions (predictions.csv from run_cv). This version pools genuine
    held-out predictions across all 5 folds instead (see
    _cv_pooled_oof_predictions), directly comparable to how baseline models
    are scored.

    max_complexity, if given, restricts each fold's candidate selection to
    formulas with complexity < max_complexity (see
    _cv_pooled_oof_predictions) -- use this for an "Interpretable DeepPySR"
    metric instead of filtering the unconstrained result after the fact.

    Returns (formula, complexity, metrics) for a single representative
    formula, taken from whichever fold generalized best on its own held-out
    split (display purposes only; the pooled metrics do not depend on it).
    """
    if task == 'regression':
        empty_metrics = (np.nan, np.nan, np.nan, np.nan)
    else:
        empty_metrics = (np.nan, np.nan, np.nan, np.nan, np.nan)

    y_values = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
    y_pred_oof, valid, fold_candidates = _cv_pooled_oof_predictions(
        folder_path, X, y_true, prefix=prefix, task=task, model_type=model_type,
        n_splits=n_splits, random_state=random_state, max_complexity=max_complexity,
        stratify=stratify, groups=groups, pareto_point=pareto_point)

    if not fold_candidates or not valid.any():
        return "", np.nan, empty_metrics

    metrics = calculate_metrics(y_values[valid], y_pred_oof[valid], task=task)

    # Representative formula for display: the fold whose OWN held-out split
    # it generalized best on. Not used for the pooled metrics above.
    _, best_formula, best_complexity, _, _ = max(
        fold_candidates, key=lambda t: t[3] if not np.isnan(t[3]) else -np.inf)

    return best_formula, best_complexity, metrics


def get_formula_fold_metrics(folder_path, X, y_true, prefix='relationships_fold', task='regression',
                              model_type='deeppysr', n_splits=5, random_state=42, max_complexity=None,
                              stratify=None, groups=None, pareto_point=None):
    """
    Per-fold metric dicts for the same leak-free formula selection used by
    get_best_formula_from_raw (each fold's own best/interpretable formula,
    evaluated only on that fold's held-out rows). Use this for SE / Wilcoxon
    on a formula-selected metric (e.g. Interpretable DeepPySR) where no
    fold_metrics.csv/predictions.csv exists for this exact selection rule
    (predictions.csv reflects the model's own unconstrained default pick,
    not a complexity-constrained one).
    """
    _, _, fold_candidates = _cv_pooled_oof_predictions(
        folder_path, X, y_true, prefix=prefix, task=task, model_type=model_type,
        n_splits=n_splits, random_state=random_state, max_complexity=max_complexity,
        stratify=stratify, groups=groups, pareto_point=pareto_point)
    if not fold_candidates:
        return None
    keys = _REGRESSION_METRIC_COLS if task == 'regression' else _CLASSIFICATION_METRIC_COLS
    return [dict(zip(keys, fc[4])) for fc in fold_candidates]


def get_oof_predictions(folder_path, X, y_true, prefix='relationships_fold', task='regression',
                         model_type='deeppysr', n_splits=5, random_state=42, max_complexity=None,
                         stratify=None, groups=None):
    """
    Pooled leak-free out-of-fold predictions for one grid-point directory,
    aligned to X's row order (NaN where no fold covered a row -- shouldn't
    happen when all 5 fold files are present). Use this for scatter
    plots/CSV export so the displayed points are exactly what the R2 in
    get_best_formula_from_raw was computed from, instead of re-evaluating a
    single formula in-sample on the full dataset. Pass the same
    max_complexity/stratify/groups used to obtain the metrics you're
    displaying.
    """
    y_pred_oof, valid, fold_candidates = _cv_pooled_oof_predictions(
        folder_path, X, y_true, prefix=prefix, task=task, model_type=model_type,
        n_splits=n_splits, random_state=random_state, max_complexity=max_complexity,
        stratify=stratify, groups=groups)
    if not fold_candidates:
        return None
    return y_pred_oof


_REGRESSION_METRIC_COLS = ['r2', 'rmse', 'mae', 'pearson_r']
_CLASSIFICATION_METRIC_COLS = ['accuracy', 'precision', 'recall', 'f1', 'auc']


def leak_free_process_and_select(source_dir, X, y, task='regression', interp_max_complexity=None,
                                  stratify=None, groups=None, n_splits=5, random_state=42,
                                  baseline_models=None):
    """
    Generic leak-free replacement for the process_results()+save_results()
    pattern repeated (with the leaky get_best_formula_from_raw) across
    test/<dataset>/analysis.py for feynman, bodyfat, bmi, studentPerformance,
    stroke, heart, diabetes, and wineQuality. Walks
    source_dir/{baselines,deeppysr,pysr} exactly like those scripts do, but
    scores every DeepPySR/PySR/KANSym formula with genuine held-out CV
    (get_best_formula_from_raw) instead of evaluating it on the full X/y it
    was fit on.

    stratify/groups must match whatever that dataset's training script
    actually passed to eval_utils.run_cv's cv_kwargs (stratify_by/groups) --
    passing the wrong one silently reconstructs the wrong folds. Check the
    dataset's test_*.py driver script before calling this.

    "Best DeepPySR" is selected from the union of unconstrained and
    complexity-constrained (interp_max_complexity) candidates, so it always
    has metrics at least as good as "Interpretable DeepPySR" by
    construction -- both selections rank fold candidates by in-fold training
    fitness (a noisy generalization proxy), not by held-out performance
    itself, so picking "Best" only from the unconstrained pool could
    otherwise lose to the interpretable pick.

    Returns (all_df, best_df):
      all_df  -- every model/variant found, one row per candidate, with
                 columns [model, family, <metric columns>, complexity,
                 formula, source_path, formula_model_type, max_complexity].
      best_df -- one row per model family (display_model column), the
                 selection baselines.csv / *_best_models_metrics.csv used to
                 be built from.
    """
    primary = 'r2' if task == 'regression' else 'f1'
    metric_cols = _REGRESSION_METRIC_COLS if task == 'regression' else _CLASSIFICATION_METRIC_COLS

    def _row(model, family, metrics, complexity=np.nan, formula="", source_path="",
              formula_model_type="", max_complexity=np.nan):
        row = {'model': model, 'family': family}
        row.update(dict(zip(metric_cols, metrics)))
        row.update({'complexity': complexity, 'formula': formula, 'source_path': source_path,
                    'formula_model_type': formula_model_type, 'max_complexity': max_complexity})
        return row

    rows = []

    baselines_dir = os.path.join(source_dir, 'baselines')
    if os.path.exists(baselines_dir):
        for model_name in sorted(os.listdir(baselines_dir)):
            model_path = os.path.join(baselines_dir, model_name)
            if not os.path.isdir(model_path):
                continue
            pred_file = os.path.join(model_path, 'predictions.csv')
            if not os.path.exists(pred_file):
                continue
            df_pred = pd.read_csv(pred_file)
            metrics = calculate_metrics(df_pred['y_true'], df_pred['y_pred'], task=task)
            family = 'kan' if model_name.lower() == 'kan' else model_name
            rows.append(_row(model_name, family, metrics, source_path=model_path))

            if model_name.lower() == 'kan' and 'y_pred_kansym' in df_pred.columns:
                # Metrics: genuine held-out y_pred_kansym column from run_cv (per-fold
                # model.predict_symbolic on its own held-out split, already pooled) --
                # not a re-evaluated formula. get_best_formula_from_raw is used here only
                # to pick a representative formula string/complexity for display.
                formula, complexity, _ = get_best_formula_from_raw(
                    model_path, X, y, prefix='formulas_fold', model_type='kan', task=task,
                    n_splits=n_splits, random_state=random_state, stratify=stratify, groups=groups)
                metrics_k = calculate_metrics(df_pred['y_true'], df_pred['y_pred_kansym'], task=task)
                rows.append(_row('KANSym', 'kansym', metrics_k, complexity, formula, model_path, 'kan'))

    for subdir, model_type in [('deeppysr', 'deeppysr'), ('pysr', 'pysr')]:
        sr_dir = os.path.join(source_dir, subdir)
        if not os.path.exists(sr_dir):
            continue
        for variant in sorted(os.listdir(sr_dir)):
            v_path = os.path.join(sr_dir, variant)
            if not os.path.isdir(v_path):
                continue
            # Metrics: genuine held-out predictions.csv from run_cv (each fold's
            # model.predict on its own held-out split, already pooled) -- not a
            # re-evaluated formula. get_best_formula_from_raw is used here only to
            # pick a representative formula string/complexity for display; it
            # reflects the model's own default (unconstrained) pareto pick, same
            # selection rule DeepPySR.predict()/PySR.get_best() use internally.
            formula, complexity, _ = get_best_formula_from_raw(
                v_path, X, y, model_type=model_type, task=task, n_splits=n_splits,
                random_state=random_state, stratify=stratify, groups=groups)
            pred_file = os.path.join(v_path, 'predictions.csv')
            if os.path.exists(pred_file):
                df_pred = pd.read_csv(pred_file)
                metrics = calculate_metrics(df_pred['y_true'], df_pred['y_pred'], task=task)
            else:
                metrics = (np.nan,) * len(metric_cols)
            rows.append(_row(variant, model_type, metrics, complexity, formula, v_path, model_type))

            if model_type == 'deeppysr' and interp_max_complexity is not None:
                formula_i, complexity_i, metrics_i = get_best_formula_from_raw(
                    v_path, X, y, model_type=model_type, task=task, n_splits=n_splits,
                    random_state=random_state, max_complexity=interp_max_complexity,
                    stratify=stratify, groups=groups)
                if formula_i:
                    rows.append(_row(f"{variant}__interp{interp_max_complexity}", 'deeppysr', metrics_i,
                                      complexity_i, formula_i, v_path, model_type, interp_max_complexity))

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        return all_df, all_df

    if task == 'regression':
        all_df['r2'] = all_df['r2'].clip(lower=0)

    best_rows = []
    deeppysr_df = all_df[all_df['family'] == 'deeppysr']
    if not deeppysr_df.empty:
        best = deeppysr_df.loc[deeppysr_df[primary].idxmax()].copy()
        best['display_model'] = 'Best DeepPySR'
        best_rows.append(best)
        if interp_max_complexity is not None:
            interp_df = deeppysr_df[deeppysr_df['max_complexity'] == interp_max_complexity]
            if not interp_df.empty:
                bi = interp_df.loc[interp_df[primary].idxmax()].copy()
                bi['display_model'] = 'Interpretable DeepPySR'
                best_rows.append(bi)

    family_labels = {'pysr': 'PySR', 'kan': 'KAN', 'kansym': 'KANSym'}
    if baseline_models is None:
        # Auto-discover: any family that isn't one of the SR/symbolic ones is
        # a baseline, whatever it's named (ElasticNet/RandomForest/... for
        # regression, LogisticRegression/... for classification -- avoids
        # silently dropping a baseline whose name doesn't match a hardcoded
        # regression-flavored default, as happened for classification tasks
        # where the baseline is 'LogisticRegression' not 'ElasticNet').
        baseline_models = sorted(set(all_df['family'].unique()) - {'deeppysr', 'pysr', 'kan', 'kansym'})
    for family in ['pysr', 'kan', 'kansym'] + list(baseline_models):
        fam_df = all_df[all_df['family'] == family]
        if fam_df.empty:
            continue
        row = fam_df.loc[fam_df[primary].idxmax()].copy()
        row['display_model'] = family_labels.get(family, family)
        best_rows.append(row)

    best_df = pd.DataFrame(best_rows).reset_index(drop=True)
    return all_df, best_df
