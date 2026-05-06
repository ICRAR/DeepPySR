import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add test/ and test/bmi to path to import load_bmi_agg_data
current_dir = os.path.dirname(os.path.abspath(__file__))
if not current_dir:
    current_dir = "."
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

from feynman_utils import equations
from analysis_utils import calculate_metrics, get_best_formula_from_raw

def select_pareto_optimal(formulas_dict):
    """
    Select the formula with highest R2, and if multiple have the same R2, the one with lowest complexity.
    """
    if not formulas_dict:
        return None, np.nan, (np.nan, np.nan, np.nan), None

    candidates = []
    for key, (formula, complexity, metrics, model_path) in formulas_dict.items():
        r2 = metrics[0]
        if not np.isnan(r2):
            candidates.append((r2, complexity, formula, metrics, model_path))

    if not candidates:
        return None, np.nan, (np.nan, np.nan, np.nan), None

    # Find max r2
    max_r2 = max(c[0] for c in candidates)
    # Get candidates with max r2
    max_r2_candidates = [c for c in candidates if c[0] == max_r2]
    # Among them, pick the one with lowest complexity
    best = min(max_r2_candidates, key=lambda x: x[1])
    return best[2], best[1], best[3], best[4]

def process_results():
    eq_names = list(equations.keys())
    all_data = []
    for eq_name in eq_names:
        eq_key = eq_name.replace('.', '_')
        base_dir = os.path.join(current_dir, f"results_{eq_key}_all")

        # Load data to get X and y
        from feynman_utils import load_feynman_data
        X_df, y_true = load_feynman_data(eq_name, n_samples=1000)

        # Baselines
        baselines_dir = os.path.join(base_dir, "baselines")
        if os.path.exists(baselines_dir):
            for model_name in os.listdir(baselines_dir):
                model_path = os.path.join(baselines_dir, model_name)
                if not os.path.isdir(model_path):
                    continue

                # Use overall_metrics.csv for all baselines if it exists
                overall_metrics_file = os.path.join(model_path, "overall_metrics.csv")
                use_overall = False
                if os.path.exists(overall_metrics_file):
                    df_metrics = pd.read_csv(overall_metrics_file)
                    r2_o = df_metrics['r2'].iloc[0]
                    rmse_o = df_metrics['rmse'].iloc[0]
                    mae_o = df_metrics['mae'].iloc[0]
                    use_overall = True

                if model_name.lower() == 'kan':
                    # KAN
                    if use_overall:
                        r2, rmse, mae = r2_o, rmse_o, mae_o
                    else:
                        pred_file = os.path.join(model_path, "predictions.csv")
                        if os.path.exists(pred_file):
                            df_pred = pd.read_csv(pred_file)
                            r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                        else:
                            r2, rmse, mae = [np.nan] * 3
                    all_data.append([eq_name, 'KAN', r2, rmse, mae, np.nan, "", model_path])

                    # KANSym
                    overall_metrics_sym_file = os.path.join(model_path, "overall_metrics_sym.csv")
                    use_overall_sym = False
                    if os.path.exists(overall_metrics_sym_file):
                        df_metrics_sym = pd.read_csv(overall_metrics_sym_file)
                        r2_s = df_metrics_sym['r2'].iloc[0]
                        rmse_s = df_metrics_sym['rmse'].iloc[0]
                        mae_s = df_metrics_sym['mae'].iloc[0]
                        use_overall_sym = True

                    pred_file = os.path.join(model_path, "predictions.csv")
                    if os.path.exists(pred_file):
                        df_pred = pd.read_csv(pred_file)
                        if 'y_pred_kansym' in df_pred.columns:
                            formula, complexity, metrics = get_best_formula_from_raw(model_path, X_df, y_true, prefix='formulas_fold', model_type='kan')
                            r2, rmse, mae = metrics
                            if use_overall_sym:
                                r2, rmse, mae = r2_s, rmse_s, mae_s

                            all_data.append([eq_name, 'KANSym', r2, rmse, mae, complexity, formula, model_path])
                else:
                    # Other baselines
                    if use_overall:
                        r2, rmse, mae = r2_o, rmse_o, mae_o
                    else:
                        pred_file = os.path.join(model_path, "predictions.csv")
                        if os.path.exists(pred_file):
                            df_pred = pd.read_csv(pred_file)
                            r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                        else:
                            r2, rmse, mae = [np.nan] * 3
                    all_data.append([eq_name, model_name, r2, rmse, mae, np.nan, "", model_path])

        # DeepPySR - collect all variants and select Pareto optimal across all
        deeppysr_formulas = {}
        deeppysr_dir = os.path.join(base_dir, "deeppysr")
        if os.path.exists(deeppysr_dir):
            for variant in os.listdir(deeppysr_dir):
                v_path = os.path.join(deeppysr_dir, variant)
                if not os.path.isdir(v_path): continue

                # Use overall_metrics.csv for DeepPySR if it exists
                overall_metrics_file = os.path.join(v_path, "overall_metrics.csv")
                use_overall = False
                if os.path.exists(overall_metrics_file):
                    df_metrics = pd.read_csv(overall_metrics_file)
                    r2_o = df_metrics['r2'].iloc[0]
                    rmse_o = df_metrics['rmse'].iloc[0]
                    mae_o = df_metrics['mae'].iloc[0]
                    use_overall = True

                res = get_best_formula_from_raw(v_path, X_df, y_true, model_type='deeppysr')

                if isinstance(res, dict):
                    for key, value in res.items():
                        formula, complexity, metrics = value
                        if use_overall:
                            metrics = (r2_o, rmse_o, mae_o)
                        deeppysr_formulas[(variant, key[0], key[1])] = (formula, complexity, metrics, v_path)
                else:
                    # If single, add with default key
                    formula, complexity, metrics = res
                    if use_overall:
                        metrics = (r2_o, rmse_o, mae_o)
                    deeppysr_formulas[(variant, 1.0, 0.001)] = (formula, complexity, metrics, v_path)

        if deeppysr_formulas:
            formula, complexity, metrics, model_path = select_pareto_optimal(deeppysr_formulas)
            r2, rmse, mae = metrics
            all_data.append([eq_name, 'DeepPySR', r2, rmse, mae, complexity, formula, model_path])

        # PySR - collect all variants and select Pareto optimal across all
        pysr_formulas = {}
        pysr_dir = os.path.join(base_dir, "pysr")
        if os.path.exists(pysr_dir):
            for variant in os.listdir(pysr_dir):
                v_path = os.path.join(pysr_dir, variant)
                if not os.path.isdir(v_path): continue

                # Use overall_metrics.csv for PySR if it exists
                overall_metrics_file = os.path.join(v_path, "overall_metrics.csv")
                use_overall = False
                if os.path.exists(overall_metrics_file):
                    df_metrics = pd.read_csv(overall_metrics_file)
                    r2_o = df_metrics['r2'].iloc[0]
                    rmse_o = df_metrics['rmse'].iloc[0]
                    mae_o = df_metrics['mae'].iloc[0]
                    use_overall = True

                res = get_best_formula_from_raw(v_path, X_df, y_true, model_type='pysr')

                if isinstance(res, dict):
                    for key, value in res.items():
                        formula, complexity, metrics = value
                        if use_overall:
                            metrics = (r2_o, rmse_o, mae_o)
                        pysr_formulas[(variant, key[0], key[1])] = (formula, complexity, metrics, v_path)
                else:
                    formula, complexity, metrics = res
                    if use_overall:
                        metrics = (r2_o, rmse_o, mae_o)
                    pysr_formulas[(variant, 1.0, 0.001)] = (formula, complexity, metrics, v_path)

        if pysr_formulas:
            formula, complexity, metrics, model_path = select_pareto_optimal(pysr_formulas)
            r2, rmse, mae = metrics
            all_data.append([eq_name, 'PySR', r2, rmse, mae, complexity, formula, model_path])

    # Create DataFrame and save
    result_df = pd.DataFrame(all_data, columns=['equation', 'model', 'r2', 'rmse', 'mae', 'complexity', 'formula', 'model_path'])
    # Clip r2 to 0
    result_df['r2'] = result_df['r2'].clip(lower=0)
    result_df.to_csv(os.path.join(current_dir, "aggregated_results.csv"), index=False)
    print(f"Results saved to {os.path.join(current_dir, 'aggregated_results.csv')}")
    return result_df

def save_best_formulas(df):
    """
    Save the true formula and best formulas for each equation.
    """
    best_formulas = []

    for eq_name in equations.keys():
        eq_df = df[df['equation'] == eq_name]

        # True formula
        true_formula = equations[eq_name]['formula']

        # Best DeepPySR: highest r2
        deeppysr_df = eq_df[eq_df['model'].str.contains('DeepPySR', na=False)]
        if not deeppysr_df.empty:
            best_deeppysr = deeppysr_df.loc[deeppysr_df['r2'].idxmax()]
            best_deeppysr_formula = best_deeppysr['formula']
            best_deeppysr_r2 = best_deeppysr['r2']
            best_deeppysr_complexity = best_deeppysr['complexity']
        else:
            best_deeppysr_formula = ""
            best_deeppysr_r2 = np.nan
            best_deeppysr_complexity = np.nan

        # Best PySR: highest r2
        pysr_df = eq_df[eq_df['model'].str.contains('PySR', na=False)]
        if not pysr_df.empty:
            best_pysr = pysr_df.loc[pysr_df['r2'].idxmax()]
            best_pysr_formula = best_pysr['formula']
            best_pysr_r2 = best_pysr['r2']
            best_pysr_complexity = best_pysr['complexity']
        else:
            best_pysr_formula = ""
            best_pysr_r2 = np.nan
            best_pysr_complexity = np.nan

        # Best KanSym: highest r2
        kansym_df = eq_df[eq_df['model'] == 'KANSym']
        if not kansym_df.empty:
            best_kansym = kansym_df.loc[kansym_df['r2'].idxmax()]
            best_kansym_formula = best_kansym['formula']
            best_kansym_r2 = best_kansym['r2']
            best_kansym_complexity = best_kansym['complexity']
        else:
            best_kansym_formula = ""
            best_kansym_r2 = np.nan
            best_kansym_complexity = np.nan

        best_formulas.append({
            'equation': eq_name,
            'true_formula': true_formula,
            'best_deeppysr_formula': best_deeppysr_formula,
            'best_deeppysr_r2': best_deeppysr_r2,
            'best_deeppysr_complexity': best_deeppysr_complexity,
            'best_pysr_formula': best_pysr_formula,
            'best_pysr_r2': best_pysr_r2,
            'best_pysr_complexity': best_pysr_complexity,
            'best_kansym_formula': best_kansym_formula,
            'best_kansym_r2': best_kansym_r2,
            'best_kansym_complexity': best_kansym_complexity
        })

    formulas_df = pd.DataFrame(best_formulas)
    formulas_csv_path = os.path.join(current_dir, 'best_formulas.csv')
    formulas_df.to_csv(formulas_csv_path, index=False)
    print(f"Best formulas saved to {formulas_csv_path}")

    # Print
    print("\n--- Best Formulas ---")
    for _, row in formulas_df.iterrows():
        print(f"Equation: {row['equation']}")
        print(f"  True: {row['true_formula']}")
        print(f"  DeepPySR: {row['best_deeppysr_formula']} (R2: {row['best_deeppysr_r2']:.3f}, Complexity: {row['best_deeppysr_complexity']})")
        print(f"  PySR: {row['best_pysr_formula']} (R2: {row['best_pysr_r2']:.3f}, Complexity: {row['best_pysr_complexity']})")
        print(f"  KanSym: {row['best_kansym_formula']} (R2: {row['best_kansym_r2']:.3f}, Complexity: {row['best_kansym_complexity']})")
        print()

def plot_best_models():
    """
    Create a plot with 5 rows (equations) and 4 columns (r2, rmse, mae, complexity).
    Select best models similar to wine analysis.
    """
    df = pd.read_csv(os.path.join(current_dir, 'aggregated_results.csv'))

    equations_list = list(equations.keys())
    metrics = ['r2', 'rmse', 'mae', 'complexity']
    models_to_include_for_complexity = ['DeepPySR', 'PySR', 'KANSym']

    fig, axes = plt.subplots(5, 4, figsize=(20, 15))

    for i, eq_name in enumerate(equations_list):
        eq_df = df[df['equation'] == eq_name]

        # Select models: DeepPySR, PySR, KAN, KANSym, and baselines
        selected_data = []

        # DeepPySR
        deeppysr_df = eq_df[eq_df['model'] == 'DeepPySR']
        if not deeppysr_df.empty:
            selected_data.append(deeppysr_df.iloc[0])

        # PySR
        pysr_df = eq_df[eq_df['model'] == 'PySR']
        if not pysr_df.empty:
            selected_data.append(pysr_df.iloc[0])

        # KAN and KANSym
        for m in ['KAN', 'KANSym']:
            m_df = eq_df[eq_df['model'] == m]
            if not m_df.empty:
                selected_data.append(m_df.iloc[0])

        # Baselines: ElasticNet, ExtraTrees, MLP, RandomForest, XGBoost
        baselines = ['ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']
        for b in baselines:
            b_df = eq_df[eq_df['model'] == b]
            if not b_df.empty:
                selected_data.append(b_df.iloc[0])

        plot_df_all = pd.DataFrame(selected_data)
        plot_df_complexity = plot_df_all[plot_df_all['model'].isin(models_to_include_for_complexity)]

        for j, metric in enumerate(metrics):
            ax = axes[i, j]

            if metric == 'complexity':
                plot_df = plot_df_complexity.copy()
            else:
                plot_df = plot_df_all.copy()

            if plot_df.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
                ax.set_title(f'{eq_name} - {metric.upper()}')
                ax.set_xlabel('Model', fontsize=8)
                ax.set_ylabel(metric.upper())
                ax.set_xticks([])
                continue

            ax.bar(range(len(plot_df)), plot_df[metric])
            ax.set_title(f'{eq_name} - {metric.upper()}')
            ax.set_xlabel('Model', fontsize=8)
            ax.set_ylabel(metric.upper())
            ax.set_xticks(range(len(plot_df)))
            ax.set_xticklabels(plot_df['model'], rotation=90, ha='center', fontsize=6)

    plt.tight_layout()
    plot_path = os.path.join(current_dir, 'best_models_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    df = process_results()
    save_best_formulas(df)
    plot_best_models()