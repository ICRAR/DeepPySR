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

                pred_file = os.path.join(model_path, "predictions.csv")
                if os.path.exists(pred_file):
                    df_pred = pd.read_csv(pred_file)
                    if model_name.lower() == 'kan':
                        # KAN
                        r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                        all_data.append([eq_name, 'KAN', r2, rmse, mae, np.nan, "", model_path])

                        # KANSym
                        if 'y_pred_kansym' in df_pred.columns:
                            formula, complexity, metrics = get_best_formula_from_raw(model_path, X_df, y_true, prefix='formulas_fold', model_type='kan')
                            r2, rmse, mae = metrics
                            all_data.append([eq_name, 'KANSym', r2, rmse, mae, complexity, formula, model_path])
                    else:
                        # Other baselines
                        r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                        all_data.append([eq_name, model_name, r2, rmse, mae, np.nan, "", model_path])

        # DeepPySR - collect all variants and select Pareto optimal across all
        deeppysr_formulas = {}
        deeppysr_dir = os.path.join(base_dir, "deeppysr")
        if os.path.exists(deeppysr_dir):
            for variant in os.listdir(deeppysr_dir):
                v_path = os.path.join(deeppysr_dir, variant)
                if not os.path.isdir(v_path): continue

                res = get_best_formula_from_raw(v_path, X_df, y_true, model_type='deeppysr')

                if isinstance(res, dict):
                    for key, value in res.items():
                        formula, complexity, metrics = value
                        deeppysr_formulas[(variant, key[0], key[1])] = (formula, complexity, metrics, v_path)
                else:
                    # If single, add with default key
                    formula, complexity, metrics = res
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

                res = get_best_formula_from_raw(v_path, X_df, y_true, model_type='pysr')

                if isinstance(res, dict):
                    for key, value in res.items():
                        formula, complexity, metrics = value
                        pysr_formulas[(variant, key[0], key[1])] = (formula, complexity, metrics, v_path)
                else:
                    formula, complexity, metrics = res
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

def process_ablation_data():
    """Collect variant-level results for ablation studies (keeps VPS/VPR/pareto info)."""
    from feynman_utils import load_feynman_data
    all_data = []
    for eq_name in equations.keys():
        eq_key = eq_name.replace('.', '_')
        base_dir = os.path.join(current_dir, f"results_{eq_key}_all")
        X_df, y_true = load_feynman_data(eq_name, n_samples=1000)

        deeppysr_dir = os.path.join(base_dir, "deeppysr")
        if os.path.exists(deeppysr_dir):
            for variant in os.listdir(deeppysr_dir):
                v_path = os.path.join(deeppysr_dir, variant)
                if not os.path.isdir(v_path):
                    continue
                res = get_best_formula_from_raw(v_path, X_df, y_true, model_type='deeppysr')
                if isinstance(res, dict):
                    for (r2w, lamb), (formula, complexity, metrics) in res.items():
                        r2, rmse, mae = metrics
                        all_data.append([eq_name, f"{variant}_r2w{r2w}_L{lamb}", r2, rmse, mae, complexity])
                else:
                    formula, complexity, mets = res
                    r2, rmse, mae = mets
                    all_data.append([eq_name, variant, r2, rmse, mae, complexity])

        pysr_dir = os.path.join(base_dir, "pysr")
        if os.path.exists(pysr_dir):
            for variant in os.listdir(pysr_dir):
                v_path = os.path.join(pysr_dir, variant)
                if not os.path.isdir(v_path):
                    continue
                res = get_best_formula_from_raw(v_path, X_df, y_true, model_type='pysr')
                if isinstance(res, dict):
                    for (r2w, lamb), (formula, complexity, metrics) in res.items():
                        r2, rmse, mae = metrics
                        all_data.append([eq_name, f"{variant}_r2w{r2w}_L{lamb}", r2, rmse, mae, complexity])
                else:
                    formula, complexity, mets = res
                    r2, rmse, mae = mets
                    all_data.append([eq_name, variant, r2, rmse, mae, complexity])

    result_df = pd.DataFrame(all_data, columns=['equation', 'model', 'r2', 'rmse', 'mae', 'complexity'])
    result_df['r2'] = result_df['r2'].clip(lower=0)
    return result_df


def plot_vps_vpr_ablation(ablation_df):
    """Ablation: VPS/VPR effect with fixed APS=10.0, r2w=1.0, λ=0.01. Per-equation bar charts."""
    import re
    from matplotlib.patches import Patch

    metrics = ['r2', 'rmse', 'mae', 'complexity']
    metric_labels = ['R²', 'RMSE', 'MAE', 'Complexity']
    df = ablation_df

    deep_mask = (df['model'].str.contains('fullsr', regex=False, na=False) &
                 df['model'].str.contains('aps10.0', regex=False, na=False) &
                 df['model'].str.contains('_r2w1.0_L0.01', regex=False, na=False))
    deep_df = df[deep_mask].copy()

    def vps_vpr_label(m):
        match = re.search(r'vps(\d+)_vpr(\d+)', m)
        return f"vps{match.group(1)}/vpr{match.group(2)}" if match else m
    deep_df['label'] = deep_df['model'].apply(vps_vpr_label)

    pysr_mask = (df['model'].str.contains(r'^pysr', regex=True, na=False) &
                 df['model'].str.contains('aps10.0', regex=False, na=False))
    pysr_df = df[pysr_mask].copy()
    pysr_df['label'] = 'PySR (no VPS/VPR)'

    # Save CSV
    csv_df = pd.concat([deep_df[['equation', 'label'] + metrics],
                        pysr_df[['equation', 'label'] + metrics]], ignore_index=True)
    csv_df.to_csv(os.path.join(current_dir, 'ablation_vps_vpr.csv'), index=False)
    print(f"VPS/VPR ablation data saved to {os.path.join(current_dir, 'ablation_vps_vpr.csv')}")

    if deep_df.empty and pysr_df.empty:
        print("No data for VPS/VPR ablation")
        return

    def sort_key(lbl):
        m = re.search(r'vps(\d+)/vpr(\d+)', lbl)
        return (int(m.group(1)), int(m.group(2))) if m else (999, 999)

    deep_labels = sorted(deep_df['label'].unique(), key=sort_key)
    order = deep_labels + ['PySR (no VPS/VPR)']
    colors = ['#4878CF'] * len(deep_labels) + ['#E87722']
    eq_names = list(equations.keys())
    n_eq = len(eq_names)

    fig, axes = plt.subplots(n_eq, 4, figsize=(24, 5 * n_eq))
    if n_eq == 1:
        axes = axes.reshape(1, -1)
    plt.rcParams.update({'font.size': 11})

    for row_i, eq_name in enumerate(eq_names):
        deep_eq = deep_df[deep_df['equation'] == eq_name]
        pysr_eq = pysr_df[pysr_df['equation'] == eq_name]
        parts = []
        if not deep_eq.empty:
            parts.append(deep_eq[['label'] + metrics])
        if not pysr_eq.empty:
            pr = pysr_eq[metrics].mean().to_frame().T
            pr['label'] = 'PySR (no VPS/VPR)'
            parts.append(pr[['label'] + metrics])
        if not parts:
            continue
        plot_eq = pd.concat(parts, ignore_index=True)

        for col_j, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row_i, col_j]
            vals = [plot_eq[plot_eq['label'] == lbl][metric].mean() for lbl in order]
            ax.bar(range(len(order)), vals, color=colors, edgecolor='white', linewidth=0.5)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=45, ha='right', fontsize=8)
            ax.set_title(f'{eq_name} – {mlabel}', fontsize=12, fontweight='bold')
            ax.set_ylabel(mlabel, fontsize=10)

    fig.legend(handles=[Patch(facecolor='#4878CF', label='DeepPySR'),
                        Patch(facecolor='#E87722', label='PySR (reference)')],
               loc='upper right', fontsize=11, frameon=True)
    plt.suptitle('Ablation: VPS/VPR Effect (APS=10.0, r2w=1.0, λ=0.01)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(current_dir, 'ablation_vps_vpr.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"VPS/VPR ablation plot saved to {out}")


def plot_pareto_ablation(ablation_df):
    """Ablation: pareto r2w/λ effect with fixed VPS=25, VPR=100, APS=10.0. Per-equation bar charts."""
    import re
    from matplotlib.patches import Patch

    metrics = ['r2', 'rmse', 'mae', 'complexity']
    metric_labels = ['R²', 'RMSE', 'MAE', 'Complexity']
    df = ablation_df

    deep_mask = (df['model'].str.contains('fullsr', regex=False, na=False) &
                 df['model'].str.contains('_vps25_', regex=False, na=False) &
                 df['model'].str.contains('_vpr100_', regex=False, na=False) &
                 df['model'].str.contains('aps10.0', regex=False, na=False))
    deep_df = df[deep_mask].copy()

    def pareto_label(m):
        r2w_m = re.search(r'_r2w([\d.]+)_L', m)
        l_m = re.search(r'_L([\d.]+)$', m)
        if r2w_m and l_m:
            return f"r2w={r2w_m.group(1)}, λ={l_m.group(1)}"
        return m
    deep_df['label'] = deep_df['model'].apply(pareto_label)

    pysr_mask = (df['model'].str.contains(r'^pysr', regex=True, na=False) &
                 df['model'].str.contains('aps10.0', regex=False, na=False))
    pysr_df = df[pysr_mask].copy()
    pysr_df['label'] = 'PySR (reference)'

    # Save CSV
    csv_df = pd.concat([deep_df[['equation', 'label'] + metrics],
                        pysr_df[['equation', 'label'] + metrics]], ignore_index=True)
    csv_df.to_csv(os.path.join(current_dir, 'ablation_pareto.csv'), index=False)
    print(f"Pareto ablation data saved to {os.path.join(current_dir, 'ablation_pareto.csv')}")

    if deep_df.empty and pysr_df.empty:
        print("No data for pareto ablation")
        return

    def sort_key(lbl):
        r2w_m = re.search(r'r2w=([\d.]+)', lbl)
        l_m = re.search(r'λ=([\d.]+)', lbl)
        if r2w_m and l_m:
            return (float(r2w_m.group(1)), float(l_m.group(1)))
        return (999, 999)

    deep_labels = sorted(deep_df['label'].unique(), key=sort_key)
    order = deep_labels + ['PySR (reference)']
    r2w_vals = sorted(set(float(re.search(r'r2w=([\d.]+)', l).group(1))
                          for l in deep_labels if re.search(r'r2w=([\d.]+)', l)))
    r2w_palette = dict(zip(r2w_vals, ['#2166ac', '#4dac26', '#d6604d']))
    colors = []
    for lbl in order:
        m = re.search(r'r2w=([\d.]+)', lbl)
        colors.append(r2w_palette.get(float(m.group(1)), '#888') if m else '#E87722')

    eq_names = list(equations.keys())
    n_eq = len(eq_names)
    fig, axes = plt.subplots(n_eq, 4, figsize=(26, 5 * n_eq))
    if n_eq == 1:
        axes = axes.reshape(1, -1)
    plt.rcParams.update({'font.size': 11})

    for row_i, eq_name in enumerate(eq_names):
        deep_eq = deep_df[deep_df['equation'] == eq_name]
        pysr_eq = pysr_df[pysr_df['equation'] == eq_name]
        parts = []
        if not deep_eq.empty:
            parts.append(deep_eq[['label'] + metrics])
        if not pysr_eq.empty:
            pr = pysr_eq[metrics].mean().to_frame().T
            pr['label'] = 'PySR (reference)'
            parts.append(pr[['label'] + metrics])
        if not parts:
            continue
        plot_eq = pd.concat(parts, ignore_index=True)

        for col_j, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row_i, col_j]
            vals = [plot_eq[plot_eq['label'] == lbl][metric].mean() for lbl in order]
            ax.bar(range(len(order)), vals, color=colors, edgecolor='white', linewidth=0.5)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=45, ha='right', fontsize=8)
            ax.set_title(f'{eq_name} – {mlabel}', fontsize=12, fontweight='bold')
            ax.set_ylabel(mlabel, fontsize=10)

    legend_elements = [Patch(facecolor=r2w_palette[v], label=f'DeepPySR r2w={v}') for v in r2w_vals]
    legend_elements.append(Patch(facecolor='#E87722', label='PySR (reference)'))
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True)
    plt.suptitle('Ablation: Pareto r2w/λ Effect (VPS=25, VPR=100, APS=10.0)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(current_dir, 'ablation_pareto.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pareto ablation plot saved to {out}")


def _pareto_front_steps(complexity, error):
    """Return Pareto-optimal (complexity, error) pairs sorted by complexity (both minimize)."""
    points = sorted(zip(complexity, error), key=lambda p: (p[0], p[1]))
    pareto = []
    min_error = float('inf')
    for c, e in points:
        if e < min_error:
            min_error = e
            pareto.append((c, e))
    return pareto


def plot_pareto_front_rmse(ablation_df):
    """Scatter plot of complexity vs RMSE showing the Pareto front per Feynman equation."""
    eq_names = list(equations.keys())
    n_eq = len(eq_names)
    ncols = min(n_eq, 3)
    nrows = (n_eq + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, eq_name in enumerate(eq_names):
        ax = axes_flat[idx]
        sub_df = ablation_df[ablation_df['equation'] == eq_name]
        deep_df = sub_df[sub_df['model'].str.contains('fullsr', regex=False, na=False)].copy()
        pysr_df = sub_df[sub_df['model'].str.contains(r'^pysr', regex=True, na=False)].copy()

        deep_df = deep_df[deep_df['rmse'].notna() & deep_df['complexity'].notna()]
        pysr_df = pysr_df[pysr_df['rmse'].notna() & pysr_df['complexity'].notna()]

        if not deep_df.empty:
            pf = _pareto_front_steps(deep_df['complexity'].tolist(), deep_df['rmse'].tolist())
            if pf:
                px, py = zip(*pf)
                ax.step(px, py, where='post', color='#2166ac', linewidth=2, zorder=4)
                ax.scatter(px, py, c='#2166ac', s=100, zorder=5, marker='D', label='DeepPySR')

        if not pysr_df.empty:
            pf_pysr = _pareto_front_steps(pysr_df['complexity'].tolist(), pysr_df['rmse'].tolist())
            if pf_pysr:
                px, py = zip(*pf_pysr)
                ax.step(px, py, where='post', color='#cc4400', linewidth=2, zorder=4)
                ax.scatter(px, py, c='#cc4400', s=100, zorder=5, marker='D', label='PySR')

        ax.set_xlabel('Complexity', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title(f'{eq_name} – Pareto Front: Complexity vs RMSE', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for idx in range(len(eq_names), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    out = os.path.join(current_dir, 'pareto_front_rmse.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pareto front RMSE plot saved to {out}")


if __name__ == "__main__":
    df = process_results()
    save_best_formulas(df)
    plot_best_models()
    ablation_df = process_ablation_data()
    plot_vps_vpr_ablation(ablation_df)
    plot_pareto_ablation(ablation_df)
    plot_pareto_front_rmse(ablation_df)