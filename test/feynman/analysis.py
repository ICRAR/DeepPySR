import os
import re
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

current_dir = os.path.dirname(os.path.abspath(__file__))
if not current_dir:
    current_dir = "."
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

from feynman_utils import equations, load_feynman_data
from analysis_v1_utils import (calculate_metrics, leak_free_process_and_select,
                                get_best_formula_from_raw, collect_model_fold_data,
                                get_formula_fold_metrics, compute_fold_metrics_from_predictions,
                                se_from_fold_data, run_wilcoxon_analysis, compute_se)

TASK = 'regression'
R2W_LIST = [1, 1.5, 2]
LAMBDA_LIST = [0.001, 0.005, 0.01]
# test_all_models_feynman.py / test_baselines_pysr_feynman.py: cv_kwargs stratify_by=None
# (plain regression, no stratification) for every equation.
STRATIFY = None


def process_results():
    """One row per (equation, model). DeepPySR/PySR/KAN(Sym) metrics come from
    predictions.csv (genuine held-out, pooled) -- no formula re-evaluation.
    Feynman has no Interpretable-DeepPySR concept (matches old code)."""
    all_rows = []
    for eq_name in equations.keys():
        eq_key = eq_name.replace('.', '_')
        base_dir = os.path.join(current_dir, f"results_{eq_key}_all")
        if not os.path.isdir(base_dir):
            continue
        X_df, y_true = load_feynman_data(eq_name, n_samples=1000)
        all_df, best_df = leak_free_process_and_select(
            base_dir, X_df, y_true, task=TASK, interp_max_complexity=None, stratify=STRATIFY)
        if all_df.empty:
            continue
        all_df.insert(0, 'equation', eq_name)
        all_rows.append(all_df)

    result_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not result_df.empty:
        result_df['r2'] = result_df['r2'].clip(lower=0)
    result_df.to_csv(os.path.join(current_dir, "aggregated_results.csv"), index=False)
    print(f"Results saved to {os.path.join(current_dir, 'aggregated_results.csv')}")
    return result_df


def _display_label(family):
    return {'deeppysr': 'DeepPySR', 'pysr': 'PySR', 'kan': 'KAN', 'kansym': 'KANSym'}.get(family, family)


def save_best_formulas(df):
    """Save the true formula and best formula per model, per equation."""
    best_formulas = []
    for eq_name in equations.keys():
        eq_df = df[df['equation'] == eq_name]
        true_formula = equations[eq_name]['formula']
        row = {'equation': eq_name, 'true_formula': true_formula}
        for family, prefix in [('deeppysr', 'best_deeppysr'), ('pysr', 'best_pysr'), ('kansym', 'best_kansym')]:
            fam_df = eq_df[eq_df['family'] == family]
            if not fam_df.empty:
                best = fam_df.loc[fam_df['r2'].idxmax()]
                row[f'{prefix}_formula'] = best['formula']
                row[f'{prefix}_r2'] = best['r2']
                row[f'{prefix}_complexity'] = best['complexity']
            else:
                row[f'{prefix}_formula'] = ""
                row[f'{prefix}_r2'] = np.nan
                row[f'{prefix}_complexity'] = np.nan
        best_formulas.append(row)

    formulas_df = pd.DataFrame(best_formulas)
    formulas_csv_path = os.path.join(current_dir, 'best_formulas.csv')
    formulas_df.to_csv(formulas_csv_path, index=False)
    print(f"Best formulas saved to {formulas_csv_path}")

    print("\n--- Best Formulas ---")
    for _, row in formulas_df.iterrows():
        print(f"Equation: {row['equation']}")
        print(f"  True: {row['true_formula']}")
        print(f"  DeepPySR: {row['best_deeppysr_formula']} (R2: {row['best_deeppysr_r2']:.3f}, Complexity: {row['best_deeppysr_complexity']})")
        print(f"  PySR: {row['best_pysr_formula']} (R2: {row['best_pysr_r2']:.3f}, Complexity: {row['best_pysr_complexity']})")
        print(f"  KanSym: {row['best_kansym_formula']} (R2: {row['best_kansym_r2']:.3f}, Complexity: {row['best_kansym_complexity']})")
        print()


def _fold_data_for_family(eq_df, base_dir, family, X_df, y_true):
    fam_df = eq_df[eq_df['family'] == family]
    if fam_df.empty:
        return None
    best = fam_df.loc[fam_df['r2'].idxmax()]
    source_path = best.get('source_path', '')
    if not source_path or not os.path.isdir(str(source_path)):
        return None
    if family == 'kansym':
        return compute_fold_metrics_from_predictions(
            source_path, X_df, y_true, task=TASK, stratify=STRATIFY, pred_col='y_pred_kansym')
    return collect_model_fold_data(source_path, "", X_df, y_true, TASK, model_type=family, stratify=STRATIFY)


def compute_se_and_wilcoxon(result_df):
    """Compute per-fold SE and Wilcoxon vs DeepPySR for each Feynman equation."""
    all_se_data = {}  # (eq_name, model_name) -> se_dict

    for eq_name in equations.keys():
        eq_key = eq_name.replace('.', '_')
        base_dir = os.path.join(current_dir, f"results_{eq_key}_all")
        if not os.path.isdir(base_dir):
            continue
        X_df, y_true = load_feynman_data(eq_name, n_samples=1000)
        eq_df = result_df[result_df['equation'] == eq_name]

        fold_data = {}
        baselines_dir = os.path.join(base_dir, "baselines")
        if os.path.exists(baselines_dir):
            for model_name in os.listdir(baselines_dir):
                model_path = os.path.join(baselines_dir, model_name)
                if not os.path.isdir(model_path) or model_name.lower() == 'kan':
                    continue
                fold_data[model_name] = collect_model_fold_data(
                    model_path, "", X_df, y_true, TASK, model_type=model_name, stratify=STRATIFY)
            kan_path = os.path.join(baselines_dir, 'KAN')
            if os.path.isdir(kan_path):
                fold_data['KAN'] = collect_model_fold_data(
                    kan_path, "", X_df, y_true, TASK, model_type='kan', stratify=STRATIFY)

        for family, label in [('deeppysr', 'DeepPySR'), ('pysr', 'PySR'), ('kansym', 'KANSym')]:
            fd = _fold_data_for_family(eq_df, base_dir, family, X_df, y_true)
            if fd is not None:
                fold_data[label] = fd

        for model_name, fd in fold_data.items():
            if fd is not None:
                ses = se_from_fold_data(fd)
                ses['n_folds'] = len(fd)
                all_se_data[(eq_name, model_name)] = ses

        if 'DeepPySR' in fold_data:
            run_wilcoxon_analysis(fold_data, 'DeepPySR', TASK,
                                  output_file=os.path.join(current_dir, f"wilcoxon_results_{eq_key}.csv"))

    agg_csv = os.path.join(current_dir, 'aggregated_results.csv')
    if all_se_data and os.path.exists(agg_csv):
        agg_df = pd.read_csv(agg_csv)
        first_se = next(iter(all_se_data.values()))
        se_cols = [c for c in first_se.keys() if c != 'n_folds']
        for col in se_cols + ['n_folds']:
            agg_df[col] = np.nan
        for i, row in agg_df.iterrows():
            key = (row['equation'], _display_label(row['family']))
            if key in all_se_data:
                for col in se_cols + ['n_folds']:
                    agg_df.at[i, col] = all_se_data[key].get(col, np.nan)
        agg_df.to_csv(agg_csv, index=False)
        print(f"SE merged into {agg_csv}")


def plot_best_models():
    """5 rows (equations) x 4 columns (r2, rmse, mae, complexity)."""
    df = pd.read_csv(os.path.join(current_dir, 'aggregated_results.csv'))
    df['display'] = df['family'].map(_display_label)

    equations_list = list(equations.keys())
    metrics = ['r2', 'rmse', 'mae', 'complexity']
    models_to_include_for_complexity = ['DeepPySR', 'PySR', 'KANSym']
    baselines = ['ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']

    fig, axes = plt.subplots(5, 4, figsize=(20, 15))

    for i, eq_name in enumerate(equations_list):
        eq_df = df[df['equation'] == eq_name]
        selected_data = []
        for label in ['DeepPySR', 'PySR', 'KAN', 'KANSym']:
            m_df = eq_df[eq_df['display'] == label]
            if not m_df.empty:
                row = m_df.loc[m_df['r2'].idxmax()].copy()
                row['model'] = label
                selected_data.append(row)
        for b in baselines:
            b_df = eq_df[eq_df['model'] == b]
            if not b_df.empty:
                selected_data.append(b_df.iloc[0])

        plot_df_all = pd.DataFrame(selected_data)
        plot_df_complexity = plot_df_all[plot_df_all['model'].isin(models_to_include_for_complexity)] if not plot_df_all.empty else plot_df_all

        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            plot_df = plot_df_complexity.copy() if metric == 'complexity' else plot_df_all.copy()
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
    """Per-variant genuine held-out results for the VPS/VPR ablation (default
    r2w=1/lambda=0.001 operating point, per fold), keyed by (equation, model)."""
    all_data = []
    for eq_name in equations.keys():
        eq_key = eq_name.replace('.', '_')
        base_dir = os.path.join(current_dir, f"results_{eq_key}_all")
        if not os.path.isdir(base_dir):
            continue
        X_df, y_true = load_feynman_data(eq_name, n_samples=1000)

        for subdir, model_type in [('deeppysr', 'deeppysr'), ('pysr', 'pysr')]:
            sr_dir = os.path.join(base_dir, subdir)
            if not os.path.exists(sr_dir):
                continue
            for variant in os.listdir(sr_dir):
                v_path = os.path.join(sr_dir, variant)
                if not os.path.isdir(v_path):
                    continue
                pred_file = os.path.join(v_path, 'predictions.csv')
                if not os.path.exists(pred_file):
                    continue
                df_pred = pd.read_csv(pred_file)
                r2, rmse, mae, _ = calculate_metrics(df_pred['y_true'], df_pred['y_pred'], task=TASK)
                _, complexity, _ = get_best_formula_from_raw(
                    v_path, X_df, y_true, model_type=model_type, task=TASK, stratify=STRATIFY)
                all_data.append([eq_name, variant, r2, rmse, mae, complexity])

    result_df = pd.DataFrame(all_data, columns=['equation', 'model', 'r2', 'rmse', 'mae', 'complexity'])
    if not result_df.empty:
        result_df['r2'] = result_df['r2'].clip(lower=0)
    return result_df


def process_pareto_ablation_data():
    """Genuine held-out r2w/lambda ablation (fixed VPS=25/VPR=100/APS=10.0),
    each grid point scored per-fold via get_best_formula_from_raw(pareto_point=)."""
    all_data = []
    for eq_name in equations.keys():
        eq_key = eq_name.replace('.', '_')
        base_dir = os.path.join(current_dir, f"results_{eq_key}_all")
        if not os.path.isdir(base_dir):
            continue
        X_df, y_true = load_feynman_data(eq_name, n_samples=1000)

        deep_dir = os.path.join(base_dir, 'deeppysr')
        target_variant = None
        if os.path.isdir(deep_dir):
            for variant in os.listdir(deep_dir):
                if '_vps25_' in variant and '_vpr100_' in variant and 'aps10.0' in variant:
                    target_variant = variant
                    break
        if target_variant is not None:
            v_path = os.path.join(deep_dir, target_variant)
            for r2w in R2W_LIST:
                for lam in LAMBDA_LIST:
                    formula, complexity, m = get_best_formula_from_raw(
                        v_path, X_df, y_true, task=TASK, model_type='deeppysr', stratify=STRATIFY,
                        pareto_point=(r2w, lam))
                    if not formula:
                        continue
                    all_data.append([eq_name, f"r2w={r2w}, λ={lam}", m[0], m[1], m[2], complexity])

        pysr_dir = os.path.join(base_dir, 'pysr')
        if os.path.isdir(pysr_dir):
            for variant in os.listdir(pysr_dir):
                if 'aps10.0' not in variant:
                    continue
                v_path = os.path.join(pysr_dir, variant)
                pred_file = os.path.join(v_path, 'predictions.csv')
                if not os.path.exists(pred_file):
                    continue
                df_pred = pd.read_csv(pred_file)
                r2, rmse, mae, _ = calculate_metrics(df_pred['y_true'], df_pred['y_pred'], task=TASK)
                _, complexity, _ = get_best_formula_from_raw(
                    v_path, X_df, y_true, model_type='pysr', task=TASK, stratify=STRATIFY)
                all_data.append([eq_name, 'PySR (reference)', r2, rmse, mae, complexity])

    result_df = pd.DataFrame(all_data, columns=['equation', 'label', 'r2', 'rmse', 'mae', 'complexity'])
    if not result_df.empty:
        result_df['r2'] = result_df['r2'].clip(lower=0)
    return result_df


def plot_vps_vpr_ablation(ablation_df):
    """Ablation: VPS/VPR effect (default r2w/λ) -- best R² across all aps per config."""
    metrics = ['r2', 'rmse', 'mae', 'complexity']
    metric_labels = ['R²', 'RMSE', 'MAE', 'Complexity']
    df = ablation_df

    deep_mask = df['model'].str.contains('fullsr', regex=False, na=False)
    deep_df = df[deep_mask].copy()

    def vps_vpr_label(m):
        match = re.search(r'vps(\d+)_vpr(\d+)', m)
        return f"vps{match.group(1)}/vpr{match.group(2)}" if match else m
    deep_df['label'] = deep_df['model'].apply(vps_vpr_label)
    deep_df = deep_df.loc[deep_df.groupby(['equation', 'label'])['r2'].idxmax()].reset_index(drop=True)

    pysr_mask = df['model'].str.contains(r'^pysr', regex=True, na=False)
    pysr_df = df[pysr_mask].copy()
    pysr_df = pysr_df.loc[pysr_df.groupby('equation')['r2'].idxmax()].reset_index(drop=True) if not pysr_df.empty else pysr_df
    pysr_df['label'] = 'PySR (no VPS/VPR)'

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
    plt.suptitle('Ablation: VPS/VPR Effect (best APS per config, default r2w/λ)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(current_dir, 'ablation_vps_vpr.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"VPS/VPR ablation plot saved to {out}")


def plot_pareto_ablation(pareto_df):
    """Ablation: pareto r2w/λ effect with fixed VPS=25, VPR=100, APS=10.0
    (genuine per-fold held-out, see process_pareto_ablation_data)."""
    metrics = ['r2', 'rmse', 'mae', 'complexity']
    metric_labels = ['R²', 'RMSE', 'MAE', 'Complexity']
    df = pareto_df

    df.to_csv(os.path.join(current_dir, 'ablation_pareto.csv'), index=False)
    print(f"Pareto ablation data saved to {os.path.join(current_dir, 'ablation_pareto.csv')}")

    if df.empty:
        print("No data for pareto ablation")
        return

    deep_df = df[df['label'] != 'PySR (reference)'].copy()
    pysr_df = df[df['label'] == 'PySR (reference)'].copy()

    def sort_key(lbl):
        r2w_m = re.search(r'r2w=([\d.]+)', lbl)
        l_m = re.search(r'λ=([\d.]+)', lbl)
        if r2w_m and l_m:
            return (float(r2w_m.group(1)), float(l_m.group(1)))
        return (999, 999)

    deep_labels = sorted(deep_df['label'].unique(), key=sort_key)
    order = deep_labels + (['PySR (reference)'] if not pysr_df.empty else [])
    r2w_vals = sorted(set(float(re.search(r'r2w=([\d.]+)', l).group(1))
                          for l in deep_labels if re.search(r'r2w=([\d.]+)', l)))
    palette_colors = ['#2166ac', '#4dac26', '#d6604d']
    r2w_palette = dict(zip(r2w_vals, palette_colors * (len(r2w_vals) // len(palette_colors) + 1)))
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
        eq_sub = df[df['equation'] == eq_name]
        if eq_sub.empty:
            continue
        for col_j, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row_i, col_j]
            vals = [eq_sub[eq_sub['label'] == lbl][metric].mean() for lbl in order]
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
    points = sorted(zip(complexity, error), key=lambda p: (p[0], p[1]))
    pareto = []
    min_error = float('inf')
    for c, e in points:
        if e < min_error:
            min_error = e
            pareto.append((c, e))
    return pareto


def _load_hof_data(model_dir):
    """SR search's own training-time complexity/loss log -- optimizer
    dynamics, not a held-out generalization claim, unaffected by the leak."""
    pysr_out = os.path.join(model_dir, 'pysr_outputs', 'y')
    if not os.path.exists(pysr_out):
        return pd.DataFrame()
    rows = []
    for ts in sorted(os.listdir(pysr_out)):
        hof_file = os.path.join(pysr_out, ts, 'hall_of_fame.csv')
        if not os.path.exists(hof_file):
            continue
        hof = pd.read_csv(hof_file)
        if 'Complexity' not in hof.columns or 'Loss' not in hof.columns:
            continue
        for _, row in hof.iterrows():
            rows.append({'complexity': int(row['Complexity']), 'loss': float(row['Loss'])})
    if not rows:
        return pd.DataFrame()
    hof_df = pd.DataFrame(rows)
    agg = hof_df.groupby('complexity')['loss'].mean().reset_index()
    agg['rmse'] = np.sqrt(agg['loss'])
    return agg[['complexity', 'rmse']]


def plot_pareto_front_rmse(ablation_df):
    """Pareto front per Feynman equation: DeepPySR/PySR from hall_of_fame (training log)."""
    eq_names = [e for e in equations.keys() if e != 'I.8.14']
    n_eq = len(eq_names)
    ncols = 2
    nrows = (n_eq + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, eq_name in enumerate(eq_names):
        ax = axes_flat[idx]
        sub_df = ablation_df[ablation_df['equation'] == eq_name]
        deep_df = sub_df[sub_df['model'].str.contains('fullsr', regex=False, na=False)].copy()
        pysr_df = sub_df[sub_df['model'].str.contains(r'^pysr', regex=True, na=False)].copy()
        pysr_df = pysr_df[pysr_df['rmse'].notna() & pysr_df['complexity'].notna()]

        hof_data = pd.DataFrame()
        eq_key = eq_name.replace('.', '_')
        if not deep_df.empty:
            best_name = deep_df.loc[deep_df['r2'].idxmax(), 'model']
            model_dir = os.path.join(current_dir, f'results_{eq_key}_all', 'deeppysr', best_name)
            hof_data = _load_hof_data(model_dir)

        hof_pysr = pd.DataFrame()
        if not pysr_df.empty:
            for _, pysr_row in pysr_df.sort_values('r2', ascending=False).iterrows():
                candidate_dir = os.path.join(current_dir, f'results_{eq_key}_all', 'pysr', pysr_row['model'])
                hof_pysr = _load_hof_data(candidate_dir)
                if not hof_pysr.empty:
                    break

        if not hof_data.empty:
            pf = _pareto_front_steps(hof_data['complexity'].tolist(), hof_data['rmse'].tolist())
            if pf:
                px, py = zip(*pf)
                ax.step(px, py, where='post', color='#2166ac', linewidth=2, zorder=4)
                ax.scatter(px, py, c='#2166ac', s=100, zorder=5, marker='D', label='DeepPySR')

        if not hof_pysr.empty:
            pf_pysr = _pareto_front_steps(hof_pysr['complexity'].tolist(), hof_pysr['rmse'].tolist())
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
    compute_se_and_wilcoxon(df)
    plot_best_models()
    ablation_df = process_ablation_data()
    plot_vps_vpr_ablation(ablation_df)
    pareto_df = process_pareto_ablation_data()
    plot_pareto_ablation(pareto_df)
    plot_pareto_front_rmse(ablation_df)
