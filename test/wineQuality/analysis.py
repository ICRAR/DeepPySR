import os
import re
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

current_dir = os.path.dirname(os.path.abspath(__file__))
if not current_dir:
    current_dir = "."
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

from wine_utils import load_wine_data
from analysis_v1_utils import (calculate_metrics, leak_free_process_and_select,
                                get_best_formula_from_raw, collect_model_fold_data,
                                get_formula_fold_metrics, compute_fold_metrics_from_predictions,
                                se_from_fold_data, run_wilcoxon_analysis, compute_se)

TASK = 'regression'
WINE_TYPES = ['red', 'white']
INTERP_MAX_COMPLEXITY = 25
R2W_LIST = [1, 1.5, 2]
LAMBDA_LIST = [0.001, 0.005, 0.01]


def _load_xy(wine_type):
    df = load_wine_data(wine_type)
    X = df.drop(columns=['quality'])
    y = df['quality']
    return X, y


def process_results():
    """Aggregate every model's genuine held-out metrics for both wine types.

    DeepPySR/PySR/KAN/KANSym metrics come from predictions.csv (each fold's
    model.predict on its own held-out split, pooled) -- never from
    re-evaluating a formula against data it was fit on. Interpretable
    DeepPySR is scored by evaluating each fold's own low-complexity candidate
    only on that fold's own held-out rows (analysis_v1_utils.get_best_formula_from_raw).
    Training used stratify_by=y (StratifiedKFold on the discrete quality
    score) for both baselines/PySR and DeepPySR -- reconstructed folds here
    must use the same stratify=y or they won't match.
    """
    all_rows, best_rows = [], []
    for wine_type in WINE_TYPES:
        X, y = _load_xy(wine_type)
        source_dir = os.path.join(current_dir, f"results_{wine_type}_all")
        all_df, best_df = leak_free_process_and_select(
            source_dir, X, y, task=TASK, interp_max_complexity=INTERP_MAX_COMPLEXITY,
            stratify=y)
        all_df.insert(0, 'wine type', wine_type)
        best_df.insert(0, 'wine type', wine_type)
        all_rows.append(all_df)
        best_rows.append(best_df)

    all_df = pd.concat(all_rows, ignore_index=True)
    best_df = pd.concat(best_rows, ignore_index=True)
    all_df.to_csv(os.path.join(current_dir, "aggregated_results.csv"), index=False)
    print(f"Results saved to {os.path.join(current_dir, 'aggregated_results.csv')}")
    return all_df, best_df


def save_results(best_df):
    plot_csv_path = os.path.join(current_dir, 'wine_best_models_metrics.csv')
    best_df.to_csv(plot_csv_path, index=False)
    print(f"Best models plot data saved to {plot_csv_path}")

    interp_rows = best_df[best_df['display_model'] == 'Interpretable DeepPySR']
    print(f"\n--- Interpretable DeepPySR Formulas (Complexity < {INTERP_MAX_COMPLEXITY}) ---")
    cols = ['wine type', 'model', 'formula', 'r2', 'complexity']
    print(interp_rows[cols].to_string(index=False))
    interp_rows[cols].to_csv(os.path.join(current_dir, 'interpretable_deeppysr_formulas.csv'), index=False)


def _fold_data_for_row(row, X, y, stratify):
    source_path = row.get('source_path', '')
    if not source_path or not os.path.isdir(str(source_path)):
        return None
    family = row['family']
    max_c = row.get('max_complexity', np.nan)
    is_interp = pd.notna(max_c)

    if family == 'kansym':
        return compute_fold_metrics_from_predictions(
            source_path, X, y, task=TASK, stratify=stratify, pred_col='y_pred_kansym')
    if family in ('deeppysr', 'pysr') and is_interp:
        return get_formula_fold_metrics(
            source_path, X, y, task=TASK, model_type=family, stratify=stratify,
            max_complexity=int(max_c))
    return collect_model_fold_data(source_path, "", X, y, TASK, model_type=family, stratify=stratify)


def compute_se_and_wilcoxon(best_df):
    """Compute per-fold SE for each best-model row and run Wilcoxon vs DeepPySR (best), per wine type."""
    all_se_maps = {}

    for wine_type in WINE_TYPES:
        X, y = _load_xy(wine_type)
        sub = best_df[best_df['wine type'] == wine_type]

        fold_data = {}
        for _, row in sub.iterrows():
            fd = _fold_data_for_row(row, X, y, y)
            if fd is not None:
                fold_data[row['display_model']] = fd

        se_map = {}
        for model_name, fd in fold_data.items():
            if fd is not None:
                ses = se_from_fold_data(fd)
                ses['n_folds'] = len(fd)
                se_map[model_name] = ses
        all_se_maps[wine_type] = se_map

        if 'Best DeepPySR' in fold_data:
            run_wilcoxon_analysis(fold_data, 'Best DeepPySR', TASK,
                                  output_file=os.path.join(current_dir, f"wilcoxon_results_{wine_type}.csv"))

    metrics_csv_path = os.path.join(current_dir, 'wine_best_models_metrics.csv')
    if all_se_maps and os.path.exists(metrics_csv_path):
        metrics_df = pd.read_csv(metrics_csv_path)
        first_se = next((s for sm in all_se_maps.values() for s in sm.values()), {})
        se_cols = [c for c in first_se.keys() if c != 'n_folds']
        for col in se_cols + ['n_folds']:
            metrics_df[col] = np.nan
        for i, row in metrics_df.iterrows():
            sm = all_se_maps.get(row['wine type'], {})
            if row['display_model'] in sm:
                for col in se_cols + ['n_folds']:
                    metrics_df.at[i, col] = sm[row['display_model']].get(col, np.nan)
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"SE merged into {metrics_csv_path}")


def aggregate_feature_importance():
    """Aggregate feature importance for ElasticNet, ExtraTrees, RandomForest, XGBoost, KAN
    per wine type. (Unrelated to the formula-selection leak.)"""
    for wine_type in WINE_TYPES:
        base_dir = os.path.join(current_dir, f"results_{wine_type}_all")
        importance_data = []

        def process_importance(path, model_name):
            if os.path.exists(path):
                df_imp = pd.read_csv(path)
                if 'feature' in df_imp.columns and 'importance' in df_imp.columns:
                    total = df_imp['importance'].sum()
                    df_imp['importance_pct'] = (df_imp['importance'] / total * 100) if total > 0 else 0
                    for _, row in df_imp.iterrows():
                        importance_data.append({'model': model_name, 'variable': row['feature'],
                                                 'weight': row['importance_pct']})

        baselines_dir = os.path.join(base_dir, "baselines")
        if os.path.exists(baselines_dir):
            for m in os.listdir(baselines_dir):
                if m in ['ElasticNet', 'ExtraTrees', 'RandomForest', 'XGBoost', 'KAN']:
                    process_importance(os.path.join(baselines_dir, m, "feature_importance.csv"), m)

        imp_df = pd.DataFrame(importance_data)
        imp_df.to_csv(os.path.join(base_dir, "feature_importance_aggregated.csv"), index=False)
        print(f"Feature importance ({wine_type}) aggregated to feature_importance_aggregated.csv")

        if not imp_df.empty:
            agg_imp = imp_df.groupby(['model', 'variable'])['weight'].mean().reset_index()
            top_features = agg_imp.groupby('variable')['weight'].mean().sort_values(ascending=False).head(15).index
            plot_df = agg_imp[agg_imp['variable'].isin(top_features)].copy()
            plot_df['variable'] = pd.Categorical(plot_df['variable'], categories=top_features, ordered=True)

            plt.figure(figsize=(14, 10))
            sns.barplot(data=plot_df, x='weight', y='variable', hue='model', palette="bright")
            plt.title(f'Feature Importance Comparison across Models ({wine_type})', fontsize=22, fontweight='bold', pad=20)
            plt.xlabel('Average Percentage Importance (%)', fontsize=18)
            plt.ylabel('Feature', fontsize=18)
            plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
            plt.tick_params(labelsize=14)
            plt.tight_layout()
            plot_path = os.path.join(base_dir, "feature_importance_by_model.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Combined feature importance plot saved to {plot_path}")


def plot_best_models():
    """2 rows (red/white) x 4 cols (r2, rmse, mae, complexity)."""
    df = pd.read_csv(os.path.join(current_dir, 'wine_best_models_metrics.csv'))

    metrics = ['r2', 'rmse', 'mae', 'complexity']
    models_to_include_for_complexity = ['Best DeepPySR', 'Interpretable DeepPySR', 'PySR', 'KANSym']
    label_map = {'Best DeepPySR': 'DeepPySR', 'Interpretable DeepPySR': 'InterpDeepPySR'}

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for i, wine in enumerate(WINE_TYPES):
        wine_df_all = df[df['wine type'] == wine].sort_values('display_model').copy()
        wine_df_complexity = wine_df_all[wine_df_all['display_model'].isin(models_to_include_for_complexity)].copy()

        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            plot_df = wine_df_complexity.copy() if metric == 'complexity' else wine_df_all.copy()
            if plot_df.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
                ax.set_title(f'{wine.capitalize()} Wine - {metric.upper()}')
                ax.set_xlabel('Model', fontsize=8)
                ax.set_ylabel(metric.upper())
                ax.set_xticks([])
                continue
            plot_df['plot_label'] = plot_df['display_model'].replace(label_map)
            ax.bar(plot_df['plot_label'], plot_df[metric])
            ax.set_title(f'{wine.capitalize()} Wine - {metric.upper()}')
            ax.set_xlabel('Model', fontsize=8)
            ax.set_ylabel(metric.upper())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

    plt.tight_layout()
    plot_path = os.path.join(current_dir, 'best_models_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_vps_vpr_ablation(all_df):
    """Ablation: VPS/VPR effect -- best R² across all aps per config at the
    model's own default (r2w=1, lambda=0.001) operating point, genuinely
    held-out per fold, averaged over wine types."""
    metrics = ['r2', 'rmse', 'mae', 'complexity']
    metric_labels = ['R²', 'RMSE', 'MAE', 'Complexity']

    deep_df = all_df[(all_df['family'] == 'deeppysr') & all_df['max_complexity'].isna()].copy()

    def vps_vpr_label(m):
        match = re.search(r'vps(\d+)_vpr(\d+)', m)
        return f"vps{match.group(1)}/vpr{match.group(2)}" if match else m
    deep_df['label'] = deep_df['model'].apply(vps_vpr_label)
    deep_df = deep_df.loc[deep_df.groupby(['wine type', 'label'])['r2'].idxmax()].reset_index(drop=True)
    deep_agg = deep_df.groupby('label')[metrics].mean().reset_index()

    pysr_sub = all_df[all_df['family'] == 'pysr'].copy()
    if not pysr_sub.empty:
        pysr_sub = pysr_sub.loc[pysr_sub.groupby('wine type')['r2'].idxmax()].reset_index(drop=True)
    pysr_sub['label'] = 'PySR (no VPS/VPR)'

    csv_df = pd.concat([deep_df[['label'] + metrics], pysr_sub[['label'] + metrics]], ignore_index=True)
    csv_df.to_csv(os.path.join(current_dir, 'ablation_vps_vpr.csv'), index=False)
    print(f"VPS/VPR ablation data saved to {os.path.join(current_dir, 'ablation_vps_vpr.csv')}")

    if deep_agg.empty and pysr_sub.empty:
        print("No data for VPS/VPR ablation")
        return

    pysr_row = pysr_sub[metrics].mean().to_frame().T
    pysr_row['label'] = 'PySR\n(no VPS/VPR)'
    plot_df = pd.concat([deep_agg, pysr_row[['label'] + metrics]], ignore_index=True)

    def sort_key(lbl):
        m = re.search(r'vps(\d+)/vpr(\d+)', lbl)
        return (int(m.group(1)), int(m.group(2))) if m else (999, 999)

    labels = sorted([l for l in plot_df['label'].unique() if 'PySR' not in l], key=sort_key)
    order = labels + [l for l in plot_df['label'].unique() if 'PySR' in l]
    colors = ['#4878CF'] * len(labels) + ['#E87722'] * (len(order) - len(labels))

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    plt.rcParams.update({'font.size': 12})
    for j, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[j]
        vals = [plot_df[plot_df['label'] == lbl][metric].mean() for lbl in order]
        ax.bar(range(len(order)), vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha='right', fontsize=9)
        ax.set_title(f'Wine Quality – {mlabel}', fontsize=13, fontweight='bold')
        ax.set_ylabel(mlabel, fontsize=11)

    fig.legend(handles=[Patch(facecolor='#4878CF', label='DeepPySR'),
                        Patch(facecolor='#E87722', label='PySR (reference)')],
               loc='upper right', fontsize=11, frameon=True)
    plt.suptitle('Ablation: VPS/VPR Effect (best APS per config, default r2w/λ)', fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout()
    out = os.path.join(current_dir, 'ablation_vps_vpr.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"VPS/VPR ablation saved to {out}")


def plot_pareto_ablation(all_df):
    """Ablation: pareto r2w/λ effect with fixed VPS=25, VPR=100, APS=10.0,
    averaged over wine types. Each r2w/λ grid point is scored genuinely:
    per fold, restrict candidates to that exact pareto_r2_weight/pareto_lambda,
    pick by in-fold fitness, evaluate only on that fold's held-out rows."""
    metrics = ['r2', 'rmse', 'mae', 'complexity']
    metric_labels = ['R²', 'RMSE', 'MAE', 'Complexity']

    rows = []
    for wine_type in WINE_TYPES:
        X, y = _load_xy(wine_type)
        deep_dir = os.path.join(current_dir, f"results_{wine_type}_all", 'deeppysr')
        target_variant = None
        if os.path.isdir(deep_dir):
            for variant in os.listdir(deep_dir):
                if '_vps25_' in variant and '_vpr100_' in variant and 'aps10.0' in variant:
                    target_variant = variant
                    break
        if target_variant is None:
            continue
        v_path = os.path.join(deep_dir, target_variant)
        for r2w in R2W_LIST:
            for lam in LAMBDA_LIST:
                formula, complexity, m = get_best_formula_from_raw(
                    v_path, X, y, task=TASK, model_type='deeppysr', stratify=y,
                    pareto_point=(r2w, lam))
                if not formula:
                    continue
                rows.append({'wine type': wine_type, 'label': f"r2w={r2w}, λ={lam}",
                             'r2': m[0], 'rmse': m[1], 'mae': m[2], 'complexity': complexity})
    deep_df = pd.DataFrame(rows)
    deep_agg = deep_df.groupby('label')[metrics].mean().reset_index() if not deep_df.empty else pd.DataFrame()

    pysr_sub = all_df[(all_df['family'] == 'pysr') & all_df['model'].str.contains('aps10.0', na=False)].copy()
    pysr_sub['label'] = 'PySR (reference)'

    csv_df = pd.concat([deep_df, pysr_sub[['wine type', 'label'] + metrics] if not pysr_sub.empty else pd.DataFrame()],
                        ignore_index=True)
    csv_df.to_csv(os.path.join(current_dir, 'ablation_pareto.csv'), index=False)
    print(f"Pareto ablation data saved to {os.path.join(current_dir, 'ablation_pareto.csv')}")

    if deep_agg.empty and pysr_sub.empty:
        print("No data for pareto ablation")
        return

    pysr_row = pysr_sub[metrics].mean().to_frame().T if not pysr_sub.empty else pd.DataFrame()
    if not pysr_row.empty:
        pysr_row['label'] = 'PySR\n(reference)'
    plot_df = pd.concat([deep_agg, pysr_row], ignore_index=True) if not deep_agg.empty else pysr_row

    def sort_key(lbl):
        r2w_m = re.search(r'r2w=([\d.]+)', lbl)
        l_m = re.search(r'λ=([\d.]+)', lbl)
        if r2w_m and l_m:
            return (float(r2w_m.group(1)), float(l_m.group(1)))
        return (999, 999)

    labels = sorted([l for l in plot_df['label'].unique() if 'PySR' not in l], key=sort_key)
    order = labels + [l for l in plot_df['label'].unique() if 'PySR' in l]

    r2w_vals = sorted(set(float(re.search(r'r2w=([\d.]+)', l).group(1))
                          for l in labels if re.search(r'r2w=([\d.]+)', l)))
    palette_colors = ['#2166ac', '#4dac26', '#d6604d']
    r2w_palette = dict(zip(r2w_vals, palette_colors * (len(r2w_vals) // len(palette_colors) + 1)))
    colors = []
    for lbl in order:
        m = re.search(r'r2w=([\d.]+)', lbl)
        colors.append(r2w_palette.get(float(m.group(1)), '#888') if m else '#E87722')

    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    plt.rcParams.update({'font.size': 12})
    for j, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[j]
        vals = [plot_df[plot_df['label'] == lbl][metric].mean() for lbl in order]
        ax.bar(range(len(order)), vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha='right', fontsize=9)
        ax.set_title(f'Wine Quality – {mlabel}', fontsize=13, fontweight='bold')
        ax.set_ylabel(mlabel, fontsize=11)

    legend_elements = [Patch(facecolor=r2w_palette[v], label=f'DeepPySR r2w={v}') for v in r2w_vals]
    legend_elements.append(Patch(facecolor='#E87722', label='PySR (reference)'))
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True)
    plt.suptitle('Ablation: Pareto r2w/λ Effect (VPS=25, VPR=100, APS=10.0)', fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout()
    out = os.path.join(current_dir, 'ablation_pareto.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pareto ablation saved to {out}")


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
    """Load hall_of_fame CSVs from pysr_outputs/y/ (the SR search's own
    training-time complexity/loss log -- optimizer dynamics, not a held-out
    generalization claim, so unaffected by the formula-selection leak)."""
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


def plot_pareto_front_rmse(all_df):
    """Pareto front per wine type: DeepPySR from hall_of_fame, PySR from aggregated variants."""
    n_types = len(WINE_TYPES)
    fig, axes = plt.subplots(1, n_types, figsize=(8 * n_types, 7), squeeze=False)

    for ax, wtype in zip(axes[0], WINE_TYPES):
        sub_df = all_df[all_df['wine type'] == wtype]
        deep_df = sub_df[(sub_df['family'] == 'deeppysr') & sub_df['max_complexity'].isna()].copy()
        pysr_df = sub_df[sub_df['family'] == 'pysr'].copy()
        pysr_df = pysr_df[pysr_df['rmse'].notna() & pysr_df['complexity'].notna()]

        hof_data = pd.DataFrame()
        if not deep_df.empty:
            model_dir = deep_df.loc[deep_df['r2'].idxmax(), 'source_path']
            hof_data = _load_hof_data(model_dir)

        hof_pysr = pd.DataFrame()
        if not pysr_df.empty:
            pysr_model_dir = pysr_df.loc[pysr_df['r2'].idxmax(), 'source_path']
            hof_pysr = _load_hof_data(pysr_model_dir)

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

        ax.set_xlabel('Complexity', fontsize=13)
        ax.set_ylabel('RMSE', fontsize=13)
        ax.set_title(f'Wine Quality ({wtype}) – Pareto Front: Complexity vs RMSE', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(current_dir, 'pareto_front_rmse.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pareto front RMSE plot saved to {out}")


if __name__ == "__main__":
    # process_results: every model's metrics come from genuine held-out CV
    # (predictions.csv for Best DeepPySR/PySR/KAN/KANSym, matching how
    # baselines are already scored; per-fold held-out formula evaluation only
    # for Interpretable DeepPySR, which has no predictions.csv of its own).
    all_df, best_df = process_results()

    save_results(best_df)
    compute_se_and_wilcoxon(best_df)
    aggregate_feature_importance()
    plot_best_models()
    plot_vps_vpr_ablation(all_df)
    plot_pareto_ablation(all_df)
    plot_pareto_front_rmse(all_df)
