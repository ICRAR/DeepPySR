import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add test/ and current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if not current_dir:
    current_dir = "."
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

from student_utils import load_student_data
from analysis_utils import (calculate_metrics, get_best_formula_from_raw,
                             collect_model_fold_data, se_from_fold_data,
                             run_wilcoxon_analysis, compute_se)

def process_results():
    subjects = ['mat', 'por']
    all_data = []
    for subject in subjects:
        base_dir = os.path.join(current_dir, f"results_{subject}_all")

        df = load_student_data(subject)
        X = df.drop(columns=['G3'])
        y = df['G3']
        
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
                        all_data.append(['KAN', subject, r2, rmse, mae, np.nan, ""])

                        # KANSym
                        if 'y_pred_kansym' in df_pred.columns:
                            formula, complexity, metrics = get_best_formula_from_raw(model_path, X, y, prefix='formulas_fold', model_type='kan')
                            r2, rmse, mae = metrics

                            if not formula:
                                r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred_kansym'])

                            all_data.append(['KANSym', subject, r2, rmse, mae, complexity, formula])
                    else:
                        # Other baselines
                        r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                        all_data.append([model_name, subject, r2, rmse, mae, np.nan, ""])

        # DeepPySR
        deeppysr_dir = os.path.join(base_dir, "deeppysr")
        if os.path.exists(deeppysr_dir):
            for variant in os.listdir(deeppysr_dir):
                v_path = os.path.join(deeppysr_dir, variant)
                if not os.path.isdir(v_path): continue

                res = get_best_formula_from_raw(v_path, X, y, model_type='deeppysr')

                if isinstance(res, dict):
                    for (r2w, lamb), (formula, complexity, metrics) in res.items():
                        r2, rmse, mae = metrics
                        model_name = f"{variant}_r2w{r2w}_L{lamb}"
                        all_data.append([model_name, subject, r2, rmse, mae, complexity, formula])
                else:
                    formula, complexity, metrics = res
                    r2, rmse, mae = metrics
                    if not formula:
                        pred_file = os.path.join(v_path, "predictions.csv")
                        if os.path.exists(pred_file):
                            df_pred = pd.read_csv(pred_file)
                            r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                    all_data.append([variant, subject, r2, rmse, mae, complexity, formula])

        # PySR
        pysr_dir = os.path.join(base_dir, "pysr")
        if os.path.exists(pysr_dir):
            for variant in os.listdir(pysr_dir):
                v_path = os.path.join(pysr_dir, variant)
                if not os.path.isdir(v_path): continue

                res = get_best_formula_from_raw(v_path, X, y, model_type='pysr')

                if isinstance(res, dict):
                    for (r2w, lamb), (formula, complexity, metrics) in res.items():
                        r2, rmse, mae = metrics
                        model_name = f"{variant}_r2w{r2w}_L{lamb}"
                        all_data.append([model_name, subject, r2, rmse, mae, complexity, formula])
                else:
                    formula, complexity, metrics = res
                    r2, rmse, mae = metrics
                    if not formula:
                        pred_file = os.path.join(v_path, "predictions.csv")
                        if os.path.exists(pred_file):
                            df_pred = pd.read_csv(pred_file)
                            r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                    all_data.append([variant, subject, r2, rmse, mae, complexity, formula])

    # Create DataFrame and save
    result_df = pd.DataFrame(all_data, columns=['model', 'subject', 'r2', 'rmse', 'mae', 'complexity', 'formula'])
    result_df['r2'] = result_df['r2'].clip(lower=0)
    result_df.to_csv(os.path.join(current_dir, "aggregated_results.csv"), index=False)
    print(f"Results saved to {os.path.join(current_dir, 'aggregated_results.csv')}")
    return result_df


def compute_se_and_wilcoxon(result_df):
    """Compute per-fold SE and Wilcoxon vs DeepPySR (best R²) for each subject."""
    task = 'regression'

    display_to_fold = {
        'Best DeepPySR': 'DeepPySR_best',
        'Interpretable DeepPySR': 'DeepPySR_best',
        'Best PySR': 'PySR',
    }

    all_se_maps = {}  # subject -> se_map

    for subject in ['mat', 'por']:
        df_sub = load_student_data(subject)
        X = df_sub.drop(columns=['G3'])
        y = df_sub['G3']
        base_dir = os.path.join(current_dir, f"results_{subject}_all")
        sub_df = result_df[result_df['subject'] == subject]

        fold_data = {}

        baselines_dir = os.path.join(base_dir, "baselines")
        if os.path.exists(baselines_dir):
            for model_name in os.listdir(baselines_dir):
                model_path = os.path.join(baselines_dir, model_name)
                if not os.path.isdir(model_path):
                    continue
                row = sub_df[sub_df['model'] == model_name]
                formula = row['formula'].iloc[0] if not row.empty else ""
                fold_data[model_name] = collect_model_fold_data(
                    model_path, formula, X, y, task,
                    model_type='kan' if model_name.lower() == 'kansym' else 'pysr')

        deeppysr_sub = sub_df[sub_df['model'].str.contains('fullsr|stdsr|srprn|srpsm', na=False)]
        if not deeppysr_sub.empty:
            best_row = deeppysr_sub.loc[deeppysr_sub['r2'].idxmax()]
            fold_data['DeepPySR_best'] = collect_model_fold_data(
                base_dir, best_row['formula'], X, y, task, model_type='deeppysr')

        pysr_sub = sub_df[sub_df['model'].str.contains('pysr', na=False)]
        if not pysr_sub.empty:
            best_pysr = pysr_sub.loc[pysr_sub['r2'].idxmax()]
            fold_data['PySR'] = collect_model_fold_data(
                base_dir, best_pysr['formula'], X, y, task, model_type='pysr')

        se_map = {}
        for model_name, fd in fold_data.items():
            if fd is not None:
                ses = se_from_fold_data(fd)
                ses['n_folds'] = len(fd)
                se_map[model_name] = ses
        all_se_maps[subject] = se_map

        run_wilcoxon_analysis(fold_data, 'DeepPySR_best', task,
                              output_file=os.path.join(current_dir, f"wilcoxon_results_{subject}.csv"))

    # Merge SE into student_best_models_metrics.csv
    metrics_csv_path = os.path.join(current_dir, 'student_best_models_metrics.csv')
    if all_se_maps and os.path.exists(metrics_csv_path):
        metrics_df = pd.read_csv(metrics_csv_path)
        first_se = next((s for sm in all_se_maps.values() for s in sm.values()), {})
        se_cols = [c for c in first_se.keys() if c != 'n_folds']
        for col in se_cols + ['n_folds']:
            metrics_df[col] = np.nan
        for i, row in metrics_df.iterrows():
            subj = row['subject']
            sm = all_se_maps.get(subj, {})
            fold_key = display_to_fold.get(row['display_model'], row['display_model'])
            if fold_key in sm:
                for col in se_cols + ['n_folds']:
                    metrics_df.at[i, col] = sm[fold_key].get(col, np.nan)
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"SE merged into {metrics_csv_path}")


def save_results(df):
    df = df.copy()
    df['r2'] = df['r2'].clip(lower=0)

    subjects = ['mat', 'por']
    selected_data = []
    interpretable_formulas = []

    for s in subjects:
        sub_df = df[df['subject'] == s]
        if sub_df.empty:
            continue

        # DeepPySR variants
        deeppysr_df = sub_df[sub_df['model'].str.contains('fullsr|stdsr|srprn|srpsm', na=False)]
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
                    'subject': s, 'model': interp_deeppysr['model'],
                    'formula': interp_deeppysr['formula'], 'r2': interp_deeppysr['r2'], 'complexity': interp_deeppysr['complexity']
                })

        # PySR variants
        pysr_df = sub_df[sub_df['model'].str.contains('pysr', na=False)]
        if not pysr_df.empty:
            best_pysr = pysr_df.loc[pysr_df['r2'].idxmax()].copy()
            best_pysr['display_model'] = 'Best PySR'
            selected_data.append(best_pysr)

        # KAN and KANSym
        for m in ['KAN', 'KANSym']:
            m_df = sub_df[sub_df['model'] == m]
            if not m_df.empty:
                m_row = m_df.iloc[0].copy()
                m_row['display_model'] = m
                selected_data.append(m_row)

        # Other baselines
        baselines = ['ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']
        for b in baselines:
            b_df = sub_df[sub_df['model'] == b]
            if not b_df.empty:
                b_row = b_df.iloc[0].copy()
                b_row['display_model'] = b
                selected_data.append(b_row)

    plot_df = pd.DataFrame(selected_data)
    plot_csv_path = os.path.join(current_dir, 'student_best_models_metrics.csv')
    plot_df.to_csv(plot_csv_path, index=False)
    print(f"Best models plot data saved to {plot_csv_path}")

    print("\n--- Interpretable DeepPySR Formulas (Complexity < 25) ---")
    interp_df = pd.DataFrame(interpretable_formulas)
    print(interp_df.to_string(index=False))
    interp_csv_path = os.path.join(current_dir, 'interpretable_deeppysr_formulas.csv')
    interp_df.to_csv(interp_csv_path, index=False)

def aggregate_feature_importance():
    subjects = ['mat', 'por']
    importance_data = []
    for sub in subjects:
        base_dir = os.path.join(current_dir, f"results_{sub}_all")

        def process_importance(path, model_name, sub):
            if os.path.exists(path):
                df_imp = pd.read_csv(path)
                if 'feature' in df_imp.columns and 'importance' in df_imp.columns:
                    total = df_imp['importance'].sum()
                    if total > 0:
                        df_imp['importance_pct'] = (df_imp['importance'] / total) * 100
                    else:
                        df_imp['importance_pct'] = 0

                    for _, row in df_imp.iterrows():
                        importance_data.append({
                            'model': model_name,
                            'subject': sub,
                            'variable': row['feature'],
                            'weight': row['importance_pct']
                        })

        if os.path.exists(base_dir):
            baselines_dir = os.path.join(base_dir, "baselines")
            if os.path.exists(baselines_dir):
                for m in os.listdir(baselines_dir):
                    if m in ['ElasticNet', 'ExtraTrees', 'RandomForest', 'XGBoost', 'KAN']:
                        imp_file = os.path.join(baselines_dir, m, "feature_importance.csv")
                        process_importance(imp_file, m, sub)

    if not importance_data:
        print("No feature importance data found.")
        return

    imp_df = pd.DataFrame(importance_data)
    imp_df.to_csv(os.path.join(current_dir, "feature_importance_aggregated.csv"), index=False)
    print("Feature importance aggregated to feature_importance_aggregated.csv")

    agg_imp = imp_df.groupby(['model', 'variable'])['weight'].mean().reset_index()
    top_features = agg_imp.groupby('variable')['weight'].mean().sort_values(ascending=False).head(15).index

    plot_df = agg_imp[agg_imp['variable'].isin(top_features)].copy()
    plot_df['variable'] = pd.Categorical(plot_df['variable'], categories=top_features, ordered=True)

    plt.figure(figsize=(14, 10))
    sns.barplot(data=plot_df, x='weight', y='variable', hue='model', palette="bright")
    plt.title('Feature Importance Comparison across Models (Student)', fontsize=22, fontweight='bold', pad=20)
    plt.xlabel('Average Percentage Importance (%)', fontsize=18)
    plt.ylabel('Feature', fontsize=18)
    plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(current_dir, "feature_importance_by_model.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined feature importance plot saved to {plot_path}")

def plot_best_models():
    csv_path = os.path.join(current_dir, 'student_best_models_metrics.csv')
    if not os.path.exists(csv_path):
        print(f"No best models data found at {csv_path}")
        return
    df = pd.read_csv(csv_path)

    subjects = ['mat', 'por']
    metrics = ['r2', 'rmse', 'mae', 'complexity']
    models_to_include_for_complexity = ['Best DeepPySR', 'Interpretable DeepPySR', 'Best PySR', 'KANSym']
    label_map = {
        'Best DeepPySR': 'DeepPySR',
        'Interpretable DeepPySR': 'InterpDeepPySR'
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    for i, sub in enumerate(subjects):
        sub_df_all = df[df['subject'] == sub].copy()
        sub_df_all = sub_df_all.sort_values('display_model')

        sub_df_complexity = sub_df_all[sub_df_all['display_model'].isin(models_to_include_for_complexity)].copy()

        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            if metric == 'complexity':
                plot_df = sub_df_complexity.copy()
            else:
                plot_df = sub_df_all.copy()

            if plot_df.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
                ax.set_title(f'{sub.upper()} - {metric.upper()}')
                continue

            plot_df['plot_label'] = plot_df['display_model'].replace(label_map)
            ax.bar(plot_df['plot_label'], plot_df[metric])
            ax.set_title(f'{sub.upper()} - {metric.upper()}')
            ax.set_ylabel(metric.upper())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

    plt.tight_layout()
    plot_path = os.path.join(current_dir, 'best_models_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")

def plot_vps_vpr_ablation(df):
    """Ablation: VPS/VPR effect with fixed APS=10.0, r2w=1.0, λ=0.01. Averaged over subjects."""
    import re
    from matplotlib.patches import Patch

    metrics = ['r2', 'rmse', 'mae', 'complexity']
    metric_labels = ['R²', 'RMSE', 'MAE', 'Complexity']

    deep_mask = df['model'].str.contains('fullsr', regex=False, na=False)
    deep_df = df[deep_mask].copy()

    def vps_vpr_label(m):
        match = re.search(r'vps(\d+)_vpr(\d+)', m)
        return f"vps{match.group(1)}/vpr{match.group(2)}" if match else m
    deep_df['label'] = deep_df['model'].apply(vps_vpr_label)
    # Best R² per (subject, vps/vpr config) across all aps/r2w/λ
    deep_df = deep_df.loc[deep_df.groupby(['subject', 'label'])['r2'].idxmax()].reset_index(drop=True)
    deep_agg = deep_df.groupby('label')[metrics].mean().reset_index()

    pysr_mask = df['model'].str.contains(r'^pysr', regex=True, na=False)
    pysr_sub = df[pysr_mask].copy()
    pysr_sub = pysr_sub.loc[pysr_sub.groupby('subject')['r2'].idxmax()].reset_index(drop=True)
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
        ax.set_title(f'Student Performance – {mlabel}', fontsize=13, fontweight='bold')
        ax.set_ylabel(mlabel, fontsize=11)

    fig.legend(handles=[Patch(facecolor='#4878CF', label='DeepPySR'),
                        Patch(facecolor='#E87722', label='PySR (reference)')],
               loc='upper right', fontsize=11, frameon=True)
    plt.suptitle('Ablation: VPS/VPR Effect (APS=10.0, r2w=1.0, λ=0.01)', fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout()
    out = os.path.join(current_dir, 'ablation_vps_vpr.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"VPS/VPR ablation saved to {out}")


def plot_pareto_ablation(df):
    """Ablation: pareto r2w/λ effect with fixed VPS=25, VPR=100, APS=10.0. Averaged over subjects."""
    import re
    from matplotlib.patches import Patch

    metrics = ['r2', 'rmse', 'mae', 'complexity']
    metric_labels = ['R²', 'RMSE', 'MAE', 'Complexity']

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
    deep_agg = deep_df.groupby('label')[metrics].mean().reset_index()

    pysr_mask = (df['model'].str.contains(r'^pysr', regex=True, na=False) &
                 df['model'].str.contains('aps10.0', regex=False, na=False))
    pysr_sub = df[pysr_mask].copy()
    pysr_sub['label'] = 'PySR (reference)'

    csv_df = pd.concat([deep_df[['label'] + metrics], pysr_sub[['label'] + metrics]], ignore_index=True)
    csv_df.to_csv(os.path.join(current_dir, 'ablation_pareto.csv'), index=False)
    print(f"Pareto ablation data saved to {os.path.join(current_dir, 'ablation_pareto.csv')}")

    if deep_agg.empty and pysr_sub.empty:
        print("No data for pareto ablation")
        return

    pysr_row = pysr_sub[metrics].mean().to_frame().T
    pysr_row['label'] = 'PySR\n(reference)'
    plot_df = pd.concat([deep_agg, pysr_row[['label'] + metrics]], ignore_index=True)

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
    r2w_palette = dict(zip(r2w_vals, ['#2166ac', '#4dac26', '#d6604d']))
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
        ax.set_title(f'Student Performance – {mlabel}', fontsize=13, fontweight='bold')
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
    """Return Pareto-optimal (complexity, error) pairs sorted by complexity (both minimize)."""
    points = sorted(zip(complexity, error), key=lambda p: (p[0], p[1]))
    pareto = []
    min_error = float('inf')
    for c, e in points:
        if e < min_error:
            min_error = e
            pareto.append((c, e))
    return pareto


def _load_hof_data(model_dir):
    """Load hall_of_fame CSVs from pysr_outputs/y/ sorted by timestamp (fold order).
    Returns DataFrame with (complexity, rmse) where rmse = mean sqrt(Loss) across folds."""
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


def plot_pareto_front_rmse(df):
    """Pareto front per subject: DeepPySR from hall_of_fame, PySR from aggregated variants."""
    import re
    subjects = df['subject'].unique() if 'subject' in df.columns else [None]
    n_sub = len(subjects)
    fig, axes = plt.subplots(1, n_sub, figsize=(8 * n_sub, 7), squeeze=False)

    subject_to_results_dir = {'mat': 'results_mat_all', 'por': 'results_por_all'}

    for ax, subject in zip(axes[0], subjects):
        sub_df = df[df['subject'] == subject] if subject is not None else df
        deep_df = sub_df[sub_df['model'].str.contains('fullsr', regex=False, na=False)].copy()
        pysr_df = sub_df[sub_df['model'].str.contains(r'^pysr', regex=True, na=False)].copy()
        pysr_df = pysr_df[pysr_df['rmse'].notna() & pysr_df['complexity'].notna()]

        # Load full Pareto hall_of_fame for the best DeepPySR model
        hof_data = pd.DataFrame()
        if not deep_df.empty:
            best_name = deep_df.loc[deep_df['r2'].idxmax(), 'model']
            base_model = re.sub(r'_r2w[\d.]+_L[\d.]+$', '', best_name)
            res_dir = subject_to_results_dir.get(subject, f'results_{subject}_all') if subject else 'results_all'
            model_dir = os.path.join(current_dir, res_dir, 'deeppysr', base_model)
            hof_data = _load_hof_data(model_dir)

        # Load hall_of_fame for best PySR model (only plotted once saved by save_pysr_hof.py)
        hof_pysr = pd.DataFrame()
        if not pysr_df.empty:
            best_pysr_name = re.sub(r'_r2w[\d.]+_L[\d.]+$', '',
                                    pysr_df.loc[pysr_df['r2'].idxmax(), 'model'])
            pysr_model_dir = os.path.join(current_dir, res_dir, 'pysr', best_pysr_name)
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

        title = f'Student ({subject}) – Pareto Front: Complexity vs RMSE' if subject else 'Student – Pareto Front: Complexity vs RMSE'
        ax.set_xlabel('Complexity', fontsize=13)
        ax.set_ylabel('RMSE', fontsize=13)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(current_dir, 'pareto_front_rmse.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pareto front RMSE plot saved to {out}")


if __name__ == "__main__":
    df = process_results()
    save_results(df)
    compute_se_and_wilcoxon(df)
    aggregate_feature_importance()
    plot_best_models()
    plot_vps_vpr_ablation(df)
    plot_pareto_ablation(df)
    plot_pareto_front_rmse(df)
