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

from diabetes_utils import load_diabetes_brfss_data
from analysis_utils import (calculate_metrics, get_best_formula_from_raw,
                             collect_model_fold_data, se_from_fold_data,
                             run_wilcoxon_analysis, compute_se)

def process_results():
    all_data = []
    base_dir = os.path.join(current_dir, "results_diabetes_all")

    X, y = load_diabetes_brfss_data()
    task = 'classification'

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
                    acc, prec, rec, f1, auc = calculate_metrics(df_pred['y_true'], df_pred['y_pred'], task=task)
                    all_data.append(['KAN', acc, prec, rec, f1, auc, np.nan, ""])

                    # KANSym
                    if 'y_pred_kansym' in df_pred.columns:
                        # For KANSym, we need formula and complexity
                        # The user wants us to check all formulas and pick the best one
                        formula, complexity, metrics = get_best_formula_from_raw(model_path, X, y, prefix='formulas_fold', model_type='kan', task=task)
                        acc, prec, rec, f1, auc = metrics

                        all_data.append(['KANSym', acc, prec, rec, f1, auc, complexity, formula])
                else:
                    # Other baselines
                    acc, prec, rec, f1, auc = calculate_metrics(df_pred['y_true'], df_pred['y_pred'], task=task)
                    all_data.append([model_name, acc, prec, rec, f1, auc, np.nan, ""])

    # DeepPySR
    deeppysr_dir = os.path.join(base_dir, "deeppysr")
    if os.path.exists(deeppysr_dir):
        for variant in os.listdir(deeppysr_dir):
            v_path = os.path.join(deeppysr_dir, variant)
            if not os.path.isdir(v_path): continue

            res = get_best_formula_from_raw(v_path, X, y, task=task, model_type='deeppysr')

            if isinstance(res, dict):
                for (r2w, lamb), (formula, complexity, metrics) in res.items():
                    acc, prec, rec, f1, auc = metrics
                    model_name = f"{variant}_r2w{r2w}_L{lamb}"
                    all_data.append([model_name, acc, prec, rec, f1, auc, complexity, formula])
            else:
                formula, complexity, metrics = res
                acc, prec, rec, f1, auc = metrics
                if not formula:
                    pred_file = os.path.join(v_path, "predictions.csv")
                    if os.path.exists(pred_file):
                        df_pred = pd.read_csv(pred_file)
                        acc, prec, rec, f1, auc = calculate_metrics(df_pred['y_true'], df_pred['y_pred'], task=task)
                all_data.append([variant, acc, prec, rec, f1, auc, complexity, formula])

    # PySR
    pysr_dir = os.path.join(base_dir, "pysr")
    if os.path.exists(pysr_dir):
        for variant in os.listdir(pysr_dir):
            v_path = os.path.join(pysr_dir, variant)
            if not os.path.isdir(v_path): continue

            # Use overall_metrics.csv for PySR if it exists
            overall_metrics_file = os.path.join(v_path, "overall_metrics.csv")
            if os.path.exists(overall_metrics_file):
                df_metrics = pd.read_csv(overall_metrics_file)
                acc = df_metrics['accuracy'].iloc[0]
                prec = df_metrics['precision'].iloc[0]
                rec = df_metrics['recall'].iloc[0]
                f1 = df_metrics['f1'].iloc[0]
                auc = df_metrics['auc'].iloc[0]
                
                # We still might want formula and complexity for display
                res = get_best_formula_from_raw(v_path, X, y, task=task, model_type='pysr')
                if isinstance(res, dict):
                    # Pick the first one or best one for complexity/formula
                    key = list(res.keys())[0]
                    formula, complexity, _ = res[key]
                else:
                    formula, complexity, _ = res
                
                all_data.append([variant, acc, prec, rec, f1, auc, complexity, formula])
            else:
                res = get_best_formula_from_raw(v_path, X, y, task=task, model_type='pysr')

                if isinstance(res, dict):
                    for (r2w, lamb), (formula, complexity, metrics) in res.items():
                        acc, prec, rec, f1, auc = metrics
                        model_name = f"{variant}_r2w{r2w}_L{lamb}"
                        all_data.append([model_name, acc, prec, rec, f1, auc, complexity, formula])
                else:
                    formula, complexity, metrics = res
                    acc, prec, rec, f1, auc = metrics
                    if not formula:
                        pred_file = os.path.join(v_path, "predictions.csv")
                        if os.path.exists(pred_file):
                            df_pred = pd.read_csv(pred_file)
                            acc, prec, rec, f1, auc = calculate_metrics(df_pred['y_true'], df_pred['y_pred'], task=task)
                    all_data.append([variant, acc, prec, rec, f1, auc, complexity, formula])

    # Create DataFrame and save
    result_df = pd.DataFrame(all_data, columns=['model', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'complexity', 'formula'])
    result_df.to_csv(os.path.join(current_dir, "aggregated_results.csv"), index=False)
    print(f"Results saved to {os.path.join(current_dir, 'aggregated_results.csv')}")
    return result_df


def compute_se_and_wilcoxon(result_df):
    """Compute per-fold SE and run Wilcoxon vs DeepPySR (best F1)."""
    X, y = load_diabetes_brfss_data()
    task = 'classification'
    base_dir = os.path.join(current_dir, "results_diabetes_all")

    fold_data = {}

    baselines_dir = os.path.join(base_dir, "baselines")
    if os.path.exists(baselines_dir):
        for model_name in os.listdir(baselines_dir):
            model_path = os.path.join(baselines_dir, model_name)
            if not os.path.isdir(model_path):
                continue
            row = result_df[result_df['model'] == model_name]
            formula = row['formula'].iloc[0] if not row.empty else ""
            fold_data[model_name] = collect_model_fold_data(
                model_path, formula, X, y, task,
                model_type='kan' if model_name.lower() == 'kansym' else 'pysr')

    deeppysr_df = result_df[result_df['model'].str.contains('fullsr|stdsr|srprn|srpsm', na=False)]
    if not deeppysr_df.empty:
        best_row = deeppysr_df.loc[deeppysr_df['f1'].idxmax()]
        fold_data['DeepPySR_best'] = collect_model_fold_data(
            base_dir, best_row['formula'], X, y, task, model_type='deeppysr')

    pysr_df = result_df[result_df['model'].str.contains('pysr', na=False)]
    if not pysr_df.empty:
        best_pysr = pysr_df.loc[pysr_df['f1'].idxmax()]
        fold_data['PySR'] = collect_model_fold_data(
            base_dir, best_pysr['formula'], X, y, task, model_type='pysr')

    # SE summary — merge into best_models_metrics CSV
    se_map = {}
    for model_name, fd in fold_data.items():
        if fd is not None:
            ses = se_from_fold_data(fd)
            ses['n_folds'] = len(fd)
            se_map[model_name] = ses

    display_to_fold = {
        'Best DeepPySR': 'DeepPySR_best',
        'Interpretable DeepPySR': 'DeepPySR_best',
        'Best PySR': 'PySR',
    }

    metrics_csv_path = os.path.join(current_dir, 'diabetes_best_models_metrics.csv')
    if se_map and os.path.exists(metrics_csv_path):
        metrics_df = pd.read_csv(metrics_csv_path)
        se_cols = [c for c in next(iter(se_map.values())).keys() if c != 'n_folds']
        for col in se_cols + ['n_folds']:
            metrics_df[col] = np.nan
        for i, row in metrics_df.iterrows():
            fold_key = display_to_fold.get(row['display_model'], row['display_model'])
            if fold_key in se_map:
                for col in se_cols + ['n_folds']:
                    metrics_df.at[i, col] = se_map[fold_key].get(col, np.nan)
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"SE merged into {metrics_csv_path}")

    run_wilcoxon_analysis(fold_data, 'DeepPySR_best', task,
                          output_file=os.path.join(current_dir, "wilcoxon_results.csv"))


def save_results(df):
    """
    Select best models and save interpretable formulas.
    """
    selected_data = []
    interpretable_formulas = []

    # DeepPySR variants
    deeppysr_df = df[df['model'].str.contains('fullsr|stdsr|srprn|srpsm', na=False)]
    if not deeppysr_df.empty:
        best_deeppysr = deeppysr_df.loc[deeppysr_df['f1'].idxmax()].copy()
        best_deeppysr['display_model'] = 'Best DeepPySR'
        selected_data.append(best_deeppysr)

        interp_deeppysr_df = deeppysr_df[deeppysr_df['complexity'] < 35]
        if not interp_deeppysr_df.empty:
            interp_deeppysr = interp_deeppysr_df.loc[interp_deeppysr_df['f1'].idxmax()].copy()
            interp_deeppysr['display_model'] = 'Interpretable DeepPySR'
            selected_data.append(interp_deeppysr)
            interpretable_formulas.append({
                'model': interp_deeppysr['model'],
                'formula': interp_deeppysr['formula'], 'f1': interp_deeppysr['f1'], 'complexity': interp_deeppysr['complexity']
            })

    # PySR variants
    pysr_df = df[df['model'].str.contains('pysr', na=False)]
    if not pysr_df.empty:
        best_pysr = pysr_df.loc[pysr_df['f1'].idxmax()].copy()
        best_pysr['display_model'] = 'Best PySR'
        selected_data.append(best_pysr)

    # KAN and KANSym
    for m in ['KAN', 'KANSym']:
        m_df = df[df['model'] == m]
        if not m_df.empty:
            m_row = m_df.iloc[0].copy()
            m_row['display_model'] = m
            selected_data.append(m_row)

    # Other baselines (ElasticNet, ExtraTrees, MLP, RandomForest, XGBoost)
    baselines = ['ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']
    for b in baselines:
        b_df = df[df['model'] == b]
        if not b_df.empty:
            b_row = b_df.iloc[0].copy()
            b_row['display_model'] = b
            selected_data.append(b_row)

    plot_df = pd.DataFrame(selected_data)

    # Save the plot data for the best models to CSV
    plot_csv_path = os.path.join(current_dir, 'diabetes_best_models_metrics.csv')
    plot_df.to_csv(plot_csv_path, index=False)
    print(f"Best models plot data saved to {plot_csv_path}")

    # Print interpretable DeepPySR formulas
    print("\n--- Interpretable DeepPySR Formulas (Complexity < 35) ---")
    interp_df = pd.DataFrame(interpretable_formulas)
    print(interp_df.to_string(index=False))
    interp_csv_path = os.path.join(current_dir, 'interpretable_deeppysr_formulas.csv')
    interp_df.to_csv(interp_csv_path, index=False)

def aggregate_feature_importance():
    """
    Aggregate feature importance for ElasticNet, ExtraTrees, RandomForest, XGBoost, KAN.
    Average across folds, percentage it.
    """
    importance_data = []
    base_dir = os.path.join(current_dir, "results_diabetes_all")

    # Helper to process importance file
    def process_importance(path, model_name):
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
                        'variable': row['feature'],
                        'weight': row['importance_pct']
                    })

    if os.path.exists(base_dir):
        baselines_dir = os.path.join(base_dir, "baselines")
        if os.path.exists(baselines_dir):
            for m in os.listdir(baselines_dir):
                if m in ['ElasticNet', 'ExtraTrees', 'RandomForest', 'XGBoost', 'KAN']:
                    imp_file = os.path.join(baselines_dir, m, "feature_importance.csv")
                    process_importance(imp_file, m)

    imp_df = pd.DataFrame(importance_data)
    imp_df.to_csv(os.path.join(base_dir, "feature_importance_aggregated.csv"), index=False)
    print("Feature importance aggregated to feature_importance_aggregated.csv")

    # Grouped bar plot for all models comparison
    if not imp_df.empty:
        # Average importance across models per variable
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

def plot_best_models():
    """
    Create a plot with 1 row and 5 columns (accuracy, precision, recall, f1, complexity).
    Each subplot shows metric values for the models.
    """
    df = pd.read_csv(os.path.join(current_dir, 'diabetes_best_models_metrics.csv'))

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'complexity']
    models_to_include_for_complexity = ['Best DeepPySR', 'Interpretable DeepPySR', 'Best PySR', 'KANSym']
    label_map = {
        'Best DeepPySR': 'DeepPySR',
        'Interpretable DeepPySR': 'InterpDeepPySR'
    }

    fig, axes = plt.subplots(1, 5, figsize=(20, 6))

    df_all = df.copy()
    df_all = df_all.sort_values('display_model')

    df_complexity = df_all[df_all['display_model'].isin(models_to_include_for_complexity)].copy()

    for j, metric in enumerate(metrics):
        ax = axes[j]

        if metric == 'complexity':
            plot_df = df_complexity.copy()
        else:
            plot_df = df_all.copy()

        if plot_df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            ax.set_title(f'Diabetes Disease - {metric.upper()}')
            ax.set_xlabel('Model', fontsize=8)
            ax.set_ylabel(metric.upper())
            ax.set_xticks([])
            continue

        plot_df['plot_label'] = plot_df['display_model'].replace(label_map)
        ax.bar(plot_df['plot_label'], plot_df[metric])
        ax.set_title(f'Diabetes Disease - {metric.upper()}')
        ax.set_xlabel('Model', fontsize=8)
        ax.set_ylabel(metric.upper())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

    plt.tight_layout()
    plot_path = os.path.join(current_dir, 'best_models_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")

def plot_vps_vpr_ablation(df):
    """Ablation: VPS/VPR effect with fixed APS=10.0, r2w=1.0, λ=0.01."""
    import re
    from matplotlib.patches import Patch

    metrics = ['accuracy', 'f1', 'auc', 'complexity']
    metric_labels = ['Accuracy', 'F1', 'AUC', 'Complexity']

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
    pysr_sub = df[pysr_mask].copy()
    pysr_sub['label'] = 'PySR (no VPS/VPR)'

    csv_df = pd.concat([deep_df[['label'] + metrics], pysr_sub[['label'] + metrics]], ignore_index=True)
    csv_df.to_csv(os.path.join(current_dir, 'results_diabetes_all', 'ablation_vps_vpr.csv'), index=False)
    print(f"VPS/VPR ablation data saved to {os.path.join(current_dir, 'results_diabetes_all', 'ablation_vps_vpr.csv')}")

    if deep_df.empty and pysr_sub.empty:
        print("No data for VPS/VPR ablation")
        return

    pysr_row = pysr_sub[metrics].mean().to_frame().T
    pysr_row['label'] = 'PySR\n(no VPS/VPR)'
    plot_df = pd.concat([deep_df[['label'] + metrics], pysr_row[['label'] + metrics]], ignore_index=True)

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
        ax.set_title(f'Diabetes – {mlabel}', fontsize=13, fontweight='bold')
        ax.set_ylabel(mlabel, fontsize=11)

    fig.legend(handles=[Patch(facecolor='#4878CF', label='DeepPySR'),
                        Patch(facecolor='#E87722', label='PySR (reference)')],
               loc='upper right', fontsize=11, frameon=True)
    plt.suptitle('Ablation: VPS/VPR Effect (APS=10.0, r2w=1.0, λ=0.01)', fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout()
    out = os.path.join(current_dir, 'results_diabetes_all', 'ablation_vps_vpr.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"VPS/VPR ablation saved to {out}")


def plot_pareto_ablation(df):
    """Ablation: pareto r2w/λ effect with fixed VPS=25, VPR=100, APS=10.0."""
    import re
    from matplotlib.patches import Patch

    metrics = ['accuracy', 'f1', 'auc', 'complexity']
    metric_labels = ['Accuracy', 'F1', 'AUC', 'Complexity']

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
    pysr_sub = df[pysr_mask].copy()
    pysr_sub['label'] = 'PySR (reference)'

    csv_df = pd.concat([deep_df[['label'] + metrics], pysr_sub[['label'] + metrics]], ignore_index=True)
    csv_df.to_csv(os.path.join(current_dir, 'results_diabetes_all', 'ablation_pareto.csv'), index=False)
    print(f"Pareto ablation data saved to {os.path.join(current_dir, 'results_diabetes_all', 'ablation_pareto.csv')}")

    if deep_df.empty and pysr_sub.empty:
        print("No data for pareto ablation")
        return

    pysr_row = pysr_sub[metrics].mean().to_frame().T
    pysr_row['label'] = 'PySR\n(reference)'
    plot_df = pd.concat([deep_df[['label'] + metrics], pysr_row[['label'] + metrics]], ignore_index=True)

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
        ax.set_title(f'Diabetes – {mlabel}', fontsize=13, fontweight='bold')
        ax.set_ylabel(mlabel, fontsize=11)

    legend_elements = [Patch(facecolor=r2w_palette[v], label=f'DeepPySR r2w={v}') for v in r2w_vals]
    legend_elements.append(Patch(facecolor='#E87722', label='PySR (reference)'))
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True)
    plt.suptitle('Ablation: Pareto r2w/λ Effect (VPS=25, VPR=100, APS=10.0)', fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout()
    out = os.path.join(current_dir, 'results_diabetes_all', 'ablation_pareto.png')
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
    """Pareto front: DeepPySR uses hall_of_fame (training RMSE), PySR uses 1-F1 from aggregated variants."""
    import re
    results_dir = os.path.join(current_dir, 'results_diabetes_all')

    deep_df = df[df['model'].str.contains('fullsr', regex=False, na=False)].copy()
    pysr_df = df[df['model'].str.contains(r'^pysr', regex=True, na=False)].copy()
    pysr_df = pysr_df[pysr_df['complexity'].notna()].copy()
    pysr_df['error'] = 1.0 - pysr_df['f1'].fillna(0)

    # Load full Pareto hall_of_fame for the best DeepPySR model
    hof_data = pd.DataFrame()
    if not deep_df.empty:
        best_name = deep_df.loc[deep_df['f1'].idxmax(), 'model']
        base_model = re.sub(r'_r2w[\d.]+_L[\d.]+$', '', best_name)
        model_dir = os.path.join(current_dir, 'results_diabetes_all', 'deeppysr', base_model)
        hof_data = _load_hof_data(model_dir)

    # Load hall_of_fame for best PySR model (only plotted once saved by save_pysr_hof.py)
    hof_pysr = pd.DataFrame()
    if not pysr_df.empty:
        best_pysr_name = re.sub(r'_r2w[\d.]+_L[\d.]+$', '',
                                pysr_df.loc[pysr_df['f1'].idxmax(), 'model'])
        pysr_model_dir = os.path.join(current_dir, 'results_diabetes_all', 'pysr', best_pysr_name)
        hof_pysr = _load_hof_data(pysr_model_dir)

    if hof_data.empty and hof_pysr.empty:
        print("No data for pareto front plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    if not hof_data.empty:
        pf = _pareto_front_steps(hof_data['complexity'].tolist(), hof_data['rmse'].tolist())
        if pf:
            px, py = zip(*pf)
            ax.step(px, py, where='post', color='#2166ac', linewidth=2, zorder=4)
            ax.scatter(px, py, c='#2166ac', s=100, zorder=5, marker='D', label='DeepPySR (train RMSE)')

    if not hof_pysr.empty:
        pf_pysr = _pareto_front_steps(hof_pysr['complexity'].tolist(), hof_pysr['rmse'].tolist())
        if pf_pysr:
            px, py = zip(*pf_pysr)
            ax.step(px, py, where='post', color='#cc4400', linewidth=2, zorder=4)
            ax.scatter(px, py, c='#cc4400', s=100, zorder=5, marker='D', label='PySR (train RMSE)')

    ax.set_xlabel('Complexity', fontsize=13)
    ax.set_ylabel('Error', fontsize=13)
    ax.set_title('Diabetes – Pareto Front: Complexity vs Error', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(results_dir, 'pareto_front_rmse.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pareto front plot saved to {out}")


if __name__ == "__main__":
    # process_results: aggregate all the results from the 5 fold cv, select one formula among the 5 which achieves the highest f1.
    # The f1 is calculated by applying this formula on the entire dataset, not the fold.

    df = process_results()

    save_results(df)
    compute_se_and_wilcoxon(df)
    aggregate_feature_importance()
    plot_best_models()
    plot_vps_vpr_ablation(df)
    plot_pareto_ablation(df)
    plot_pareto_front_rmse(df)