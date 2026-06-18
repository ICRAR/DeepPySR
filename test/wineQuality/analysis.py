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

from wine_utils import load_wine_data
from analysis_utils import calculate_metrics, get_best_formula_from_raw

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
                            formula, complexity, metrics = get_best_formula_from_raw(model_path, X, y, prefix='formulas_fold',model_type='kan')
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

                res = get_best_formula_from_raw(v_path, X, y,model_type='deeppysr')

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

        # PySR
        pysr_dir = os.path.join(base_dir, "pysr")
        if os.path.exists(pysr_dir):
            for variant in os.listdir(pysr_dir):
                v_path = os.path.join(pysr_dir, variant)
                if not os.path.isdir(v_path): continue

                res = get_best_formula_from_raw(v_path, X, y,model_type='pysr')

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
        deeppysr_df = type_df[type_df['model'].str.contains('fullsr|stdsr|srprn|srpsm', na=False)]
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

        # PySR variants
        pysr_df = type_df[type_df['model'].str.contains('pysr', na=False)]
        if not pysr_df.empty:
            best_pysr = pysr_df.loc[pysr_df['r2'].idxmax()].copy()
            best_pysr['display_model'] = 'Best PySR'
            selected_data.append(best_pysr)

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
    print("\n--- Interpretable DeepPySR Formulas (Complexity < 25) ---")
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

def plot_best_models():
    """
    Create a plot with 2 rows (red and white wine) and 4 columns (r2, rmse, mae, complexity).
    Each subplot shows metric values for the models, with R2/RMSE/MAE showing all models.
    """
    df = pd.read_csv(os.path.join(current_dir, 'wine_best_models_metrics.csv'))

    wine_types = ['red', 'white']
    metrics = ['r2', 'rmse', 'mae', 'complexity']
    models_to_include_for_complexity = ['Best DeepPySR', 'Interpretable DeepPySR', 'Best PySR', 'KANSym']
    label_map = {
        'Best DeepPySR': 'DeepPySR',
        'Interpretable DeepPySR': 'InterpDeepPySR'
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))

    for i, wine in enumerate(wine_types):
        wine_df_all = df[df['wine type'] == wine].copy()
        wine_df_all = wine_df_all.sort_values('display_model')

        wine_df_complexity = wine_df_all[wine_df_all['display_model'].isin(models_to_include_for_complexity)].copy()

        for j, metric in enumerate(metrics):
            ax = axes[i, j]

            if metric == 'complexity':
                plot_df = wine_df_complexity.copy()
            else:
                plot_df = wine_df_all.copy()

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

def plot_vps_vpr_ablation(df):
    """Ablation: VPS/VPR effect with fixed APS=10.0, r2w=1.0, λ=0.01. Averaged over wine types."""
    import re
    from matplotlib.patches import Patch

    metrics = ['r2', 'rmse', 'mae', 'complexity']
    metric_labels = ['R²', 'RMSE', 'MAE', 'Complexity']

    deep_mask = (df['model'].str.contains('fullsr', regex=False, na=False) &
                 df['model'].str.contains('aps10.0', regex=False, na=False) &
                 df['model'].str.contains('_r2w1.0_L0.01', regex=False, na=False))
    deep_df = df[deep_mask].copy()

    def vps_vpr_label(m):
        match = re.search(r'vps(\d+)_vpr(\d+)', m)
        return f"vps{match.group(1)}/vpr{match.group(2)}" if match else m
    deep_df['label'] = deep_df['model'].apply(vps_vpr_label)
    deep_agg = deep_df.groupby('label')[metrics].mean().reset_index()

    pysr_mask = (df['model'].str.contains(r'^pysr', regex=True, na=False) &
                 df['model'].str.contains('aps10.0', regex=False, na=False))
    pysr_sub = df[pysr_mask].copy()
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
    plt.suptitle('Ablation: VPS/VPR Effect (APS=10.0, r2w=1.0, λ=0.01)', fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout()
    out = os.path.join(current_dir, 'ablation_vps_vpr.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"VPS/VPR ablation saved to {out}")


def plot_pareto_ablation(df):
    """Ablation: pareto r2w/λ effect with fixed VPS=25, VPR=100, APS=10.0. Averaged over wine types."""
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
    """Return Pareto-optimal (complexity, error) pairs sorted by complexity (both minimize)."""
    points = sorted(zip(complexity, error), key=lambda p: (p[0], p[1]))
    pareto = []
    min_error = float('inf')
    for c, e in points:
        if e < min_error:
            min_error = e
            pareto.append((c, e))
    return pareto


def plot_pareto_front_rmse(df):
    """Scatter plot of complexity vs RMSE showing the Pareto front per wine type."""
    wine_types = df['wine type'].unique() if 'wine type' in df.columns else [None]
    n_types = len(wine_types)
    fig, axes = plt.subplots(1, n_types, figsize=(8 * n_types, 7), squeeze=False)

    for ax, wtype in zip(axes[0], wine_types):
        sub_df = df[df['wine type'] == wtype] if wtype is not None else df
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

        title = f'Wine Quality ({wtype}) – Pareto Front: Complexity vs RMSE' if wtype else 'Wine Quality – Pareto Front: Complexity vs RMSE'
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
    # process_results: aggregate all the results from the 5 fold cv, select one formula among the 5 which achieves the highest r2.
    # The r2 is calculated by applying this formula on the entire dataset, not the fold.

    df = process_results()

    save_results(df)
    aggregate_feature_importance()
    plot_best_models()
    plot_vps_vpr_ablation(df)
    plot_pareto_ablation(df)
    plot_pareto_front_rmse(df)
