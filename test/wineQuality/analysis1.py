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

                # Use overall_metrics.csv/overall_metrics_sym.csv if they exist
                overall_metrics_file = os.path.join(model_path, "overall_metrics.csv")
                overall_metrics_sym_file = os.path.join(model_path, "overall_metrics_sym.csv")

                if model_name.lower() == 'kan':
                    # KAN
                    if os.path.exists(overall_metrics_file):
                        df_metrics = pd.read_csv(overall_metrics_file)
                        r2, rmse, mae = df_metrics['r2'].iloc[0], df_metrics['rmse'].iloc[0], df_metrics['mae'].iloc[0]
                    else:
                        pred_file = os.path.join(model_path, "predictions.csv")
                        if os.path.exists(pred_file):
                            df_pred = pd.read_csv(pred_file)
                            r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                        else:
                            r2, rmse, mae = [np.nan] * 3
                    all_data.append(['KAN', wine_type, r2, rmse, mae, np.nan, ""])

                    # KANSym
                    if os.path.exists(overall_metrics_sym_file):
                        df_metrics_sym = pd.read_csv(overall_metrics_sym_file)
                        r2, rmse, mae = df_metrics_sym['r2'].iloc[0], df_metrics_sym['rmse'].iloc[0], df_metrics_sym['mae'].iloc[0]
                    else:
                        _, _, metrics = get_best_formula_from_raw(model_path, X, y, prefix='formulas_fold', model_type='kan')
                        r2, rmse, mae = metrics

                    formula, complexity, _ = get_best_formula_from_raw(model_path, X, y, prefix='formulas_fold', model_type='kan')
                    all_data.append(['KANSym', wine_type, r2, rmse, mae, complexity, formula])
                else:
                    # Other baselines
                    if os.path.exists(overall_metrics_file):
                        df_metrics = pd.read_csv(overall_metrics_file)
                        r2, rmse, mae = df_metrics['r2'].iloc[0], df_metrics['rmse'].iloc[0], df_metrics['mae'].iloc[0]
                    else:
                        pred_file = os.path.join(model_path, "predictions.csv")
                        if os.path.exists(pred_file):
                            df_pred = pd.read_csv(pred_file)
                            r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                        else:
                            r2, rmse, mae = [np.nan] * 3
                    all_data.append([model_name, wine_type, r2, rmse, mae, np.nan, ""])

        # DeepPySR
        deeppysr_dir = os.path.join(base_dir, "deeppysr")
        if os.path.exists(deeppysr_dir):
            for variant in os.listdir(deeppysr_dir):
                v_path = os.path.join(deeppysr_dir, variant)
                if not os.path.isdir(v_path): continue

                # Use overall_metrics.csv for DeepPySR if it exists
                overall_metrics_file = os.path.join(v_path, "overall_metrics.csv")
                if os.path.exists(overall_metrics_file):
                    df_metrics = pd.read_csv(overall_metrics_file)
                    r2_o, rmse_o, mae_o = df_metrics['r2'].iloc[0], df_metrics['rmse'].iloc[0], df_metrics['mae'].iloc[0]
                    use_overall = True
                else:
                    use_overall = False

                res = get_best_formula_from_raw(v_path, X, y,model_type='deeppysr')

                if isinstance(res, dict):
                    for (r2w, lamb), (formula, complexity, metrics) in res.items():
                        r2, rmse, mae = metrics
                        if use_overall:
                            r2, rmse, mae = r2_o, rmse_o, mae_o
                        model_name = f"{variant}_r2w{r2w}_L{lamb}"
                        all_data.append([model_name, wine_type, r2, rmse, mae, complexity, formula])
                else:
                    formula, complexity, metrics = res
                    r2, rmse, mae = metrics
                    if use_overall:
                        r2, rmse, mae = r2_o, rmse_o, mae_o
                    elif not formula:
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

                # Use overall_metrics.csv for PySR if it exists
                overall_metrics_file = os.path.join(v_path, "overall_metrics.csv")
                if os.path.exists(overall_metrics_file):
                    df_metrics = pd.read_csv(overall_metrics_file)
                    r2_o, rmse_o, mae_o = df_metrics['r2'].iloc[0], df_metrics['rmse'].iloc[0], df_metrics['mae'].iloc[0]
                    use_overall = True
                else:
                    use_overall = False

                res = get_best_formula_from_raw(v_path, X, y,model_type='pysr')

                if isinstance(res, dict):
                    for (r2w, lamb), (formula, complexity, metrics) in res.items():
                        r2, rmse, mae = metrics
                        if use_overall:
                            r2, rmse, mae = r2_o, rmse_o, mae_o
                        model_name = f"{variant}_r2w{r2w}_L{lamb}"
                        all_data.append([model_name, wine_type, r2, rmse, mae, complexity, formula])
                else:
                    formula, complexity, metrics = res
                    r2, rmse, mae = metrics
                    if use_overall:
                        r2, rmse, mae = r2_o, rmse_o, mae_o
                    elif not formula:
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

if __name__ == "__main__":
    # process_results: aggregate all the results from the 5 fold cv, select one formula among the 5 which achieves the highest r2.\
    # The r2 is calculated by applying this formula on the entire dataset, not the fold.

    df = process_results()

    save_results(df)
    aggregate_feature_importance()
    plot_best_models()
