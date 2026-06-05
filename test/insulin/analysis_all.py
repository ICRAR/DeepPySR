import os
import pandas as pd
import numpy as np
import glob
import sys
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
if not current_dir:
    current_dir = "."
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

from data_utils import load_data, load_data_longitudinal
from analysis_utils import calculate_metrics, evaluate_formula, get_best_formula_from_raw

# Insulin ages and default target
AGES = [14, 17, 20, 22, 27, 28]
TARGET = 'homa_ir'


def _load_age(age):
    import pandas as pd
    ids, X, y_df = load_data(["insulin", "glucose"], age)
    insulin_col = [c for c in y_df.columns if "insulin" in c][0]
    glucose_col = [c for c in y_df.columns if "glucose" in c][0]
    y = (y_df[insulin_col] * y_df[glucose_col] / 22.5).rename("homa_ir")
    return ids, X, y


def _load_longitudinal():
    ids, X, y_df = load_data_longitudinal(["insulin", "glucose"])
    insulin_col = [c for c in y_df.columns if "insulin" in c][0]
    glucose_col = [c for c in y_df.columns if "glucose" in c][0]
    y = (y_df[insulin_col] * y_df[glucose_col] / 22.5).rename("homa_ir")
    return ids, X, y


def process_results():
    base_dir = os.path.join(current_dir, "results_insulin")
    all_data = []

    # 1. Process age-specific  (results_insulin/age_{age}_{target}/)
    for age in AGES:
        age_path = os.path.join(base_dir, f"age_{age}_{TARGET}")
        if not os.path.exists(age_path):
            continue

        # Baselines
        baselines_dir = os.path.join(age_path, "baselines")
        if os.path.exists(baselines_dir):
            for model_name in os.listdir(baselines_dir):
                model_path = os.path.join(baselines_dir, model_name)
                if not os.path.isdir(model_path):
                    continue
                pred_file = os.path.join(model_path, "predictions.csv")
                if os.path.exists(pred_file):
                    df_pred = pd.read_csv(pred_file)
                    if model_name.lower() == 'kan':
                        r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                        all_data.append([age, 'KAN', 'age-specific', r2, rmse, mae, np.nan, ""])
                        if 'y_pred_kansym' in df_pred.columns:
                            _, X_age, y_age = _load_age(age)
                            formula, complexity, metrics = get_best_formula_from_raw(
                                model_path, X_age, y_age, prefix='formulas_fold', model_type='kan')
                            if not formula:
                                r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred_kansym'])
                            else:
                                r2, rmse, mae = metrics
                            all_data.append([age, 'KANSym', 'age-specific', r2, rmse, mae, complexity, formula])
                    else:
                        r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                        all_data.append([age, model_name, 'age-specific', r2, rmse, mae, np.nan, ""])

        # DeepPySR
        deeppysr_dir = os.path.join(age_path, "deeppysr")
        if os.path.exists(deeppysr_dir):
            for variant in os.listdir(deeppysr_dir):
                v_path = os.path.join(deeppysr_dir, variant)
                if not os.path.isdir(v_path):
                    continue
                _, X_age, y_age = _load_age(age)
                res = get_best_formula_from_raw(v_path, X_age, y_age, model_type='deeppysr')
                if isinstance(res, dict):
                    for (r2w, lamb), (formula, complexity, metrics) in res.items():
                        r2, rmse, mae = metrics
                        all_data.append([age, f"{variant}_r2w{r2w}_L{lamb}", 'age-specific', r2, rmse, mae, complexity, formula])
                else:
                    formula, complexity, metrics = res
                    r2, rmse, mae = metrics
                    if not formula:
                        pred_file = os.path.join(v_path, "predictions.csv")
                        if os.path.exists(pred_file):
                            df_pred = pd.read_csv(pred_file)
                            r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                    all_data.append([age, variant, 'age-specific', r2, rmse, mae, complexity, formula])

        # PySR
        pysr_dir = os.path.join(age_path, "pysr")
        if os.path.exists(pysr_dir):
            for variant in os.listdir(pysr_dir):
                v_path = os.path.join(pysr_dir, variant)
                if not os.path.isdir(v_path):
                    continue
                _, X_age, y_age = _load_age(age)
                res = get_best_formula_from_raw(v_path, X_age, y_age, model_type='pysr')
                if isinstance(res, dict):
                    for (r2w, lamb), (formula, complexity, metrics) in res.items():
                        r2, rmse, mae = metrics
                        all_data.append([age, f"{variant}_r2w{r2w}_L{lamb}", 'age-specific', r2, rmse, mae, complexity, formula])
                else:
                    formula, complexity, metrics = res
                    r2, rmse, mae = metrics
                    if not formula:
                        pred_file = os.path.join(v_path, "predictions.csv")
                        if os.path.exists(pred_file):
                            df_pred = pd.read_csv(pred_file)
                            r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                    all_data.append([age, variant, 'age-specific', r2, rmse, mae, complexity, formula])

    # 2. Process longitudinal  (results_insulin/longitudinal_{target}/)
    long_path = os.path.join(base_dir, f"longitudinal_{TARGET}")
    if os.path.exists(long_path):
        _, X_long, y_long = _load_longitudinal()

        for sd in ['baselines', 'deeppysr', 'pysr']:
            sd_path = os.path.join(long_path, sd)
            if not os.path.exists(sd_path):
                continue

            for model_folder in os.listdir(sd_path):
                m_path = os.path.join(sd_path, model_folder)
                if not os.path.isdir(m_path):
                    continue

                pred_file = os.path.join(m_path, "predictions.csv")
                if not os.path.exists(pred_file):
                    continue
                df_pred = pd.read_csv(pred_file)

                for age in AGES:
                    age_df = df_pred[df_pred['age'] == age] if 'age' in df_pred.columns else df_pred
                    if age_df.empty:
                        continue

                    if sd == 'baselines':
                        if model_folder.lower() == 'kan':
                            r2, rmse, mae = calculate_metrics(age_df['y_true'], age_df['y_pred'])
                            all_data.append([age, 'KAN', 'longitudinal', r2, rmse, mae, np.nan, ""])
                            if 'y_pred_kansym' in age_df.columns:
                                formula, complexity, metrics = get_best_formula_from_raw(
                                    m_path, X_long, y_long, prefix='formulas_fold', model_type='kan')
                                if formula:
                                    X_age_data = X_long[X_long['age'] == age]
                                    y_age_data = y_long[X_long['age'] == age]
                                    y_pred_best = evaluate_formula(formula, X_age_data, model_type='kan')
                                    r2, rmse, mae = calculate_metrics(y_age_data, y_pred_best)
                                else:
                                    r2, rmse, mae = calculate_metrics(age_df['y_true'], age_df['y_pred_kansym'])
                                all_data.append([age, 'KANSym', 'longitudinal', r2, rmse, mae, complexity, formula])
                        else:
                            r2, rmse, mae = calculate_metrics(age_df['y_true'], age_df['y_pred'])
                            all_data.append([age, model_folder, 'longitudinal', r2, rmse, mae, np.nan, ""])
                    else:
                        res = get_best_formula_from_raw(m_path, X_long, y_long, model_type=sd)
                        if isinstance(res, dict):
                            for (r2w, lamb), (formula, complexity, _) in res.items():
                                if formula:
                                    X_age_data = X_long[X_long['age'] == age]
                                    y_age_data = y_long[X_long['age'] == age]
                                    y_pred_best = evaluate_formula(formula, X_age_data, model_type=sd)
                                    r2, rmse, mae = calculate_metrics(y_age_data, y_pred_best)
                                else:
                                    r2, rmse, mae = calculate_metrics(age_df['y_true'], age_df['y_pred'])
                                all_data.append([age, f"{model_folder}_r2w{r2w}_L{lamb}", 'longitudinal', r2, rmse, mae, complexity, formula])
                        else:
                            formula, complexity, _ = res
                            if formula:
                                X_age_data = X_long[X_long['age'] == age]
                                y_age_data = y_long[X_long['age'] == age]
                                y_pred_best = evaluate_formula(formula, X_age_data, model_type=sd)
                                r2, rmse, mae = calculate_metrics(y_age_data, y_pred_best)
                            else:
                                r2, rmse, mae = calculate_metrics(age_df['y_true'], age_df['y_pred'])
                            all_data.append([age, model_folder, 'longitudinal', r2, rmse, mae, complexity, formula])

    result_df = pd.DataFrame(all_data, columns=['age', 'model', 'type', 'r2', 'rmse', 'mae', 'complexity', 'formula'])
    result_df['r2'] = result_df['r2'].clip(lower=0)
    out_csv = os.path.join(base_dir, "insulin_aggregated_results.csv")
    result_df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    return result_df


def plot_results(df):
    df = df.copy()
    df['r2'] = df['r2'].clip(lower=0)

    metrics = ['r2', 'rmse', 'mae']
    types = ['longitudinal', 'age-specific']
    selected_data = []
    interpretable_formulas = []

    _, X_all, y_all = _load_longitudinal()

    for t in types:
        type_df = df[df['type'] == t]
        if type_df.empty:
            continue
        ages = sorted(type_df['age'].unique())

        if t == 'longitudinal':
            deeppysr_long = type_df[type_df['model'].str.contains('fullsr|stdsr|srprn|srpsm', na=False)]
            if not deeppysr_long.empty:
                model_variants = deeppysr_long.groupby('model').agg({'formula': 'first', 'complexity': 'first'}).reset_index()
                best_model_name, best_r2 = None, -np.inf
                for _, row in model_variants.iterrows():
                    y_pred = evaluate_formula(row['formula'], X_all)
                    r2, _, _ = calculate_metrics(y_all, y_pred)
                    if r2 > best_r2:
                        best_r2, best_model_name = r2, row['model']
                if best_model_name:
                    for age in ages:
                        rows = type_df[(type_df['age'] == age) & (type_df['model'] == best_model_name)]
                        if not rows.empty:
                            row = rows.iloc[0].copy()
                            row['display_model'] = 'Best DeepPySR'
                            selected_data.append(row)

                interp_candidates = deeppysr_long[deeppysr_long['complexity'] < 30].groupby('model').agg({'formula': 'first', 'complexity': 'first'}).reset_index()
                best_interp_name, best_interp_r2 = None, -np.inf
                for _, row in interp_candidates.iterrows():
                    y_pred = evaluate_formula(row['formula'], X_all)
                    r2, _, _ = calculate_metrics(y_all, y_pred)
                    if r2 > best_interp_r2:
                        best_interp_r2, best_interp_name = r2, row['model']
                if best_interp_name:
                    formula_info = interp_candidates[interp_candidates['model'] == best_interp_name].iloc[0]
                    for age in ages:
                        rows = type_df[(type_df['age'] == age) & (type_df['model'] == best_interp_name)]
                        if not rows.empty:
                            row = rows.iloc[0].copy()
                            row['display_model'] = 'Interpretable DeepPySR'
                            selected_data.append(row)
                            interpretable_formulas.append({
                                'age': age, 'type': t, 'model': best_interp_name,
                                'formula': formula_info['formula'], 'r2': row['r2'],
                                'complexity': formula_info['complexity']
                            })

            pysr_long = type_df[type_df['model'].str.contains('pysr', na=False)]
            if not pysr_long.empty:
                model_variants = pysr_long.groupby('model').agg({'formula': 'first', 'complexity': 'first'}).reset_index()
                best_model_name, best_r2 = None, -np.inf
                for _, row in model_variants.iterrows():
                    y_pred = evaluate_formula(row['formula'], X_all, model_type='pysr')
                    r2, _, _ = calculate_metrics(y_all, y_pred)
                    if r2 > best_r2:
                        best_r2, best_model_name = r2, row['model']
                if best_model_name:
                    for age in ages:
                        rows = type_df[(type_df['age'] == age) & (type_df['model'] == best_model_name)]
                        if not rows.empty:
                            row = rows.iloc[0].copy()
                            row['display_model'] = 'PySR'
                            selected_data.append(row)

            for b in ['ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']:
                b_df = type_df[type_df['model'] == b]
                for age in ages:
                    age_row = b_df[b_df['age'] == age]
                    if not age_row.empty:
                        row = age_row.iloc[0].copy()
                        row['display_model'] = b
                        selected_data.append(row)

        else:  # age-specific
            for age in ages:
                age_df = type_df[type_df['age'] == age]

                deeppysr_df = age_df[age_df['model'].str.contains('fullsr|stdsr|srprn|srpsm', na=False)]
                if not deeppysr_df.empty:
                    best = deeppysr_df.loc[deeppysr_df['r2'].idxmax()].copy()
                    best['display_model'] = 'Best DeepPySR'
                    selected_data.append(best)
                    interp = deeppysr_df[deeppysr_df['complexity'] < 30]
                    if not interp.empty:
                        best_interp = interp.loc[interp['r2'].idxmax()].copy()
                        best_interp['display_model'] = 'Interpretable DeepPySR'
                        selected_data.append(best_interp)
                        interpretable_formulas.append({
                            'age': age, 'type': t, 'model': best_interp['model'],
                            'formula': best_interp['formula'], 'r2': best_interp['r2'],
                            'complexity': best_interp['complexity']
                        })

                pysr_df = age_df[age_df['model'].str.contains('pysr', na=False)]
                if not pysr_df.empty:
                    best_pysr = pysr_df.loc[pysr_df['r2'].idxmax()].copy()
                    best_pysr['display_model'] = 'PySR'
                    selected_data.append(best_pysr)

                for m in ['KAN', 'KANSym']:
                    m_df = age_df[age_df['model'] == m]
                    if not m_df.empty:
                        row = m_df.iloc[0].copy()
                        row['display_model'] = m
                        selected_data.append(row)

                for b in ['ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']:
                    b_df = age_df[age_df['model'] == b]
                    if not b_df.empty:
                        row = b_df.iloc[0].copy()
                        row['display_model'] = b
                        selected_data.append(row)

    plot_df = pd.DataFrame(selected_data)
    if plot_df.empty:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    plt.rcParams.update({'font.size': 14})

    palette = sns.color_palette("tab10", n_colors=len(plot_df['display_model'].unique()))
    models = sorted(plot_df['display_model'].unique())
    model_colors = dict(zip(models, palette))

    for t in types:
        current_row = 0 if t == 'age-specific' else 1
        linestyle = '--' if t == 'age-specific' else '-'
        for col, metric in enumerate(metrics):
            ax = axes[current_row, col]
            sns.lineplot(data=plot_df[plot_df['type'] == t],
                         x='age', y=metric, hue='display_model', ax=ax,
                         linestyle=linestyle, linewidth=3.0, palette=model_colors,
                         marker='o', markersize=8)
            type_label = "Age-specific" if t == 'age-specific' else "Longitudinal"
            ax.set_title(f'{type_label}: {metric.upper()} vs Age', fontsize=20, fontweight='bold', pad=15)
            ax.set_ylabel(metric.upper(), fontsize=16)
            ax.set_xlabel('Age', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=12)
            if ax.get_legend():
                ax.get_legend().remove()

    legend_elements = [Line2D([0], [0], color=model_colors[m], lw=3, label=m) for m in models]
    legend_elements.append(Line2D([0], [0], color='white', label=''))
    legend_elements.append(Line2D([0], [0], color='black', lw=3, ls='--', label='Age-specific'))
    legend_elements.append(Line2D([0], [0], color='black', lw=3, ls='-', label='Longitudinal'))
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.91, 0.5),
               fontsize=14, frameon=True, title='Models & Types', title_fontsize=16, handlelength=4.0)

    plt.suptitle('Insulin (HOMA-IR) Prediction Performance: Best Models Comparison',
                 fontsize=26, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    base_dir = os.path.join(current_dir, "results_insulin")
    plot_path = os.path.join(base_dir, 'insulin_metrics_vs_age.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Combined metrics plot saved to {plot_path}")

    plot_csv_path = os.path.join(base_dir, 'insulin_best_models_metrics.csv')
    plot_df.to_csv(plot_csv_path, index=False)
    print(f"Best models plot data saved to {plot_csv_path}")

    if interpretable_formulas:
        print("\n--- Interpretable DeepPySR Formulas (Complexity < 30) ---")
        interp_df = pd.DataFrame(interpretable_formulas)
        print(interp_df.to_string(index=False))
        interp_df.to_csv(os.path.join(base_dir, 'interpretable_deeppysr_formulas.csv'), index=False)


def aggregate_feature_importance():
    base_dir = os.path.join(current_dir, "results_insulin")
    importance_data = []

    def process_importance(path, model_name, age, type_str):
        if os.path.exists(path):
            df_imp = pd.read_csv(path)
            if 'feature' in df_imp.columns and 'importance' in df_imp.columns:
                total = df_imp['importance'].sum()
                for _, row in df_imp.iterrows():
                    importance_data.append({
                        'age': age, 'model': model_name, 'type': type_str,
                        'variable': row['feature'],
                        'weight': (row['importance'] / total * 100) if total > 0 else 0
                    })

    for age in AGES:
        baselines_dir = os.path.join(base_dir, f"age_{age}_{TARGET}", "baselines")
        if os.path.exists(baselines_dir):
            for m in os.listdir(baselines_dir):
                if m in ['ElasticNet', 'ExtraTrees', 'RandomForest', 'XGBoost', 'KAN']:
                    process_importance(os.path.join(baselines_dir, m, "feature_importance.csv"),
                                       m, age, 'age-specific')

    long_baselines_dir = os.path.join(base_dir, f"longitudinal_{TARGET}", "baselines")
    if os.path.exists(long_baselines_dir):
        for m in os.listdir(long_baselines_dir):
            if m in ['ElasticNet', 'ExtraTrees', 'RandomForest', 'XGBoost', 'KAN']:
                imp_file = os.path.join(long_baselines_dir, m, "feature_importance.csv")
                if os.path.exists(imp_file):
                    df_imp = pd.read_csv(imp_file)
                    total = df_imp['importance'].sum()
                    for _, row in df_imp.iterrows():
                        importance_data.append({
                            'age': 'all', 'model': m, 'type': 'longitudinal',
                            'variable': row['feature'],
                            'weight': (row['importance'] / total * 100) if total > 0 else 0
                        })

    imp_df = pd.DataFrame(importance_data)
    imp_df.to_csv(os.path.join(base_dir, "feature_importance_aggregated.csv"), index=False)
    print("Feature importance aggregated to results_insulin/feature_importance_aggregated.csv")

    if not imp_df.empty:
        agg_imp = imp_df.groupby(['model', 'variable'])['weight'].mean().reset_index()
        top_features = agg_imp.groupby('variable')['weight'].mean().sort_values(ascending=False).head(15).index
        plot_df = agg_imp[agg_imp['variable'].isin(top_features)].copy()
        plot_df['variable'] = pd.Categorical(plot_df['variable'], categories=top_features, ordered=True)
        plt.figure(figsize=(14, 10))
        sns.barplot(data=plot_df, x='weight', y='variable', hue='model', palette="bright")
        plt.title('Top 15 Feature Importance Comparison across Models', fontsize=22, fontweight='bold', pad=20)
        plt.xlabel('Average Percentage Importance (%)', fontsize=18)
        plt.ylabel('Feature', fontsize=18)
        plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
        plt.tick_params(labelsize=14)
        plt.tight_layout()
        plot_path = os.path.join(base_dir, "feature_importance_by_model.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {plot_path}")


if __name__ == "__main__":
    df = process_results()
    plot_results(df)
    aggregate_feature_importance()
