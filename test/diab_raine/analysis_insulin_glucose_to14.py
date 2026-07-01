import os
import pandas as pd
import numpy as np
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)

current_dir = os.path.dirname(os.path.abspath(__file__))
if not current_dir:
    current_dir = "."
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

from data_utils import load_data_keepto14, load_data_longitudinal_keepto14
from analysis_utils import calculate_metrics, evaluate_formula, get_best_formula_from_raw

AGES = [14, 17, 20, 22, 27, 28]
TARGETS = ['diab_raine', 'glucose']

RESULTS_DIR = os.path.join(current_dir, "results_insulin_glucose_to14")

# ─── Classification thresholds ──────────────────────────────────────────────

GLUCOSE_LABELS = {0: 'Normal', 1: 'Prediabetes', 2: 'Diabetes'}
INSULIN_LABELS = {0: 'Optimal', 1: 'Normal/Moderate', 2: 'High Risk'}
HOMAIR_LABELS  = {0: 'Optimal', 1: 'Normal/Healthy', 2: 'Early IR', 3: 'Significant IR'}


def classify_glucose(v):
    """Classify fasting glucose in mmol/L."""
    if v < 6.1:  return 0  # Normal
    if v < 7.0:  return 1  # Prediabetes
    return 2                # Diabetes (>=7.0)


def classify_insulin(v):
    """Classify fasting diab_raine in μU/mL."""
    if v <= 6:   return 0  # Optimal
    if v <= 10:  return 1  # Normal/Moderate Risk
    return 2                # High Risk (>10)


def compute_homa_ir(glucose_mmol, insulin_uU):
    """HOMA-IR = (glucose × diab_raine) / 22.5"""
    return (glucose_mmol * insulin_uU) / 22.5


def classify_homa_ir(v):
    if v < 1.0:  return 0  # Optimal
    if v < 2.0:  return 1  # Normal/Healthy
    if v < 3.0:  return 2  # Early Insulin Resistance
    return 3                # Significant Insulin Resistance (>=3.0)


# ─── Data helpers ───────────────────────────────────────────────────────────

def _extract_y(y_df, target):
    col = [c for c in y_df.columns if target in c][0]
    return y_df[col].rename(target)


def _load_age(age, target):
    ids, X, y_df = load_data_keepto14(["diab_raine", "glucose"], age)
    return ids, X, _extract_y(y_df, target)


def _load_longitudinal(target):
    ids, X, y_df = load_data_longitudinal_keepto14(["diab_raine", "glucose"])
    return ids, X, _extract_y(y_df, target)


# ─── Regression analysis ────────────────────────────────────────────────────

def _process_target(target):
    """Collect regression result rows for one target."""
    base_dir = RESULTS_DIR
    all_data = []

    for age in AGES:
        age_path = os.path.join(base_dir, f"age_{age}_{target}")
        if not os.path.exists(age_path):
            continue

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
                            _, X_age, y_age = _load_age(age, target)
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

        deeppysr_dir = os.path.join(age_path, "deeppysr")
        if os.path.exists(deeppysr_dir):
            for variant in os.listdir(deeppysr_dir):
                v_path = os.path.join(deeppysr_dir, variant)
                if not os.path.isdir(v_path):
                    continue
                _, X_age, y_age = _load_age(age, target)
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

        pysr_dir = os.path.join(age_path, "pysr")
        if os.path.exists(pysr_dir):
            for variant in os.listdir(pysr_dir):
                v_path = os.path.join(pysr_dir, variant)
                if not os.path.isdir(v_path):
                    continue
                _, X_age, y_age = _load_age(age, target)
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

    long_path = os.path.join(base_dir, f"longitudinal_{target}")
    if os.path.exists(long_path):
        _, X_long, y_long = _load_longitudinal(target)

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

    df = pd.DataFrame(all_data, columns=['age', 'model', 'type', 'r2', 'rmse', 'mae', 'complexity', 'formula'])
    df['r2'] = df['r2'].clip(lower=0)
    return df


def process_results():
    base_dir = RESULTS_DIR
    dfs = {}
    for target in TARGETS:
        out_csv = os.path.join(base_dir, f"insulin_{target}_aggregated_results.csv")
        if os.path.exists(out_csv):
            dfs[target] = pd.read_csv(out_csv)
            print(f"Results loaded from {out_csv}")
        else:
            df = _process_target(target)
            dfs[target] = df
            df.to_csv(out_csv, index=False)
            print(f"Results saved to {out_csv}")
    return dfs


def _select_best_models(df):
    """Return plot_df and interpretable_formulas for one target's result df."""
    df = df.copy()
    df['r2'] = df['r2'].clip(lower=0)
    types = ['longitudinal', 'age-specific']
    selected_data = []
    interpretable_formulas = []

    _, X_all, _ = _load_longitudinal(TARGETS[0])

    for t in types:
        type_df = df[df['type'] == t]
        if type_df.empty:
            continue
        ages = sorted(type_df['age'].unique())

        if t == 'longitudinal':
            deeppysr_long = type_df[type_df['model'].str.contains('fullsr|stdsr|srprn|srpsm', na=False)]
            if not deeppysr_long.empty:
                model_variants = deeppysr_long.groupby('model').agg({'formula': 'first', 'complexity': 'first', 'r2': 'mean'}).reset_index()
                best_model_name, best_r2 = None, -np.inf
                for _, row in model_variants.iterrows():
                    if row['r2'] > best_r2:
                        best_r2, best_model_name = row['r2'], row['model']
                if best_model_name:
                    for age in ages:
                        rows = type_df[(type_df['age'] == age) & (type_df['model'] == best_model_name)]
                        if not rows.empty:
                            row = rows.iloc[0].copy(); row['display_model'] = 'Best DeepPySR'
                            selected_data.append(row)

                interp = deeppysr_long[deeppysr_long['complexity'] < 30].groupby('model').agg({'formula': 'first', 'complexity': 'first', 'r2': 'max'}).reset_index()
                if not interp.empty:
                    best_interp = interp.loc[interp['r2'].idxmax()]
                    for age in ages:
                        rows = type_df[(type_df['age'] == age) & (type_df['model'] == best_interp['model'])]
                        if not rows.empty:
                            row = rows.iloc[0].copy(); row['display_model'] = 'Interpretable DeepPySR'
                            selected_data.append(row)
                            interpretable_formulas.append({
                                'age': age, 'type': t, 'model': best_interp['model'],
                                'formula': best_interp['formula'], 'r2': row['r2'],
                                'complexity': best_interp['complexity']
                            })

            pysr_long = type_df[type_df['model'].str.contains('pysr', na=False)]
            if not pysr_long.empty:
                best_pysr = pysr_long.groupby('model')['r2'].mean().idxmax()
                for age in ages:
                    rows = type_df[(type_df['age'] == age) & (type_df['model'] == best_pysr)]
                    if not rows.empty:
                        row = rows.iloc[0].copy(); row['display_model'] = 'PySR'
                        selected_data.append(row)

            for b in ['ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']:
                b_df = type_df[type_df['model'] == b]
                for age in ages:
                    age_row = b_df[b_df['age'] == age]
                    if not age_row.empty:
                        row = age_row.iloc[0].copy(); row['display_model'] = b
                        selected_data.append(row)

        else:
            for age in ages:
                age_df = type_df[type_df['age'] == age]
                deeppysr_df = age_df[age_df['model'].str.contains('fullsr|stdsr|srprn|srpsm', na=False)]
                if not deeppysr_df.empty:
                    best = deeppysr_df.loc[deeppysr_df['r2'].idxmax()].copy()
                    best['display_model'] = 'Best DeepPySR'
                    selected_data.append(best)
                    interp = deeppysr_df[deeppysr_df['complexity'] < 30]
                    if not interp.empty:
                        bi = interp.loc[interp['r2'].idxmax()].copy()
                        bi['display_model'] = 'Interpretable DeepPySR'
                        selected_data.append(bi)
                        interpretable_formulas.append({'age': age, 'type': t, 'model': bi['model'],
                                                       'formula': bi['formula'], 'r2': bi['r2'], 'complexity': bi['complexity']})
                pysr_df = age_df[age_df['model'].str.contains('pysr', na=False)]
                if not pysr_df.empty:
                    best_pysr = pysr_df.loc[pysr_df['r2'].idxmax()].copy()
                    best_pysr['display_model'] = 'PySR'; selected_data.append(best_pysr)
                for m in ['KAN', 'KANSym']:
                    m_df = age_df[age_df['model'] == m]
                    if not m_df.empty:
                        row = m_df.iloc[0].copy(); row['display_model'] = m; selected_data.append(row)
                for b in ['ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']:
                    b_df = age_df[age_df['model'] == b]
                    if not b_df.empty:
                        row = b_df.iloc[0].copy(); row['display_model'] = b; selected_data.append(row)

    return pd.DataFrame(selected_data), interpretable_formulas


def plot_results(dfs):
    base_dir = RESULTS_DIR
    metrics = ['r2', 'rmse', 'mae']
    types = ['longitudinal', 'age-specific']

    for target, df in dfs.items():
        plot_df, interpretable_formulas = _select_best_models(df)
        if plot_df.empty:
            print(f"No data to plot for target={target}.")
            continue

        plot_csv_path = os.path.join(base_dir, f'insulin_{target}_best_models_metrics.csv')
        plot_df.to_csv(plot_csv_path, index=False)
        print(f"Best models saved to {plot_csv_path}")

        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        plt.rcParams.update({'font.size': 14})
        models = sorted(plot_df['display_model'].unique())
        palette = sns.color_palette("tab10", n_colors=len(models))
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
        plt.suptitle(f'Insulin ({target}) Prediction Performance: Best Models Comparison',
                     fontsize=26, fontweight='bold', y=0.99)
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        plot_path = os.path.join(base_dir, f'insulin_{target}_metrics_vs_age.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to {plot_path}")

        if interpretable_formulas:
            print(f"\n--- Interpretable DeepPySR Formulas for {target} (Complexity < 30) ---")
            interp_df = pd.DataFrame(interpretable_formulas)
            print(interp_df.to_string(index=False))
            interp_df.to_csv(os.path.join(base_dir, f'interpretable_deeppysr_formulas_{target}.csv'), index=False)


# ─── HOMA-IR regression analysis ────────────────────────────────────────────

def _process_homa_ir():
    """
    Compute HOMA-IR = (glucose × diab_raine) / 22.5 from matched predictions,
    then calculate regression metrics (R², RMSE, MAE) per model/age/type.
    """
    all_preds = _collect_all_predictions()

    glu_map = {}
    for rec in all_preds['glucose']:
        key = (rec['age'], rec['model'], rec['model_type'])
        glu_map[key] = rec

    rows = []
    for rec in all_preds['diab_raine']:
        key = (rec['age'], rec['model'], rec['model_type'])
        if key not in glu_map:
            continue
        glu_rec = glu_map[key]

        glu_df = pd.DataFrame({'id': glu_rec['id'], 'glu_true': glu_rec['y_true'], 'glu_pred': glu_rec['y_pred']})
        ins_df = pd.DataFrame({'id': rec['id'], 'ins_true': rec['y_true'], 'ins_pred': rec['y_pred']})
        merged = glu_df.merge(ins_df, on='id', how='inner')
        if merged.empty:
            continue

        homa_true = compute_homa_ir(merged['glu_true'].values, merged['ins_true'].values)
        homa_pred = compute_homa_ir(merged['glu_pred'].values, merged['ins_pred'].values)

        r2, rmse, mae = calculate_metrics(homa_true, homa_pred)
        age, model, mtype = key
        rows.append({
            'age': age,
            'model': model,
            'type': mtype,
            'display_model': _model_display_name(model),
            'r2': max(0.0, r2),
            'rmse': rmse,
            'mae': mae,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        out_csv = os.path.join(RESULTS_DIR, "insulin_homa_ir_aggregated_results.csv")
        df.to_csv(out_csv, index=False)
        print(f"HOMA-IR regression results saved to {out_csv}")
    return df


def _select_best_homa_ir_models(df):
    """Select best DeepPySR, best PySR, and all baselines — mirrors _select_best_models."""
    types = ['longitudinal', 'age-specific']
    selected = []

    for t in types:
        type_df = df[df['type'] == t]
        if type_df.empty:
            continue
        ages = sorted(type_df['age'].unique())

        deeppysr_df = type_df[type_df['model'].str.contains('fullsr|stdsr|srprn|srpsm', na=False)]
        pysr_df     = type_df[type_df['model'].str.contains('pysr', na=False)]

        if t == 'longitudinal':
            if not deeppysr_df.empty:
                best_name = deeppysr_df.groupby('model')['r2'].mean().idxmax()
                for age in ages:
                    rows = type_df[(type_df['age'] == age) & (type_df['model'] == best_name)]
                    if not rows.empty:
                        r = rows.iloc[0].copy(); r['display_model'] = 'Best DeepPySR'; selected.append(r)
            if not pysr_df.empty:
                best_pysr = pysr_df.groupby('model')['r2'].mean().idxmax()
                for age in ages:
                    rows = type_df[(type_df['age'] == age) & (type_df['model'] == best_pysr)]
                    if not rows.empty:
                        r = rows.iloc[0].copy(); r['display_model'] = 'PySR'; selected.append(r)
            for b in ['ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']:
                b_df = type_df[type_df['model'] == b]
                for age in ages:
                    age_row = b_df[b_df['age'] == age]
                    if not age_row.empty:
                        r = age_row.iloc[0].copy(); r['display_model'] = b; selected.append(r)
        else:
            for age in ages:
                age_df = type_df[type_df['age'] == age]
                dp = age_df[age_df['model'].str.contains('fullsr|stdsr|srprn|srpsm', na=False)]
                if not dp.empty:
                    r = dp.loc[dp['r2'].idxmax()].copy(); r['display_model'] = 'Best DeepPySR'; selected.append(r)
                py = age_df[age_df['model'].str.contains('pysr', na=False)]
                if not py.empty:
                    r = py.loc[py['r2'].idxmax()].copy(); r['display_model'] = 'PySR'; selected.append(r)
                for m in ['KAN', 'KANSym']:
                    m_df = age_df[age_df['model'] == m]
                    if not m_df.empty:
                        r = m_df.iloc[0].copy(); r['display_model'] = m; selected.append(r)
                for b in ['ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']:
                    b_df = age_df[age_df['model'] == b]
                    if not b_df.empty:
                        r = b_df.iloc[0].copy(); r['display_model'] = b; selected.append(r)

    return pd.DataFrame(selected)


def plot_homa_ir_results():
    """Plot R², RMSE, MAE vs Age for HOMA-IR — same layout as insulin_glucose/diab_raine plots."""
    base_dir = RESULTS_DIR
    df = _process_homa_ir()
    if df.empty:
        print("No HOMA-IR data to plot.")
        return

    plot_df = _select_best_homa_ir_models(df)
    if plot_df.empty:
        print("No HOMA-IR models selected for plot.")
        return

    plot_csv_path = os.path.join(base_dir, 'insulin_homa_ir_best_models_metrics.csv')
    plot_df.to_csv(plot_csv_path, index=False)
    print(f"HOMA-IR best models saved to {plot_csv_path}")

    metrics = ['r2', 'rmse', 'mae']
    types = ['longitudinal', 'age-specific']

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    plt.rcParams.update({'font.size': 14})
    models = sorted(plot_df['display_model'].unique())
    palette = sns.color_palette("tab10", n_colors=len(models))
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
    plt.suptitle('HOMA-IR Prediction Performance: Best Models Comparison\n'
                 'HOMA-IR = (Glucose × Insulin) / 22.5',
                 fontsize=24, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plot_path = os.path.join(base_dir, 'insulin_homa_ir_metrics_vs_age.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"HOMA-IR plot saved to {plot_path}")


# ─── Classification analysis ────────────────────────────────────────────────

def _load_pred_file(path):
    """Load a predictions.csv; return None if missing or malformed."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'y_true' not in df.columns or 'y_pred' not in df.columns:
        return None
    return df


def _collect_all_predictions():
    """
    Gather raw predictions from all predictions.csv files.

    Returns a dict: {target: list of dicts with keys
        age, model, model_type ('age-specific'|'longitudinal'), y_true, y_pred, id}
    """
    base_dir = RESULTS_DIR
    records = {t: [] for t in TARGETS}

    def _add(target, age, model, mtype, df):
        ids = df['id'].values if 'id' in df.columns else np.arange(len(df))
        records[target].append({
            'age': age, 'model': model, 'model_type': mtype,
            'y_true': df['y_true'].values,
            'y_pred': df['y_pred'].values,
            'id': ids,
        })

    for target in TARGETS:
        # age-specific
        for age in AGES:
            age_path = os.path.join(base_dir, f"age_{age}_{target}")
            if not os.path.exists(age_path):
                continue
            for sub in ['baselines', 'deeppysr', 'pysr']:
                sub_path = os.path.join(age_path, sub)
                if not os.path.exists(sub_path):
                    continue
                for model_name in os.listdir(sub_path):
                    pred_file = os.path.join(sub_path, model_name, "predictions.csv")
                    df = _load_pred_file(pred_file)
                    if df is not None:
                        _add(target, age, model_name, 'age-specific', df)

        # longitudinal
        long_path = os.path.join(base_dir, f"longitudinal_{target}")
        if not os.path.exists(long_path):
            continue
        for sub in ['baselines', 'deeppysr', 'pysr']:
            sub_path = os.path.join(long_path, sub)
            if not os.path.exists(sub_path):
                continue
            for model_name in os.listdir(sub_path):
                pred_file = os.path.join(sub_path, model_name, "predictions.csv")
                df = _load_pred_file(pred_file)
                if df is None:
                    continue
                for age in AGES:
                    age_df = df[df['age'] == age] if 'age' in df.columns else df
                    if not age_df.empty:
                        _add(target, age, model_name, 'longitudinal', age_df)

    return records


def _classify_metrics(y_true_cls, y_pred_cls, n_classes):
    labels = list(range(n_classes))
    acc = accuracy_score(y_true_cls, y_pred_cls)
    f1_macro = f1_score(y_true_cls, y_pred_cls, average='macro', labels=labels, zero_division=0)
    f1_weighted = f1_score(y_true_cls, y_pred_cls, average='weighted', labels=labels, zero_division=0)
    return acc, f1_macro, f1_weighted


def _model_display_name(model_name):
    """Map raw folder names to display names for plots."""
    mn = model_name.lower()
    if any(k in mn for k in ['fullsr', 'stdsr', 'srprn', 'srpsm']):
        return 'DeepPySR'
    if 'pysr' in mn:
        return 'PySR'
    if 'kansym' in mn:
        return 'KANSym'
    if mn == 'kan':
        return 'KAN'
    return model_name  # ElasticNet, ExtraTrees, MLP, RandomForest, XGBoost


def process_classification():
    """
    Apply clinical classification thresholds to model predictions for
    glucose (mmol/L), diab_raine (μU/mL), and derived HOMA-IR, then compute
    accuracy and F1 metrics.

    Glucose:  <6.1 Normal | 6.1–<7.0 Prediabetes | ≥7.0 Diabetes
    Insulin:  ≤6 Optimal  | 6–10 Normal/Moderate  | >10 High Risk
    HOMA-IR:  <1 Optimal  | 1–<2 Normal/Healthy   | 2–<3 Early IR | ≥3 Significant IR
    """
    all_preds = _collect_all_predictions()
    rows = []

    # Per-target classification
    for target in TARGETS:
        classify_fn = classify_glucose if target == 'glucose' else classify_insulin
        n_cls = len(GLUCOSE_LABELS) if target == 'glucose' else len(INSULIN_LABELS)
        for rec in all_preds[target]:
            y_true_cls = np.array([classify_fn(v) for v in rec['y_true']])
            y_pred_cls = np.array([classify_fn(v) for v in rec['y_pred']])
            acc, f1_mac, f1_wt = _classify_metrics(y_true_cls, y_pred_cls, n_cls)
            rows.append({
                'target': target,
                'age': rec['age'],
                'model': rec['model'],
                'display_model': _model_display_name(rec['model']),
                'model_type': rec['model_type'],
                'accuracy': acc,
                'f1_macro': f1_mac,
                'f1_weighted': f1_wt,
            })

    # HOMA-IR: match glucose & diab_raine predictions on id
    glu_map = {}
    for rec in all_preds['glucose']:
        key = (rec['age'], rec['model'], rec['model_type'])
        glu_map[key] = rec

    for rec in all_preds['diab_raine']:
        key = (rec['age'], rec['model'], rec['model_type'])
        if key not in glu_map:
            continue
        glu_rec = glu_map[key]

        # Align on shared IDs
        glu_df = pd.DataFrame({'id': glu_rec['id'], 'glu_true': glu_rec['y_true'], 'glu_pred': glu_rec['y_pred']})
        ins_df = pd.DataFrame({'id': rec['id'], 'ins_true': rec['y_true'], 'ins_pred': rec['y_pred']})
        merged = glu_df.merge(ins_df, on='id', how='inner')
        if merged.empty:
            continue

        homa_true = compute_homa_ir(merged['glu_true'].values, merged['ins_true'].values)
        homa_pred = compute_homa_ir(merged['glu_pred'].values, merged['ins_pred'].values)

        true_cls = np.array([classify_homa_ir(v) for v in homa_true])
        pred_cls = np.array([classify_homa_ir(v) for v in homa_pred])
        acc, f1_mac, f1_wt = _classify_metrics(true_cls, pred_cls, len(HOMAIR_LABELS))

        age, model, mtype = key
        rows.append({
            'target': 'homa_ir',
            'age': age,
            'model': model,
            'display_model': _model_display_name(model),
            'model_type': mtype,
            'accuracy': acc,
            'f1_macro': f1_mac,
            'f1_weighted': f1_wt,
        })

    cls_df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "classification_results.csv")
    cls_df.to_csv(out_csv, index=False)
    print(f"Classification results saved to {out_csv}")
    return cls_df


def _select_best_cls_models(cls_df, target, mtype):
    """
    For a given target and model_type, select the best variant per display_model
    (highest mean f1_macro across ages).
    """
    sub = cls_df[(cls_df['target'] == target) & (cls_df['model_type'] == mtype)].copy()
    if sub.empty:
        return pd.DataFrame()

    # Pick best raw model per display_model group
    best_models = (sub.groupby(['display_model', 'model'])['f1_macro']
                   .mean().reset_index()
                   .sort_values('f1_macro', ascending=False)
                   .drop_duplicates('display_model'))

    selected = sub[sub['model'].isin(best_models['model'])]
    return selected


def plot_classification_results(cls_df):
    """
    For each target (glucose, diab_raine, homa_ir) and each model_type
    (age-specific, longitudinal), plot accuracy and F1 vs age.
    Also plot a summary confusion-matrix heatmap for the best model.
    """
    base_dir = RESULTS_DIR
    all_targets = ['glucose', 'diab_raine', 'homa_ir']
    mtypes = ['age-specific', 'longitudinal']
    metrics = ['accuracy', 'f1_macro', 'f1_weighted']
    metric_labels = {'accuracy': 'Accuracy', 'f1_macro': 'F1 (Macro)', 'f1_weighted': 'F1 (Weighted)'}

    target_labels = {
        'glucose': 'Glucose (mmol/L)\nNormal | Prediabetes | Diabetes',
        'diab_raine': 'Insulin (μU/mL)\nOptimal | Normal/Moderate | High Risk',
        'homa_ir': 'HOMA-IR\nOptimal | Normal | Early IR | Significant IR',
    }

    for target in all_targets:
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        plt.rcParams.update({'font.size': 14})

        all_display = cls_df[cls_df['target'] == target]['display_model'].unique()
        palette = sns.color_palette("tab10", n_colors=len(all_display))
        model_colors = dict(zip(sorted(all_display), palette))

        has_data = False
        for row_i, mtype in enumerate(mtypes):
            plot_data = _select_best_cls_models(cls_df, target, mtype)
            if plot_data.empty:
                continue
            has_data = True
            linestyle = '--' if mtype == 'age-specific' else '-'
            for col_i, metric in enumerate(metrics):
                ax = axes[row_i, col_i]
                sns.lineplot(data=plot_data, x='age', y=metric, hue='display_model',
                             ax=ax, linestyle=linestyle, linewidth=3.0,
                             palette=model_colors, marker='o', markersize=8)
                mtype_label = 'Age-specific' if mtype == 'age-specific' else 'Longitudinal'
                ax.set_title(f'{mtype_label}: {metric_labels[metric]} vs Age',
                             fontsize=18, fontweight='bold', pad=12)
                ax.set_ylabel(metric_labels[metric], fontsize=15)
                ax.set_xlabel('Age', fontsize=15)
                ax.set_ylim(-0.05, 1.05)
                ax.tick_params(labelsize=12)
                if ax.get_legend():
                    ax.get_legend().remove()

        if not has_data:
            plt.close(fig)
            continue

        legend_elements = [Line2D([0], [0], color=model_colors[m], lw=3, label=m)
                           for m in sorted(model_colors)]
        legend_elements += [
            Line2D([0], [0], color='white', label=''),
            Line2D([0], [0], color='black', lw=3, ls='--', label='Age-specific'),
            Line2D([0], [0], color='black', lw=3, ls='-', label='Longitudinal'),
        ]
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.91, 0.5),
                   fontsize=13, frameon=True, title='Models & Types', title_fontsize=15, handlelength=4.0)
        plt.suptitle(f'Classification Performance — {target_labels[target]}',
                     fontsize=22, fontweight='bold', y=0.99)
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        plot_path = os.path.join(base_dir, f'classification_{target}_metrics_vs_age.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Classification plot saved to {plot_path}")

    # Summary bar chart: mean F1 macro across ages per target × display_model × model_type
    summary = (cls_df.groupby(['target', 'display_model', 'model_type'])
               ['f1_macro'].mean().reset_index())
    summary['target_mtype'] = summary['target'] + '\n(' + summary['model_type'] + ')'

    n_groups = summary['target_mtype'].nunique()
    fig, ax = plt.subplots(figsize=(max(14, n_groups * 2), 7))
    sns.barplot(data=summary, x='target_mtype', y='f1_macro', hue='display_model',
                palette='tab10', ax=ax)
    ax.set_title('Mean F1 (Macro) by Target, Model, and Training Type',
                 fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('Target (Training Type)', fontsize=14)
    ax.set_ylabel('Mean F1 Macro', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=11)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plot_path = os.path.join(base_dir, 'classification_f1_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Classification summary bar chart saved to {plot_path}")

    # Print summary table
    print("\n=== Classification Summary (Mean F1 Macro across ages) ===")
    pivot = (summary.pivot_table(index=['target', 'model_type'], columns='display_model',
                                 values='f1_macro', aggfunc='mean')
             .round(3))
    print(pivot.to_string())


def _print_confusion_matrices(cls_df):
    """
    Recompute and print confusion matrices for the overall best model
    (across ages, age-specific) for each target.
    """
    all_preds = _collect_all_predictions()

    targets_info = {
        'glucose':  (classify_glucose,  GLUCOSE_LABELS),
        'diab_raine':  (classify_insulin,  INSULIN_LABELS),
    }

    for target, (classify_fn, labels) in targets_info.items():
        sub = cls_df[(cls_df['target'] == target) & (cls_df['model_type'] == 'age-specific')]
        if sub.empty:
            continue
        best_raw_model = sub.groupby('model')['f1_macro'].mean().idxmax()
        best = sub[sub['model'] == best_raw_model]['display_model'].iloc[0]

        all_true, all_pred = [], []
        for rec in all_preds[target]:
            if rec['model'] == best_raw_model and rec['model_type'] == 'age-specific':
                all_true.extend([classify_fn(v) for v in rec['y_true']])
                all_pred.extend([classify_fn(v) for v in rec['y_pred']])

        if not all_true:
            continue
        cm = confusion_matrix(all_true, all_pred, labels=list(labels.keys()))
        label_names = list(labels.values())
        print(f"\n=== Confusion Matrix: {target} — best model: {best} ({best_raw_model}) ===")
        cm_df = pd.DataFrame(cm, index=[f'True: {l}' for l in label_names],
                             columns=[f'Pred: {l}' for l in label_names])
        print(cm_df.to_string())

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=label_names, yticklabels=label_names)
        ax.set_title(f'{target.capitalize()} Classification — {best}\n(all age-specific folds combined)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{target}_{best.replace("/", "_")}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Confusion matrix plot saved to {plot_path}")

    # HOMA-IR confusion matrix: best age-specific model
    sub_h = cls_df[(cls_df['target'] == 'homa_ir') & (cls_df['model_type'] == 'age-specific')]
    if not sub_h.empty:
        best_raw = sub_h.groupby('model')['f1_macro'].mean().idxmax()
        best_disp = sub_h[sub_h['model'] == best_raw]['display_model'].iloc[0]

        glu_map = {(r['age'], r['model'], r['model_type']): r for r in all_preds['glucose']}
        all_true, all_pred = [], []
        for rec in all_preds['diab_raine']:
            if rec['model'] != best_raw or rec['model_type'] != 'age-specific':
                continue
            key = (rec['age'], rec['model'], rec['model_type'])
            if key not in glu_map:
                continue
            glu_rec = glu_map[key]
            glu_df = pd.DataFrame({'id': glu_rec['id'], 'glu_true': glu_rec['y_true'], 'glu_pred': glu_rec['y_pred']})
            ins_df = pd.DataFrame({'id': rec['id'], 'ins_true': rec['y_true'], 'ins_pred': rec['y_pred']})
            merged = glu_df.merge(ins_df, on='id', how='inner')
            if merged.empty:
                continue
            homa_true = compute_homa_ir(merged['glu_true'].values, merged['ins_true'].values)
            homa_pred = compute_homa_ir(merged['glu_pred'].values, merged['ins_pred'].values)
            all_true.extend([classify_homa_ir(v) for v in homa_true])
            all_pred.extend([classify_homa_ir(v) for v in homa_pred])

        if all_true:
            label_names = list(HOMAIR_LABELS.values())
            cm = confusion_matrix(all_true, all_pred, labels=list(HOMAIR_LABELS.keys()))
            print(f"\n=== Confusion Matrix: HOMA-IR — best model: {best_disp} ({best_raw}) ===")
            cm_df = pd.DataFrame(cm, index=[f'True: {l}' for l in label_names],
                                 columns=[f'Pred: {l}' for l in label_names])
            print(cm_df.to_string())

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=label_names, yticklabels=label_names)
            ax.set_title(f'HOMA-IR Classification — {best_disp}\n(all age-specific folds combined)',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('True', fontsize=12)
            plt.tight_layout()
            plot_path = os.path.join(RESULTS_DIR, f'confusion_matrix_homa_ir_{best_disp.replace("/", "_")}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Confusion matrix plot saved to {plot_path}")


# ─── Per-age confusion matrices ─────────────────────────────────────────────

def _plot_confusion_matrices_per_age(cls_df):
    """
    For each target (glucose, diab_raine, homa_ir), plot a 2×3 grid of confusion
    matrices — one per age — using the best-performing model for that specific age.
    """
    all_preds = _collect_all_predictions()

    targets_info = {
        'glucose': (classify_glucose, GLUCOSE_LABELS),
        'diab_raine': (classify_insulin, INSULIN_LABELS),
    }

    for target, (classify_fn, labels) in targets_info.items():
        label_names = list(labels.values())

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        for ax_i, age in enumerate(AGES):
            ax = axes[ax_i]

            sub = cls_df[
                (cls_df['target'] == target) &
                (cls_df['model_type'] == 'age-specific') &
                (cls_df['age'] == age)
            ]
            if sub.empty:
                ax.set_visible(False)
                continue

            best_raw = sub.groupby('model')['f1_macro'].mean().idxmax()
            best_display = sub[sub['model'] == best_raw]['display_model'].iloc[0]

            y_true_cls, y_pred_cls = [], []
            for rec in all_preds[target]:
                if (rec['model'] == best_raw and
                        rec['model_type'] == 'age-specific' and
                        rec['age'] == age):
                    y_true_cls.extend([classify_fn(v) for v in rec['y_true']])
                    y_pred_cls.extend([classify_fn(v) for v in rec['y_pred']])

            if not y_true_cls:
                ax.set_visible(False)
                continue

            cm = confusion_matrix(y_true_cls, y_pred_cls, labels=list(labels.keys()))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=label_names, yticklabels=label_names,
                        annot_kws={'size': 11})
            ax.set_title(f'Age {age} — {best_display}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('True', fontsize=11)
            ax.tick_params(labelsize=10)

        plt.suptitle(f'{target.capitalize()} Confusion Matrices per Age (Best Model per Age)',
                     fontsize=18, fontweight='bold', y=1.01)
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{target}_per_age.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Per-age confusion matrix plot saved to {plot_path}")

    # HOMA-IR per age
    label_names = list(HOMAIR_LABELS.values())
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    axes = axes.flatten()

    glu_map = {}
    for rec in all_preds['glucose']:
        key = (rec['age'], rec['model'], rec['model_type'])
        glu_map[key] = rec

    for ax_i, age in enumerate(AGES):
        ax = axes[ax_i]

        sub_h = cls_df[
            (cls_df['target'] == 'homa_ir') &
            (cls_df['model_type'] == 'age-specific') &
            (cls_df['age'] == age)
        ]
        if sub_h.empty:
            ax.set_visible(False)
            continue

        best_raw = sub_h.groupby('model')['f1_macro'].mean().idxmax()
        best_display = sub_h[sub_h['model'] == best_raw]['display_model'].iloc[0]

        all_true, all_pred = [], []
        for rec in all_preds['diab_raine']:
            if (rec['model'] != best_raw or
                    rec['model_type'] != 'age-specific' or
                    rec['age'] != age):
                continue
            key = (rec['age'], rec['model'], rec['model_type'])
            if key not in glu_map:
                continue
            glu_rec = glu_map[key]
            glu_df = pd.DataFrame({'id': glu_rec['id'],
                                   'glu_true': glu_rec['y_true'],
                                   'glu_pred': glu_rec['y_pred']})
            ins_df = pd.DataFrame({'id': rec['id'],
                                   'ins_true': rec['y_true'],
                                   'ins_pred': rec['y_pred']})
            merged = glu_df.merge(ins_df, on='id', how='inner')
            if merged.empty:
                continue
            homa_true = compute_homa_ir(merged['glu_true'].values, merged['ins_true'].values)
            homa_pred = compute_homa_ir(merged['glu_pred'].values, merged['ins_pred'].values)
            all_true.extend([classify_homa_ir(v) for v in homa_true])
            all_pred.extend([classify_homa_ir(v) for v in homa_pred])

        if not all_true:
            ax.set_visible(False)
            continue

        cm = confusion_matrix(all_true, all_pred, labels=list(HOMAIR_LABELS.keys()))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=label_names, yticklabels=label_names,
                    annot_kws={'size': 10})
        ax.set_title(f'Age {age} — {best_display}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.tick_params(labelsize=10)

    plt.suptitle('HOMA-IR Confusion Matrices per Age (Best Model per Age)',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'confusion_matrix_homa_ir_per_age.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Per-age HOMA-IR confusion matrix plot saved to {plot_path}")


# ─── Feature importance ─────────────────────────────────────────────────────

def aggregate_feature_importance():
    base_dir = RESULTS_DIR
    importance_data = []

    def process_importance(path, model_name, age, type_str, target):
        if os.path.exists(path):
            df_imp = pd.read_csv(path)
            if 'feature' in df_imp.columns and 'importance' in df_imp.columns:
                total = df_imp['importance'].sum()
                for _, row in df_imp.iterrows():
                    importance_data.append({
                        'target': target, 'age': age, 'model': model_name, 'type': type_str,
                        'variable': row['feature'],
                        'weight': (row['importance'] / total * 100) if total > 0 else 0
                    })

    for target in TARGETS:
        for age in AGES:
            baselines_dir = os.path.join(base_dir, f"age_{age}_{target}", "baselines")
            if os.path.exists(baselines_dir):
                for m in os.listdir(baselines_dir):
                    if m in ['ElasticNet', 'ExtraTrees', 'RandomForest', 'XGBoost', 'KAN']:
                        process_importance(os.path.join(baselines_dir, m, "feature_importance.csv"),
                                           m, age, 'age-specific', target)

        long_baselines_dir = os.path.join(base_dir, f"longitudinal_{target}", "baselines")
        if os.path.exists(long_baselines_dir):
            for m in os.listdir(long_baselines_dir):
                if m in ['ElasticNet', 'ExtraTrees', 'RandomForest', 'XGBoost', 'KAN']:
                    imp_file = os.path.join(long_baselines_dir, m, "feature_importance.csv")
                    if os.path.exists(imp_file):
                        df_imp = pd.read_csv(imp_file)
                        total = df_imp['importance'].sum()
                        for _, row in df_imp.iterrows():
                            importance_data.append({
                                'target': target, 'age': 'all', 'model': m, 'type': 'longitudinal',
                                'variable': row['feature'],
                                'weight': (row['importance'] / total * 100) if total > 0 else 0
                            })

    imp_df = pd.DataFrame(importance_data)
    imp_df.to_csv(os.path.join(base_dir, "feature_importance_aggregated.csv"), index=False)
    print("Feature importance aggregated to results_insulin_glucose_to14/feature_importance_aggregated.csv")

    if not imp_df.empty:
        for target in TARGETS:
            t_df = imp_df[imp_df['target'] == target]
            if t_df.empty:
                continue
            agg_imp = t_df.groupby(['model', 'variable'])['weight'].mean().reset_index()
            top_features = agg_imp.groupby('variable')['weight'].mean().sort_values(ascending=False).head(15).index
            plot_df = agg_imp[agg_imp['variable'].isin(top_features)].copy()
            plot_df['variable'] = pd.Categorical(plot_df['variable'], categories=top_features, ordered=True)
            plt.figure(figsize=(14, 10))
            sns.barplot(data=plot_df, x='weight', y='variable', hue='model', palette="bright")
            plt.title(f'Top 15 Feature Importance ({target}) across Models', fontsize=22, fontweight='bold', pad=20)
            plt.xlabel('Average Percentage Importance (%)', fontsize=18)
            plt.ylabel('Feature', fontsize=18)
            plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
            plt.tick_params(labelsize=14)
            plt.tight_layout()
            plot_path = os.path.join(base_dir, f"feature_importance_by_model_{target}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Feature importance plot saved to {plot_path}")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results_insulin_glucose_to14',
                        help="Results folder to analyse (default: results_insulin_glucose_to14)")
    parser.add_argument('--skip_regression', action='store_true',
                        help="Skip regression analysis (faster if only classification needed)")
    args = parser.parse_args()

    if args.results_dir is not None:
        RESULTS_DIR = os.path.join(current_dir, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir

    if not args.skip_regression:
        dfs = process_results()
        plot_results(dfs)
        plot_homa_ir_results()
        aggregate_feature_importance()

    print("\n" + "=" * 60)
    print("CLASSIFICATION ANALYSIS")
    print("Glucose: <6.1 Normal | 6.1–<7.0 Prediabetes | ≥7.0 Diabetes")
    print("Insulin: ≤6 Optimal  | 6–10 Normal/Moderate  | >10 High Risk")
    print("HOMA-IR: <1 Optimal  | 1–<2 Normal | 2–<3 Early IR | ≥3 Significant IR")
    print("=" * 60)
    cls_df = process_classification()
    plot_classification_results(cls_df)
    _print_confusion_matrices(cls_df)
    _plot_confusion_matrices_per_age(cls_df)