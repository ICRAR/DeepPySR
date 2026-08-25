import os
import pandas as pd
import numpy as np
import glob
import sys
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Add test/ and test/bmi to path to import load_bmi_agg_data
current_dir = os.path.dirname(os.path.abspath(__file__))
if not current_dir:
    current_dir = "."
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

from bmi_utils import load_bmi_agg_data
from analysis_v1_utils import (calculate_metrics, leak_free_process_and_select,
                                get_best_formula_from_raw, get_oof_predictions,
                                get_formula_fold_metrics, collect_model_fold_data,
                                compute_fold_metrics_from_predictions,
                                se_from_fold_data, run_wilcoxon_analysis, compute_se,
                                _cv_pooled_oof_predictions)

TASK = 'regression'
BASE_DIR = os.path.join(current_dir, "results_bmi_all")
AGES = [8, 10, 14, 17, 20, 23, 27]
INTERP_MAX_COMPLEXITY = 30


def process_results():
    """Aggregate every model's genuine held-out metrics for both age-specific
    and longitudinal BMI settings.

    Age-specific runs used plain KFold (no groups/stratify -- see
    test_all_models_bmi.py's cv_kwargs for setting='age_specific'), so each
    age is handled directly via leak_free_process_and_select.

    Longitudinal used StratifiedGroupKFold (groups=child_id, stratify_by=age
    -- see cv_kwargs for setting='longitudinal'). DeepPySR/PySR/KAN/KANSym
    metrics there come from the single longitudinal model's predictions.csv
    (genuine held-out, pooled across all ages), then sliced by age -- exactly
    how baselines are already scored, never a re-evaluated formula. The one
    exception is Interpretable DeepPySR, which needs pooled leak-free OOF
    predictions at a complexity-constrained operating point that
    predictions.csv doesn't cover (see get_oof_predictions), then sliced by
    age the same way.
    """
    all_rows = []
    best_rows = []

    # --- Age-specific ---
    age_spec_dir = os.path.join(BASE_DIR, "age_specific")
    for age in AGES:
        age_path = os.path.join(age_spec_dir, f"age_{age}")
        if not os.path.isdir(age_path):
            continue
        _, X_age, y_age = load_bmi_agg_data(age=age)
        all_df, best_df = leak_free_process_and_select(
            age_path, X_age, y_age, task=TASK, interp_max_complexity=INTERP_MAX_COMPLEXITY)
        if not all_df.empty:
            all_df['age'] = age
            all_df['type'] = 'age-specific'
            all_rows.append(all_df)
        if not best_df.empty:
            best_df['age'] = age
            best_df['type'] = 'age-specific'
            best_rows.append(best_df)

    # --- Longitudinal ---
    ids_all, X_all, y_all = load_bmi_agg_data()
    age_col = X_all['age']
    long_dir = os.path.join(BASE_DIR, "longitudinal")

    def _slice_metrics(y_true_all, y_pred_all, age, age_ref):
        """age_ref must be the age array in the SAME row order as
        y_true_all/y_pred_all. predictions.csv rows are in fold-concatenated
        order (not X_all's original order) but carry their own 'age' column
        (from extra_data) -- use that, not X_all['age'], when slicing
        predictions.csv. y_pred_oof from get_oof_predictions IS aligned to
        X_all's original row order, so age_col is correct there instead."""
        age_ref = np.asarray(age_ref)
        mask = (age_ref == age)
        if mask.sum() == 0:
            return None
        return calculate_metrics(np.asarray(y_true_all)[mask], np.asarray(y_pred_all)[mask], task=TASK)

    long_all = []
    long_best = {a: [] for a in AGES}

    baselines_dir = os.path.join(long_dir, "baselines")
    if os.path.isdir(baselines_dir):
        for model_name in sorted(os.listdir(baselines_dir)):
            model_path = os.path.join(baselines_dir, model_name)
            if not os.path.isdir(model_path):
                continue
            pred_file = os.path.join(model_path, "predictions.csv")
            if not os.path.exists(pred_file):
                continue
            df_pred = pd.read_csv(pred_file)
            family = 'kan' if model_name.lower() == 'kan' else model_name
            pred_age = df_pred['age']
            for age in AGES:
                m = _slice_metrics(df_pred['y_true'], df_pred['y_pred'], age, pred_age)
                if m is None:
                    continue
                row = {'model': model_name, 'family': family, 'r2': m[0], 'rmse': m[1],
                       'mae': m[2], 'pearson_r': m[3], 'complexity': np.nan, 'formula': '',
                       'source_path': model_path, 'formula_model_type': '', 'max_complexity': np.nan,
                       'age': age, 'type': 'longitudinal'}
                long_all.append(row)
                long_best[age].append(dict(row, display_model=model_name))

            if model_name.lower() == 'kan' and 'y_pred_kansym' in df_pred.columns:
                formula, complexity, _ = get_best_formula_from_raw(
                    model_path, X_all, y_all, prefix='formulas_fold', model_type='kan', task=TASK,
                    stratify=age_col, groups=ids_all)
                for age in AGES:
                    m = _slice_metrics(df_pred['y_true'], df_pred['y_pred_kansym'], age, pred_age)
                    if m is None:
                        continue
                    row = {'model': 'KANSym', 'family': 'kansym', 'r2': m[0], 'rmse': m[1],
                           'mae': m[2], 'pearson_r': m[3], 'complexity': complexity, 'formula': formula,
                           'source_path': model_path, 'formula_model_type': 'kan', 'max_complexity': np.nan,
                           'age': age, 'type': 'longitudinal'}
                    long_all.append(row)
                    long_best[age].append(dict(row, display_model='KANSym'))

    for subdir, model_type in [('deeppysr', 'deeppysr'), ('pysr', 'pysr')]:
        sr_dir = os.path.join(long_dir, subdir)
        if not os.path.isdir(sr_dir):
            continue
        # Longitudinal is ONE model fit on the pooled all-ages dataset (age
        # is a feature), so there must be exactly one winning variant/formula
        # for the whole family -- never a different one per age. Rank
        # variants by their POOLED (unsliced) r2/predictions.csv, matching
        # how leak_free_process_and_select picks a winner everywhere else;
        # only after that pick do we slice the winner's predictions by age
        # for the per-age reporting table.
        uncon_variants = []   # (pooled_r2, variant, complexity, formula, v_path, df_pred)
        interp_variants = []  # (pooled_r2, variant, complexity, formula, v_path, y_pred_oof)

        def _emit_all(variant, complexity, formula, v_path, y_true_all, y_pred_all, pred_age_all,
                      max_complexity):
            """Every variant's per-age slice goes into long_all (needed intact
            for the ablation plots and aggregated_results.csv), regardless of
            whether this variant ends up winning long_best below."""
            for age in AGES:
                m = _slice_metrics(y_true_all, y_pred_all, age, pred_age_all)
                if m is None or np.isnan(m[0]):
                    continue
                row = {'model': variant, 'family': model_type, 'r2': m[0], 'rmse': m[1],
                       'mae': m[2], 'pearson_r': m[3], 'complexity': complexity, 'formula': formula,
                       'source_path': v_path, 'formula_model_type': model_type,
                       'max_complexity': max_complexity, 'age': age, 'type': 'longitudinal'}
                long_all.append(row)

        for variant in sorted(os.listdir(sr_dir)):
            v_path = os.path.join(sr_dir, variant)
            if not os.path.isdir(v_path):
                continue
            pred_file = os.path.join(v_path, "predictions.csv")
            if not os.path.exists(pred_file):
                continue
            df_pred = pd.read_csv(pred_file)
            pooled_m = calculate_metrics(df_pred['y_true'], df_pred['y_pred'], task=TASK)
            formula, complexity, _ = get_best_formula_from_raw(
                v_path, X_all, y_all, model_type=model_type, task=TASK, stratify=age_col, groups=ids_all)
            uncon_variants.append((pooled_m[0], variant, complexity, formula, v_path, df_pred))
            _emit_all(variant, complexity, formula, v_path, df_pred['y_true'], df_pred['y_pred'],
                      df_pred['age'], np.nan)

            if model_type == 'deeppysr':
                y_pred_oof = get_oof_predictions(
                    v_path, X_all, y_all, model_type=model_type, task=TASK, stratify=age_col,
                    groups=ids_all, max_complexity=INTERP_MAX_COMPLEXITY)
                if y_pred_oof is not None:
                    valid = ~np.isnan(y_pred_oof)
                    pooled_mi = calculate_metrics(y_all.values[valid], y_pred_oof[valid], task=TASK)
                    formula_i, complexity_i, _ = get_best_formula_from_raw(
                        v_path, X_all, y_all, model_type=model_type, task=TASK, stratify=age_col,
                        groups=ids_all, max_complexity=INTERP_MAX_COMPLEXITY)
                    interp_variants.append((pooled_mi[0], variant, complexity_i, formula_i, v_path, y_pred_oof))
                    _emit_all(f"{variant}__interp{INTERP_MAX_COMPLEXITY}", complexity_i, formula_i, v_path,
                              y_all.values, y_pred_oof, age_col, INTERP_MAX_COMPLEXITY)

        # Longitudinal is ONE model fit on the pooled all-ages dataset (age is
        # a feature), so exactly one variant/formula must win for the whole
        # family -- never a different one per age. Pick by POOLED (unsliced)
        # r2, matching how leak_free_process_and_select picks a winner
        # everywhere else; only then slice that single winner's predictions
        # by age for long_best's per-age reporting rows.
        def _emit_best(pooled_r2, variant, complexity, formula, v_path, y_true_all, y_pred_all, pred_age_all,
                       max_complexity, display_model):
            for age in AGES:
                m = _slice_metrics(y_true_all, y_pred_all, age, pred_age_all)
                if m is None or np.isnan(m[0]):
                    continue
                row = {'model': variant, 'family': model_type, 'r2': m[0], 'rmse': m[1],
                       'mae': m[2], 'pearson_r': m[3], 'complexity': complexity, 'formula': formula,
                       'source_path': v_path, 'formula_model_type': model_type,
                       'max_complexity': max_complexity, 'age': age, 'type': 'longitudinal'}
                long_best[age].append(dict(row, display_model=display_model))

        if uncon_variants:
            pooled_r2, variant, complexity, formula, v_path, df_pred = max(uncon_variants, key=lambda t: t[0])
            _emit_best(pooled_r2, variant, complexity, formula, v_path, df_pred['y_true'], df_pred['y_pred'],
                       df_pred['age'], np.nan, 'PySR' if model_type == 'pysr' else 'Best DeepPySR')

        if interp_variants and model_type == 'deeppysr':
            pooled_ri, variant_i, complexity_i, formula_i, v_path_i, y_pred_oof_i = max(
                interp_variants, key=lambda t: t[0])
            _emit_best(pooled_ri, variant_i, complexity_i, formula_i, v_path_i, y_all.values, y_pred_oof_i,
                       age_col, INTERP_MAX_COMPLEXITY, 'Interpretable DeepPySR')
            # "Best" >= "Interpretable" by construction, matching leak_free_process_and_select
            if uncon_variants and pooled_ri > max(t[0] for t in uncon_variants):
                for age in AGES:
                    long_best[age] = [r for r in long_best[age] if r.get('display_model') != 'Best DeepPySR']
                _emit_best(pooled_ri, variant_i, complexity_i, formula_i, v_path_i, y_all.values, y_pred_oof_i,
                           age_col, np.nan, 'Best DeepPySR')

    if long_all:
        long_all_df = pd.DataFrame(long_all)
        all_rows.append(long_all_df)
    long_best_df = pd.DataFrame([r for rows in long_best.values() for r in rows])
    if not long_best_df.empty:
        best_rows.append(long_best_df)

    all_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    best_df = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()

    if not all_df.empty:
        all_df['r2'] = all_df['r2'].clip(lower=0)
    if not best_df.empty:
        best_df['r2'] = best_df['r2'].clip(lower=0)

    all_df.to_csv(os.path.join(BASE_DIR, "bmi_aggregated_results.csv"), index=False)
    print(f"Results saved to {os.path.join(BASE_DIR, 'bmi_aggregated_results.csv')}")
    return all_df, best_df


def save_best_models(best_df):
    plot_csv_path = os.path.join(BASE_DIR, 'bmi_best_models_metrics.csv')
    best_df.to_csv(plot_csv_path, index=False)
    print(f"Best models plot data saved to {plot_csv_path}")

    interp_df = best_df[best_df['display_model'] == 'Interpretable DeepPySR']
    print(f"\n--- Interpretable DeepPySR Formulas (Complexity < {INTERP_MAX_COMPLEXITY}) ---")
    print(interp_df[['age', 'type', 'model', 'formula', 'r2', 'complexity']].to_string(index=False))
    interp_df[['age', 'type', 'model', 'formula', 'r2', 'complexity']].to_csv(
        os.path.join(BASE_DIR, 'interpretable_deeppysr_formulas.csv'), index=False)


def _fold_data_for_row(row, X_age, y_age, stratify, groups):
    source_path = row.get('source_path', '')
    if not source_path or not os.path.isdir(str(source_path)):
        return None
    family = row['family']
    max_c = row.get('max_complexity', np.nan)
    is_interp = pd.notna(max_c)

    if family == 'kansym':
        return compute_fold_metrics_from_predictions(
            source_path, X_age, y_age, task=TASK, stratify=stratify, groups=groups, pred_col='y_pred_kansym')
    if family in ('deeppysr', 'pysr') and is_interp:
        return get_formula_fold_metrics(
            source_path, X_age, y_age, task=TASK, model_type=family, stratify=stratify, groups=groups,
            max_complexity=int(max_c))
    return collect_model_fold_data(source_path, "", X_age, y_age, TASK, model_type=family,
                                    stratify=stratify, groups=groups)


def compute_se_and_wilcoxon(best_df):
    """Compute per-fold SE and Wilcoxon vs Best DeepPySR for each BMI age/type target."""
    ids_all, X_all, y_all = load_bmi_agg_data()
    age_col_all = X_all['age']

    se_frames = []
    for age in AGES:
        _, X_age, y_age = load_bmi_agg_data(age=age)
        sub = best_df[(best_df['age'] == age) & (best_df['type'] == 'age-specific')]
        fold_data = {row['display_model']: _fold_data_for_row(row, X_age, y_age, None, None)
                     for _, row in sub.iterrows()}
        fold_data = {k: v for k, v in fold_data.items() if v is not None}
        se_map = {}
        for model_name, fd in fold_data.items():
            ses = se_from_fold_data(fd)
            ses['n_folds'] = len(fd)
            se_map[model_name] = ses
        if 'Best DeepPySR' in fold_data:
            run_wilcoxon_analysis(fold_data, 'Best DeepPySR', TASK,
                                  output_file=os.path.join(current_dir, f"wilcoxon_results_age{age}.csv"))
        se_frames.append((age, 'age-specific', se_map))

    long_sub = best_df[best_df['type'] == 'longitudinal']
    for age in AGES:
        age_rows = long_sub[long_sub['age'] == age]
        fold_data = {row['display_model']: _fold_data_for_row(row, X_all, y_all, age_col_all, ids_all)
                     for _, row in age_rows.iterrows()}
        fold_data = {k: v for k, v in fold_data.items() if v is not None}
        se_map = {}
        for model_name, fd in fold_data.items():
            ses = se_from_fold_data(fd)
            ses['n_folds'] = len(fd)
            se_map[model_name] = ses
        if age == AGES[-1] and 'Best DeepPySR' in fold_data:
            run_wilcoxon_analysis(fold_data, 'Best DeepPySR', TASK,
                                  output_file=os.path.join(current_dir, 'wilcoxon_results_longitudinal.csv'))
        se_frames.append((age, 'longitudinal', se_map))

    metrics_csv_path = os.path.join(BASE_DIR, 'bmi_best_models_metrics.csv')
    if os.path.exists(metrics_csv_path):
        metrics_df = pd.read_csv(metrics_csv_path)
        first_se = next((s for _, _, sm in se_frames for s in sm.values()), {})
        se_cols = [c for c in first_se.keys() if c != 'n_folds']
        for col in se_cols + ['n_folds']:
            metrics_df[col] = np.nan
        for i, row in metrics_df.iterrows():
            sm = next((sm for a, t, sm in se_frames if a == row['age'] and t == row['type']), {})
            if row['display_model'] in sm:
                for col in se_cols + ['n_folds']:
                    metrics_df.at[i, col] = sm[row['display_model']].get(col, np.nan)
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"SE merged into {metrics_csv_path}")


def plot_results(best_df):
    """Line plots of r2/rmse/mae vs age, for the pre-selected best models per age/type."""
    plot_df = best_df.copy()
    plot_df['r2'] = plot_df['r2'].clip(lower=0)
    metrics = ['r2', 'rmse', 'mae']
    types = ['longitudinal', 'age-specific']

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
            sns.lineplot(data=plot_df[plot_df['type'] == t], x='age', y=metric, hue='display_model', ax=ax,
                         linestyle=linestyle, linewidth=3.0, palette=model_colors, marker='o', markersize=8)
            type_label = "Age-specific" if t == 'age-specific' else "Longitudinal"
            ax.set_title(f'{type_label}: {metric.upper()} vs Age', fontsize=20, fontweight='bold', pad=15)
            ax.set_ylabel(metric.upper(), fontsize=16)
            ax.set_xlabel('Age', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=12)
            if metric in ['rmse', 'mae']:
                ax.set_ylim(0, 10)
            if ax.get_legend():
                ax.get_legend().remove()

    legend_elements = [Line2D([0], [0], color=model_colors[m], lw=3, label=m) for m in models]
    legend_elements.append(Line2D([0], [0], color='white', label=''))
    legend_elements.append(Line2D([0], [0], color='black', lw=3, ls='--', label='Age-specific'))
    legend_elements.append(Line2D([0], [0], color='black', lw=3, ls='-', label='Longitudinal'))
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.91, 0.5),
               fontsize=14, frameon=True, title='Models & Types', title_fontsize=16, handlelength=4.0)

    plt.suptitle('BMI Prediction Performance: Best Models Comparison', fontsize=26, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plot_path = os.path.join(BASE_DIR, 'bmi_metrics_vs_age.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Combined metrics plot saved to {plot_path}")


def aggregate_feature_importance():
    """Feature importance for baselines -- unrelated to the formula-selection leak."""
    importance_data = []

    def process_importance(path, model_name, age, type_str):
        if os.path.exists(path):
            df_imp = pd.read_csv(path)
            if 'feature' in df_imp.columns and 'importance' in df_imp.columns:
                total = df_imp['importance'].sum()
                df_imp['importance_pct'] = (df_imp['importance'] / total * 100) if total > 0 else 0
                for _, row in df_imp.iterrows():
                    importance_data.append({'age': age, 'model': model_name, 'type': type_str,
                                             'variable': row['feature'], 'weight': row['importance_pct']})

    age_spec_dir = os.path.join(BASE_DIR, "age_specific")
    for age in AGES:
        baselines_dir = os.path.join(age_spec_dir, f"age_{age}", "baselines")
        if os.path.exists(baselines_dir):
            for m in os.listdir(baselines_dir):
                if m in ['ElasticNet', 'ExtraTrees', 'RandomForest', 'XGBoost', 'KAN']:
                    process_importance(os.path.join(baselines_dir, m, "feature_importance.csv"), m, age, 'age-specific')

    long_baselines_dir = os.path.join(BASE_DIR, "longitudinal", "baselines")
    if os.path.exists(long_baselines_dir):
        for m in os.listdir(long_baselines_dir):
            if m in ['ElasticNet', 'ExtraTrees', 'RandomForest', 'XGBoost', 'KAN']:
                process_importance(os.path.join(long_baselines_dir, m, "feature_importance.csv"), m, 'all', 'longitudinal')

    imp_df = pd.DataFrame(importance_data)
    imp_df.to_csv(os.path.join(BASE_DIR, "feature_importance_aggregated.csv"), index=False)
    print("Feature importance aggregated to results_bmi_all/feature_importance_aggregated.csv")

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
        plot_path = os.path.join(BASE_DIR, "feature_importance_by_model.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Combined feature importance plot saved to {plot_path}")


def plot_vps_vpr_ablation(all_df):
    """Ablation: VPS/VPR effect -- best R² per (age, type, vps/vpr) at the
    model's own default operating point, genuinely held-out."""
    metrics = ['r2', 'rmse', 'mae', 'complexity']
    metric_labels = ['R²', 'RMSE', 'MAE', 'Complexity']
    types = ['longitudinal', 'age-specific']

    deep_df = all_df[(all_df['family'] == 'deeppysr') & all_df['max_complexity'].isna()].copy()

    def vps_vpr_label(m):
        match = re.search(r'vps(\d+)_vpr(\d+)', m)
        return f"vps{match.group(1)}/vpr{match.group(2)}" if match else m
    deep_df['label'] = deep_df['model'].apply(vps_vpr_label)
    deep_df = deep_df.loc[deep_df.groupby(['age', 'type', 'label'])['r2'].idxmax()].reset_index(drop=True)

    pysr_df = all_df[all_df['family'] == 'pysr'].copy()
    if not pysr_df.empty:
        pysr_df = pysr_df.loc[pysr_df.groupby(['age', 'type'])['r2'].idxmax()].reset_index(drop=True)
    pysr_df['label'] = 'PySR (no VPS/VPR)'

    csv_df = pd.concat([deep_df[['age', 'type', 'label'] + metrics],
                        pysr_df[['age', 'type', 'label'] + metrics]], ignore_index=True)
    csv_df.to_csv(os.path.join(BASE_DIR, 'ablation_vps_vpr.csv'), index=False)
    print(f"VPS/VPR ablation data saved to {BASE_DIR}/ablation_vps_vpr.csv")

    if deep_df.empty and pysr_df.empty:
        print("No data for VPS/VPR ablation")
        return

    def sort_key(lbl):
        m = re.search(r'vps(\d+)/vpr(\d+)', lbl)
        return (int(m.group(1)), int(m.group(2))) if m else (999, 999)

    all_deep_labels = sorted(deep_df['label'].unique(), key=sort_key)
    palette = sns.color_palette("tab10", n_colors=len(all_deep_labels))
    label_colors = dict(zip(all_deep_labels, palette))
    ages = sorted(all_df['age'].dropna().unique())

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    plt.rcParams.update({'font.size': 11})

    for row_i, t in enumerate(types):
        deep_t = deep_df[deep_df['type'] == t]
        pysr_t = pysr_df[pysr_df['type'] == t]
        type_label = 'Longitudinal' if t == 'longitudinal' else 'Age-Specific'
        for col_j, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row_i, col_j]
            for lbl in all_deep_labels:
                sub = deep_t[deep_t['label'] == lbl]
                if not sub.empty:
                    pts = sub.groupby('age')[metric].mean().reset_index()
                    ax.plot(pts['age'], pts[metric], marker='o', markersize=4,
                            color=label_colors[lbl], label=lbl, linewidth=1.5)
            if not pysr_t.empty:
                pts = pysr_t.groupby('age')[metric].mean().reset_index()
                ax.plot(pts['age'], pts[metric], marker='s', markersize=6, color='black',
                        label='PySR (no VPS/VPR)', linewidth=2, linestyle='--')
            ax.set_title(f'BMI {type_label} – {mlabel}', fontsize=12, fontweight='bold')
            ax.set_ylabel(mlabel, fontsize=10)
            ax.set_xlabel('Age', fontsize=10)
            ax.set_xticks(ages)
            if ax.get_legend():
                ax.get_legend().remove()

    legend_elements = [Line2D([0], [0], color=label_colors[lbl], marker='o', lw=1.5, label=lbl)
                       for lbl in all_deep_labels]
    legend_elements.append(Line2D([0], [0], color='black', marker='s', lw=2, ls='--', label='PySR (no VPS/VPR)'))
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize=9, frameon=True, title='Setting', title_fontsize=11)
    plt.suptitle('Ablation: VPS/VPR Effect (best APS per config, default r2w/λ)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.88, 1.0])
    out = os.path.join(BASE_DIR, 'ablation_vps_vpr.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"VPS/VPR ablation plot saved to {out}")


def plot_pareto_ablation():
    """Ablation: pareto r2w/λ effect, fixed VPS=25/VPR=100/APS=10.0, per age &
    type -- reconstructed genuinely held-out per fold via pareto_point=."""
    metrics = ['r2', 'rmse', 'mae', 'complexity']
    metric_labels = ['R²', 'RMSE', 'MAE', 'Complexity']
    types = ['longitudinal', 'age-specific']
    r2w_list = [1, 1.5, 2]
    lambda_list = [0.001, 0.005, 0.01]

    ids_all, X_all, y_all = load_bmi_agg_data()
    age_col_all = X_all['age']
    rows = []

    def find_variant(sr_dir):
        if not os.path.isdir(sr_dir):
            return None
        for v in os.listdir(sr_dir):
            if '_vps25_' in v and '_vpr100_' in v and 'aps10.0' in v:
                return v
        return None

    # age-specific
    for age in AGES:
        _, X_age, y_age = load_bmi_agg_data(age=age)
        v = find_variant(os.path.join(BASE_DIR, 'age_specific', f'age_{age}', 'deeppysr'))
        if v is None:
            continue
        v_path = os.path.join(BASE_DIR, 'age_specific', f'age_{age}', 'deeppysr', v)
        for r2w in r2w_list:
            for lam in lambda_list:
                formula, complexity, m = get_best_formula_from_raw(
                    v_path, X_age, y_age, task=TASK, model_type='deeppysr', pareto_point=(r2w, lam))
                if not formula:
                    continue
                rows.append({'age': age, 'type': 'age-specific', 'label': f"r2w={r2w}, λ={lam}",
                             'r2': m[0], 'rmse': m[1], 'mae': m[2], 'complexity': complexity})

    # longitudinal: pool genuine held-out OOF predictions per (r2w, lambda)
    # point across all folds, then slice by age -- same pattern as the
    # Interpretable-DeepPySR longitudinal computation in process_results().
    v = find_variant(os.path.join(BASE_DIR, 'longitudinal', 'deeppysr'))
    if v is not None:
        v_path = os.path.join(BASE_DIR, 'longitudinal', 'deeppysr', v)
        for r2w in r2w_list:
            for lam in lambda_list:
                y_pred_oof, valid, _ = _cv_pooled_oof_predictions(
                    v_path, X_all, y_all, task=TASK, model_type='deeppysr',
                    stratify=age_col_all, groups=ids_all, pareto_point=(r2w, lam))
                for age in AGES:
                    mask = (age_col_all.values == age) & valid
                    if mask.sum() == 0:
                        continue
                    m = calculate_metrics(y_all.values[mask], y_pred_oof[mask], task=TASK)
                    rows.append({'age': age, 'type': 'longitudinal', 'label': f"r2w={r2w}, λ={lam}",
                                 'r2': m[0], 'rmse': m[1], 'mae': m[2], 'complexity': np.nan})

    deep_df = pd.DataFrame(rows)

    all_df = pd.read_csv(os.path.join(BASE_DIR, 'bmi_aggregated_results.csv'))
    pysr_mask = (all_df['family'] == 'pysr') & all_df['model'].str.contains('aps10.0', na=False)
    pysr_df = all_df[pysr_mask].copy()
    pysr_df['label'] = 'PySR (reference)'

    csv_cols = ['age', 'type', 'label'] + metrics
    csv_df = pd.concat([deep_df[csv_cols] if not deep_df.empty else pd.DataFrame(columns=csv_cols),
                        pysr_df[csv_cols] if not pysr_df.empty else pd.DataFrame(columns=csv_cols)],
                       ignore_index=True)
    csv_df.to_csv(os.path.join(BASE_DIR, 'ablation_pareto.csv'), index=False)
    print(f"Pareto ablation data saved to {BASE_DIR}/ablation_pareto.csv")

    if deep_df.empty and pysr_df.empty:
        print("No data for pareto ablation")
        return

    def sort_key(lbl):
        r2w_m = re.search(r'r2w=([\d.]+)', lbl)
        l_m = re.search(r'λ=([\d.]+)', lbl)
        return (float(r2w_m.group(1)), float(l_m.group(1))) if r2w_m and l_m else (999, 999)

    all_deep_labels = sorted(deep_df['label'].unique(), key=sort_key) if not deep_df.empty else []
    r2w_vals = sorted(set(float(re.search(r'r2w=([\d.]+)', l).group(1)) for l in all_deep_labels))
    palette_colors = ['#2166ac', '#4dac26', '#d6604d']
    r2w_palette = dict(zip(r2w_vals, palette_colors * (len(r2w_vals) // len(palette_colors) + 1)))
    lambda_markers = {0.001: 'o', 0.005: 's', 0.01: '^'}

    def label_color(lbl):
        m = re.search(r'r2w=([\d.]+)', lbl)
        return r2w_palette.get(float(m.group(1)), '#888') if m else 'black'

    def label_marker(lbl):
        m = re.search(r'λ=([\d.]+)', lbl)
        return lambda_markers.get(float(m.group(1)), 'o') if m else 'D'

    ages = sorted(set(csv_df['age'].dropna().unique()))

    fig, axes = plt.subplots(2, 4, figsize=(26, 12))
    plt.rcParams.update({'font.size': 11})
    for row_i, t in enumerate(types):
        deep_t = deep_df[deep_df['type'] == t] if not deep_df.empty else deep_df
        pysr_t = pysr_df[pysr_df['type'] == t] if not pysr_df.empty else pysr_df
        type_label = 'Longitudinal' if t == 'longitudinal' else 'Age-Specific'
        for col_j, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row_i, col_j]
            for lbl in all_deep_labels:
                sub = deep_t[deep_t['label'] == lbl]
                if not sub.empty:
                    pts = sub.groupby('age')[metric].mean().reset_index()
                    ax.plot(pts['age'], pts[metric], marker=label_marker(lbl), markersize=5,
                            color=label_color(lbl), label=lbl, linewidth=1.5)
            if not pysr_t.empty:
                pts = pysr_t.groupby('age')[metric].mean().reset_index()
                ax.plot(pts['age'], pts[metric], marker='D', markersize=6, color='black',
                        label='PySR (reference)', linewidth=2, linestyle='--')
            ax.set_title(f'BMI {type_label} – {mlabel}', fontsize=12, fontweight='bold')
            ax.set_ylabel(mlabel, fontsize=10)
            ax.set_xlabel('Age', fontsize=10)
            ax.set_xticks(ages)
            if ax.get_legend():
                ax.get_legend().remove()

    legend_elements = [Line2D([0], [0], color=label_color(lbl), marker=label_marker(lbl), lw=1.5, label=lbl)
                       for lbl in all_deep_labels]
    legend_elements.append(Line2D([0], [0], color='black', marker='D', lw=2, ls='--', label='PySR (reference)'))
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize=9, frameon=True, title='Setting', title_fontsize=11)
    plt.suptitle('Ablation: Pareto r2w/λ Effect (VPS=25, VPR=100, APS=10.0)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.88, 1.0])
    out = os.path.join(BASE_DIR, 'ablation_pareto.png')
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


def _load_hof_data(model_dir, single_fold=False):
    """Load each fold's hall_of_fame.csv (raw PySR search log). By default
    aggregates (mean loss per complexity) across every fold's timestamp
    subdirectory; with single_fold=True, uses only the earliest one (fold 0),
    avoiding the cross-fold-averaging artefact where folds that converge on
    different large-formula structures make the averaged front look like it
    plateaus early even though each fold individually keeps improving."""
    pysr_out = os.path.join(model_dir, 'pysr_outputs', 'y')
    if not os.path.exists(pysr_out):
        return pd.DataFrame()
    timestamps = sorted(os.listdir(pysr_out))
    if single_fold and timestamps:
        timestamps = timestamps[:1]
    rows = []
    for ts in timestamps:
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


def plot_pareto_front_rmse(best_df):
    """Single Pareto front for the longitudinal (all-ages-stacked) model:
    DeepPySR from the winning variant's hall_of_fame, PySR likewise, each
    fold-averaged (mean loss per complexity across all 5 folds' search
    logs) -- the same convention used for every other dataset's Pareto
    front. Sized as a compact inset for Fig. 3c in the main text.
    (Training-time complexity/loss log -- optimizer dynamics, unaffected by the leak.)"""
    long_best = best_df[best_df['type'] == 'longitudinal'] if 'type' in best_df.columns else pd.DataFrame()
    if long_best.empty:
        print("No longitudinal data for pareto front RMSE plot")
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    deep_rows = long_best[long_best['display_model'] == 'Best DeepPySR']
    if not deep_rows.empty:
        model_dir = deep_rows.iloc[0]['source_path']
        hof_data = _load_hof_data(model_dir)
        if not hof_data.empty:
            pf = _pareto_front_steps(hof_data['complexity'].tolist(), hof_data['rmse'].tolist())
            if pf:
                px, py = zip(*pf)
                ax.step(px, py, where='post', color='#2166ac', linewidth=1.3, zorder=4)
                ax.scatter(px, py, c='#2166ac', s=16, zorder=5, marker='D', label='DeepPySR')

    pysr_rows = long_best[long_best['display_model'] == 'PySR']
    if not pysr_rows.empty:
        pysr_model_dir = pysr_rows.iloc[0]['source_path']
        hof_pysr = _load_hof_data(pysr_model_dir)
        if not hof_pysr.empty:
            pf_pysr = _pareto_front_steps(hof_pysr['complexity'].tolist(), hof_pysr['rmse'].tolist())
            if pf_pysr:
                px, py = zip(*pf_pysr)
                ax.step(px, py, where='post', color='#cc4400', linewidth=1.3, zorder=4)
                ax.scatter(px, py, c='#cc4400', s=16, zorder=5, marker='D', label='PySR')

    ax.set_xlabel('Complexity', fontsize=8)
    ax.set_ylabel('RMSE', fontsize=8)
    ax.set_title('Pareto Front: Complexity vs. RMSE', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'pareto_front_rmse.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pareto front RMSE plot saved to {out}")


if __name__ == "__main__":
    all_df, best_df = process_results()

    save_best_models(best_df)
    plot_results(best_df)
    compute_se_and_wilcoxon(best_df)
    aggregate_feature_importance()
    plot_vps_vpr_ablation(all_df)
    plot_pareto_ablation()
    plot_pareto_front_rmse(best_df)
