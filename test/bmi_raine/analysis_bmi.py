"""Regression + clinical-bin classification analysis for BMI-prediction
feature-set variants (PGS, to8, PGSto8, recent) under results_bmi/.

Single target (bmi_raine = child BMI, kg/m2), all_features only -- no
feature-selection/top100 variant exists for this dataset (see
test_baselines_pysr_bmi_*.py/test_deeppysr_bmi_*.py, which never produce one).

Unlike analysis_lipids.py/analysis_bp.py, DeepPySR/PySR r2/rmse/mae/pearson_r
are NOT computed by re-evaluating a formula in-sample on the full dataset --
they use genuine leak-free out-of-fold predictions pooled across all 5 CV
folds (analysis_v1_utils.get_best_formula_from_raw/get_oof_predictions),
directly comparable to how baseline models are scored from their own
predictions.csv. "Interpretable DeepPySR" is its own leak-free CV pass with
each fold's candidate pool restricted to complexity < INTERP_MAX_COMPLEXITY
*before* ranking by held-out fit (not a post-hoc filter of the unconstrained
result). This whole leak-free-CV architecture is ported from
diab_raine/analysis_insulin.py.

Unlike insulin (pure regression, no established clinical categories), BMI
has well-known adult clinical categories, so this analysis ALSO scores every
row by **F1 (macro)** on 3 clinical bins (see BMI_BIN_EDGES) and selects each
age's "best" model by max F1 rather than max r2 -- mirroring bp_raine's/
lipids_raine's pattern (adapted into this file's single-target row/column
structure).

Produces, per variant, under results_bmi/results_bmi_<variant>/:
  bmi_aggregated_results.csv   -- every (age, model) row found on disk.
  bmi_best_models_metrics.csv  -- one row per (age, display_model): the
                                   CV-pooled winner (max F1) within that
                                   model family (Best DeepPySR, Interpretable
                                   DeepPySR, PySR, KAN, and each baseline).
  bmi_metrics_vs_age.png       -- r2/pearson_r/f1_macro vs age, lines=
                                   display_model.
  interpretable_deeppysr_formulas.csv
  formula_predictions/age_<age>.csv + age_<age>_scatter.png
  feature_importance_aggregated.csv + feature_importance_by_model.png
  age_<age>_bmi_raine/bmi_sensitivity.csv + .png
                                    -- feature-importance heatmap combining
                                       ElasticNet/ExtraTrees/RandomForest/
                                       XGBoost's own feature_importance.csv
                                       with a subprocess-isolated MLP SHAP
                                       refit (see
                                       _bmi_mlp_shap_importance_subprocess
                                       for why it's a subprocess).
  age_<age>_bmi_raine/bmi_deeppysr_sensitivity.csv + .png
                                    -- Best vs Interpretable DeepPySR
                                       permutation sensitivity
                                       (common.formula_sensitivity), reusing
                                       the exact formula string/r2/pearson_r/
                                       f1_macro already selected into
                                       bmi_best_models_metrics.csv.
Combined (across all 4 variants), under results_bmi/:
  bmi_deeppysr_combined_metrics.csv
  bmi_deeppysr_metrics_vs_age_combined.png
  bmi_deeppysr_sensitivity_overview.csv + .png
                                    -- which variables drive DeepPySR's
                                       interpretable formulas, and does that
                                       change by data source (see
                                       aggregate_permutation_sensitivity).
"""
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from sklearn.metrics import f1_score

SCATTER_AXIS_LIMIT = 50

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

RESULTS_BASE_DIR = os.path.join(current_dir, "results_bmi")

from data_utils import (
    load_data_PGS_only, load_data_keepto8, load_data_PGSto8, load_data_recent,
    _BMI_AGES, TARGET,
)
from analysis_v1_utils import calculate_metrics, get_best_formula_from_raw, get_oof_predictions

sys.path.append(os.path.join(current_dir, "..", "..", "deeppysr_paper", "codes"))
from common import formula_sensitivity

INTERP_MAX_COMPLEXITY = 30

AGES = _BMI_AGES

VARIANTS = {
    'PGS':    (load_data_PGS_only,  'results_bmi_PGS'),
    'to8':    (load_data_keepto8,   'results_bmi_to8'),
    'PGSto8': (load_data_PGSto8,    'results_bmi_PGSto8'),
    'recent': (load_data_recent,    'results_bmi_recent'),
}

# ── Clinical binning for classification-style metrics ────────────────────────
# Standard adult WHO/CDC BMI categories (kg/m2): Underweight <18.5,
# Normal 18.5-24.9, Overweight 25-29.9, Obese >=30. No pediatric
# age/sex-specific percentile norms are applied even at the younger ages in
# this cohort -- same simplification analysis_bp.py/analysis_lipids.py make
# for their own adolescent ages -- and it matters more here than for those:
# checking real population counts at each target age confirmed the
# "Underweight" tier is dominated by developmentally-normal children at
# young ages (752/1284 = 58.6% "underweight" at age 10, 298/1277 = 23.3% at
# age 14) simply because normal childhood BMI sits well below the adult
# 18.5 cutoff -- not because these children are clinically underweight. By
# age 22/28 this tier shrinks to ~2% as the cohort approaches adulthood.
# Underweight and Normal are therefore collapsed into a single "Normal or
# below" tier (mirroring how lipids_raine/bp_raine each collapse their own
# least clinically distinct tier down to 3), leaving Overweight/Obese --
# the two categories that stay clinically meaningful across the whole age
# range -- as their own tiers. This F1_macro is a rough clinical-relevance
# signal for model comparison, not a diagnostic label.
BMI_BIN_EDGES = (25.0, 30.0)
BMI_CLASS_LABELS = ['Normal or below', 'Overweight', 'Obese']


def _bin_bmi(values):
    """Values (kg/m2) -> class index 0/1/2 per BMI_BIN_EDGES."""
    return np.digitize(np.asarray(values, dtype=float), list(BMI_BIN_EDGES))


def _bmi_f1_macro(y_true, y_pred):
    """Macro-F1 of the 3-class clinical bins, applying the same cutoffs to
    both y_true and y_pred (so a regression model is scored on whether its
    continuous prediction lands in the right clinical category, not on a
    separately-fit classifier)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not valid.any():
        return np.nan
    y_true_bin = _bin_bmi(y_true[valid])
    y_pred_bin = _bin_bmi(y_pred[valid])
    return float(f1_score(y_true_bin, y_pred_bin, average='macro', labels=[0, 1, 2], zero_division=0))


def _load_age(load_fn, age):
    ids, X, y = load_fn(age)
    return ids, X, y.rename(TARGET) if hasattr(y, 'rename') else y


def _process_baselines(age_path, age, all_data):
    baselines_dir = os.path.join(age_path, "baselines")
    if not os.path.exists(baselines_dir):
        return
    for model_name in os.listdir(baselines_dir):
        model_dir = os.path.join(baselines_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        pred_file = os.path.join(model_dir, "all_features", "predictions.csv")
        if not os.path.exists(pred_file):
            continue
        df_pred = pd.read_csv(pred_file)
        tag = f"{model_name}_all_features"
        r2, rmse, mae, pearson_r = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
        f1_macro = _bmi_f1_macro(df_pred['y_true'], df_pred['y_pred'])
        all_data.append([age, tag, r2, rmse, mae, pearson_r, f1_macro, np.nan, "", "", "", np.nan])


def _process_sr_dir(age_path, age, subdir, model_type, load_fn, all_data):
    """variant is the SR config directory name (e.g.
    'fullsr_nit500_..._grid'); the actual relationships_fold*.csv /
    formulas_fold*.csv / predictions.csv files sit one level deeper, under
    that config's own all_features/ subfolder -- same convention
    test_*_bmi_*.py's baselines already nest under (see
    _process_baselines), matching diab_raine/analysis_insulin.py's identical
    fix (a real bug there before it: SR configs weren't being found at all
    because this descent into all_features/ was missing)."""
    sr_dir = os.path.join(age_path, subdir)
    if not os.path.exists(sr_dir):
        return
    for variant in os.listdir(sr_dir):
        v_path = os.path.join(sr_dir, variant, "all_features")
        if not os.path.isdir(v_path):
            continue
        _, X_age, y_age = _load_age(load_fn, age)
        formula, complexity, metrics = get_best_formula_from_raw(v_path, X_age, y_age, model_type=model_type)
        r2, rmse, mae, pearson_r = metrics
        y_pred = get_oof_predictions(v_path, X_age, y_age, model_type=model_type)
        f1_macro = _bmi_f1_macro(y_age, y_pred) if y_pred is not None else np.nan
        if not formula:
            pred_file = os.path.join(v_path, "predictions.csv")
            if os.path.exists(pred_file):
                df_pred = pd.read_csv(pred_file)
                r2, rmse, mae, pearson_r = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                f1_macro = _bmi_f1_macro(df_pred['y_true'], df_pred['y_pred'])
        all_data.append([age, variant, r2, rmse, mae, pearson_r, f1_macro, complexity, formula, v_path, model_type, np.nan])

        # Separate complexity-constrained candidate for "Interpretable
        # DeepPySR": selection must be constrained to complexity < 30 per
        # fold *before* ranking by r2 (see get_best_formula_from_raw), so
        # this needs its own CV pass rather than filtering the row above.
        if model_type == 'deeppysr':
            formula_i, complexity_i, metrics_i = get_best_formula_from_raw(
                v_path, X_age, y_age, model_type=model_type, max_complexity=INTERP_MAX_COMPLEXITY)
            if formula_i:
                r2_i, rmse_i, mae_i, pearson_r_i = metrics_i
                y_pred_i = get_oof_predictions(v_path, X_age, y_age, model_type=model_type,
                                                max_complexity=INTERP_MAX_COMPLEXITY)
                f1_i = _bmi_f1_macro(y_age, y_pred_i) if y_pred_i is not None else np.nan
                all_data.append([age, f"{variant}__interp{INTERP_MAX_COMPLEXITY}", r2_i, rmse_i, mae_i, pearson_r_i,
                                 f1_i, complexity_i, formula_i, v_path, model_type, INTERP_MAX_COMPLEXITY])


def process_results(load_fn, results_dir):
    all_data = []
    for age in AGES:
        age_path = os.path.join(results_dir, f"age_{age}_{TARGET}")
        if not os.path.exists(age_path):
            continue
        _process_baselines(age_path, age, all_data)
        _process_sr_dir(age_path, age, "deeppysr", "deeppysr", load_fn, all_data)
        _process_sr_dir(age_path, age, "pysr", "pysr", load_fn, all_data)

    df = pd.DataFrame(all_data, columns=['age', 'model', 'r2', 'rmse', 'mae', 'pearson_r', 'f1_macro', 'complexity',
                                          'formula', 'source_path', 'formula_model_type', 'max_complexity'])
    df['r2'] = df['r2'].clip(lower=0)
    return df


def _select_best_models(df):
    """Return plot_df and interpretable_formulas, age-specific only. Each
    age's winner within a model family is the max-F1_macro (clinical-bin
    classification) row, not max-r2 -- mirroring bp_raine's/lipids_raine's
    selection rule."""
    df = df.copy()
    df['r2'] = df['r2'].clip(lower=0)
    ages = sorted(df['age'].unique())
    selected_data = []
    interpretable_formulas = []

    for age in ages:
        age_df = df[df['age'] == age]

        is_deeppysr = age_df['model'].str.contains('fullsr|stdsr|srprn|srpsm', na=False)
        # "Best DeepPySR" must dominate "Interpretable DeepPySR" by
        # definition (interpretable is a complexity-restricted special
        # case), so its candidate pool includes both the unconstrained rows
        # AND the complexity-constrained rows -- not just the unconstrained
        # ones.
        deeppysr_df = age_df[is_deeppysr]
        if not deeppysr_df.empty:
            best = deeppysr_df.loc[deeppysr_df['f1_macro'].idxmax()].copy()
            best['display_model'] = 'Best DeepPySR'
            selected_data.append(best)
            interp = age_df[is_deeppysr & (age_df['max_complexity'] == INTERP_MAX_COMPLEXITY)]
            if not interp.empty:
                bi = interp.loc[interp['f1_macro'].idxmax()].copy()
                bi['display_model'] = 'Interpretable DeepPySR'
                selected_data.append(bi)
                interpretable_formulas.append({'age': age, 'model': bi['model'],
                                               'formula': bi['formula'], 'r2': bi['r2'],
                                               'f1_macro': bi['f1_macro'], 'complexity': bi['complexity']})

        pysr_df = age_df[age_df['model'].str.contains('pysr', na=False)]
        if not pysr_df.empty:
            best_pysr = pysr_df.loc[pysr_df['f1_macro'].idxmax()].copy()
            best_pysr['display_model'] = 'PySR'
            selected_data.append(best_pysr)

        for m in ['KAN']:
            m_df = age_df[age_df['model'].str.startswith(f"{m}_")]
            if not m_df.empty:
                row = m_df.loc[m_df['f1_macro'].idxmax()].copy()
                row['display_model'] = m
                selected_data.append(row)

        for b in ['ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']:
            b_df = age_df[age_df['model'].str.startswith(f"{b}_")]
            if not b_df.empty:
                row = b_df.loc[b_df['f1_macro'].idxmax()].copy()
                row['display_model'] = b
                selected_data.append(row)

    return pd.DataFrame(selected_data).reset_index(drop=True), interpretable_formulas


def plot_results(df, results_dir, variant_name):
    plot_df, interpretable_formulas = _select_best_models(df)
    if plot_df.empty:
        print(f"No data to plot for variant={variant_name}.")
        return plot_df

    plot_csv_path = os.path.join(results_dir, 'bmi_best_models_metrics.csv')
    plot_df.to_csv(plot_csv_path, index=False)
    print(f"Best models saved to {plot_csv_path}")

    metrics = ['r2', 'pearson_r', 'f1_macro']
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    plt.rcParams.update({'font.size': 14})
    models = sorted(plot_df['display_model'].unique())
    palette = sns.color_palette("tab10", n_colors=len(models))
    model_colors = dict(zip(models, palette))

    metric_labels = {'r2': 'R2', 'pearson_r': 'Pearson r', 'f1_macro': 'F1 (macro, clinical bins)'}
    for col, metric in enumerate(metrics):
        ax = axes[col]
        label = metric_labels.get(metric, metric.upper())
        sns.lineplot(data=plot_df, x='age', y=metric, hue='display_model', ax=ax,
                     linestyle='--', linewidth=3.0, palette=model_colors,
                     marker='o', markersize=8)
        ax.set_title(f'{label} vs Age', fontsize=20, fontweight='bold', pad=15)
        ax.set_ylabel(label, fontsize=16)
        ax.set_xlabel('Age', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        if ax.get_legend():
            ax.get_legend().remove()

    legend_elements = [Line2D([0], [0], color=model_colors[m], lw=3, label=m) for m in models]
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.91, 0.5),
               fontsize=14, frameon=True, title='Models', title_fontsize=16, handlelength=4.0)
    plt.suptitle(f'BMI Prediction Performance ({variant_name}): Best Models Comparison',
                 fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plot_path = os.path.join(results_dir, 'bmi_metrics_vs_age.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

    if interpretable_formulas:
        print(f"\n--- Interpretable DeepPySR Formulas for {variant_name} (Complexity < {INTERP_MAX_COMPLEXITY}) ---")
        interp_df = pd.DataFrame(interpretable_formulas)
        print(interp_df.to_string(index=False))
        interp_df.to_csv(os.path.join(results_dir, 'interpretable_deeppysr_formulas.csv'), index=False)

    return plot_df


MODEL_ORDER = ['Best DeepPySR', 'Interpretable DeepPySR', 'PySR', 'KAN',
               'ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']


def _predictions_from_disk(model_tag, age_path):
    """Look up (y_true, y_pred) for a baseline/KAN row (no formula) from its
    original predictions.csv under age_path/baselines/<model>/all_features/."""
    suffix = "_all_features"
    if not model_tag.endswith(suffix):
        return None, None
    base_name = model_tag[: -len(suffix)]
    pred_file = os.path.join(age_path, "baselines", base_name, "all_features", "predictions.csv")
    if not os.path.exists(pred_file):
        return None, None
    df_pred = pd.read_csv(pred_file)
    return df_pred['y_true'].values, df_pred['y_pred'].values


def _get_model_predictions(row, age_path, X_full, y_full):
    """Return (y_true, y_pred) arrays for one selected model row.

    Formula-based models (DeepPySR, PySR) use the same pooled leak-free
    out-of-fold predictions the reported R2/F1 was computed from
    (get_oof_predictions on row['source_path']) -- NOT a re-evaluation of a
    single formula on the full in-sample dataset. Everything else falls
    back to the CV predictions.csv saved during training.
    """
    formula = row.get('formula')
    source_path = row.get('source_path')
    if isinstance(formula, str) and formula.strip() and isinstance(source_path, str) and source_path.strip():
        formula_model_type = row.get('formula_model_type') or 'deeppysr'
        prefix = 'relationships_fold'
        max_complexity = row.get('max_complexity')
        max_complexity = None if pd.isna(max_complexity) else max_complexity
        y_pred = get_oof_predictions(source_path, X_full, y_full, prefix=prefix, model_type=formula_model_type,
                                      max_complexity=max_complexity)
        if y_pred is None:
            return None, None
        # Should be fully covered when all 5 fold files are present; guard
        # against a missing fold leaving a gap rather than crashing the plot.
        y_pred = np.nan_to_num(y_pred, nan=0.0)
        return None, y_pred
    return _predictions_from_disk(row['model'], age_path)


def save_predictions_and_scatter(plot_df, load_fn, results_dir):
    """Save per-age formula predictions (DeepPySR/PySR) with the raw
    features, and a true-vs-predicted scatter plot per age with one subplot
    per selected model (DeepPySR, PySR, KAN, and all baselines)."""
    if plot_df.empty:
        return

    pred_dir = os.path.join(results_dir, "formula_predictions")
    os.makedirs(pred_dir, exist_ok=True)
    palette = sns.color_palette("tab10", n_colors=len(MODEL_ORDER))
    model_colors = dict(zip(MODEL_ORDER, palette))

    for age in sorted(plot_df['age'].unique()):
        age_df = plot_df[plot_df['age'] == age]
        age_path = os.path.join(results_dir, f"age_{age}_{TARGET}")
        ids, X_full, y_full = _load_age(load_fn, age)

        pred_table = pd.concat([
            ids.reset_index(drop=True),
            X_full.reset_index(drop=True),
            pd.DataFrame({'y_true': y_full.reset_index(drop=True)}),
        ], axis=1)

        present_models = [m for m in MODEL_ORDER if m in age_df['display_model'].values]
        if not present_models:
            continue

        ncols = 5
        nrows = int(np.ceil(len(present_models) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
        axes = axes.reshape(-1)

        for i, m in enumerate(present_models):
            row = age_df[age_df['display_model'] == m].iloc[0]
            y_true_arr, y_pred_arr = _get_model_predictions(row, age_path, X_full, y_full)
            ax = axes[i]
            if y_pred_arr is None:
                ax.set_visible(False)
                continue
            if y_true_arr is None:
                y_true_arr = y_full.values
                pred_table[f'y_pred_{m.replace(" ", "_")}'] = y_pred_arr

            lo = min(y_true_arr.min(), y_pred_arr.min())
            hi = max(y_true_arr.max(), y_pred_arr.max())
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
            ax.scatter(y_true_arr, y_pred_arr, alpha=0.5, color=model_colors[m], s=20)
            ax.set_title(f"{m} (R2={row['r2']:.2f}, F1={row['f1_macro']:.2f})", fontsize=13, fontweight='bold')
            ax.set_xlabel('True BMI', fontsize=11)
            ax.set_ylabel('Predicted BMI', fontsize=11)
            ax.set_xlim(0, SCATTER_AXIS_LIMIT)
            ax.set_ylim(0, SCATTER_AXIS_LIMIT)

            # Sane bound so a handful of formula-overflow predictions
            # (e.g. exp() blowing up to ~1e300) don't collapse the inset
            # scale; genuine outliers in y_true are still fully shown.
            # Any prediction beyond the bound is pinned to the edge
            # (visible as a dot at the border) rather than hidden.
            sane_pred = y_pred_arr[np.abs(y_pred_arr) < 1e6]
            pred_cap = sane_pred.max() if sane_pred.size else y_pred_arr.max()
            raw_lo = min(lo, 0)
            raw_hi = max(y_true_arr.max(), pred_cap, SCATTER_AXIS_LIMIT)
            pad = (raw_hi - raw_lo) * 0.08 if raw_hi > raw_lo else 1.0
            lo_display = raw_lo - pad
            hi_display = raw_hi + pad
            y_pred_display = np.clip(y_pred_arr, lo_display, hi_display)

            # Points already visible in the main 0-50 panel are shown
            # faint/small; points that fall outside it (the reason the
            # inset exists) are emphasized so they stand out.
            in_main = (y_true_arr <= SCATTER_AXIS_LIMIT) & (y_pred_display <= SCATTER_AXIS_LIMIT)
            extreme = ~in_main

            axins = inset_axes(ax, width="42%", height="42%", loc='upper right', borderpad=1.2)
            axins.plot([lo_display, hi_display], [lo_display, hi_display], 'k--', lw=1)
            axins.scatter(y_true_arr[in_main], y_pred_display[in_main],
                          alpha=0.25, color=model_colors[m], s=6, zorder=2)
            axins.scatter(y_true_arr[extreme], y_pred_display[extreme],
                          alpha=0.9, color=model_colors[m], s=90, marker='*',
                          edgecolors='black', linewidths=0.6, zorder=3)
            axins.set_xlim(lo_display, hi_display)
            axins.set_ylim(lo_display, hi_display)
            axins.tick_params(axis='both', which='major', labelsize=7)
            axins.set_title(f'full range ({extreme.sum()} extreme)', fontsize=8)

        for j in range(len(present_models), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f'Age {age}: True vs Predicted (all models)', fontsize=20, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        scatter_path = os.path.join(pred_dir, f"age_{age}_scatter.png")
        plt.savefig(scatter_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Scatter plot saved to {scatter_path}")

        pred_path = os.path.join(pred_dir, f"age_{age}.csv")
        pred_table.to_csv(pred_path, index=False)
        print(f"Formula predictions saved to {pred_path}")


def aggregate_feature_importance(results_dir):
    importance_data = []
    for age in AGES:
        baselines_dir = os.path.join(results_dir, f"age_{age}_{TARGET}", "baselines")
        if not os.path.exists(baselines_dir):
            continue
        for m in os.listdir(baselines_dir):
            if m not in ['ElasticNet', 'ExtraTrees', 'RandomForest', 'XGBoost']:
                continue
            imp_file = os.path.join(baselines_dir, m, "all_features", "feature_importance.csv")
            if not os.path.exists(imp_file):
                continue
            df_imp = pd.read_csv(imp_file)
            if 'feature' not in df_imp.columns or 'importance' not in df_imp.columns:
                continue
            total = df_imp['importance'].sum()
            for _, row in df_imp.iterrows():
                importance_data.append({
                    'age': age, 'model': m,
                    'variable': row['feature'],
                    'weight': (row['importance'] / total * 100) if total > 0 else 0
                })

    imp_df = pd.DataFrame(importance_data)
    imp_df.to_csv(os.path.join(results_dir, "feature_importance_aggregated.csv"), index=False)
    print(f"Feature importance aggregated to {results_dir}/feature_importance_aggregated.csv")

    if imp_df.empty:
        return
    agg_imp = imp_df.groupby(['model', 'variable'])['weight'].mean().reset_index()
    top_features = agg_imp.groupby('variable')['weight'].mean().sort_values(ascending=False).head(15).index
    plot_df = agg_imp[agg_imp['variable'].isin(top_features)].copy()
    plot_df['variable'] = pd.Categorical(plot_df['variable'], categories=top_features, ordered=True)
    plt.figure(figsize=(14, 10))
    sns.barplot(data=plot_df, x='weight', y='variable', hue='model', palette="bright")
    plt.title('Top 15 Feature Importance across Models', fontsize=22, fontweight='bold', pad=20)
    plt.xlabel('Average Percentage Importance (%)', fontsize=18)
    plt.ylabel('Feature', fontsize=18)
    plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "feature_importance_by_model.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {plot_path}")


# ── Per-(variant, age) sensitivity: conventional + MLP-SHAP heatmap, and
#    DeepPySR Best-vs-Interpretable permutation sensitivity ─────────────────

FEATURE_IMPORTANCE_MODELS = ['ElasticNet', 'ExtraTrees', 'RandomForest', 'XGBoost']
HEATMAP_MODEL_ORDER = ['RandomForest', 'ExtraTrees', 'XGBoost', 'ElasticNet', 'MLP (SHAP)']

_SENS_INK = "#0b0b0b"
_SENS_BLUE_STEPS = ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                    "#256abf", "#184f95", "#104281", "#0d366b"]
_SENS_CMAP = LinearSegmentedColormap.from_list("bmi_seq_blue", _SENS_BLUE_STEPS, N=256)

DEEPPYSR_VARIANT_COLORS = {'Best DeepPySR': '#3987e5', 'Interpretable DeepPySR': '#e58939'}


def _bmi_mlp_shap_importance(X, y, task='regression', use_smote=False, epochs=80, batch_size=64,
                              max_train_rows=None, n_background=50, n_explain=200,
                              random_state=42):
    """Same as deeppysr_paper/codes/common.py::mlp_shap_importance, except the
    SHAP Permutation explainer is given an explicit max_evals sized to the
    feature count (common.py's version leaves it at SHAP's internal default
    of 500, which is too low once a variant has more than ~250 features)."""
    import shap
    from model_utils import MLPRegressorWrapper, MLPClassifierWrapper

    feature_names = list(X.columns)
    rng = np.random.RandomState(random_state)

    Xv = X.values.astype(float)
    yv = y.values.astype(float) if hasattr(y, "values") else np.asarray(y, dtype=float)

    if max_train_rows and len(Xv) > max_train_rows:
        idx = rng.choice(len(Xv), max_train_rows, replace=False)
        Xv, yv = Xv[idx], yv[idx]

    if use_smote:
        from imblearn.over_sampling import SMOTE
        Xv, yv = SMOTE(random_state=random_state).fit_resample(Xv, yv)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xv)
    Xs = scaler.transform(Xv)

    if task == "classification":
        model = MLPClassifierWrapper(hidden_layer_sizes=(256, 128, 64), dropout=0.1,
                                      activation="leaky_relu", random_state=random_state,
                                      epochs=epochs, batch_size=batch_size)
    else:
        model = MLPRegressorWrapper(hidden_layer_sizes=(256, 128, 64), dropout=0.1,
                                     activation="leaky_relu", random_state=random_state,
                                     epochs=epochs, batch_size=batch_size)
    model.fit(Xs, yv)

    if task == "classification":
        def predict_fn(Xraw):
            return model.predict_proba(scaler.transform(Xraw))
    else:
        def predict_fn(Xraw):
            return model.predict(scaler.transform(Xraw))

    n_bg = min(n_background, len(Xv))
    background = shap.sample(Xv, n_bg, random_state=random_state)
    n_ex = min(n_explain, len(Xv))
    explain_idx = rng.choice(len(Xv), n_ex, replace=False)
    Xexplain = Xv[explain_idx]

    max_evals = max(500, 2 * len(feature_names) + 1)
    explainer = shap.Explainer(predict_fn, background, feature_names=feature_names)
    sv = explainer(Xexplain, max_evals=max_evals)
    vals = np.asarray(sv.values)
    if vals.ndim == 3:  # (n_samples, n_features, n_classes)
        vals = np.abs(vals).mean(axis=2)
    importance = np.abs(vals).mean(axis=0)
    total = importance.sum()
    pct = 100.0 * importance / total if total > 0 else importance
    return dict(zip(feature_names, pct))


def _bmi_mlp_shap_importance_subprocess(variant_name, age, random_state=42, n_explain=60, timeout=1800):
    """Runs _bmi_mlp_shap_importance in a fresh subprocess
    (_mlp_shap_worker.py) instead of in-process. Importing model_utils
    (which pulls in torch AND, via its own module-level
    `from pysr import PySRRegressor`, juliacall) reliably segfaults once
    this long-running analysis process has already done enough prior
    matplotlib/numpy work -- the exact same Julia-embedding fragility fixed
    the same way in test/bp_raine/analysis_bp.py's
    _bp_mlp_shap_importance_subprocess (see that docstring for the isolated
    repro that root-caused it). Isolating the SHAP call into its own
    subprocess sidesteps it entirely by always running under the conditions
    where it works."""
    worker = os.path.join(current_dir, "_mlp_shap_worker.py")
    fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        result = subprocess.run(
            [sys.executable, worker,
             "--variant", variant_name, "--age", str(age),
             "--random_state", str(random_state), "--n_explain", str(n_explain),
             "--out", out_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"worker exited {result.returncode}: {result.stderr[-2000:]}")
        with open(out_path) as f:
            return json.load(f)
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def _bmi_conventional_importance(age_path, model_name):
    """{feature: pct} read from that model's own all_features
    feature_importance.csv, or {} if absent."""
    imp_file = os.path.join(age_path, "baselines", model_name, "all_features", "feature_importance.csv")
    if not os.path.exists(imp_file):
        return {}
    df_imp = pd.read_csv(imp_file)
    if 'feature' not in df_imp.columns or 'importance' not in df_imp.columns:
        return {}
    total = df_imp['importance'].sum()
    if total <= 0:
        return {}
    return dict(zip(df_imp['feature'], 100.0 * df_imp['importance'] / total))


def _plot_sensitivity_heatmap(table, out_path, title, top_n=10,
                               cbar_label="% importance (within model)"):
    """Heatmap of top_n features (by max importance across models) x models,
    styled after deeppysr_paper/codes/feature_importance_comparison.py's
    plot_heatmap. `table`'s columns don't have to be models -- reused by
    aggregate_permutation_sensitivity with variant columns instead."""
    top_features = table.max(axis=1).sort_values(ascending=False).head(top_n).index
    plot_df = table.loc[top_features]
    arr = plot_df.values.astype(float)
    vmax = max(arr.max(), 1e-6)

    fig, ax = plt.subplots(figsize=(1.6 * len(plot_df.columns) + 3.0, 0.5 * len(plot_df) + 2.2))
    im = ax.imshow(arr, cmap=_SENS_CMAP, vmin=0, vmax=vmax, aspect="auto")

    for r in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            v = arr[r, col]
            frac = v / vmax
            color = "white" if frac > 0.62 else _SENS_INK
            label = f"{v:.1f}" if v >= 0.1 else ("" if v == 0 else "<0.1")
            ax.text(col, r, label, ha="center", va="center", fontsize=10, color=color)

    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, rotation=40, ha="right", fontsize=11)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df.index, fontsize=11)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(plot_df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(plot_df), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label(cbar_label, fontsize=11)

    ax.set_title(title, fontsize=13, fontweight='bold', loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _plot_deeppysr_sensitivity(age_df, X_full, out_dir, age, variant_name, n_repeats=15, random_state=42):
    """Combined Best-vs-Interpretable DeepPySR permutation sensitivity plot
    for one (variant, age) combo: grouped horizontal bars over the union of
    every variable either formula references, annotated with each formula's
    own complexity and r2/pearson_r/f1_macro -- reusing the exact leak-free
    (CV-pooled OOF) values already selected into bmi_best_models_metrics.csv,
    not a fresh re-evaluation.

    Saves under out_dir (the same age_<age>_bmi_raine directory as
    bmi_sensitivity.csv/.png):
      bmi_deeppysr_sensitivity.csv
      bmi_deeppysr_sensitivity.png
    """
    variants = []
    for label in ['Best DeepPySR', 'Interpretable DeepPySR']:
        row = age_df[age_df['display_model'] == label]
        if row.empty:
            continue
        row = row.iloc[0]
        formula = row.get('formula')
        if not formula or pd.isna(formula):
            continue
        formula = str(formula)
        try:
            sens = formula_sensitivity(formula, X_full, n_repeats=n_repeats, random_state=random_state)
            sens = {k: v for k, v in sens.items() if v > 0}
        except Exception as e:
            print(f"    DeepPySR {label} sensitivity failed for {variant_name}/age_{age}: {e}")
            continue
        if not sens:
            continue
        variants.append({
            'label': label, 'formula': formula, 'complexity': row['complexity'],
            'r2': row['r2'], 'pearson_r': row['pearson_r'], 'f1_macro': row['f1_macro'], 'sens': sens,
        })

    if not variants:
        return None

    all_vars = sorted(set().union(*[v['sens'].keys() for v in variants]))
    order = sorted(all_vars, key=lambda vv: max(v['sens'].get(vv, 0.0) for v in variants))

    n_vars, n_variants = len(order), len(variants)
    bar_h = 0.8 / n_variants
    y_pos = np.arange(n_vars)

    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * n_vars)))
    for i, v in enumerate(variants):
        offset = (i - (n_variants - 1) / 2) * bar_h
        values = [v['sens'].get(vv, 0.0) for vv in order]
        ax.barh(y_pos + offset, values, height=bar_h, label=v['label'],
                 color=DEEPPYSR_VARIANT_COLORS.get(v['label'], 'gray'))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(order, fontsize=11)
    ax.set_xlabel('Sensitivity (%)', fontsize=12)
    ax.set_title(f'BMI ({variant_name}, age {age}): DeepPySR sensitivity — Best vs Interpretable',
                 fontsize=13, fontweight='bold')

    legend_labels = [
        f"{v['label']} (complexity={v['complexity']:.0f}, R2={v['r2']:.2f}, "
        f"r={v['pearson_r']:.2f}, F1={v['f1_macro']:.2f})"
        for v in variants
    ]
    handles = [plt.Rectangle((0, 0), 1, 1, color=DEEPPYSR_VARIANT_COLORS.get(v['label'], 'gray'))
               for v in variants]
    ax.legend(handles, legend_labels, loc='lower right', fontsize=10, frameon=True)

    plt.tight_layout()
    png_path = os.path.join(out_dir, 'bmi_deeppysr_sensitivity.png')
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    csv_rows = [
        {'variant': variant_name, 'age': age, 'formula_type': v['label'],
         'variable': var, 'sensitivity_pct': pct, 'complexity': v['complexity'],
         'r2': v['r2'], 'pearson_r': v['pearson_r'], 'f1_macro': v['f1_macro']}
        for v in variants for var, pct in v['sens'].items()
    ]
    pd.DataFrame(csv_rows).to_csv(os.path.join(out_dir, 'bmi_deeppysr_sensitivity.csv'), index=False)

    return png_path


def compute_bmi_sensitivity(plot_df, load_fn, results_dir, variant_name, n_repeats=15, random_state=42, top_n=10):
    """Per-age feature-importance comparison: ElasticNet/ExtraTrees/
    RandomForest/XGBoost's own feature_importance.csv, plus a subprocess-
    isolated MLP SHAP refit. Also calls _plot_deeppysr_sensitivity for every
    age with a Best/Interpretable DeepPySR row. All vectors normalised to
    sum to 100 within their own model."""
    if plot_df.empty:
        print("  No data to compute sensitivity for.")
        return

    for age in sorted(plot_df['age'].unique()):
        age_df = plot_df[plot_df['age'] == age]
        age_path = os.path.join(results_dir, f"age_{age}_{TARGET}")
        if not os.path.exists(age_path):
            continue

        importances = {}
        for model_name in FEATURE_IMPORTANCE_MODELS:
            imp = _bmi_conventional_importance(age_path, model_name)
            if imp:
                importances[model_name] = imp

        print(f"  Loading {variant_name}/age_{age} for sensitivity analysis...")
        _, X_full, y_full = _load_age(load_fn, age)

        try:
            mlp_imp = _bmi_mlp_shap_importance_subprocess(variant_name, age,
                                                           random_state=random_state, n_explain=60)
            if mlp_imp:
                importances['MLP (SHAP)'] = mlp_imp
        except Exception as e:
            print(f"    MLP SHAP failed for {variant_name}/age_{age}: {e}")

        _plot_deeppysr_sensitivity(age_df, X_full, age_path, age, variant_name,
                                    n_repeats=n_repeats, random_state=random_state)

        if not importances:
            continue

        cols = [m for m in HEATMAP_MODEL_ORDER if m in importances]
        all_features = sorted(set().union(*[d.keys() for d in importances.values()]))
        table = pd.DataFrame({m: {f: importances[m].get(f, 0.0) for f in all_features} for m in cols})

        long_df = table.reset_index().melt(id_vars='index', var_name='model', value_name='pct')
        long_df = long_df.rename(columns={'index': 'variable'})
        long_df = long_df[long_df['pct'] > 0].copy()
        long_df.insert(0, 'variant', variant_name)
        long_df.insert(0, 'age', age)
        csv_path = os.path.join(age_path, "bmi_sensitivity.csv")
        long_df.to_csv(csv_path, index=False)

        title = f'BMI ({variant_name}, age {age}): feature importance across models'
        png_path = os.path.join(age_path, "bmi_sensitivity.png")
        _plot_sensitivity_heatmap(table, png_path, title, top_n=top_n)

        print(f"  Sensitivity saved to {age_path}")


def run_variant(name):
    load_fn, results_subdir = VARIANTS[name]
    results_dir = os.path.join(RESULTS_BASE_DIR, results_subdir)

    out_csv = os.path.join(results_dir, "bmi_aggregated_results.csv")
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv)
        print(f"Results loaded from {out_csv}")
    else:
        df = process_results(load_fn, results_dir)
        df.to_csv(out_csv, index=False)
        print(f"Results saved to {out_csv}")

    plot_df = plot_results(df, results_dir, name)
    save_predictions_and_scatter(plot_df, load_fn, results_dir)
    aggregate_feature_importance(results_dir)

    print(f"\n--- Computing formula/feature sensitivity for {name} ---")
    compute_bmi_sensitivity(plot_df, load_fn, results_dir, name)


def aggregate_permutation_sensitivity(top_n=15, min_pct=1.0):
    """Combine every per-age DeepPySR permutation-sensitivity CSV
    (bmi_deeppysr_sensitivity.csv, written by _plot_deeppysr_sensitivity)
    across all 4 variants into one overview: which variables drive
    DeepPySR's formulas, and does that change depending on which data
    source (variant) the model was given? Only the 'Interpretable DeepPySR'
    formula_type is kept -- the complexity-capped formula a clinician could
    actually read, not the unconstrained 'Best DeepPySR' one.

    Ranking is by *how often* a variable is a driver, not its mean
    sensitivity_pct share -- a plain mean is dominated by ages whose formula
    happens to have very few terms (a variable alone in a 2-term formula
    gets ~90%+ of that formula's normalised share by construction,
    regardless of how rarely it shows up anywhere else). Instead, for each
    variant, each variable's cell is "in what % of the ages evaluated for
    that variant does this variable appear as a driver (sensitivity_pct >
    min_pct)" -- top_n variables are chosen by total appearance count
    summed across every variant.

    Saves, under RESULTS_BASE_DIR:
      bmi_deeppysr_sensitivity_overview.csv  -- every kept age's rows,
                                                 concatenated (unaveraged)
      bmi_deeppysr_sensitivity_overview.png  -- heatmap, top_n variables
                                                 x variant, % of that
                                                 variant's ages where the
                                                 variable is a driver
    """
    rows = []
    for variant_name, (_, results_subdir) in VARIANTS.items():
        results_dir = os.path.join(RESULTS_BASE_DIR, results_subdir)
        for age in AGES:
            f = os.path.join(results_dir, f"age_{age}_{TARGET}", "bmi_deeppysr_sensitivity.csv")
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f)
            df = df[df['formula_type'] == 'Interpretable DeepPySR']
            if not df.empty:
                rows.append(df)
    if not rows:
        print("No DeepPySR sensitivity data to aggregate.")
        return pd.DataFrame()

    long_df = pd.concat(rows, ignore_index=True)
    csv_path = os.path.join(RESULTS_BASE_DIR, "bmi_deeppysr_sensitivity_overview.csv")
    long_df.to_csv(csv_path, index=False)
    print(f"DeepPySR sensitivity overview saved to {csv_path}")

    ages_per_variant = long_df[['variant', 'age']].drop_duplicates().groupby('variant').size()
    present = (long_df[long_df['sensitivity_pct'] > min_pct]
               [['variable', 'variant', 'age']].drop_duplicates())
    counts = present.groupby(['variable', 'variant']).size().unstack(fill_value=0)
    variant_order = [v for v in VARIANTS if v in ages_per_variant.index]
    counts = counts.reindex(columns=variant_order, fill_value=0)
    pct_table = counts.div(ages_per_variant.reindex(variant_order), axis=1) * 100.0

    top_vars = counts.sum(axis=1).sort_values(ascending=False).head(top_n).index
    plot_table = pct_table.loc[top_vars]

    png_path = os.path.join(RESULTS_BASE_DIR, "bmi_deeppysr_sensitivity_overview.png")
    _plot_sensitivity_heatmap(
        plot_table, png_path,
        title="BMI: how often each variable drives DeepPySR's interpretable formula, by data source"
              f" (n ages: {', '.join(f'{v}={n}' for v, n in ages_per_variant.items())})",
        top_n=top_n, cbar_label="% of that variant's ages where this variable is a driver")
    print(f"DeepPySR sensitivity overview plot saved to {png_path}")
    return long_df


# ─── Combined comparison across all 4 variants ──────────────────────────────

COMBINED_DISPLAY_MODELS = ['Best DeepPySR', 'Interpretable DeepPySR']


def load_combined():
    rows = []
    for variant_name, (_, results_subdir) in VARIANTS.items():
        csv_path = os.path.join(RESULTS_BASE_DIR, results_subdir, 'bmi_best_models_metrics.csv')
        if not os.path.exists(csv_path):
            print(f"Missing {csv_path}, run analysis for variant {variant_name} first.")
            continue
        df = pd.read_csv(csv_path)
        df = df[df['display_model'].isin(COMBINED_DISPLAY_MODELS)].copy()
        df['test'] = variant_name
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_combined(combined_df):
    metrics = ['r2', 'pearson_r', 'f1_macro']
    metric_labels = {'r2': 'R2', 'pearson_r': 'Pearson r', 'f1_macro': 'F1 (macro, clinical bins)'}
    tests = sorted(combined_df['test'].unique())
    palette = sns.color_palette("tab10", n_colors=len(tests))
    test_colors = dict(zip(tests, palette))

    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    plt.rcParams.update({'font.size': 14})

    for row_i, display_model in enumerate(COMBINED_DISPLAY_MODELS):
        sub = combined_df[combined_df['display_model'] == display_model]
        for col_i, metric in enumerate(metrics):
            ax = axes[row_i, col_i]
            label = metric_labels.get(metric, metric.upper())
            sns.lineplot(data=sub, x='age', y=metric, hue='test', ax=ax,
                         linewidth=3.0, palette=test_colors, marker='o', markersize=8)
            ax.set_title(f'{display_model}: {label} vs Age', fontsize=18, fontweight='bold', pad=15)
            ax.set_ylabel(label, fontsize=15)
            ax.set_xlabel('Age', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=12)
            if ax.get_legend():
                ax.get_legend().remove()

    legend_elements = [Line2D([0], [0], color=test_colors[t], lw=3, marker='o', label=t) for t in tests]
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.91, 0.5),
               fontsize=14, frameon=True, title='Feature set', title_fontsize=16, handlelength=4.0)
    plt.suptitle('BMI Prediction: DeepPySR Performance Across Feature-Set Variants\n'
                 f'(Top: Best DeepPySR — Bottom: Interpretable DeepPySR, complexity < {INTERP_MAX_COMPLEXITY})',
                 fontsize=24, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plot_path = os.path.join(RESULTS_BASE_DIR, 'bmi_deeppysr_metrics_vs_age_combined.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Combined plot saved to {plot_path}")


def run_combined():
    combined_df = load_combined()
    if combined_df.empty:
        print("No data available for combined analysis.")
        return

    out_csv = os.path.join(RESULTS_BASE_DIR, 'bmi_deeppysr_combined_metrics.csv')
    combined_df.to_csv(out_csv, index=False)
    print(f"Combined metrics saved to {out_csv}")

    plot_combined(combined_df)


def main():
    for name in VARIANTS:
        print("\n" + "=" * 60)
        print(f"ANALYSIS: {name}")
        print("=" * 60)
        run_variant(name)

    print("\n" + "=" * 60)
    print("COMBINED ANALYSIS")
    print("=" * 60)
    run_combined()

    print("\n" + "=" * 60)
    print("AGGREGATING DEEPPYSR PERMUTATION SENSITIVITY OVERVIEW")
    print("=" * 60)
    aggregate_permutation_sensitivity()


if __name__ == "__main__":
    main()
