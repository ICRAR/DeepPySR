"""Regression analysis for insulin-prediction feature-set variants
(PGS, to8, PGSto8, recent). DeepPySR/PySR R2 (and other metrics) are
computed with genuine held-out cross-validation instead of evaluating the
winning formula on the full dataset it was fit on. See
analysis_v1_utils.get_best_formula_from_raw for the leak-free method.
Results are written directly into the results_insulin_* subfolders under
whichever base results folder is selected via --results_base (e.g.
results_insulin_nfeateng or results_insulin_feateng).
"""
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

SCATTER_AXIS_LIMIT = 50

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../.."))
sys.path.append(current_dir)

# Base results folder to analyse (e.g. results_insulin_nfeateng or
# results_insulin_feateng); set from --results_base in __main__.
RESULTS_BASE_DIR = None

from data_utils import (
    load_data_PGS_only, load_data_keepto8, load_data_PGSto8, load_data_recent,
    _INSULIN_AGES,
)
from analysis_v1_utils import calculate_metrics, get_best_formula_from_raw, evaluate_formula, get_oof_predictions

INTERP_MAX_COMPLEXITY = 30

AGES = _INSULIN_AGES
TARGET = 'diab_raine'
FS_SUBFOLDERS = ['all_features', 'top50']

VARIANTS = {
    'PGS':    (load_data_PGS_only,  'results_insulin_PGS'),
    'to8':    (load_data_keepto8,   'results_insulin_to8'),
    'PGSto8': (load_data_PGSto8,    'results_insulin_PGSto8'),
    'recent': (load_data_recent,    'results_insulin_recent'),
}


def _load_age(load_fn, age):
    ids, X, y = load_fn(age)
    return ids, X, y.rename(TARGET) if hasattr(y, 'rename') else y


def _process_baselines(age_path, age, load_fn, all_data):
    baselines_dir = os.path.join(age_path, "baselines")
    if not os.path.exists(baselines_dir):
        return
    for model_name in os.listdir(baselines_dir):
        model_dir = os.path.join(baselines_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        for fs in FS_SUBFOLDERS:
            model_path = os.path.join(model_dir, fs)
            pred_file = os.path.join(model_path, "predictions.csv")
            if not os.path.exists(pred_file):
                continue
            df_pred = pd.read_csv(pred_file)
            tag = f"{model_name}_{fs}"
            r2, rmse, mae, pearson_r = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
            all_data.append([age, tag, r2, rmse, mae, pearson_r, np.nan, "", "", "", np.nan])


def _process_sr_dir(age_path, age, subdir, model_type, load_fn, all_data):
    sr_dir = os.path.join(age_path, subdir)
    if not os.path.exists(sr_dir):
        return
    for variant in os.listdir(sr_dir):
        v_path = os.path.join(sr_dir, variant)
        if not os.path.isdir(v_path):
            continue
        _, X_age, y_age = _load_age(load_fn, age)
        formula, complexity, metrics = get_best_formula_from_raw(v_path, X_age, y_age, model_type=model_type)
        r2, rmse, mae, pearson_r = metrics
        if not formula:
            pred_file = os.path.join(v_path, "predictions.csv")
            if os.path.exists(pred_file):
                df_pred = pd.read_csv(pred_file)
                r2, rmse, mae, pearson_r = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
        all_data.append([age, variant, r2, rmse, mae, pearson_r, complexity, formula, v_path, model_type, np.nan])

        # Separate complexity-constrained candidate for "Interpretable
        # DeepPySR": selection must be constrained to complexity < 30 per
        # fold *before* ranking by r2 (see get_best_formula_from_raw), so
        # this needs its own CV pass rather than filtering the row above.
        if model_type == 'deeppysr':
            formula_i, complexity_i, metrics_i = get_best_formula_from_raw(
                v_path, X_age, y_age, model_type=model_type, max_complexity=INTERP_MAX_COMPLEXITY)
            if formula_i:
                r2_i, rmse_i, mae_i, pearson_r_i = metrics_i
                all_data.append([age, f"{variant}__interp{INTERP_MAX_COMPLEXITY}", r2_i, rmse_i, mae_i, pearson_r_i,
                                 complexity_i, formula_i, v_path, model_type, INTERP_MAX_COMPLEXITY])


def process_results(load_fn, results_dir):
    all_data = []
    for age in AGES:
        age_path = os.path.join(results_dir, f"age_{age}_diab_raine")
        if not os.path.exists(age_path):
            continue
        _process_baselines(age_path, age, load_fn, all_data)
        _process_sr_dir(age_path, age, "deeppysr", "deeppysr", load_fn, all_data)
        _process_sr_dir(age_path, age, "pysr", "pysr", load_fn, all_data)

    df = pd.DataFrame(all_data, columns=['age', 'model', 'r2', 'rmse', 'mae', 'pearson_r', 'complexity', 'formula',
                                          'source_path', 'formula_model_type', 'max_complexity'])
    df['r2'] = df['r2'].clip(lower=0)
    return df


def _select_best_models(df):
    """Return plot_df and interpretable_formulas, age-specific only."""
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
        # ones. Picking only from unconstrained rows could otherwise lose to
        # the interpretable pick, since both selections rank candidates by
        # in-fold training fitness (a noisy proxy for held-out
        # generalization), not by held-out performance itself.
        deeppysr_df = age_df[is_deeppysr]
        if not deeppysr_df.empty:
            best = deeppysr_df.loc[deeppysr_df['r2'].idxmax()].copy()
            best['display_model'] = 'Best DeepPySR'
            selected_data.append(best)
            # Complexity-constrained candidates were computed separately
            # (own CV pass per fold, see _process_sr_dir) rather than by
            # filtering deeppysr_df post-hoc.
            interp = age_df[is_deeppysr & (age_df['max_complexity'] == INTERP_MAX_COMPLEXITY)]
            if not interp.empty:
                bi = interp.loc[interp['r2'].idxmax()].copy()
                bi['display_model'] = 'Interpretable DeepPySR'
                selected_data.append(bi)
                interpretable_formulas.append({'age': age, 'model': bi['model'],
                                               'formula': bi['formula'], 'r2': bi['r2'],
                                               'complexity': bi['complexity']})

        pysr_df = age_df[age_df['model'].str.contains('pysr', na=False)]
        if not pysr_df.empty:
            best_pysr = pysr_df.loc[pysr_df['r2'].idxmax()].copy()
            best_pysr['display_model'] = 'PySR'
            selected_data.append(best_pysr)

        for m in ['KAN']:
            m_df = age_df[age_df['model'].str.startswith(f"{m}_")]
            if not m_df.empty:
                row = m_df.loc[m_df['r2'].idxmax()].copy()
                row['display_model'] = m
                selected_data.append(row)

        for b in ['ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']:
            b_df = age_df[age_df['model'].str.startswith(f"{b}_")]
            if not b_df.empty:
                row = b_df.loc[b_df['r2'].idxmax()].copy()
                row['display_model'] = b
                selected_data.append(row)

    return pd.DataFrame(selected_data).reset_index(drop=True), interpretable_formulas


def plot_results(df, results_dir, variant_name):
    plot_df, interpretable_formulas = _select_best_models(df)
    if plot_df.empty:
        print(f"No data to plot for variant={variant_name}.")
        return plot_df

    plot_csv_path = os.path.join(results_dir, 'insulin_best_models_metrics.csv')
    plot_df.to_csv(plot_csv_path, index=False)
    print(f"Best models saved to {plot_csv_path}")

    metrics = ['r2', 'rmse', 'mae', 'pearson_r']
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    plt.rcParams.update({'font.size': 14})
    models = sorted(plot_df['display_model'].unique())
    palette = sns.color_palette("tab10", n_colors=len(models))
    model_colors = dict(zip(models, palette))

    metric_labels = {'r2': 'R2', 'rmse': 'RMSE', 'mae': 'MAE', 'pearson_r': 'Pearson r'}
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
    plt.suptitle(f'Insulin Prediction Performance ({variant_name}): Best Models Comparison',
                 fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plot_path = os.path.join(results_dir, 'insulin_metrics_vs_age.png')
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
    original predictions.csv under age_path/baselines/<model>/<fs>/."""
    for fs in FS_SUBFOLDERS:
        suffix = f"_{fs}"
        if not model_tag.endswith(suffix):
            continue
        base_name = model_tag[: -len(suffix)]
        pred_file = os.path.join(age_path, "baselines", base_name, fs, "predictions.csv")
        if not os.path.exists(pred_file):
            return None, None
        df_pred = pd.read_csv(pred_file)
        return df_pred['y_true'].values, df_pred['y_pred'].values
    return None, None


def _get_model_predictions(row, age_path, X_full, y_full):
    """Return (y_true, y_pred) arrays for one selected model row.

    Formula-based models (DeepPySR, PySR) use the same
    pooled leak-free out-of-fold predictions the reported R2 was computed
    from (get_oof_predictions on row['source_path']) -- NOT a re-evaluation
    of a single formula on the full in-sample dataset. Everything else falls
    back to the CV predictions.csv saved during training.
    """
    formula = row.get('formula')
    source_path = row.get('source_path')
    if isinstance(formula, str) and formula.strip() and isinstance(source_path, str) and source_path.strip():
        formula_model_type = row.get('formula_model_type') or 'deeppysr'
        prefix = 'formulas_fold' if formula_model_type == 'kan' else 'relationships_fold'
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
        age_path = os.path.join(results_dir, f"age_{age}_diab_raine")
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
            ax.set_title(f"{m} (R2={row['r2']:.2f}, r={row['pearson_r']:.2f})", fontsize=13, fontweight='bold')
            ax.set_xlabel('True Insulin', fontsize=11)
            ax.set_ylabel('Predicted Insulin', fontsize=11)
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

            # Points already visible in the main 0-100 panel are shown
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
        baselines_dir = os.path.join(results_dir, f"age_{age}_diab_raine", "baselines")
        if not os.path.exists(baselines_dir):
            continue
        for m in os.listdir(baselines_dir):
            if m not in ['ElasticNet', 'ExtraTrees', 'RandomForest', 'XGBoost', 'KAN']:
                continue
            for fs in FS_SUBFOLDERS:
                imp_file = os.path.join(baselines_dir, m, fs, "feature_importance.csv")
                if not os.path.exists(imp_file):
                    continue
                df_imp = pd.read_csv(imp_file)
                if 'feature' not in df_imp.columns or 'importance' not in df_imp.columns:
                    continue
                total = df_imp['importance'].sum()
                for _, row in df_imp.iterrows():
                    importance_data.append({
                        'age': age, 'model': f"{m}_{fs}",
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


def run_variant(name):
    load_fn, results_subdir = VARIANTS[name]
    results_dir = os.path.join(RESULTS_BASE_DIR, results_subdir)

    out_csv = os.path.join(results_dir, "insulin_aggregated_results.csv")
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


# ─── Combined comparison across all 4 variants ──────────────────────────────

COMBINED_DISPLAY_MODELS = ['Best DeepPySR', 'Interpretable DeepPySR']


def load_combined():
    rows = []
    for variant_name, (_, results_subdir) in VARIANTS.items():
        csv_path = os.path.join(RESULTS_BASE_DIR, results_subdir, 'insulin_best_models_metrics.csv')
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
    metrics = ['r2', 'rmse', 'mae', 'pearson_r']
    metric_labels = {'r2': 'R2', 'rmse': 'RMSE', 'mae': 'MAE', 'pearson_r': 'Pearson r'}
    tests = sorted(combined_df['test'].unique())
    palette = sns.color_palette("tab10", n_colors=len(tests))
    test_colors = dict(zip(tests, palette))

    fig, axes = plt.subplots(2, 4, figsize=(28, 14))
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
    plt.suptitle('Insulin Prediction: DeepPySR Performance Across Feature-Set Variants\n'
                 '(Top: Best DeepPySR — Bottom: Interpretable DeepPySR, complexity < 30)',
                 fontsize=24, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plot_path = os.path.join(RESULTS_BASE_DIR, 'insulin_deeppysr_metrics_vs_age_combined.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Combined plot saved to {plot_path}")


def run_combined():
    combined_df = load_combined()
    if combined_df.empty:
        print("No data available for combined analysis.")
        return

    out_csv = os.path.join(RESULTS_BASE_DIR, 'insulin_deeppysr_combined_metrics.csv')
    combined_df.to_csv(out_csv, index=False)
    print(f"Combined metrics saved to {out_csv}")

    plot_combined(combined_df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_base', type=str, default='results_insulin_nfeateng',
                        help="Base results folder to analyse, relative to this script's "
                             "directory (e.g. results_insulin_nfeateng or results_insulin_feateng).")
    parser.add_argument('--variant', type=str, default=None, choices=list(VARIANTS.keys()),
                        help="Which variant to analyse. Default: all four.")
    parser.add_argument('--skip_combined', action='store_true',
                        help="Skip the cross-variant combined comparison plot.")
    args = parser.parse_args()

    RESULTS_BASE_DIR = os.path.join(current_dir, args.results_base)

    names = [args.variant] if args.variant else list(VARIANTS.keys())
    for name in names:
        print("\n" + "=" * 60)
        print(f"ANALYSIS: {name}")
        print("=" * 60)
        run_variant(name)

    if not args.variant and not args.skip_combined:
        print("\n" + "=" * 60)
        print("COMBINED ANALYSIS")
        print("=" * 60)
        run_combined()
