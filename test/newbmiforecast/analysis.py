"""
Simplified analysis for newbmiforecast results.

Metrics (r2/rmse/mae/pearson_r) for DeepPySR/PySR/baselines come straight from
each age's cv_metrics_summary.csv -- already genuine, leak-free out-of-fold CV
numbers (see test_newbmiforecast.py). No re-evaluation of any formula happens
here.

Complexity for DeepPySR / DeepPySR (Interpretable) / PySR is a display-only
attribute layered on top: the winning grid config (the one cv_metrics_summary.csv
already picked, re-derived here by the same rule -- highest cv r2) is scanned
for the formula with the highest in-fold fitness, and that formula's complexity
is reported. DeepPySR (Interpretable) is a SEPARATE config selection: the
highest-cv-r2 config that has SOME formula with complexity < INTERP_COMPLEXITY_THRESHOLD.
Because both selections rank configs by the exact same cv r2 metric, and
Interpretable's candidate configs are a subset of DeepPySR's, DeepPySR's r2 is
always >= DeepPySR (Interpretable)'s r2 by construction.

Four comparison sources:
  - full_bmiforecast: this project's own results (native ages 8,10,14,17,20,22,27).
  - base_bmiforecast: test/bmiforecast's own saved comparison CSV (run its
    analysis.py first) -- already leak-free, already has age 23 relabeled to 22.
  - BMI Age-Specific / BMI Longitudinal: read directly from
    bmi/analysis_v1/bmi_best_models_metrics.csv, with age 23 relabeled to 22
    (same underlying wave as this project's native age 22, different source
    naming). KANSym is excluded everywhere.

Trajectory plots read predictions directly from rolling_dataset.csv's
y{year}bmi_{model}_pred columns (the pipeline's own best full-data fit) and
observed values from base_dataset.csv's raw y{year}bmi columns -- no formula
evaluation of any kind.
"""
import juliacall  # noqa: F401 -- must precede torch import (via model_utils'
# joblib-pickled MLPRegressorWrapper) to avoid the torch/juliacall segfault
# documented at https://github.com/pytorch/pytorch/issues/78829

import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr, norm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from analysis_utils import map_variable_names, evaluate_formula
from analysis_v1_utils import calculate_complexity
from data_utils import YEARS, actual_age, _is_bmi_col, get_age_filtered_feature_cols

FORECAST_RESULTS_DIR = os.path.join(current_dir, 'results_newbmiforecast')
BMI_BEST_MODELS_CSV = os.path.join(
    current_dir, '..', 'bmi', 'analysis_v1', 'bmi_best_models_metrics.csv')
BASE_BMIFORECAST_CSV = os.path.join(
    current_dir, '..', 'bmiforecast', 'analysis_v1', 'bmiforecast_comparison_metrics.csv')

FORECAST_YEARS = list(YEARS)   # [8, 10, 14, 17, 20, 22, 27] -- native at every age
FORECAST_AGES = [actual_age(y) for y in FORECAST_YEARS]

INTERP_COMPLEXITY_THRESHOLD = 30
# bmi's own "age 23" wave is the same underlying measurement as this project's
# native "age 22" wave -- different source naming for the same thing.
AGE_RELABEL = {23: 22}

FAMILY_MODEL_TO_DISPLAY = {
    ('deeppysr', 'DeepPySR'): 'DeepPySR',
    ('pysr', 'PySR'): 'PySR',
    ('baseline', 'ElasticNet'): 'ElasticNet',
    ('baseline', 'ExtraTrees'): 'ExtraTrees',
    ('baseline', 'MLP'): 'MLP',
    ('baseline', 'RandomForest'): 'RandomForest',
    ('baseline', 'XGBoost'): 'XGBoost',
    ('baseline', 'KAN'): 'KAN',
    # family == 'kan' (KANSym) is intentionally absent -- excluded everywhere.
}

MODELS_TO_PLOT = ['DeepPySR', 'PySR', 'ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']
MODELS_TO_PLOT_INTERP = MODELS_TO_PLOT + ['DeepPySR (Interpretable)']
SYMBOLIC_MODELS = ['DeepPySR', 'DeepPySR (Interpretable)', 'PySR']
HIGHLIGHT_MODELS = {'DeepPySR': 'D', 'DeepPySR (Interpretable)': 'X'}


# ── Metric / complexity extraction ─────────────────────────────────────────────

def _read_overall_metrics(cfg_path):
    """r2/mae/rmse from a grid config's overall_metrics.csv, plus pearson_r
    computed fresh from its sibling predictions.csv (genuine out-of-fold
    predictions from run_cv)."""
    om_path = os.path.join(cfg_path, 'overall_metrics.csv')
    if not os.path.exists(om_path):
        return None
    try:
        row = pd.read_csv(om_path).iloc[0]
    except Exception:
        return None
    pearson_r = np.nan
    pred_path = os.path.join(cfg_path, 'predictions.csv')
    if os.path.exists(pred_path):
        try:
            dfp = pd.read_csv(pred_path)
            yt = dfp['y_true'].values.astype(float)
            yp = dfp['y_pred'].values.astype(float)
            mask = np.isfinite(yt) & np.isfinite(yp)
            if mask.sum() >= 2 and np.std(yt[mask]) > 0 and np.std(yp[mask]) > 0:
                pearson_r = float(pearsonr(yt[mask], yp[mask])[0])
        except Exception:
            pass
    return {'r2': float(row['r2']), 'mae': float(row['mae']), 'rmse': float(row['rmse']),
            'pearson_r': pearson_r}


def _read_predictions_metrics(pred_path):
    """r2 computed directly from a raw y_true/y_pred predictions CSV (e.g.
    full_models' predictions_foldnocv.csv, which has no separate
    overall_metrics.csv of its own)."""
    if not os.path.exists(pred_path):
        return None
    try:
        df = pd.read_csv(pred_path)
        yt = df['y_true'].values.astype(float)
        yp = df['y_pred'].values.astype(float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() < 2:
            return None
        from sklearn.metrics import r2_score
        return {'r2': float(r2_score(yt[mask], yp[mask]))}
    except Exception:
        return None


def _best_formula_row(fold_files, max_complexity=None):
    """Across a config's relationships_fold*.csv / formulas_fold*.csv files,
    find the formula with the highest in-fold fitness ('r2' column, the SR
    search's own training-set score for that candidate -- used only to pick a
    REPRESENTATIVE formula, never to compute a reported metric), optionally
    restricted to complexity < max_complexity. None if no candidate qualifies.

    Complexity is always computed fresh from the formula string via
    calculate_complexity (token count) -- PySR's formulas_fold*.csv has no
    'complexity' column at all (only 'fold,formula'), so relying on the
    file's own column silently dropped every PySR candidate. Recomputing
    uniformly also keeps DeepPySR and PySR complexity on the same scale.
    """
    best = None
    for f in fold_files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if 'formula' not in df.columns:
            continue
        df = df.copy()
        df['_complexity'] = df['formula'].astype(str).map(calculate_complexity)
        cand = df[df['_complexity'] < max_complexity] if max_complexity is not None else df
        if cand.empty:
            continue
        row = cand.loc[cand['r2'].idxmax()] if 'r2' in cand.columns else cand.iloc[0]
        fold_r2 = float(row['r2']) if 'r2' in cand.columns else -np.inf
        if best is None or fold_r2 > best['fold_r2']:
            best = {'formula': str(row['formula']), 'complexity': float(row['_complexity']),
                    'fold_r2': fold_r2}
    return best


def _get_feature_cols_for_year(rolling_df, year, model_type='deeppysr'):
    """Reconstruct the feature columns used when training for a given year and
    model type -- used only to map x_i tokens back to real feature names for
    display. Never used to compute metrics."""
    non_bmi_cols = [c for c in rolling_df.columns
                    if c != 'child_id' and not _is_bmi_col(c) and not c.endswith('_pred')]
    non_bmi_cols = get_age_filtered_feature_cols(non_bmi_cols, year)
    prior_years = [y for y in YEARS if y < year]
    prior_pred_cols = []
    for py in prior_years:
        pred_col = f'y{py}bmi_{model_type}_pred'
        if pred_col in rolling_df.columns:
            prior_pred_cols.append(pred_col)
        elif f'y{py}bmi' in rolling_df.columns:
            prior_pred_cols.append(f'y{py}bmi')
    return non_bmi_cols + prior_pred_cols


def load_age_metrics(age_dir, rolling_df, year):
    """Metric rows for one age directory: {display_model: {r2, rmse, mae,
    pearson_r, complexity, formula}}. r2/rmse/mae/pearson_r for DeepPySR/PySR/
    baselines come straight from cv_metrics_summary.csv; complexity/formula
    for DeepPySR, DeepPySR (Interpretable) and PySR are derived separately
    (see module docstring).
    """
    rows = {}
    csv_path = os.path.join(age_dir, 'cv_metrics_summary.csv')
    if not os.path.exists(csv_path):
        return rows

    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        family, model = str(row['family']), str(row['model'])
        if family == 'kan':  # KANSym -- excluded everywhere
            continue
        display = FAMILY_MODEL_TO_DISPLAY.get((family, model))
        if display is None:
            continue
        rows[display] = {
            'r2': float(row['cv_r2']), 'rmse': float(row['cv_rmse']), 'mae': float(row['cv_mae']),
            'pearson_r': float(row['cv_pearson_r']) if 'cv_pearson_r' in row and pd.notna(row['cv_pearson_r']) else np.nan,
            'complexity': np.nan, 'formula': '',
        }

    def _winning_config(family_dir, max_complexity=None):
        best_cfg, best_metrics, best_formula, best_r2 = None, None, None, -np.inf
        for cfg in sorted(os.listdir(family_dir)):
            cfg_path = os.path.join(family_dir, cfg)
            prefix = 'relationships_fold*.csv' if 'deeppysr' in family_dir else 'formulas_fold*.csv'
            fold_files = sorted(glob.glob(os.path.join(cfg_path, prefix)))
            fr = _best_formula_row(fold_files, max_complexity=max_complexity)
            if fr is None:
                continue
            m = _read_overall_metrics(cfg_path)
            if m is None or m['r2'] <= best_r2:
                continue
            best_cfg, best_metrics, best_formula, best_r2 = cfg, m, fr, m['r2']
        return best_cfg, best_metrics, best_formula

    dsr_dir = os.path.join(age_dir, 'deeppysr')
    if os.path.isdir(dsr_dir) and 'DeepPySR' in rows:
        fn = _get_feature_cols_for_year(rolling_df, year, model_type='deeppysr')

        _, _, fr = _winning_config(dsr_dir)
        if fr:
            rows['DeepPySR']['complexity'] = fr['complexity']
            rows['DeepPySR']['formula'] = map_variable_names(fr['formula'], fn, model_type='deeppysr')

        cfg, m, fr = _winning_config(dsr_dir, max_complexity=INTERP_COMPLEXITY_THRESHOLD)
        if cfg:
            rows['DeepPySR (Interpretable)'] = {
                'r2': m['r2'], 'rmse': m['rmse'], 'mae': m['mae'], 'pearson_r': m['pearson_r'],
                'complexity': fr['complexity'],
                'formula': map_variable_names(fr['formula'], fn, model_type='deeppysr'),
            }

    psr_dir = os.path.join(age_dir, 'pysr')
    if os.path.isdir(psr_dir) and 'PySR' in rows:
        fn = _get_feature_cols_for_year(rolling_df, year, model_type='pysr')
        _, _, fr = _winning_config(psr_dir)
        if fr:
            rows['PySR']['complexity'] = fr['complexity']
            rows['PySR']['formula'] = map_variable_names(fr['formula'], fn, model_type='pysr')

    return rows


def load_newbmiforecast_results(rolling_df):
    """One row per (age, display_model) across all age_* directories."""
    rows = []
    for year, age in zip(FORECAST_YEARS, FORECAST_AGES):
        age_dir = os.path.join(FORECAST_RESULTS_DIR, f'age_{age}')
        if not os.path.isdir(age_dir):
            continue
        for display, m in load_age_metrics(age_dir, rolling_df, year).items():
            rows.append({'age': age, 'display_model': display, 'source': 'full_bmiforecast', **m})
    df = pd.DataFrame(rows)
    df['r2'] = df['r2'].clip(lower=0)
    return df


def load_base_bmiforecast_results():
    """test/bmiforecast's own precomputed comparison metrics -- run its
    analysis.py first. Already leak-free, already has age 23 relabeled to 22,
    already excludes KANSym."""
    if not os.path.exists(BASE_BMIFORECAST_CSV):
        raise FileNotFoundError(
            f'base bmiforecast comparison CSV not found: {BASE_BMIFORECAST_CSV} '
            f'-- run test/bmiforecast/analysis.py first.')
    df = pd.read_csv(BASE_BMIFORECAST_CSV)
    df = df[df['source'] == 'BMI Forecast'].copy()
    df['source'] = 'base_bmiforecast'
    df['r2'] = df['r2'].clip(lower=0)
    return df


def load_bmi_comparison():
    """bmi's leak-free age-specific / longitudinal results, read directly
    from bmi/analysis_v1/bmi_best_models_metrics.csv -- no re-computation.
    Drops KANSym (degenerate at every age)."""
    if not os.path.exists(BMI_BEST_MODELS_CSV):
        raise FileNotFoundError(f'BMI best models CSV not found: {BMI_BEST_MODELS_CSV}')
    df = pd.read_csv(BMI_BEST_MODELS_CSV)
    df['display_model'] = df['display_model'].replace(
        {'Best DeepPySR': 'DeepPySR', 'Interpretable DeepPySR': 'DeepPySR (Interpretable)'})
    df = df[df['display_model'] != 'KANSym'].copy()
    df['r2'] = df['r2'].clip(lower=0)
    df['age'] = df['age'].replace(AGE_RELABEL)

    cols = ['age', 'display_model', 'r2', 'rmse', 'mae', 'pearson_r', 'complexity', 'formula']
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    def _rows(sub_df, source_label):
        out = sub_df[cols].copy()
        out['source'] = source_label
        return out.reset_index(drop=True)

    return (_rows(df[df['type'] == 'age-specific'], 'BMI Age-Specific'),
            _rows(df[df['type'] == 'longitudinal'], 'BMI Longitudinal'))


def save_combined_csv(df_forecast, df_base_bmiforecast, df_age_specific, df_longitudinal, out_dir):
    combined = pd.concat(
        [df_forecast, df_base_bmiforecast, df_age_specific, df_longitudinal], ignore_index=True)
    cols = ['source', 'age', 'display_model', 'r2', 'rmse', 'mae', 'pearson_r', 'complexity', 'formula']
    combined = combined[[c for c in cols if c in combined.columns]]
    csv_path = os.path.join(out_dir, 'newbmiforecast_comparison_metrics.csv')
    combined.to_csv(csv_path, index=False)
    print(f'Saved combined metrics CSV: {csv_path}')
    return combined


# ── Plotting helpers ────────────────────────────────────────────────────────────

def _build_palette(models):
    palette = sns.color_palette('tab10', n_colors=max(len(models), 1))
    return dict(zip(sorted(models), palette))


def _metric_label(metric):
    return 'Pearson r' if metric == 'pearson_r' else metric.upper()


def _highlight_lines(ax, model_colors, models=HIGHLIGHT_MODELS, lw=4.2, ms=12):
    """Boost linewidth/marker size/zorder (and give a distinct marker shape)
    for specific model lines drawn by sns.lineplot, so they stay visible even
    when another series' values coincide exactly.

    sns.lineplot's actual plotted Line2D objects carry useless auto-generated
    labels ('_child0', ...) -- the real hue-category labels are only set on
    separate, empty proxy lines it adds purely for the legend. So matching
    goes by color (via the same palette dict passed to sns.lineplot), not by
    line.get_label().
    """
    import matplotlib.colors as mcolors
    targets = {m: mcolors.to_rgba(model_colors[m]) for m in models if m in model_colors}
    if not targets:
        return
    for line in ax.get_lines():
        if len(line.get_xdata()) == 0:
            continue
        color = mcolors.to_rgba(line.get_color())
        for m, rgba in targets.items():
            if all(abs(a - b) < 1e-3 for a, b in zip(color, rgba)):
                line.set_linewidth(lw)
                line.set_markersize(ms)
                line.set_marker(models[m])
                line.set_markeredgecolor('black')
                line.set_markeredgewidth(1.2)
                line.set_zorder(10)
                break


def _plot_line(ax, x, y, color, label, linewidth=2.5, markersize=7, marker='o',
                highlight_markersize=12, highlight_zorder=10, highlight_marker_alpha=1.0,
                **kwargs):
    """highlight_markersize/highlight_zorder/highlight_marker_alpha only apply
    when label is in HIGHLIGHT_MODELS -- callers with tightly clustered series
    (e.g. plot_sparse_trajectories, where all models can converge to nearly
    the same value at a given age) should pass a smaller size/lower zorder/
    partial alpha so the highlighted marker doesn't fully occlude other
    series' markers sitting at coincident points."""
    if label in HIGHLIGHT_MODELS:
        linewidth = max(linewidth, 4.2)
        markersize = max(markersize, highlight_markersize)
        marker = HIGHLIGHT_MODELS[label]
        kwargs.setdefault('markeredgecolor', 'black')
        kwargs.setdefault('markeredgewidth', 1.2)
        kwargs.setdefault('zorder', highlight_zorder)
        if highlight_marker_alpha < 1.0:
            import matplotlib.colors as mcolors
            r, g, b = mcolors.to_rgb(color)
            kwargs.setdefault('markerfacecolor', (r, g, b, highlight_marker_alpha))
    return ax.plot(x, y, color=color, linewidth=linewidth, marker=marker,
                   markersize=markersize, label=label, **kwargs)


def _legend_handles(models, model_colors):
    handles = []
    for m in models:
        marker = HIGHLIGHT_MODELS.get(m, 'o')
        lw = 4.2 if m in HIGHLIGHT_MODELS else 2.5
        ms = 12 if m in HIGHLIGHT_MODELS else 7
        handles.append(mlines.Line2D([0], [0], color=model_colors[m], lw=lw, marker=marker,
                                     markersize=ms,
                                     markeredgecolor='black' if m in HIGHLIGHT_MODELS else None,
                                     label=m))
    return handles


FORECAST_AGES_DISPLAY = sorted(set(FORECAST_AGES) | set(AGE_RELABEL.values()))


def _add_complexity_subplot(ax, df, model_colors, title_prefix):
    sym_df = df[df['display_model'].isin(SYMBOLIC_MODELS) & df['complexity'].notna()].copy()
    for model in sorted(sym_df['display_model'].unique()):
        m_df = sym_df[sym_df['display_model'] == model].sort_values('age')
        _plot_line(ax, m_df['age'], m_df['complexity'], model_colors[model], model)
    ax.axhline(INTERP_COMPLEXITY_THRESHOLD, color='gray', linestyle='--', linewidth=1.5)
    ax.set_title(f'{title_prefix}: Complexity vs Age', fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel('Age', fontsize=13)
    ax.set_ylabel('Complexity', fontsize=13)
    ax.set_xticks(FORECAST_AGES_DISPLAY)


def plot_comparison_vs_age(df_forecast, df_base_bmiforecast, df_age_specific, df_longitudinal, out_dir):
    """4×5 grid: rows = source, cols = R2 / RMSE / MAE / Pearson r / Complexity."""
    sources = [
        ('full_bmiforecast', df_forecast),
        ('base_bmiforecast', df_base_bmiforecast),
        ('BMI Age-Specific', df_age_specific),
        ('BMI Longitudinal', df_longitudinal),
    ]
    filtered = []
    for label, df in sources:
        d = df[df['display_model'].isin(MODELS_TO_PLOT_INTERP)].copy()
        d['source'] = label
        filtered.append(d)

    all_models = sorted(set(m for d in filtered for m in d['display_model'].unique()))
    model_colors = _build_palette(all_models)
    metrics = ['r2', 'rmse', 'mae', 'pearson_r']

    fig, axes = plt.subplots(4, 5, figsize=(35, 24))
    plt.rcParams.update({'font.size': 13})

    for row_idx, (label, df) in enumerate(zip([s[0] for s in sources], filtered)):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            models_here = sorted(df['display_model'].unique())
            row_colors = {m: model_colors[m] for m in models_here}
            sns.lineplot(data=df, x='age', y=metric, hue='display_model', ax=ax,
                         linewidth=2.5, palette=row_colors, marker='o', markersize=7)
            _highlight_lines(ax, row_colors)
            ax.set_title(f'{label}: {_metric_label(metric)} vs Age', fontsize=15, fontweight='bold', pad=10)
            ax.set_xlabel('Age', fontsize=13)
            ax.set_ylabel(_metric_label(metric), fontsize=13)
            ax.set_xticks(FORECAST_AGES_DISPLAY)
            if ax.get_legend():
                ax.get_legend().remove()

        _add_complexity_subplot(axes[row_idx, 4], df, model_colors, label)

    handles = _legend_handles(all_models, model_colors)
    handles.append(mlines.Line2D([0], [0], color='gray', lw=1.5, linestyle='--',
                                  label=f'Complexity threshold ({INTERP_COMPLEXITY_THRESHOLD})'))
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.91, 0.5),
               fontsize=12, frameon=True, title='Model', title_fontsize=13)
    plt.suptitle('full_bmiforecast vs base_bmiforecast vs BMI Test (Age-Specific & Longitudinal)',
                 fontsize=20, fontweight='bold', y=1.005)
    plt.tight_layout(rect=[0, 0, 0.9, 1.0])
    path = os.path.join(out_dir, 'newbmiforecast_vs_bmi_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_deeppysr_only(df_forecast, df_base_bmiforecast, df_age_specific, df_longitudinal, out_dir):
    """2×5: DeepPySR (row 0) and DeepPySR Interpretable (row 1) across 4 sources.
    Cols: R2 / RMSE / MAE / Pearson r / Complexity.
    """
    source_styles = {
        'full_bmiforecast': ('-', 'tab:blue'),
        'base_bmiforecast': ('-.', 'tab:red'),
        'BMI Age-Specific': ('--', 'tab:orange'),
        'BMI Longitudinal': (':', 'tab:green'),
    }
    sources_data = [
        ('full_bmiforecast', df_forecast),
        ('base_bmiforecast', df_base_bmiforecast),
        ('BMI Age-Specific', df_age_specific),
        ('BMI Longitudinal', df_longitudinal),
    ]
    variants = ['DeepPySR', 'DeepPySR (Interpretable)']
    metrics = ['r2', 'rmse', 'mae', 'pearson_r']

    fig, axes = plt.subplots(2, 5, figsize=(35, 12))
    plt.rcParams.update({'font.size': 13})

    for row_idx, variant in enumerate(variants):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for src_label, df in sources_data:
                ls, color = source_styles[src_label]
                dsr_df = df[df['display_model'] == variant].sort_values('age')
                if dsr_df.empty:
                    continue
                ax.plot(dsr_df['age'], dsr_df[metric], color=color, linestyle=ls,
                        linewidth=2.5, marker='o', markersize=8)
            ax.set_title(f'{variant}: {_metric_label(metric)} vs Age', fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('Age', fontsize=13)
            ax.set_ylabel(_metric_label(metric), fontsize=13)
            ax.set_xticks(FORECAST_AGES_DISPLAY)

        ax_c = axes[row_idx, 4]
        for src_label, df in sources_data:
            ls, color = source_styles[src_label]
            dsr_df = df[(df['display_model'] == variant) & df['complexity'].notna()].sort_values('age')
            if dsr_df.empty:
                continue
            ax_c.plot(dsr_df['age'], dsr_df['complexity'], color=color, linestyle=ls,
                      linewidth=2.5, marker='o', markersize=8)
        ax_c.axhline(INTERP_COMPLEXITY_THRESHOLD, color='gray', linestyle='--', linewidth=1.5)
        ax_c.set_title(f'{variant}: Complexity vs Age', fontsize=14, fontweight='bold', pad=10)
        ax_c.set_xlabel('Age', fontsize=13)
        ax_c.set_ylabel('Complexity', fontsize=13)
        ax_c.set_xticks(FORECAST_AGES_DISPLAY)

    handles = [mlines.Line2D([0], [0], color=source_styles[lbl][1], lw=2.5,
                              linestyle=source_styles[lbl][0], marker='o', label=lbl)
               for lbl in source_styles]
    handles.append(mlines.Line2D([0], [0], color='gray', lw=1.5, linestyle='--',
                                  label=f'Complexity threshold ({INTERP_COMPLEXITY_THRESHOLD})'))
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.91, 0.5),
               fontsize=12, frameon=True, title='Source', title_fontsize=13, handlelength=4.0)
    plt.suptitle('DeepPySR Performance: full_bmiforecast vs base_bmiforecast vs Age-Specific vs Longitudinal',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 0.9, 1.0])
    path = os.path.join(out_dir, 'newbmiforecast_deeppysr_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ── CDC growth chart background ─────────────────────────────────────────────────

_CDC_BMI_EXT_CACHE = {}
_CDC_BMI_EXT_PATH = os.path.join(current_dir, 'bmizscore', 'bmi-age-2022.csv')


def _load_cdc_bmi_extended(sex):
    """Load pre-computed CDC extended BMI-for-age percentiles. sex: 1=male, 2=female."""
    if sex in _CDC_BMI_EXT_CACHE:
        return _CDC_BMI_EXT_CACHE[sex]
    if not os.path.exists(_CDC_BMI_EXT_PATH):
        return None
    df = pd.read_csv(_CDC_BMI_EXT_PATH)
    sub = df[df['sex'] == sex].sort_values('agemos').reset_index(drop=True)
    _CDC_BMI_EXT_CACHE[sex] = sub
    return sub


_WHO_LMS_CACHE = {}
_WHO_LMS_DIR = os.path.join(current_dir, 'bmizscore')


def _load_who_lms(sex):
    """WHO 0-2y BMI-for-age LMS table: {month: (L, M, S)}. sex: 1=male, 2=female."""
    if sex in _WHO_LMS_CACHE:
        return _WHO_LMS_CACHE[sex]
    import json
    fname = 'bmifa_boys_0_2_zscores.json' if sex == 1 else 'bmifa_girls_0_2_zscores.json'
    path = os.path.join(_WHO_LMS_DIR, fname)
    if not os.path.exists(path):
        return None
    rows = json.loads(open(path).read())
    table = {int(r['Month']): (float(r['L']), float(r['M']), float(r['S'])) for r in rows}
    _WHO_LMS_CACHE[sex] = table
    return table


def _lms_zscore_to_bmi(L, M, S, z):
    """Invert the standard LMS z-score formula (Box-Cox): z = ((bmi/M)^L - 1)/(L*S)."""
    if abs(L) >= 0.01:
        return M * (1.0 + L * S * z) ** (1.0 / L)
    return M * np.exp(S * z)


def _who_zscore_to_bmi(z, agemos, sex):
    """Raw BMI from a WHO BMI-for-age z-score, ages 0-24 months (interpolated)."""
    table = _load_who_lms(sex)
    if table is None or pd.isna(z):
        return np.nan
    lo, hi = int(agemos), min(int(agemos) + 1, 24)
    if lo not in table:
        return np.nan
    if lo == hi or hi not in table:
        L, M, S = table[lo]
    else:
        frac = agemos - lo
        L = table[lo][0] + frac * (table[hi][0] - table[lo][0])
        M = table[lo][1] + frac * (table[hi][1] - table[lo][1])
        S = table[lo][2] + frac * (table[hi][2] - table[lo][2])
    return _lms_zscore_to_bmi(L, M, S, z)


def _early_bmi_observations(base_idx, cid, sex):
    """Birth / year-5 raw BMI for a participant. Birth from base_dataset's raw
    birth_weight(g)/birth_length(cm); year-5 from base_dataset's own y5bmi
    column (already raw, no z-score inversion needed here since
    newbmiforecast's base_dataset retains it directly, unlike bmiforecast's).
    Year-1 has no retained data in base_dataset (no y1 weight/height nor
    z-score column) and is skipped. Returns {age_years: bmi}.
    """
    out = {}
    if cid not in base_idx.index:
        return out
    row = base_idx.loc[cid]
    if 'birth_weight' in row.index and 'birth_length' in row.index:
        bw, bl = row['birth_weight'], row['birth_length']
        if pd.notna(bw) and pd.notna(bl) and float(bl) > 0:
            out[0] = (float(bw) / 1000.0) / (float(bl) / 100.0) ** 2
    if 'y5bmi' in row.index and pd.notna(row['y5bmi']):
        out[5] = float(row['y5bmi'])
    return out


def _draw_cdc_background(ax, sex, x_max=27.0, y_ceil=55.0):
    """Draw sex-specific CDC BMI-for-age percentile bands and reference lines,
    filling the axes exactly from age 0 to x_max and from y_floor to y_ceil --
    callers should pass x_max/y_ceil matching the actual data range being
    plotted (and then set matching ax.set_xlim/ylim) so the background covers
    the full subplot with no white margin, and no data point falls outside it.

    Data source: bmi-age-2022.csv (CDC extended BMI-for-age, pre-computed percentiles).
    Coverage: age 2-20 yr + flat extension to x_max. No background before age 2.

    Bands: <5th underweight (blue), 5-85th healthy (green), 85-95th overweight
    (yellow), then obesity split into clinical classes per CDC/AAP severe-obesity
    convention: Class I 95th-120% of P95, Class II 120-140% of P95, Class III
    (severe) >140% of P95 -- pct120ofP95 is given directly in the CSV
    (confirmed == 1.2*P95); pct140 is derived the same way (1.4*P95).
    Bands are drawn faint (dimmed alpha) so trajectory lines stay legible on top.
    """
    df = _load_cdc_bmi_extended(int(sex))
    if df is None:
        return

    sub = df[(df['agemos'] >= 24.0) & (df['agemos'] <= 240.0)].copy()
    ages_cdc = sub['agemos'].values / 12.0

    col = {'P5': 'P5', 'P50': 'P50', 'P85': 'P85', 'P95': 'P95'}
    curves = {k: sub[c].values for k, c in col.items()}
    curves['pct120'] = 1.2 * curves['P95']
    curves['pct140'] = 1.4 * curves['P95']
    v20 = {k: arr[-1] for k, arr in curves.items()}

    fwd_ages = np.array([20.0, x_max]) if x_max > 20.0 else np.array([20.0])
    all_ages = np.concatenate([ages_cdc, fwd_ages])

    def _curve(k):
        return np.concatenate([curves[k], np.full(len(fwd_ages), v20[k])])

    y_floor = 8.0
    DIM = 0.5  # scales all band alphas down for a dimmer background

    band_defs = [
        (None,    'P5',     '#4393c3', 0.35),
        ('P5',    'P50',    '#92c5de', 0.30),
        ('P50',   'P85',    '#a8d5a2', 0.30),
        ('P85',   'P95',    '#fee090', 0.40),
        ('P95',   'pct120', '#fdae61', 0.40),  # Obese Class I
        ('pct120', 'pct140', '#f46d43', 0.40),  # Obese Class II
        ('pct140', None,    '#a50026', 0.40),  # Obese Class III (severe)
    ]
    for lo_k, hi_k, color, alpha in band_defs:
        lo_v = _curve(lo_k) if lo_k is not None else np.full(len(all_ages), y_floor)
        hi_v = _curve(hi_k) if hi_k is not None else np.full(len(all_ages), y_ceil)
        ax.fill_between(all_ages, lo_v, hi_v, color=color, alpha=alpha * DIM, zorder=0, linewidth=0)

    line_specs = {
        'P5':     ('--', 0.7, 0.40, 'dimgray'),
        'P50':    ('-',  0.8, 0.45, 'dimgray'),
        'P85':    (':',  0.7, 0.40, 'dimgray'),
        'P95':    ('--', 0.9, 0.55, '#d6604d'),
        'pct120': (':',  0.8, 0.50, '#d6604d'),
        'pct140': ('-.', 0.8, 0.50, '#b2182b'),
    }
    for k, (ls, lw, alpha, color) in line_specs.items():
        ax.plot(all_ages, _curve(k), color=color, linestyle=ls,
                linewidth=lw, alpha=alpha, zorder=1)


# ── Trajectory plots ─────────────────────────────────────────────────────────

# Display label -> model_type string used for feature-column reconstruction /
# evaluate_formula. KAN is left out: its full-data fit is saved in a
# non-joblib format (_state/_config), not worth the extra loader for a
# baseline that's already near-degenerate everywhere in this project.
MODEL_TYPES = {
    'DeepPySR': 'deeppysr', 'DeepPySR (Interpretable)': 'deeppysr', 'PySR': 'pysr',
    'ElasticNet': 'ElasticNet', 'ExtraTrees': 'ExtraTrees', 'MLP': 'MLP',
    'RandomForest': 'RandomForest', 'XGBoost': 'XGBoost',
}

_CDC_BAND_LEGEND = [
    ('Underweight (<5th pct)', '#4393c3'),
    ('Healthy weight (5th-85th pct)', '#a8d5a2'),
    ('Overweight (85th-95th pct)', '#fee090'),
    ('Obese Class I (95th pct-120% of P95)', '#fdae61'),
    ('Obese Class II (120-140% of P95)', '#f46d43'),
    ('Obese Class III / severe (>140% of P95)', '#a50026'),
]


def _load_full_baseline_models(models_dir):
    """{model_name: fitted sklearn model} from full_models/_models/*.joblib."""
    import joblib as _joblib
    models = {}
    if not os.path.isdir(models_dir):
        return models
    for fname in sorted(os.listdir(models_dir)):
        if fname.endswith('.joblib'):
            try:
                models[fname[:-len('.joblib')]] = _joblib.load(os.path.join(models_dir, fname))
            except Exception:
                pass
    return models


def _winning_nocv_formula(family_dir, formula_filename, max_complexity=None):
    """Winning no-CV formula (highest predictions_foldnocv.csv r2 across
    configs) for a formula family -- the same model the pipeline itself used
    to fill missing values in rolling_dataset.csv, but evaluated here on
    EVERY row (not just missing ones) so a genuine prediction is available
    even where the true value is already known -- unlike rolling_dataset's
    own _pred columns, which just copy the true value whenever it exists.
    max_complexity: if set, only formulas with complexity below this
    threshold are eligible (used for the 'DeepPySR (Interpretable)' variant);
    the config-level ranking (by overall no-CV r2) still runs across ALL
    configs, only skipping ones with no qualifying formula.
    """
    if not os.path.isdir(family_dir):
        return None
    best_formula, best_r2 = None, -np.inf
    for cfg in sorted(os.listdir(family_dir)):
        cfg_path = os.path.join(family_dir, cfg)
        m = _read_predictions_metrics(os.path.join(cfg_path, 'predictions_foldnocv.csv'))
        if m is None or m['r2'] <= best_r2:
            continue
        formula_path = os.path.join(cfg_path, formula_filename)
        if not os.path.exists(formula_path):
            continue
        try:
            df = pd.read_csv(formula_path)
        except Exception:
            continue
        if 'formula' not in df.columns or df.empty:
            continue
        if max_complexity is not None:
            df = df.copy()
            df['_complexity'] = df['formula'].astype(str).map(calculate_complexity)
            df = df[df['_complexity'] < max_complexity]
            if df.empty:
                continue
        row = df.loc[df['r2'].idxmax()] if 'r2' in df.columns else df.iloc[0]
        best_formula, best_r2 = str(row['formula']), m['r2']
    return best_formula


def _read_id_pred_dict(pred_path):
    """{id: y_pred} from a raw predictions CSV (y_true, y_pred, id columns)."""
    if not os.path.exists(pred_path):
        return {}
    try:
        df = pd.read_csv(pred_path)
    except Exception:
        return {}
    return {int(i): float(p) for i, p in zip(df['id'], df['y_pred'])}


def _oof_predictions_from_dir(dir_path, is_grid_family, max_complexity=None, formula_prefix=None):
    """{id: y_pred} genuine out-of-fold CV predictions for every participant
    who has an observed value at this age (i.e. was part of that age's CV
    training data) -- these participants must NOT be scored with the
    full-data (no-CV) model, since they contributed to fitting it (leakage).
    is_grid_family=True: dir_path holds multiple <config>_grid subfolders
    (DeepPySR/PySR); the one with the highest overall CV r2 is used (same
    selection rule cv_metrics_summary.csv itself applies). is_grid_family=
    False: dir_path IS the single baseline config folder -- its own
    predictions.csv is read directly. max_complexity (grid family only): only
    configs with a qualifying (complexity below threshold) fold formula are
    eligible -- used for the 'DeepPySR (Interpretable)' variant; requires
    formula_prefix ('relationships_fold*.csv' for deeppysr).
    """
    if not is_grid_family:
        return _read_id_pred_dict(os.path.join(dir_path, 'predictions.csv'))

    if not os.path.isdir(dir_path):
        return {}
    best_cfg, best_r2 = None, -np.inf
    for cfg in sorted(os.listdir(dir_path)):
        cfg_path = os.path.join(dir_path, cfg)
        om_path = os.path.join(cfg_path, 'overall_metrics.csv')
        if not os.path.exists(om_path):
            continue
        try:
            r2 = float(pd.read_csv(om_path).iloc[0]['r2'])
        except Exception:
            continue
        if r2 <= best_r2:
            continue
        if max_complexity is not None:
            fold_files = sorted(glob.glob(os.path.join(cfg_path, formula_prefix)))
            if _best_formula_row(fold_files, max_complexity=max_complexity) is None:
                continue
        best_r2, best_cfg = r2, cfg
    if best_cfg is None:
        return {}
    return _read_id_pred_dict(os.path.join(dir_path, best_cfg, 'predictions.csv'))


def _year_model_predictions(age_dir, rolling_df, sampled_ids, year):
    """{display_model: {child_id: prediction}} for every sampled participant
    at this age. Participants who provided an observed value at this age (and
    so were part of the full-data no-CV model's own training set) are scored
    with that model's genuine out-of-fold CV prediction instead (from the
    winning grid config's own predictions.csv) to avoid leakage; participants
    who did NOT provide an observation at this age (outside the training set
    entirely, for both the CV folds and the no-CV fit) are scored with the
    full-data (no-CV) model/formula, genuinely evaluated on their row.
    """
    nocv_preds = {}
    full_dir = os.path.join(age_dir, 'full_models')
    if not os.path.isdir(full_dir):
        return {}

    rolling_sub = rolling_df[rolling_df['child_id'].isin(sampled_ids)].set_index('child_id')
    rolling_sub = rolling_sub.reindex(sampled_ids)

    baseline_models = _load_full_baseline_models(os.path.join(full_dir, '_models'))
    for name, model in baseline_models.items():
        fn = _get_feature_cols_for_year(rolling_df, year, model_type=name)
        if not all(c in rolling_sub.columns for c in fn):
            continue
        X = rolling_sub[fn]
        try:
            yp = model.predict(X.values)
        except Exception:
            continue
        nocv_preds[name] = dict(zip(sampled_ids, yp))

    formula = _winning_nocv_formula(os.path.join(full_dir, 'deeppysr'), 'relationships_nocv.csv')
    if formula:
        fn = _get_feature_cols_for_year(rolling_df, year, model_type='deeppysr')
        X = rolling_sub[[c for c in fn if c in rolling_sub.columns]]
        try:
            nocv_preds['DeepPySR'] = dict(zip(sampled_ids, evaluate_formula(formula, X, model_type='deeppysr')))
        except Exception:
            pass

    formula = _winning_nocv_formula(os.path.join(full_dir, 'deeppysr'), 'relationships_nocv.csv',
                                     max_complexity=INTERP_COMPLEXITY_THRESHOLD)
    if formula:
        fn = _get_feature_cols_for_year(rolling_df, year, model_type='deeppysr')
        X = rolling_sub[[c for c in fn if c in rolling_sub.columns]]
        try:
            nocv_preds['DeepPySR (Interpretable)'] = dict(
                zip(sampled_ids, evaluate_formula(formula, X, model_type='deeppysr')))
        except Exception:
            pass

    formula = _winning_nocv_formula(os.path.join(full_dir, 'pysr'), 'formulas_foldnocv.csv')
    if formula:
        fn = _get_feature_cols_for_year(rolling_df, year, model_type='pysr')
        X = rolling_sub[[c for c in fn if c in rolling_sub.columns]]
        try:
            nocv_preds['PySR'] = dict(zip(sampled_ids, evaluate_formula(formula, X, model_type='pysr')))
        except Exception:
            pass

    oof_preds = {}
    for name in ['ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']:
        oof_preds[name] = _oof_predictions_from_dir(
            os.path.join(age_dir, 'baselines', name), is_grid_family=False)
    oof_preds['DeepPySR'] = _oof_predictions_from_dir(
        os.path.join(age_dir, 'deeppysr'), is_grid_family=True)
    oof_preds['DeepPySR (Interpretable)'] = _oof_predictions_from_dir(
        os.path.join(age_dir, 'deeppysr'), is_grid_family=True,
        max_complexity=INTERP_COMPLEXITY_THRESHOLD, formula_prefix='relationships_fold*.csv')
    oof_preds['PySR'] = _oof_predictions_from_dir(
        os.path.join(age_dir, 'pysr'), is_grid_family=True)

    preds = {}
    for model in MODEL_TYPES:
        merged = {}
        for cid in sampled_ids:
            if cid in oof_preds.get(model, {}):
                merged[cid] = oof_preds[model][cid]
            elif cid in nocv_preds.get(model, {}):
                merged[cid] = nocv_preds[model][cid]
        if merged:
            preds[model] = merged
    return preds


def _select_extreme_bmi_ids(base_df, n_obs_filter, n_sample):
    """Participants with exactly n_obs_filter observed BMIs, prioritizing the
    highest mean observed BMI (extreme/overweight cases where the CDC
    background context is most informative)."""
    base_idx = base_df.set_index('child_id')
    true_cols = [f'y{y}bmi' for y in YEARS if f'y{y}bmi' in base_idx.columns]
    obs_counts = base_idx[true_cols].notna().sum(axis=1)
    candidates = obs_counts[obs_counts == n_obs_filter].index if n_obs_filter is not None \
                 else obs_counts.index
    if len(candidates) == 0:
        return []

    def _mean_obs_bmi(cid):
        vals = base_idx.loc[cid, true_cols].dropna().values
        return float(np.mean(vals)) if len(vals) else -np.inf

    return sorted(candidates, key=_mean_obs_bmi, reverse=True)[:n_sample]


def _all_year_preds(rolling_df, sampled_ids):
    """{year: {display_model: {child_id: prediction}}} across every year in
    YEARS for the given participant ids -- see _year_model_predictions for
    the genuine, leak-free-by-construction sourcing rule.
    """
    year_preds = {}
    for year in YEARS:
        age_dir = os.path.join(FORECAST_RESULTS_DIR, f'age_{actual_age(year)}')
        year_preds[year] = _year_model_predictions(age_dir, rolling_df, sampled_ids, year)
    return year_preds


def _select_deeppysr_advantage_ids(rolling_df, base_df, n_obs_filter, n_sample):
    """Among participants with exactly n_obs_filter observed BMIs, rank by how
    much more accurate DeepPySR is than the other 6 models' average, at the
    ages where the participant actually has an observation (so accuracy is
    measured against genuine ground truth, using the same leak-free per-age
    predictions used everywhere else in this file -- see _all_year_preds).
    Score = mean over observed ages of (mean(other models' abs error) -
    DeepPySR's abs error); positive means DeepPySR is more accurate on
    average. Returns (top n_sample ids by descending score, the year_preds
    dict already computed for the full candidate pool, so callers don't need
    to recompute predictions for the selected ids).
    """
    base_idx = base_df.set_index('child_id')
    true_cols = [f'y{y}bmi' for y in YEARS if f'y{y}bmi' in base_idx.columns]
    obs_counts = base_idx[true_cols].notna().sum(axis=1)
    candidates = list(obs_counts[obs_counts == n_obs_filter].index) if n_obs_filter is not None \
                 else list(obs_counts.index)
    if not candidates:
        return [], {}

    year_preds = _all_year_preds(rolling_df, candidates)
    other_models = [m for m in MODEL_TYPES if m not in ('DeepPySR', 'DeepPySR (Interpretable)')]

    scores = {}
    for cid in candidates:
        advantages = []
        for y in YEARS:
            col = f'y{y}bmi'
            if col not in base_idx.columns:
                continue
            true_val = base_idx.loc[cid, col]
            if pd.isna(true_val):
                continue
            dsr_pred = year_preds.get(y, {}).get('DeepPySR', {}).get(cid)
            if dsr_pred is None or not np.isfinite(dsr_pred):
                continue
            other_errs = [abs(p - true_val) for m in other_models
                          if (p := year_preds.get(y, {}).get(m, {}).get(cid)) is not None
                          and np.isfinite(p)]
            if not other_errs:
                continue
            scores[cid] = scores.get(cid, [])
            scores[cid].append(float(np.mean(other_errs)) - abs(float(dsr_pred) - float(true_val)))

    mean_scores = {cid: np.mean(v) for cid, v in scores.items() if v}
    top_ids = sorted(mean_scores, key=mean_scores.get, reverse=True)[:n_sample]
    return top_ids, year_preds


def _render_trajectory_grid(sampled_ids, year_preds, base_idx, out_dir, out_filename, suptitle):
    """Shared rendering body for a grid of per-participant BMI trajectory
    subplots (CDC background + model lines + observed points), used by both
    plot_sparse_trajectories (extreme-BMI selection) and
    plot_deeppysr_best_trajectories (DeepPySR-advantage selection).
    """
    model_colors = _build_palette(list(MODEL_TYPES))

    n = len(sampled_ids)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for p_num, (ax, cid) in enumerate(zip(axes, sampled_ids), start=1):
        ax.set_box_aspect(1)
        sex = base_idx.loc[cid, 'sex_x'] if (cid in base_idx.index and 'sex_x' in base_idx.columns) else np.nan
        sex = int(sex) if pd.notna(sex) and int(sex) in (1, 2) else 1

        model_series = {}
        for model in MODEL_TYPES:
            xs, ys = [], []
            for year in YEARS:
                v = year_preds.get(year, {}).get(model, {}).get(cid)
                if v is not None and np.isfinite(v):
                    xs.append(actual_age(year))
                    ys.append(float(v))
            if xs:
                model_series[model] = (xs, ys)

        obs_x, obs_y = [], []
        for y in YEARS:
            col = f'y{y}bmi'
            if col in base_idx.columns and cid in base_idx.index:
                v = base_idx.loc[cid, col]
                if pd.notna(v):
                    obs_x.append(actual_age(y))
                    obs_y.append(float(v))
        early = _early_bmi_observations(base_idx, cid, sex)
        for age, bmi in early.items():
            obs_x.append(age)
            obs_y.append(bmi)

        all_x = obs_x + [x for xs, _ in model_series.values() for x in xs]
        all_y = obs_y + [y for _, ys in model_series.values() for y in ys]
        x_max = max([27.0] + all_x)
        y_ceil = max(55.0, (max(all_y) * 1.05 if all_y else 0.0))

        _draw_cdc_background(ax, sex, x_max=x_max + 0.5, y_ceil=y_ceil)

        for model, (xs, ys) in model_series.items():
            _plot_line(ax, xs, ys, model_colors[model], model, linewidth=2.0, markersize=5,
                      highlight_markersize=9, highlight_zorder=5, highlight_marker_alpha=0.55)

        ax.scatter(obs_x, obs_y, color='black', s=100, zorder=20, label='Observed',
                  edgecolor='white', linewidth=1)
        ax.set_xlim(-0.5, x_max + 0.5)
        ax.set_ylim(8.0, y_ceil)

        sex_label = 'M' if sex == 1 else 'F'
        ax.set_title(f'Participant {p_num} ({sex_label})', fontsize=12)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('BMI')
        ax.set_xticks(sorted({actual_age(y) for y in YEARS} | set(early.keys())))

    for ax in axes[n:]:
        ax.axis('off')

    handles = _legend_handles(list(MODEL_TYPES), model_colors)
    handles.append(mlines.Line2D([0], [0], marker='o', color='black', lw=0, markersize=8,
                                  label='Observed'))
    handles.append(mlines.Line2D([0], [0], color='none', label=''))
    for label, color in _CDC_BAND_LEGEND:
        handles.append(mpatches.Patch(facecolor=color, alpha=0.5, label=label))
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=11)
    plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, out_filename)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_sparse_trajectories(rolling_df, base_df, out_dir, n_obs_filter=2, n_sample=12, seed=42):
    """BMI trajectories for participants with few observed values, favoring
    the highest observed BMI (extreme/overweight cases where the CDC
    background is most informative). See _all_year_preds/_year_model_predictions
    for the prediction sourcing rule (genuine out-of-fold CV where the
    participant has an observation at that age, full no-CV model otherwise).
    """
    base_idx = base_df.set_index('child_id')
    sampled_ids = _select_extreme_bmi_ids(base_df, n_obs_filter, n_sample)
    if not sampled_ids:
        print(f'  No participants found with exactly {n_obs_filter} observed BMIs.')
        return
    year_preds = _all_year_preds(rolling_df, sampled_ids)
    _render_trajectory_grid(
        sampled_ids, year_preds, base_idx, out_dir,
        out_filename=f'newbmiforecast_trajectories_{n_obs_filter}_obs.png',
        suptitle=f'BMI Trajectories — Participants with {n_obs_filter} Observed Values (highest BMI first)')


def plot_deeppysr_best_trajectories(rolling_df, base_df, out_dir, n_obs_filter=2, n_sample=12):
    """BMI trajectories for participants where DeepPySR is, on average across
    their observed ages, the most accurate model by a wide margin -- see
    _select_deeppysr_advantage_ids for the ranking rule (mean of other
    models' abs error minus DeepPySR's abs error, at genuinely observed
    ages only).
    """
    base_idx = base_df.set_index('child_id')
    sampled_ids, year_preds = _select_deeppysr_advantage_ids(rolling_df, base_df, n_obs_filter, n_sample)
    if not sampled_ids:
        print(f'  No participants found with exactly {n_obs_filter} observed BMIs.')
        return
    _render_trajectory_grid(
        sampled_ids, year_preds, base_idx, out_dir,
        out_filename=f'newbmiforecast_trajectories_deeppysr_best_{n_obs_filter}_obs.png',
        suptitle=f'BMI Trajectories — Participants where DeepPySR Outperforms Other Models ({n_obs_filter} Observed Values)')


def _extract_formula_vars(formula_str, available_cols):
    """Return sorted list of column names that appear as tokens in formula_str."""
    import re as _re
    known_funcs = {'log', 'exp', 'sin', 'cos', 'sqrt', 'abs', 'inv', 'neg',
                   'square', 'cube', 'cond', 'pi', 'E', 'nan', 'inf'}
    tokens = set(_re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', str(formula_str)))
    col_set = set(available_cols)
    return sorted(t for t in tokens if t not in known_funcs and t in col_set)


def save_formula_demonstration(rolling_df, base_df, df_forecast, out_dir, n_obs_filter=2, n_sample=10):
    """For each sampled participant (same selection as plot_sparse_trajectories:
    exactly n_obs_filter observed BMIs, highest observed BMI first), show the
    DeepPySR best and interpretable formula, the value of every variable in
    the formula, population context, and the predicted vs true BMI.

    Per-variable statistics are type-aware:
      - continuous variables: population mean +/- SD, this participant's
        percentile, and a two-sided z-test p-value (distance from the
        population mean in SD units).
      - binary/categorical variables: population prevalence of the observed
        level (n / N), and a p-value equal to the empirical frequency of that
        exact level (small = rare level in the cohort). Percentile is not
        meaningful for these and is reported as N/A.

    Ages here are newbmiforecast's own native ages (including 22, not
    relabeled), unlike bmiforecast's demo which relabels 23 -> 22.

    Outputs:
      newbmiforecast_formula_demo_{n_obs_filter}_obs.csv   -- machine-readable
      newbmiforecast_formula_demo_{n_obs_filter}_obs.txt   -- human-readable
    """
    sampled_ids = _select_extreme_bmi_ids(base_df, n_obs_filter, n_sample)
    if not sampled_ids:
        print(f'  No participants found with exactly {n_obs_filter} observed BMIs.')
        return

    base_idx = base_df.set_index('child_id')
    rolling_sub = rolling_df[rolling_df['child_id'].isin(sampled_ids)].set_index('child_id')
    all_cols = list(rolling_df.columns)

    formula_map = {(int(r['age']), r['display_model']): r['formula']
                   for _, r in df_forecast.iterrows()
                   if pd.notna(r.get('formula')) and str(r['formula']) not in ('', '0.0', 'nan')}

    # Population stats, type-aware, computed once from the full rolling_df.
    col_stats = {}
    for col in all_cols:
        series = pd.to_numeric(rolling_df[col], errors='coerce').dropna()
        if series.empty:
            continue
        vals = series.values.astype(float)
        uniq = np.unique(vals)
        is_integer_valued = np.allclose(uniq, np.round(uniq))
        if len(uniq) <= 2 and set(uniq.tolist()) <= {0.0, 1.0}:
            vtype = 'binary'
        elif len(uniq) <= 10 and is_integer_valued:
            vtype = 'categorical'
        else:
            vtype = 'continuous'
        entry = {'type': vtype, 'n': len(vals)}
        if vtype == 'continuous':
            entry['mean'] = float(np.mean(vals))
            entry['std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            entry['sorted'] = np.sort(vals)
        else:
            lv, lc = np.unique(vals, return_counts=True)
            entry['freq'] = {float(a): int(b) for a, b in zip(lv, lc)}
        col_stats[col] = entry

    def _percentile(col, value):
        st = col_stats.get(col, {})
        if st.get('type') != 'continuous' or np.isnan(value):
            return np.nan
        arr = st['sorted']
        return float(np.searchsorted(arr, value, side='right')) / len(arr) * 100.0 if len(arr) else np.nan

    def _variable_summary(col, value):
        st = col_stats.get(col)
        if st is None or np.isnan(value):
            return ('unknown', 'N/A', np.nan, np.nan)
        if st['type'] == 'continuous':
            mean, std = st['mean'], st['std']
            summary = f'{mean:.4f} ± {std:.4f}'
            pct = _percentile(col, value)
            p_value = float(2.0 * norm.sf(abs((value - mean) / std))) if std > 0 else np.nan
            return (st['type'], summary, pct, p_value)
        n_total = st['n']
        level = min(st['freq'].keys(), key=lambda lv: abs(lv - value)) if st['freq'] else value
        count = st['freq'].get(level, 0)
        freq = count / n_total if n_total else np.nan
        summary = f'level={level:g}  n={count}/{n_total} ({freq*100:.1f}%)'
        return (st['type'], summary, np.nan, freq)

    DEMO_MODELS = [('DeepPySR', 'deeppysr'), ('DeepPySR (Interpretable)', 'deeppysr')]

    def true_bmi_at(cid, year):
        col = f'y{year}bmi'
        if cid in base_idx.index and col in base_idx.columns:
            v = base_idx.loc[cid, col]
            return float(v) if pd.notna(v) else np.nan
        return np.nan

    csv_rows, text_lines = [], []
    text_lines.append(
        f'Formula Demonstration — {n_obs_filter} Observed BMI Participants (highest BMI first)\n'
        + '=' * 70)

    for p_num, cid in enumerate(sampled_ids, start=1):
        text_lines.append(f'\n{"─"*70}\nParticipant {p_num}  (ID: {int(cid)})\n{"─"*70}')
        if cid not in rolling_sub.index:
            text_lines.append('  (no rolling-dataset row for this participant)')
            continue
        row_data = rolling_sub.loc[cid]
        X_row = pd.DataFrame([row_data])

        for year, age in zip(FORECAST_YEARS, FORECAST_AGES):
            true_val = true_bmi_at(cid, year)
            text_lines.append(f'\n  Age {age}   |   True BMI: '
                              + (f'{true_val:.2f}' if not np.isnan(true_val) else 'not observed'))

            for label, model_type in DEMO_MODELS:
                formula = formula_map.get((age, label))
                if not formula:
                    text_lines.append(f'    [{label}]  formula not available')
                    continue

                var_names = _extract_formula_vars(formula, all_cols)
                var_info = {}
                for v in var_names:
                    raw = row_data[v] if v in row_data.index else np.nan
                    val = float(raw) if pd.notna(raw) else np.nan
                    var_info[v] = (val, *_variable_summary(v, val))

                try:
                    yp_arr = evaluate_formula(str(formula), X_row, model_type=model_type)
                    bmi_pred = float(yp_arr[0]) if (yp_arr is not None and len(yp_arr) > 0
                                                     and np.isfinite(yp_arr[0])) else np.nan
                except Exception:
                    bmi_pred = np.nan

                text_lines.append(f'    [{label}]')
                text_lines.append(f'      Formula : {formula}')
                text_lines.append(
                    f'      {"Variable":<24} {"Type":<11} {"Value":>9}  '
                    f'{"Summary (mean±SD or level freq)":<38} {"Pctile":>7} {"p-value":>9}')
                text_lines.append(f'      {"─"*24} {"─"*11} {"─"*9}  {"─"*38} {"─"*7} {"─"*9}')
                for vname, (val, vtype, summary, pct, p_value) in sorted(var_info.items()):
                    v_str = f'{val:.4f}' if not np.isnan(val) else 'NaN'
                    pct_str = f'{pct:.1f}%' if not np.isnan(pct) else 'N/A'
                    p_str = f'{p_value:.4g}' if not np.isnan(p_value) else 'NaN'
                    text_lines.append(
                        f'      {vname:<24} {vtype:<11} {v_str:>9}  {summary:<38} {pct_str:>7} {p_str:>9}')
                text_lines.append('      Predicted BMI : '
                                  + (f'{bmi_pred:.4f}' if not np.isnan(bmi_pred) else 'NaN'))

                for vname, (val, vtype, summary, pct, p_value) in sorted(var_info.items()):
                    csv_rows.append({
                        'n_obs': n_obs_filter, 'participant_number': p_num, 'child_id': int(cid),
                        'age': age, 'model': label, 'formula': formula,
                        'variable': vname, 'var_type': vtype,
                        'value': round(val, 4) if not np.isnan(val) else np.nan,
                        'summary': summary,
                        'percentile': round(pct, 1) if not np.isnan(pct) else np.nan,
                        'p_value': p_value if not np.isnan(p_value) else np.nan,
                        'bmi_pred': round(bmi_pred, 4) if not np.isnan(bmi_pred) else np.nan,
                        'bmi_true': round(true_val, 4) if not np.isnan(true_val) else np.nan,
                    })

    csv_path = os.path.join(out_dir, f'newbmiforecast_formula_demo_{n_obs_filter}_obs.csv')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f'  Saved: {csv_path}')

    txt_path = os.path.join(out_dir, f'newbmiforecast_formula_demo_{n_obs_filter}_obs.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(text_lines) + '\n')
    print(f'  Saved: {txt_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    out_dir = os.path.join(current_dir, 'analysis_v1')
    os.makedirs(out_dir, exist_ok=True)

    rolling_csv = os.path.join(FORECAST_RESULTS_DIR, 'rolling_dataset.csv')
    base_csv = os.path.join(FORECAST_RESULTS_DIR, 'base_dataset.csv')
    rolling_df = pd.read_csv(rolling_csv)
    base_df = pd.read_csv(base_csv) if os.path.exists(base_csv) else rolling_df

    print('=== Loading newbmiforecast results ===')
    df_forecast = load_newbmiforecast_results(rolling_df)
    print(f'  {len(df_forecast)} rows')

    print('=== Loading base_bmiforecast results ===')
    df_base_bmiforecast = load_base_bmiforecast_results()
    print(f'  {len(df_base_bmiforecast)} rows')

    print('=== Loading bmi test results ===')
    df_age_specific, df_longitudinal = load_bmi_comparison()
    print(f'  age-specific: {len(df_age_specific)} rows, longitudinal: {len(df_longitudinal)} rows')

    print('=== Saving combined CSV ===')
    save_combined_csv(df_forecast, df_base_bmiforecast, df_age_specific, df_longitudinal, out_dir)

    print('=== Plotting comparison vs age ===')
    plot_comparison_vs_age(df_forecast, df_base_bmiforecast, df_age_specific, df_longitudinal, out_dir)

    print('=== Plotting DeepPySR-only comparison ===')
    plot_deeppysr_only(df_forecast, df_base_bmiforecast, df_age_specific, df_longitudinal, out_dir)

    print('=== Plotting sparse trajectories ===')
    for n in [2, 3, 4, 5]:
        plot_sparse_trajectories(rolling_df, base_df, out_dir, n_obs_filter=n)

    print('=== Plotting DeepPySR-advantage trajectories ===')
    for n in [2, 3, 4, 5]:
        plot_deeppysr_best_trajectories(rolling_df, base_df, out_dir, n_obs_filter=n)

    print('=== Saving formula demonstrations ===')
    for n in [2, 3, 4, 5]:
        save_formula_demonstration(rolling_df, base_df, df_forecast, out_dir, n_obs_filter=n)

    print('\nDone.')
