"""
Analysis for bmiforecast results.
Loads cv_metrics_summary.csv per age folder, creates plots and CSV comparing:
  - bmiforecast models vs age
  - bmi age-specific models vs age
  - bmi longitudinal models vs age
"""

import os
import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

try:
    import torch  # must be imported before juliacall to avoid segfault
except ImportError:
    pass
from analysis_utils import evaluate_formula, map_variable_names

FORECAST_RESULTS_DIR = os.path.join(current_dir, 'results_bmiforecast')
BMI_BEST_MODELS_CSV = os.path.join(
    current_dir, '..', 'bmi', 'results_bmi_all', 'bmi_best_models_metrics.csv'
)

# year → actual age label (folder name); from bmiforecast_utils.AGE_MAPPING
AGE_MAPPING = {13: 14, 16: 17, 26: 27}
# years used in bmiforecast (skip year 8 which is the base)
FORECAST_YEARS = [10, 13, 16, 20, 23, 26]
# folder age labels
FORECAST_AGES = [AGE_MAPPING.get(y, y) for y in FORECAST_YEARS]  # [10,14,17,20,23,27]
# year → bmi column name
YEAR_TO_BMI_COL = {y: f'y{y}bmi' for y in FORECAST_YEARS}
# all BMI years including the base (year 8)
ALL_YEARS = [8] + FORECAST_YEARS
ALL_AGES  = [AGE_MAPPING.get(y, y) for y in ALL_YEARS]  # [8,10,14,17,20,23,27]

# Map family/model in bmiforecast to a clean display name
FAMILY_MODEL_TO_DISPLAY = {
    ('deeppysr', 'DeepPySR'): 'DeepPySR',
    ('pysr', 'PySR'): 'PySR',
    ('kan', 'KAN'): 'KANSym',
    ('baseline', 'ElasticNet'): 'ElasticNet',
    ('baseline', 'ExtraTrees'): 'ExtraTrees',
    ('baseline', 'MLP'): 'MLP',
    ('baseline', 'RandomForest'): 'RandomForest',
    ('baseline', 'XGBoost'): 'XGBoost',
    ('baseline', 'KAN'): 'KAN',
}

MODELS_TO_PLOT = ['DeepPySR', 'PySR', 'ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']
MODELS_TO_PLOT_INTERP = MODELS_TO_PLOT + ['DeepPySR (Interpretable)']
FORMULA_MODELS = {'deeppysr': 'DeepPySR', 'pysr': 'PySR', 'kan': 'KANSym'}
INTERP_COMPLEXITY_THRESHOLD = 22


# ── Formula extraction helpers ────────────────────────────────────────────────

def _get_feature_cols_for_year(rolling_df, year, model_type='deeppysr'):
    """Reconstruct the feature columns used when training for a given year and model type."""
    from bmiforecast_utils import _is_bmi_col
    non_bmi_cols = [c for c in rolling_df.columns
                    if c != 'child_id' and not _is_bmi_col(c) and not c.endswith('_pred')]
    prior_years = FORECAST_YEARS[:FORECAST_YEARS.index(year)]
    prior_pred_cols = []
    for py in [8] + prior_years:
        pred_col = f'y{py}bmi_{model_type}_pred'
        if pred_col in rolling_df.columns:
            prior_pred_cols.append(pred_col)
        elif f'y{py}bmi' in rolling_df.columns:
            prior_pred_cols.append(f'y{py}bmi')
    return non_bmi_cols + prior_pred_cols


def _collect_formula_candidates(files, col_formula, feature_names, X, y, model_type):
    """Evaluate every formula in the given files on (X, y) using a fixed evaluation mask.

    All formulas are evaluated on the same set of samples (rows where ALL formulas
    are finite), so r2 comparisons are consistent.

    Returns a list of dicts: {r2, rmse, mae, formula, complexity}.
    """
    from sklearn.metrics import r2_score as _r2_score, mean_squared_error, mean_absolute_error

    # Collect (raw_formula, complexity) pairs, deduplicated by raw formula string
    seen = {}
    for f in files:
        try:
            df = pd.read_csv(f)
            if col_formula not in df.columns:
                continue
            has_complexity = 'complexity' in df.columns
            for _, row in df.iterrows():
                raw = str(row[col_formula])
                complexity = float(row['complexity']) if has_complexity else np.nan
                if raw not in seen:
                    seen[raw] = complexity
        except Exception:
            continue

    if not seen:
        return []

    # Evaluate all formulas; build a shared valid mask (finite predictions from all)
    y_arr = np.asarray(y, dtype=float)
    preds = {}
    for raw in seen:
        try:
            yp = evaluate_formula(raw, X, model_type=model_type)
            if yp is not None and len(yp) == len(y_arr):
                preds[raw] = np.asarray(yp, dtype=float)
        except Exception:
            continue

    if not preds:
        return []

    # Shared mask: rows that are finite in y AND in every formula's predictions
    shared_mask = np.isfinite(y_arr)
    for yp in preds.values():
        shared_mask &= np.isfinite(yp)

    if shared_mask.sum() < 2:
        return []

    yt = y_arr[shared_mask]
    candidates = []
    for raw, yp in preds.items():
        try:
            yp_masked = yp[shared_mask]
            r2 = _r2_score(yt, yp_masked)
            candidates.append({
                'r2': r2,
                'rmse': float(np.sqrt(mean_squared_error(yt, yp_masked))),
                'mae': float(mean_absolute_error(yt, yp_masked)),
                'formula': map_variable_names(raw, feature_names, model_type=model_type),
                'complexity': seen[raw],
            })
        except Exception:
            continue

    return candidates


def _pick_best(candidates, max_complexity=None):
    """Pick the candidate with highest r2, optionally filtered by complexity."""
    pool = candidates
    if max_complexity is not None:
        pool = [c for c in candidates
                if not np.isnan(c['complexity']) and c['complexity'] <= max_complexity]
    if not pool:
        return None
    return max(pool, key=lambda c: c['r2'])


def extract_bmiforecast_formulas():
    """For each forecast age, find best formula (unrestricted) and best interpretable
    formula (complexity < INTERP_COMPLEXITY_THRESHOLD) for DeepPySR, PySR, KAN.

    Returns dict: (age, display_model) -> {'formula': str, 'complexity': float}
    DeepPySR (Interpretable) entries are added separately.
    """
    rolling_csv = os.path.join(FORECAST_RESULTS_DIR, 'rolling_dataset.csv')
    if not os.path.exists(rolling_csv):
        print('  rolling_dataset.csv not found; skipping formula extraction')
        return {}

    rolling_df = pd.read_csv(rolling_csv)
    # Use base_dataset to identify rows with REAL (not predicted) target values
    base_csv = os.path.join(FORECAST_RESULTS_DIR, 'base_dataset.csv')
    base_df = pd.read_csv(base_csv) if os.path.exists(base_csv) else None

    results = {}  # (age, display_model) -> {'formula': str, 'complexity': float, ...}

    for year, age in zip(FORECAST_YEARS, FORECAST_AGES):
        age_dir = os.path.join(FORECAST_RESULTS_DIR, f'age_{age}')
        if not os.path.exists(age_dir):
            continue

        bmi_col = YEAR_TO_BMI_COL[year]
        if bmi_col not in rolling_df.columns:
            continue

        if base_df is not None and bmi_col in base_df.columns:
            real_ids = set(base_df.loc[base_df[bmi_col].notna(), 'child_id'].values)
            real_mask = rolling_df['child_id'].isin(real_ids)
            base_sub = rolling_df[real_mask].copy().dropna(subset=[bmi_col])
        else:
            base_sub = rolling_df.copy().dropna(subset=[bmi_col])

        def _make_X_y(model_type):
            fcols = _get_feature_cols_for_year(rolling_df, year, model_type=model_type)
            avail = [c for c in fcols if c in rolling_df.columns]
            sub = base_sub[['child_id'] + avail + [bmi_col]].copy()
            return sub[avail], sub[bmi_col].values, avail

        # DeepPySR — collect once, pick best and best-interpretable from same candidate pool
        dsr_files = glob.glob(os.path.join(age_dir, 'deeppysr', '**', 'relationships_fold*.csv'),
                               recursive=True)
        if not dsr_files:
            dsr_files = glob.glob(os.path.join(age_dir, 'deeppysr', '**', 'relationships.csv'),
                                   recursive=True)
        if dsr_files:
            X_dsr, y_dsr, fn_dsr = _make_X_y('deeppysr')
            dsr_candidates = _collect_formula_candidates(
                dsr_files, 'formula', fn_dsr, X_dsr, y_dsr, 'deeppysr')
            res = _pick_best(dsr_candidates)
            if res:
                results[(age, 'DeepPySR')] = res
            res_i = _pick_best(dsr_candidates, max_complexity=INTERP_COMPLEXITY_THRESHOLD)
            if res_i:
                results[(age, 'DeepPySR (Interpretable)')] = res_i

        # PySR — uses pysr_pred columns, not deeppysr_pred
        psr_files = glob.glob(os.path.join(age_dir, 'pysr', '**', 'formulas_fold*.csv'),
                               recursive=True)
        if psr_files:
            X_psr, y_psr, fn_psr = _make_X_y('pysr')
            psr_candidates = _collect_formula_candidates(
                psr_files, 'formula', fn_psr, X_psr, y_psr, 'pysr')
            res = _pick_best(psr_candidates)
            if res:
                results[(age, 'PySR')] = res

        # KAN (baseline) — no model-specific pred column, falls back to raw bmi cols
        kan_files = glob.glob(
            os.path.join(age_dir, 'baselines', 'KAN', 'formulas_fold*.csv'), recursive=True)
        if kan_files:
            X_kan, y_kan, fn_kan = _make_X_y('KAN')
            kan_candidates = _collect_formula_candidates(
                kan_files, 'formula', fn_kan, X_kan, y_kan, 'kan')
            res = _pick_best(kan_candidates)
            if res:
                results[(age, 'KANSym')] = res

        print(f'  age {age}: extracted {sum(1 for k in results if k[0] == age)} formula entries')

    return results


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bmiforecast_results(forecast_formulas=None):
    """Load cv_metrics_summary per age and inject formula/complexity from forecast_formulas.
    Also adds a 'DeepPySR (Interpretable)' row per age (same CV metrics, interpretable formula).
    """
    rows = []
    for age in FORECAST_AGES:
        age_dir = os.path.join(FORECAST_RESULTS_DIR, f'age_{age}')
        csv_path = os.path.join(age_dir, 'cv_metrics_summary.csv')
        if not os.path.exists(csv_path):
            print(f'  Missing: {csv_path}')
            continue
        df = pd.read_csv(csv_path)
        deeppysr_row = None
        for _, row in df.iterrows():
            key = (str(row['family']), str(row['model']))
            display = FAMILY_MODEL_TO_DISPLAY.get(key, f"{row['family']}_{row['model']}")
            entry = forecast_formulas.get((age, display), {}) if forecast_formulas else {}
            # For formula-based models use formula-evaluated metrics so DeepPySR and
            # DeepPySR (Interpretable) are compared on the same evaluation basis.
            use_formula_metrics = bool(entry.get('formula')) and display in FORMULA_MODELS.values()
            rec = {
                'age': age,
                'display_model': display,
                'r2': entry.get('r2', row['cv_r2']) if use_formula_metrics else row['cv_r2'],
                'rmse': entry.get('rmse', row['cv_rmse']) if use_formula_metrics else row['cv_rmse'],
                'mae': entry.get('mae', row['cv_mae']) if use_formula_metrics else row['cv_mae'],
                'source': 'BMI Forecast',
                'formula': entry.get('formula', ''),
                'complexity': entry.get('complexity', np.nan),
            }
            rows.append(rec)
            if display == 'DeepPySR':
                deeppysr_row = rec

        # Add interpretable DeepPySR row — use formula-evaluated metrics, not CV summary
        if forecast_formulas is not None:
            interp_entry = forecast_formulas.get((age, 'DeepPySR (Interpretable)'), {})
            if interp_entry.get('formula'):
                rows.append({
                    'age': age,
                    'display_model': 'DeepPySR (Interpretable)',
                    'r2': max(interp_entry.get('r2', 0.0), 0.0),
                    'rmse': interp_entry.get('rmse', np.nan),
                    'mae': interp_entry.get('mae', np.nan),
                    'source': 'BMI Forecast',
                    'formula': interp_entry.get('formula', ''),
                    'complexity': interp_entry.get('complexity', np.nan),
                })

    df_forecast = pd.DataFrame(rows)
    df_forecast['r2'] = df_forecast['r2'].clip(lower=0)
    return df_forecast


def load_bmi_results():
    if not os.path.exists(BMI_BEST_MODELS_CSV):
        raise FileNotFoundError(f'BMI best models CSV not found: {BMI_BEST_MODELS_CSV}')
    df = pd.read_csv(BMI_BEST_MODELS_CSV)
    df = df[df['age'].isin(FORECAST_AGES)].copy()
    df['r2'] = df['r2'].clip(lower=0)
    # Normalize display_model: Best DeepPySR -> DeepPySR for forecast comparison
    df['display_model'] = df['display_model'].replace({'Best DeepPySR': 'DeepPySR',
                                                        'Interpretable DeepPySR': 'DeepPySR (Interpretable)'})

    rows_age_specific, rows_longitudinal = [], []
    for _, row in df.iterrows():
        base = {
            'age': row['age'],
            'display_model': row['display_model'],
            'r2': row['r2'],
            'rmse': row['rmse'],
            'mae': row['mae'],
            'formula': row.get('formula', ''),
            'complexity': row.get('complexity', np.nan),
        }
        if row['type'] == 'age-specific':
            base['source'] = 'BMI Age-Specific'
            rows_age_specific.append(base)
        elif row['type'] == 'longitudinal':
            base['source'] = 'BMI Longitudinal'
            rows_longitudinal.append(base)

    return pd.DataFrame(rows_age_specific), pd.DataFrame(rows_longitudinal)


# ── CSV output ────────────────────────────────────────────────────────────────

def save_combined_csv(df_forecast, df_age_specific, df_longitudinal, out_dir):
    import re as _re
    combined = pd.concat([df_forecast, df_age_specific, df_longitudinal], ignore_index=True)
    # Replace bare y{N}bmi references with y{N}bmi_pysr_pred in PySR BMI Forecast formulas
    mask = (combined['source'] == 'BMI Forecast') & (combined['display_model'] == 'PySR')
    combined.loc[mask, 'formula'] = combined.loc[mask, 'formula'].apply(
        lambda f: _re.sub(r'\by(\d+)bmi\b(?!_)', lambda m: f'y{m.group(1)}bmi_pysr_pred', str(f))
        if pd.notna(f) else f
    )
    cols = ['source', 'age', 'display_model', 'r2', 'rmse', 'mae', 'complexity', 'formula']
    combined = combined[[c for c in cols if c in combined.columns]]
    csv_path = os.path.join(out_dir, 'bmiforecast_comparison_metrics.csv')
    combined.to_csv(csv_path, index=False)
    print(f'Saved combined metrics CSV: {csv_path}')
    return combined


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _build_palette(models):
    palette = sns.color_palette('tab10', n_colors=max(len(models), 1))
    return dict(zip(sorted(models), palette))


def _add_complexity_subplot(ax, df, model_colors, symbolic_models, title_prefix):
    """Plot complexity vs age for symbolic models on the given axes."""
    sym_df = df[df['display_model'].isin(symbolic_models) & df['complexity'].notna()].copy()
    for model in sorted(sym_df['display_model'].unique()):
        m_df = sym_df[sym_df['display_model'] == model].sort_values('age')
        ax.plot(m_df['age'], m_df['complexity'], color=model_colors[model],
                linewidth=2.5, marker='o', markersize=7)
    ax.axhline(INTERP_COMPLEXITY_THRESHOLD, color='gray', linestyle='--', linewidth=1.5,
               label=f'Threshold ({INTERP_COMPLEXITY_THRESHOLD})')
    ax.set_title(f'{title_prefix}: Complexity vs Age', fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel('Age', fontsize=13)
    ax.set_ylabel('Complexity', fontsize=13)
    ax.set_xticks(FORECAST_AGES)
    if ax.get_legend():
        ax.get_legend().remove()


def plot_forecast_metrics_vs_age(df_forecast, out_dir):
    """1×4 subplots: R2 / RMSE / MAE / Complexity vs age for bmiforecast models."""
    df = df_forecast[df_forecast['display_model'].isin(MODELS_TO_PLOT_INTERP)].copy()
    metrics = ['r2', 'rmse', 'mae']
    symbolic_models = ['DeepPySR', 'DeepPySR (Interpretable)', 'PySR', 'KANSym']

    model_colors = _build_palette(df['display_model'].unique())
    models = sorted(model_colors)

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    plt.rcParams.update({'font.size': 13})

    for col, metric in enumerate(metrics):
        ax = axes[col]
        sns.lineplot(data=df, x='age', y=metric, hue='display_model', ax=ax,
                     linewidth=2.5, palette=model_colors, marker='o', markersize=7)
        ax.set_title(f'BMI Forecast: {metric.upper()} vs Age', fontsize=16, fontweight='bold', pad=10)
        ax.set_xlabel('Age', fontsize=13)
        ax.set_ylabel(metric.upper(), fontsize=13)
        ax.set_xticks(FORECAST_AGES)
        if ax.get_legend():
            ax.get_legend().remove()

    _add_complexity_subplot(axes[3], df_forecast, model_colors, symbolic_models, 'BMI Forecast')

    handles = [mlines.Line2D([0], [0], color=model_colors[m], lw=2.5, marker='o', label=m)
               for m in models]
    handles.append(mlines.Line2D([0], [0], color='gray', lw=1.5, linestyle='--',
                                  label=f'Complexity threshold ({INTERP_COMPLEXITY_THRESHOLD})'))
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.91, 0.5),
               fontsize=12, frameon=True, title='Model', title_fontsize=13)
    plt.suptitle('BMI Forecast Performance vs Age', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 1.0])
    path = os.path.join(out_dir, 'bmiforecast_metrics_vs_age.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_comparison_vs_age(df_forecast, df_age_specific, df_longitudinal, out_dir):
    """3×4 grid: rows = source, cols = R2 / RMSE / MAE / Complexity."""
    sources = [
        ('BMI Forecast', df_forecast),
        ('BMI Age-Specific', df_age_specific),
        ('BMI Longitudinal', df_longitudinal),
    ]
    symbolic_models = ['DeepPySR', 'DeepPySR (Interpretable)', 'PySR', 'KANSym']
    filtered = []
    for label, df in sources:
        d = df[df['display_model'].isin(MODELS_TO_PLOT_INTERP)].copy()
        d['source'] = label
        filtered.append(d)

    all_models = sorted(set(m for d in filtered for m in d['display_model'].unique()))
    model_colors = _build_palette(all_models)
    metrics = ['r2', 'rmse', 'mae']

    fig, axes = plt.subplots(3, 4, figsize=(28, 18))
    plt.rcParams.update({'font.size': 13})

    for row_idx, (label, df) in enumerate(zip([s[0] for s in sources], filtered)):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            models_here = sorted(df['display_model'].unique())
            sns.lineplot(data=df, x='age', y=metric, hue='display_model', ax=ax,
                         linewidth=2.5, palette={m: model_colors[m] for m in models_here},
                         marker='o', markersize=7)
            ax.set_title(f'{label}: {metric.upper()} vs Age', fontsize=15, fontweight='bold', pad=10)
            ax.set_xlabel('Age', fontsize=13)
            ax.set_ylabel(metric.upper(), fontsize=13)
            ax.set_xticks(FORECAST_AGES)
            if ax.get_legend():
                ax.get_legend().remove()

        # Complexity subplot (col 3)
        _add_complexity_subplot(axes[row_idx, 3], df, model_colors, symbolic_models, label)

    handles = [mlines.Line2D([0], [0], color=model_colors[m], lw=2.5, marker='o', label=m)
               for m in all_models]
    handles.append(mlines.Line2D([0], [0], color='gray', lw=1.5, linestyle='--',
                                  label=f'Complexity threshold ({INTERP_COMPLEXITY_THRESHOLD})'))
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.91, 0.5),
               fontsize=12, frameon=True, title='Model', title_fontsize=13)
    plt.suptitle('BMI Forecast vs BMI Test (Age-Specific & Longitudinal)', fontsize=20,
                 fontweight='bold', y=1.005)
    plt.tight_layout(rect=[0, 0, 0.9, 1.0])
    path = os.path.join(out_dir, 'bmiforecast_vs_bmi_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_combined_overlay(df_forecast, df_age_specific, df_longitudinal, out_dir):
    """1×4 overlay: all sources on same axes, colour=model, linestyle=source. +complexity."""
    linestyles = {'BMI Forecast': '-', 'BMI Age-Specific': '--', 'BMI Longitudinal': ':'}
    sources_data = [
        ('BMI Forecast', df_forecast),
        ('BMI Age-Specific', df_age_specific),
        ('BMI Longitudinal', df_longitudinal),
    ]
    combined = pd.concat(
        [df[df['display_model'].isin(MODELS_TO_PLOT_INTERP)].assign(source=lbl)
         for lbl, df in sources_data], ignore_index=True)

    all_models = sorted(combined['display_model'].unique())
    model_colors = _build_palette(all_models)
    symbolic_models = ['DeepPySR', 'DeepPySR (Interpretable)', 'PySR', 'KANSym']
    metrics = ['r2', 'rmse', 'mae']

    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    plt.rcParams.update({'font.size': 13})

    for col_idx, metric in enumerate(metrics):
        ax = axes[col_idx]
        for src_label, ls in linestyles.items():
            src_df = combined[combined['source'] == src_label]
            for model in all_models:
                m_df = src_df[src_df['display_model'] == model].sort_values('age')
                if m_df.empty:
                    continue
                ax.plot(m_df['age'], m_df[metric], color=model_colors[model],
                        linestyle=ls, linewidth=2.0, marker='o', markersize=6)
        ax.set_title(f'{metric.upper()} vs Age', fontsize=15, fontweight='bold', pad=10)
        ax.set_xlabel('Age', fontsize=13)
        ax.set_ylabel(metric.upper(), fontsize=13)
        ax.set_xticks(FORECAST_AGES)

    # Complexity subplot: overlay all sources, symbolic models only
    ax_c = axes[3]
    for src_label, ls in linestyles.items():
        src_df = combined[combined['source'] == src_label]
        sym_df = src_df[src_df['display_model'].isin(symbolic_models) & src_df['complexity'].notna()]
        for model in sorted(sym_df['display_model'].unique()):
            m_df = sym_df[sym_df['display_model'] == model].sort_values('age')
            ax_c.plot(m_df['age'], m_df['complexity'], color=model_colors[model],
                      linestyle=ls, linewidth=2.0, marker='o', markersize=6)
    ax_c.axhline(INTERP_COMPLEXITY_THRESHOLD, color='gray', linestyle='--', linewidth=1.5)
    ax_c.set_title('Complexity vs Age', fontsize=15, fontweight='bold', pad=10)
    ax_c.set_xlabel('Age', fontsize=13)
    ax_c.set_ylabel('Complexity', fontsize=13)
    ax_c.set_xticks(FORECAST_AGES)

    legend_handles = [mlines.Line2D([0], [0], color=model_colors[m], lw=2.5, marker='o', label=m)
                      for m in all_models]
    legend_handles.append(mlines.Line2D([0], [0], color='white', label=''))
    for src_label, ls in linestyles.items():
        legend_handles.append(mlines.Line2D([0], [0], color='black', lw=2.5, linestyle=ls,
                                            label=src_label))
    legend_handles.append(mlines.Line2D([0], [0], color='gray', lw=1.5, linestyle='--',
                                         label=f'Complexity threshold ({INTERP_COMPLEXITY_THRESHOLD})'))
    fig.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(0.91, 0.5),
               fontsize=11, frameon=True, title='Models & Source', title_fontsize=13,
               handlelength=4.0)
    plt.suptitle('BMI Forecast vs BMI Test — Overlay Comparison', fontsize=20,
                 fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 0.9, 1.0])
    path = os.path.join(out_dir, 'bmiforecast_overlay_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_deeppysr_only(df_forecast, df_age_specific, df_longitudinal, out_dir):
    """2×4: DeepPySR (row 0) and DeepPySR Interpretable (row 1) across 3 sources.
    Cols: R2 / RMSE / MAE / Complexity.
    """
    source_styles = {
        'BMI Forecast': ('-', 'tab:blue'),
        'BMI Age-Specific': ('--', 'tab:orange'),
        'BMI Longitudinal': (':', 'tab:green'),
    }
    sources_data = [
        ('BMI Forecast', df_forecast),
        ('BMI Age-Specific', df_age_specific),
        ('BMI Longitudinal', df_longitudinal),
    ]
    variants = ['DeepPySR', 'DeepPySR (Interpretable)']
    metrics = ['r2', 'rmse', 'mae']

    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
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
            ax.set_title(f'{variant}: {metric.upper()} vs Age', fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('Age', fontsize=13)
            ax.set_ylabel(metric.upper(), fontsize=13)
            ax.set_xticks(FORECAST_AGES)

        # Complexity subplot (col 3)
        ax_c = axes[row_idx, 3]
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
        ax_c.set_xticks(FORECAST_AGES)

    handles = [mlines.Line2D([0], [0], color=source_styles[lbl][1], lw=2.5,
                              linestyle=source_styles[lbl][0], marker='o', label=lbl)
               for lbl in source_styles]
    handles.append(mlines.Line2D([0], [0], color='gray', lw=1.5, linestyle='--',
                                  label=f'Complexity threshold ({INTERP_COMPLEXITY_THRESHOLD})'))
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.91, 0.5),
               fontsize=12, frameon=True, title='Source', title_fontsize=13, handlelength=4.0)
    plt.suptitle('DeepPySR Performance: Forecast vs Age-Specific vs Longitudinal',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 0.9, 1.0])
    path = os.path.join(out_dir, 'bmiforecast_deeppysr_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ── All-model details ─────────────────────────────────────────────────────────

def save_all_model_details(out_dir):
    """Aggregate per-config metrics and best formula for every model in every age folder.

    For DeepPySR: expands each config into one row per (pareto_r2_weight, pareto_lambda)
    pair.  The best formula for each setting is evaluated on the real data to compute
    proper r2/rmse/mae.  Model name: {cfg}_r2w{r2w}_L{lamb}

    For PySR: one row per config, best formula across folds evaluated on real data.

    For baselines: r2/rmse/mae from overall_metrics.csv (no symbolic formula).

    Saves bmiforecast_all_model_details.csv with columns:
      age, family, model, r2, rmse, mae, complexity, formula
    """
    from sklearn.metrics import r2_score as _r2, mean_squared_error as _mse, mean_absolute_error as _mae

    rolling_csv = os.path.join(FORECAST_RESULTS_DIR, 'rolling_dataset.csv')
    base_csv    = os.path.join(FORECAST_RESULTS_DIR, 'base_dataset.csv')
    if not os.path.exists(rolling_csv):
        print('  rolling_dataset.csv not found; skipping all-model details')
        return pd.DataFrame()
    rolling_df = pd.read_csv(rolling_csv)
    base_df    = pd.read_csv(base_csv) if os.path.exists(base_csv) else None

    def _eval_formula(raw_formula, X, y, model_type):
        """Evaluate a raw formula on (X, y); return (r2, rmse, mae) or (nan, nan, nan)."""
        try:
            yp = evaluate_formula(raw_formula, X, model_type=model_type)
            if yp is None or len(yp) != len(y):
                return np.nan, np.nan, np.nan
            yp = np.asarray(yp, dtype=float)
            mask = np.isfinite(y) & np.isfinite(yp)
            if mask.sum() < 2:
                return np.nan, np.nan, np.nan
            yt, yp = y[mask], yp[mask]
            return float(_r2(yt, yp)), float(np.sqrt(_mse(yt, yp))), float(_mae(yt, yp))
        except Exception:
            return np.nan, np.nan, np.nan

    def _get_sub(year, model_type):
        """Return (X, y, feature_names) for known rows at this year using model-specific cols."""
        bmi_col = f'y{year}bmi'
        if bmi_col not in rolling_df.columns:
            return None, None, None
        fcols = _get_feature_cols_for_year(rolling_df, year, model_type=model_type)
        avail = [c for c in fcols if c in rolling_df.columns]
        if base_df is not None and bmi_col in base_df.columns:
            real_ids = set(base_df.loc[base_df[bmi_col].notna(), 'child_id'].values)
            sub = rolling_df[rolling_df['child_id'].isin(real_ids)].copy()
        else:
            sub = rolling_df.copy()
        sub = sub.dropna(subset=[bmi_col])
        return sub[avail], sub[bmi_col].values, avail

    rows = []

    for year, age in zip(FORECAST_YEARS, FORECAST_AGES):
        age_dir = os.path.join(FORECAST_RESULTS_DIR, f'age_{age}')
        if not os.path.exists(age_dir):
            continue

        X_dsr, y_dsr, fn_dsr = _get_sub(year, 'deeppysr')
        X_psr, y_psr, fn_psr = _get_sub(year, 'pysr')

        # ── DeepPySR configs ──────────────────────────────────────────────────
        dsr_dir = os.path.join(age_dir, 'deeppysr')
        if os.path.exists(dsr_dir) and X_dsr is not None:
            for cfg in sorted(os.listdir(dsr_dir)):
                cfg_path = os.path.join(dsr_dir, cfg)
                if not os.path.isdir(cfg_path):
                    continue
                if not os.path.exists(os.path.join(cfg_path, 'overall_metrics.csv')):
                    continue

                # Collect best formula per (r2w, lamb) across all folds
                pareto_best = {}  # (r2w, lamb) -> {'formula': str, 'complexity': float, 'fold_r2': float}
                for rel_f in sorted(glob.glob(os.path.join(cfg_path, 'relationships_fold*.csv'))):
                    try:
                        df_rel = pd.read_csv(rel_f)
                        if 'formula' not in df_rel.columns:
                            continue
                        for _, row in df_rel.iterrows():
                            r2w  = row.get('pareto_r2_weight', np.nan)
                            lamb = row.get('pareto_lambda',    np.nan)
                            key  = (r2w, lamb)
                            fold_r2 = float(row['r2']) if 'r2' in df_rel.columns else -np.inf
                            if key not in pareto_best or fold_r2 > pareto_best[key]['fold_r2']:
                                pareto_best[key] = {
                                    'formula':    str(row['formula']),
                                    'complexity': float(row['complexity']) if 'complexity' in df_rel.columns else np.nan,
                                    'fold_r2':    fold_r2,
                                }
                    except Exception:
                        continue

                for (r2w, lamb), best in sorted(pareto_best.items()):
                    r2, rmse, mae = _eval_formula(best['formula'], X_dsr, y_dsr, 'deeppysr')
                    display_formula = map_variable_names(best['formula'], fn_dsr, model_type='deeppysr')
                    rows.append({
                        'age': age, 'family': 'deeppysr',
                        'model': f'{cfg}_r2w{r2w}_L{lamb}',
                        'r2': r2, 'rmse': rmse, 'mae': mae,
                        'complexity': best['complexity'], 'formula': display_formula,
                    })

        # ── PySR configs ──────────────────────────────────────────────────────
        psr_dir = os.path.join(age_dir, 'pysr')
        if os.path.exists(psr_dir) and X_psr is not None:
            for cfg in sorted(os.listdir(psr_dir)):
                cfg_path = os.path.join(psr_dir, cfg)
                if not os.path.isdir(cfg_path):
                    continue
                if not os.path.exists(os.path.join(cfg_path, 'overall_metrics.csv')):
                    continue
                best_formula, best_complexity, best_fold_r2 = '', np.nan, -np.inf
                for form_f in sorted(glob.glob(os.path.join(cfg_path, 'formulas_fold*.csv'))):
                    try:
                        df_f = pd.read_csv(form_f)
                        if 'formula' not in df_f.columns:
                            continue
                        for _, row in df_f.iterrows():
                            fold_r2 = float(row['r2']) if 'r2' in df_f.columns else -np.inf
                            if fold_r2 > best_fold_r2:
                                best_fold_r2 = fold_r2
                                best_formula = str(row['formula'])
                                best_complexity = float(row['complexity']) if 'complexity' in df_f.columns else np.nan
                    except Exception:
                        continue
                r2, rmse, mae = _eval_formula(best_formula, X_psr, y_psr, 'pysr')
                display_formula = map_variable_names(best_formula, fn_psr, model_type='pysr') if best_formula else ''
                rows.append({
                    'age': age, 'family': 'pysr', 'model': cfg,
                    'r2': r2, 'rmse': rmse, 'mae': mae,
                    'complexity': best_complexity, 'formula': display_formula,
                })

        # ── Baselines ─────────────────────────────────────────────────────────
        bl_dir = os.path.join(age_dir, 'baselines')
        if os.path.exists(bl_dir):
            for model_name in sorted(os.listdir(bl_dir)):
                model_path = os.path.join(bl_dir, model_name)
                if not os.path.isdir(model_path):
                    continue
                metrics_path = os.path.join(model_path, 'overall_metrics.csv')
                if not os.path.exists(metrics_path):
                    continue
                m = pd.read_csv(metrics_path).iloc[0]
                # KAN has symbolic formulas
                best_formula, best_complexity, best_fold_r2 = '', np.nan, -np.inf
                X_kan, y_kan, fn_kan = _get_sub(year, model_name)
                for form_f in sorted(glob.glob(os.path.join(model_path, 'formulas_fold*.csv'))):
                    try:
                        df_f = pd.read_csv(form_f)
                        if 'formula' not in df_f.columns:
                            continue
                        for _, row in df_f.iterrows():
                            fold_r2 = float(row['r2']) if 'r2' in df_f.columns else -np.inf
                            if fold_r2 > best_fold_r2:
                                best_fold_r2 = fold_r2
                                best_formula = str(row['formula'])
                                best_complexity = float(row['complexity']) if 'complexity' in df_f.columns else np.nan
                    except Exception:
                        continue
                if best_formula and X_kan is not None:
                    r2, rmse, mae = _eval_formula(best_formula, X_kan, y_kan, 'kan')
                    display_formula = map_variable_names(best_formula, fn_kan, model_type='kan')
                else:
                    r2, rmse, mae = float(m['r2']), float(m['rmse']), float(m['mae'])
                    display_formula = best_formula
                rows.append({
                    'age': age, 'family': 'baseline', 'model': model_name,
                    'r2': r2, 'rmse': rmse, 'mae': mae,
                    'complexity': best_complexity, 'formula': display_formula,
                })

    df = pd.DataFrame(rows, columns=['age', 'family', 'model', 'r2', 'rmse', 'mae', 'complexity', 'formula'])
    df['r2'] = df['r2'].clip(lower=0)
    out_path = os.path.join(out_dir, 'bmiforecast_all_model_details.csv')
    df.to_csv(out_path, index=False)
    print(f'Saved: {out_path}  ({len(df)} rows)')
    return df


# ── Sparse trajectory plots ───────────────────────────────────────────────────

def plot_sparse_trajectories(out_dir, n_obs_filter=2, n_sample=10, seed=42, mlp_model=None):
    """Plot BMI trajectories for participants with few observed values.

    At each age:
      - If participant has observed BMI (forecast ages only):
          DeepPySR best/interpretable/PySR → formula evaluation from comparison_metrics.csv
          Baselines → predictions.csv in age_{age}/baselines/{model}/
      - Otherwise (age 8, or no observed value):
          All models → rolling_dataset y{year}bmi_{model}_pred columns
    """
    rolling_csv = os.path.join(FORECAST_RESULTS_DIR, 'rolling_dataset.csv')
    base_csv    = os.path.join(FORECAST_RESULTS_DIR, 'base_dataset.csv')
    if not os.path.exists(rolling_csv):
        print('  rolling_dataset.csv not found; skipping trajectory plot')
        return
    from bmiforecast_utils import _get_bmi_age8_feature_cols

    rolling_df = pd.read_csv(rolling_csv)
    base_df    = pd.read_csv(base_csv) if os.path.exists(base_csv) else rolling_df

    # ── Patch: add missing y{year}bmi_pysr_pred columns from best PySR formula ─
    # (run_rolling_bmiforecast.py previously looked for formulas_nocv.csv instead
    #  of formulas_foldnocv.csv, so pysr pred cols were never saved)
    _pysr_patch_needed = [
        y for y in FORECAST_YEARS if f'y{y}bmi_pysr_pred' not in rolling_df.columns
    ]
    if _pysr_patch_needed:
        _cmp = pd.read_csv(os.path.join(FORECAST_RESULTS_DIR, 'bmiforecast_comparison_metrics.csv'))
        _pysr_fc = _cmp[(_cmp['source'] == 'BMI Forecast') & (_cmp['display_model'] == 'PySR')]
        _pysr_formulas = {int(r['age']): r['formula'] for _, r in _pysr_fc.iterrows()
                         if pd.notna(r.get('formula'))}
        for _yr in _pysr_patch_needed:
            _age = ALL_AGES[ALL_YEARS.index(_yr)]
            _formula = _pysr_formulas.get(_age)
            if not _formula:
                continue
            try:
                _yp = evaluate_formula(str(_formula), rolling_df, model_type='pysr')
                if _yp is not None:
                    _pred_col = f'y{_yr}bmi_pysr_pred'
                    rolling_df[_pred_col] = rolling_df[f'y{_yr}bmi'].copy()
                    _miss = rolling_df[f'y{_yr}bmi'].isna()
                    rolling_df.loc[_miss, _pred_col] = np.array(_yp)[_miss]
                    _med = rolling_df.loc[~_miss, _pred_col].median()
                    rolling_df[_pred_col] = rolling_df[_pred_col].fillna(_med)
            except Exception:
                pass

    # ── Year 8 data from rawdata_yr8.csv ──────────────────────────────────────
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    yr8_csv = os.path.join(project_root, 'test_data', 'Health', 'bmi', 'rawdata_yr8.csv')
    yr8_full_df = (pd.read_csv(yr8_csv).drop_duplicates(subset='child_id').set_index('child_id')
                   if os.path.exists(yr8_csv) else None)
    yr8_df      = yr8_full_df[['y8bmi']] if yr8_full_df is not None else None

    # ── Load formulas (DeepPySR best, DeepPySR Interpretable, PySR) ───────────
    cmp_csv = os.path.join(FORECAST_RESULTS_DIR, 'bmiforecast_comparison_metrics.csv')
    cmp_df  = pd.read_csv(cmp_csv)
    bmi_fc  = cmp_df[cmp_df['source'] == 'BMI Forecast']
    formula_map = {(int(r['age']), r['display_model']): r['formula']
                   for _, r in bmi_fc.iterrows()
                   if pd.notna(r['formula']) and str(r['formula']) not in ('', '0.0')}

    # Age-8 formulas from bmi_best_models_metrics.csv (age-specific)
    bmi_metrics_csv = os.path.join(current_dir, '..', 'bmi', 'results_bmi_all',
                                   'bmi_best_models_metrics.csv')
    if os.path.exists(bmi_metrics_csv):
        _bm = pd.read_csv(bmi_metrics_csv)
        _age8 = _bm[(_bm['age'] == 8) & (_bm['type'] == 'age-specific')]
        for _, r in _age8.iterrows():
            if pd.notna(r.get('formula')) and str(r['formula']) not in ('', '0.0', 'nan'):
                display = r['display_model']
                display = display.replace('Best DeepPySR', 'DeepPySR').replace(
                    'Interpretable DeepPySR', 'DeepPySR (Interpretable)')
                import re as _re
                formula = _re.sub(r'\bPGS(\d+)\b', r'SUM_PGS\1', str(r['formula']))
                formula_map[(8, display)] = formula

    BASELINE_MODELS = ['ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']

    # ── Age 8: load fitted baseline models ───────────────────────────────────────
    import joblib as _jl
    import sys as _sys
    _mu_dir = os.path.abspath(os.path.join(current_dir, '..'))
    if _mu_dir not in _sys.path:
        _sys.path.insert(0, _mu_dir)
    yr8_feat_cols = _get_bmi_age8_feature_cols()
    yr8_fitted_dir = os.path.join(current_dir, '..', 'bmi', 'results_bmi_all',
                                  'age_specific', 'age_8', 'baselines', '_fitted_models')
    yr8_fitted = {}
    if mlp_model is not None:
        yr8_fitted['MLP'] = mlp_model
    for m in BASELINE_MODELS:
        if m == 'MLP':
            continue
        jpath = os.path.join(yr8_fitted_dir, f'{m}.joblib')
        if os.path.exists(jpath):
            try:
                yr8_fitted[m] = _jl.load(jpath)
            except Exception:
                pass

    # ── Pre-load baseline predictions.csv per forecast age ────────────────────
    # {age: {model_name: DataFrame indexed by id}}
    baseline_preds = {}
    for age in FORECAST_AGES:
        baseline_preds[age] = {}
        for m in BASELINE_MODELS:
            p = os.path.join(FORECAST_RESULTS_DIR, f'age_{age}', 'baselines', m, 'predictions.csv')
            if os.path.exists(p):
                try:
                    baseline_preds[age][m] = pd.read_csv(p).set_index('id')
                except Exception:
                    pass

    # ── Find participants with sparse observations ─────────────────────────────
    # Year 8 true BMI from rawdata; years 10+ from base_dataset
    base_idx   = base_df.set_index('child_id')
    true_cols_base = [f'y{y}bmi' for y in ALL_YEARS if y != 8 and f'y{y}bmi' in base_idx.columns]
    obs_counts = base_idx[true_cols_base].notna().sum(axis=1)
    if yr8_df is not None:
        yr8_obs = yr8_df['y8bmi'].notna().astype(int).reindex(obs_counts.index, fill_value=0)
        obs_counts = obs_counts.add(yr8_obs, fill_value=0)
    ids = obs_counts[obs_counts == n_obs_filter].index.tolist() if n_obs_filter is not None \
          else obs_counts.index.tolist()

    if not ids:
        print(f'  No participants found with exactly {n_obs_filter} observed BMIs.')
        return

    # Sort by mean observed BMI descending to prioritise overweight/obese cases
    def mean_obs_bmi(cid):
        vals = []
        if yr8_df is not None and cid in yr8_df.index:
            v = yr8_df.loc[cid, 'y8bmi']
            if pd.notna(v):
                vals.append(float(v))
        if cid in base_idx.index:
            for col in true_cols_base:
                if col in base_idx.columns:
                    v = base_idx.loc[cid, col]
                    if pd.notna(v):
                        vals.append(float(v))
        return np.mean(vals) if vals else 0.0

    # Exclude participants whose any BMI prediction in rolling_dataset is below 10
    all_pred_cols = [c for c in rolling_df.columns if c.endswith('_pred') and 'bmi' in c]
    rolling_idx = rolling_df.set_index('child_id')
    def _has_low_pred(cid):
        if not all_pred_cols or cid not in rolling_idx.index:
            return False
        vals = rolling_idx.loc[cid, all_pred_cols].dropna()
        return bool((vals < 10).any())

    ids_valid = [cid for cid in ids if not _has_low_pred(cid)]

    # Also pre-filter using formula models: exclude ids where any formula prediction < 10
    _formula_map_precheck = {}
    _cmp_pre = os.path.join(FORECAST_RESULTS_DIR, 'bmiforecast_comparison_metrics.csv')
    if os.path.exists(_cmp_pre):
        _cmp_df_pre = pd.read_csv(_cmp_pre)
        _bmi_fc_pre = _cmp_df_pre[_cmp_df_pre['source'] == 'BMI Forecast']
        for _, _r in _bmi_fc_pre.iterrows():
            if pd.notna(_r.get('formula')) and str(_r['formula']) not in ('', '0.0', 'nan'):
                _formula_map_precheck[(int(_r['age']), _r['display_model'])] = _r['formula']
    _rolling_candidates = rolling_df[rolling_df['child_id'].isin(ids_valid)].set_index('child_id')
    _bad_ids = set()
    for (_age, _label), _formula in _formula_map_precheck.items():
        try:
            _yp = evaluate_formula(str(_formula), _rolling_candidates.reset_index(), model_type='pysr'
                                   if 'PySR' in _label else 'deeppysr')
            if _yp is not None:
                for _cid, _val in zip(_rolling_candidates.index, _yp):
                    if np.isfinite(float(_val)) and float(_val) < 10:
                        _bad_ids.add(_cid)
        except Exception:
            pass
    ids_valid = [cid for cid in ids_valid if cid not in _bad_ids]

    ids_sorted = sorted(ids_valid, key=mean_obs_bmi, reverse=True)

    # Select extreme cases: highest mean observed BMI
    sampled_ids = ids_sorted[:n_sample]
    print(f'  {len(ids)} participants with {n_obs_filter} obs; '
          f'plotting top-{len(sampled_ids)} extreme (highest mean BMI)')

    rolling_sub = rolling_df[rolling_df['child_id'].isin(sampled_ids)].set_index('child_id')
    base_sub    = base_idx[base_idx.index.isin(sampled_ids)]
    yr8_sub     = yr8_df[yr8_df.index.isin(sampled_ids)]     if yr8_df is not None else None

    # ── Observed BMI presence per (cid, age) ──────────────────────────────────
    # Map year→age for lookup
    year_to_age = dict(zip(ALL_YEARS, ALL_AGES))
    def has_obs(cid, year):
        col = f'y{year}bmi'
        if cid not in base_sub.index or col not in base_sub.columns:
            return False
        return pd.notna(base_sub.loc[cid, col])

    # ── Rolling-dataset column suffixes for fallback ───────────────────────────
    ROLLING_SUFFIXES = {
        'DeepPySR':                 ['deeppysr'],
        'DeepPySR (Interpretable)': ['deeppysr'],
        'PySR':                     ['pysr'],
        'ElasticNet':               ['ElasticNet'],
        'ExtraTrees':               ['ExtraTrees'],
        'MLP':                      ['MLP'],
        'RandomForest':             ['RandomForest'],
        'XGBoost':                  ['XGBoost'],
    }
    FORMULA_LABELS = {
        'DeepPySR':                 ('DeepPySR',              'deeppysr'),
        'DeepPySR (Interpretable)': ('DeepPySR (Interpretable)', 'deeppysr'),
        'PySR':                     ('PySR',                  'pysr'),
    }
    ALL_LABELS = list(ROLLING_SUFFIXES.keys())

    # Initialize: {label: {cid: {age: pred}}}
    all_preds = {m: {cid: {} for cid in sampled_ids} for m in ALL_LABELS}

    # ── Year 8: predict for all sampled participants using fitted models ────────
    # Use base_dataset (already imputed) as feature source; fill any missing
    # feature columns (e.g. PGS columns absent from base_dataset) with 0.
    if yr8_feat_cols:
        base_sub_full = base_df.set_index('child_id').reindex(sampled_ids)
        X8_data = {}
        for col in yr8_feat_cols:
            if col in base_sub_full.columns:
                X8_data[col] = base_sub_full[col].values
            elif f'SUM_{col}' in base_sub_full.columns:
                X8_data[col] = base_sub_full[f'SUM_{col}'].values
            else:
                X8_data[col] = 0.0
        X8 = pd.DataFrame(X8_data).fillna(pd.DataFrame(X8_data).median())
        for m in BASELINE_MODELS:
            model = yr8_fitted.get(m)
            if model is None:
                continue
            try:
                preds = model.predict(X8.values)
                for cid, val in zip(sampled_ids, preds):
                    all_preds[m][cid][8] = float(val)
            except Exception:
                pass

        # Age-8 formula models (DeepPySR / PySR) using age-specific formulas
        X8_rolling = rolling_sub.reindex(sampled_ids)
        for label, (cmp_key, model_type) in FORMULA_LABELS.items():
            formula = formula_map.get((8, cmp_key))
            if not formula:
                continue
            try:
                yp = evaluate_formula(str(formula), X8_rolling, model_type=model_type)
                if yp is not None:
                    for cid, val in zip(sampled_ids, yp):
                        fval = float(val)
                        all_preds[label][cid][8] = fval if np.isfinite(fval) else np.nan
            except Exception:
                pass

    for year, age in zip(ALL_YEARS, ALL_AGES):

        for cid in sampled_ids:
            # Rolling fallback for participants without observed values at this age
            if not has_obs(cid, year):
                for label, suffixes in ROLLING_SUFFIXES.items():
                    for sfx in suffixes:
                        col = f'y{year}bmi_{sfx}_pred'
                        if col in rolling_sub.columns and cid in rolling_sub.index:
                            val = rolling_sub.loc[cid, col]
                            if pd.notna(val):
                                all_preds[label][cid][age] = float(val)
                            break

        # Formula evaluation — batch per age for ALL participants (formulas use only prior features)
        if year in FORECAST_YEARS:
            X_all = rolling_sub.reindex(sampled_ids)

            for label, (cmp_key, model_type) in FORMULA_LABELS.items():
                formula = formula_map.get((age, cmp_key))
                if not formula:
                    continue
                try:
                    yp = evaluate_formula(str(formula), X_all, model_type=model_type)
                    if yp is not None:
                        for cid, val in zip(sampled_ids, yp):
                            fval = float(val)
                            all_preds[label][cid][age] = fval if np.isfinite(fval) else np.nan
                except Exception:
                    pass

            # Baseline models — predictions.csv only has participants with observed values
            obs_ids = [cid for cid in sampled_ids if has_obs(cid, year)]
            for m in BASELINE_MODELS:
                df_p = baseline_preds.get(age, {}).get(m)
                if df_p is None:
                    continue
                for cid in obs_ids:
                    if cid in df_p.index:
                        all_preds[m][cid][age] = float(df_p.loc[cid, 'y_pred'])

    # ── Plot ───────────────────────────────────────────────────────────────────
    model_styles = {
        'DeepPySR':                 ('tab:blue',   '-',  2.2),
        'DeepPySR (Interpretable)': ('tab:cyan',   '--', 2.0),
        'PySR':                     ('tab:orange', '-',  1.5),
        'ElasticNet':               ('tab:green',  '-',  1.2),
        'ExtraTrees':               ('tab:red',    '-',  1.2),
        'MLP':                      ('tab:purple', '-',  1.2),
        'RandomForest':             ('tab:brown',  '-',  1.2),
        'XGBoost':                  ('tab:pink',   '-',  1.2),
    }

    ncols = 5
    nrows = int(np.ceil(len(sampled_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.8), squeeze=False)
    plt.rcParams.update({'font.size': 9})

    for i, cid in enumerate(sampled_ids):
        ax = axes[i // ncols][i % ncols]

        for label, (color, ls, lw) in model_styles.items():
            pts = all_preds[label][cid]
            if not pts:
                continue
            ages_p = sorted(pts)
            vals_p = [pts[a] for a in ages_p]
            ax.plot(ages_p, vals_p, color=color, linestyle=ls, linewidth=lw, label=label)

        # True observed BMI dots — age 8 from rawdata, ages 10+ from base_dataset
        if yr8_sub is not None and cid in yr8_sub.index:
            val = yr8_sub.loc[cid, 'y8bmi']
            if pd.notna(val):
                ax.scatter(8, float(val), color='black', s=60, zorder=6)
        if cid in base_sub.index:
            for col, yr in zip(true_cols_base, [y for y in ALL_YEARS if y != 8]):
                if col in base_sub.columns:
                    val = base_sub.loc[cid, col]
                    if pd.notna(val):
                        ax.scatter(ALL_AGES[ALL_YEARS.index(yr)], float(val),
                                   color='black', s=60, zorder=6)

        ax.axhspan(18.5, 25, color='lightgreen', alpha=0.15, zorder=0)
        ax.axhspan(25,   30, color='gold',       alpha=0.15, zorder=0)
        ax.axhspan(30,   35, color='orange',     alpha=0.15, zorder=0)
        ax.axhspan(35,   40, color='tomato',     alpha=0.15, zorder=0)
        ax.axhspan(40,   60, color='firebrick',  alpha=0.15, zorder=0)
        ax.set_title(f'ID {cid}', fontsize=8)
        ax.set_xticks(ALL_AGES)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Age', fontsize=8)
        ax.set_ylabel('BMI', fontsize=8)

    for j in range(len(sampled_ids), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    # Legend — only include models that have any data
    present = {lbl for lbl in ALL_LABELS
               if any(all_preds[lbl][cid] for cid in sampled_ids)}
    handles = [mlines.Line2D([0], [0], color=c, lw=lw, linestyle=ls, label=lbl)
               for lbl, (c, ls, lw) in model_styles.items() if lbl in present]
    handles.append(mlines.Line2D([0], [0], marker='o', color='black', lw=0,
                                  markersize=7, label='Observed BMI'))
    handles.append(mpatches.Patch(facecolor='lightgreen', alpha=0.5, label='Normal (18.5–25)'))
    handles.append(mpatches.Patch(facecolor='gold',       alpha=0.5, label='Overweight (25–30)'))
    handles.append(mpatches.Patch(facecolor='orange',     alpha=0.5, label='Obese I (30–35)'))
    handles.append(mpatches.Patch(facecolor='tomato',     alpha=0.5, label='Obese II (35–40)'))
    handles.append(mpatches.Patch(facecolor='firebrick',  alpha=0.5, label='Obese III (≥40)'))
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.91, 0.5),
               fontsize=9, frameon=True, title='Model', title_fontsize=10)

    n_label = f'{n_obs_filter}_obs' if n_obs_filter is not None else 'all'
    plt.suptitle(
        f'BMI Trajectories — Participants with {n_obs_filter} Observed Values  '
        f'(● = observed)',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 0.9, 1.0])
    path = os.path.join(out_dir, f'bmiforecast_trajectories_{n_label}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    out_dir = FORECAST_RESULTS_DIR

    # Load MLP.joblib BEFORE any evaluate_formula call (Julia init causes segfault if torch
    # is unpickled after juliacall is already running)
    import joblib as _jl
    import sys as _sys
    _mu_dir = os.path.abspath(os.path.join(current_dir, '..'))
    if _mu_dir not in _sys.path:
        _sys.path.insert(0, _mu_dir)
    _mlp_jpath = os.path.join(current_dir, '..', 'bmi', 'results_bmi_all',
                              'age_specific', 'age_8', 'baselines', '_fitted_models', 'MLP.joblib')
    _mlp_model = None
    if os.path.exists(_mlp_jpath):
        try:
            _mlp_model = _jl.load(_mlp_jpath)
        except Exception as _e:
            print(f'  WARNING: could not load MLP.joblib: {_e}')

    print('=== Extracting bmiforecast formulas ===')
    forecast_formulas = extract_bmiforecast_formulas()

    print('=== Loading bmiforecast results ===')
    df_forecast = load_bmiforecast_results(forecast_formulas)
    print(f'  {len(df_forecast)} rows')

    print('=== Loading bmi test results ===')
    df_age_specific, df_longitudinal = load_bmi_results()
    print(f'  age-specific: {len(df_age_specific)} rows, longitudinal: {len(df_longitudinal)} rows')

    print('=== Saving combined CSV ===')
    save_combined_csv(df_forecast, df_age_specific, df_longitudinal, out_dir)

    print('=== Saving all model details ===')
    save_all_model_details(out_dir)

    print('=== Plotting bmiforecast metrics vs age ===')
    plot_forecast_metrics_vs_age(df_forecast, out_dir)

    print('=== Plotting 3-row comparison ===')
    plot_comparison_vs_age(df_forecast, df_age_specific, df_longitudinal, out_dir)

    print('=== Plotting overlay comparison ===')
    plot_combined_overlay(df_forecast, df_age_specific, df_longitudinal, out_dir)

    print('=== Plotting DeepPySR-only comparison ===')
    plot_deeppysr_only(df_forecast, df_age_specific, df_longitudinal, out_dir)

    print('=== Plotting sparse trajectories ===')
    for n in [2, 3, 4, 5]:
        plot_sparse_trajectories(out_dir, n_obs_filter=n, mlp_model=_mlp_model)

    print('\nDone.')
