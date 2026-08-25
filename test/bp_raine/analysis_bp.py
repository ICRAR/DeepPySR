"""Aggregate regression results for the blood-pressure RAINE tests (sys_bp,
dia_bp) across all input-type variants under results_bp/.

Unlike analysis_insulin_variant.py, the r2/rmse/mae/pearson_r columns are NOT
leak-free CV -- they're read straight off each leaf's predictions.csv
(y_true/y_pred columns), matching how the baseline models are already
scored. For deeppysr/pysr, "complexity" is a single representative number
per (variant config) leaf:
  - deeppysr: complexity of the highest-r2 row in relationships.csv (the
    pareto candidate the SR search itself ranked best).
  - pysr: mean calculate_complexity(formula) across formulas_fold*.csv,
    since those files carry no r2/score column to rank by.

best_formula/interp_formula ARE evaluated on the full dataset (all rows, in-
sample) across every candidate formula from every fold's
relationships_fold*.csv / formulas_fold*.csv in that leaf:
  - best_formula: highest full-dataset r2 among all candidates.
  - interp_formula: highest full-dataset r2 among candidates with
    calculate_complexity() < INTERP_MAX_COMPLEXITY (empty if none qualify).
Each fold's formulas were fit on that fold's own (scaler) view of the data,
so before evaluating we reconstruct that exact view -- same
KFold(5, shuffle=True, random_state=42) split as eval_utils.run_cv, and the
StandardScaler fit on that fold's train rows (deeppysr only, and only for
variants whose test_deeppysr_bp_*.py call omits scaler=False -- see
_deeppysr_uses_scaler) -- then apply that same transform to the full dataset
before plugging into the formula.

Produces, under this script's directory:
  bp_aggregated_results.csv   -- every (target, age, input_type, model,
                                  config) row found on disk.
  bp_best_models.csv          -- best (max f1_macro, clinical-bin
                                  classification) row per
                                  (target, age, input_type, model),
                                  collapsing across config.
  bp_models_vs_age.png        -- 2 subplots (targets) x 3 metric rows
                                  (r2, pearson_r, f1_macro), lines=model,
                                  x=age, best f1_macro collapsed across
                                  input_type.
  bp_input_types_vs_age.png   -- 2 subplots (targets) x 3 metric rows,
                                  lines=input_type, x=age, best f1_macro
                                  collapsed across model.

Also computes a per-model feature-importance/sensitivity comparison at every
(target, input_type, age) combo, mirroring
deeppysr_paper/codes/feature_importance_comparison.py's build_importance_table
but scoped to that single combo instead of a whole dataset:
  - ElasticNet/ExtraTrees/RandomForest/XGBoost: that combo's own winning
    feature_importance.csv (the best_df-selected config).
  - MLP: SHAP importance from a fresh refit of the paper's MLP architecture
    (common.mlp_shap_importance).
  - PySR: permutation sensitivity (common.formula_sensitivity) of the
    family's best formula.
  - DeepPySR: permutation sensitivity of the family's interpretable formula.
All vectors normalised to sum to 100 within their own model. Saved under
each combo's own results_bp_<input_type>/age_<age>_<target>/ directory:
  bp_sensitivity.csv   -- long format: variable, model, pct
  bp_sensitivity.png   -- heatmap, top features x models present

"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

from analysis_v1_utils import calculate_metrics, calculate_complexity, evaluate_formula, map_variable_names
from data_utils import load_data_PGS_only, load_data_keepto5, load_data_PGSto5, load_data_recent

sys.path.append(os.path.join(current_dir, "..", "..", "deeppysr_paper", "codes"))
from common import formula_sensitivity

RESULTS_BASE_DIR = os.path.join(current_dir, "results_bp")

TARGETS = ["sys_bp", "dia_bp"]
AGES = [10, 14, 17, 20, 22]
FS_SUBFOLDERS = ["all_features"]
FTSL_LABELS = {"all_features": "all_feature"}

INPUT_TYPES = ["PGS", "to5", "PGSto5", "recent"]
BASELINE_MODELS = ["ElasticNet", "ExtraTrees", "KAN", "MLP", "RandomForest", "XGBoost"]

# load_fn per input_type -- mirrors the _LOAD_FN dicts in
# test_baselines_pysr_bp_to5.py / test_deeppysr_bp_to5.py (PGS, to5, PGSto5)
# and test_baselines_pysr_bp_recent.py / test_deeppysr_bp_recent.py (recent).
INPUT_TYPE_LOADERS = {
    'PGS':    load_data_PGS_only,
    'to5':    load_data_keepto5,
    'PGSto5': load_data_PGSto5,
    'recent': load_data_recent,
}

N_SPLITS = 5
RANDOM_STATE = 42
INTERP_MAX_COMPLEXITY = 25

# ── Clinical binning for classification-style metrics ────────────────────────
# RAINE sys_bp/dia_bp are raw sphygmomanometer readings in mmHg (confirmed
# from the raw data itself via data_utils.load_data_recent: sys_bp ranges
# ~79-163 across ages 10-22 with means climbing 107->123, dia_bp ~36-119 with
# means climbing 56.6->68.7 -- consistent with mmHg, not kPa or any other
# unit). Cutoffs below are the AHA/ACC 2017 adult blood-pressure categories
# (Normal / Elevated / Stage 1 / Stage 2 / Crisis), collapsed to 3 tiers per
# target the same way analysis_lipids.py collapses NCEP ATP III's 4-5 tiers
# to 3:
#   sys_bp : <120 normal        | 120-139 elevated/stage-1 | >=140 stage-2/crisis (high)
#   dia_bp : <80  normal        | 80-89  stage-1            | >=90  stage-2/crisis (high)
# (AHA's "Elevated" category is defined purely by systolic 120-129 with
# diastolic <80, i.e. it doesn't add a distinct diastolic tier -- so dia_bp's
# 3 tiers are already AHA's Normal/Stage-1/Stage-2+Crisis with no collapsing
# needed.) No pediatric (age/sex/height-percentile) norms are applied even at
# ages 10/14 -- same simplification analysis_lipids.py makes for its own
# adolescent ages, and for the same reason: this f1_macro is a rough
# clinical-relevance signal for model comparison, not a diagnostic label.
# The "high" tier is naturally small at younger ages (real, not a binning
# artifact) and shows up as noisier F1_macro there.
BP_BIN_EDGES = {
    'sys_bp': (120, 140),
    'dia_bp': (80, 90),
}
BP_CLASS_LABELS = {
    'sys_bp': ['Normal', 'Elevated', 'High'],
    'dia_bp': ['Normal', 'Elevated', 'High'],
}


def _bin_bp(target, values):
    """Values (mmHg) -> class index 0/1/2 per BP_BIN_EDGES[target]."""
    lo, hi = BP_BIN_EDGES[target]
    return np.digitize(np.asarray(values, dtype=float), [lo, hi])


def _bp_f1_macro(target, y_true, y_pred):
    """Macro-F1 of the 3-class clinical bins, applying the same target-
    specific cutoffs to both y_true and y_pred (so a regression model is
    scored on whether its continuous prediction lands in the right clinical
    category, not on a separately-fit classifier)."""
    from sklearn.metrics import f1_score
    y_true_bin = _bin_bp(target, y_true)
    y_pred_bin = _bin_bp(target, y_pred)
    return float(f1_score(y_true_bin, y_pred_bin, average='macro', labels=[0, 1, 2], zero_division=0))


def _variant_dir(input_type):
    return os.path.join(RESULTS_BASE_DIR, f"results_bp_{input_type}")


def _deeppysr_uses_scaler(input_type):
    """test_deeppysr_bp_recent.py passes scaler=False explicitly; the
    to5/PGS/PGSto5 script test_deeppysr_bp_to5.py omits scaler=..., so
    eval_utils.run_cv's default (True) applies there. PySR always runs with
    scaler=False regardless of input_type."""
    return input_type != 'recent'


def _process_baselines(age_path, target, age, input_type, rows):
    baselines_dir = os.path.join(age_path, "baselines")
    if not os.path.exists(baselines_dir):
        return
    for model_name in sorted(os.listdir(baselines_dir)):
        model_dir = os.path.join(baselines_dir, model_name)
        if not os.path.isdir(model_dir) or model_name not in BASELINE_MODELS:
            continue
        for fs in FS_SUBFOLDERS:
            pred_file = os.path.join(model_dir, fs, "predictions.csv")
            if not os.path.exists(pred_file):
                continue
            df_pred = pd.read_csv(pred_file)
            r2, rmse, mae, pearson_r = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
            f1_macro = _bp_f1_macro(target, df_pred['y_true'], df_pred['y_pred'])
            rows.append({
                'target': target, 'age': age, 'input_type': input_type, 'model': model_name,
                'ftsl': FTSL_LABELS[fs], 'config': '', 'r2': r2, 'rmse': rmse, 'mae': mae,
                'pearson_r': pearson_r, 'f1_macro': f1_macro, 'complexity': np.nan,
                'best_formula': '', 'best_formula_r2': np.nan, 'best_formula_complexity': np.nan,
                'interp_formula': '', 'interp_formula_r2': np.nan, 'interp_formula_complexity': np.nan,
                'source_path': os.path.join(model_dir, fs),
            })


def _deeppysr_complexity(config_fs_dir):
    """Complexity of the highest-r2 row in relationships.csv (falls back to
    concatenating relationships_fold*.csv if the combined file is absent)."""
    rel_file = os.path.join(config_fs_dir, "relationships.csv")
    if not os.path.exists(rel_file):
        fold_files = sorted(f for f in os.listdir(config_fs_dir)
                             if f.startswith("relationships_fold") and f.endswith(".csv"))
        if not fold_files:
            return np.nan
        dfs = [pd.read_csv(os.path.join(config_fs_dir, f)) for f in fold_files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(rel_file)
    if df.empty or 'r2' not in df.columns or 'complexity' not in df.columns:
        return np.nan
    return float(df.loc[df['r2'].idxmax(), 'complexity'])


def _pysr_complexity(config_fs_dir):
    """Mean calculate_complexity(formula) across formulas_fold*.csv (no
    r2/score column exists there to rank candidates by)."""
    fold_files = sorted(f for f in os.listdir(config_fs_dir)
                         if f.startswith("formulas_fold") and f.endswith(".csv"))
    if not fold_files:
        return np.nan
    complexities = []
    for f in fold_files:
        df = pd.read_csv(os.path.join(config_fs_dir, f))
        if 'formula' not in df.columns:
            continue
        complexities.extend(calculate_complexity(fm) for fm in df['formula'])
    return float(np.mean(complexities)) if complexities else np.nan


def _fold_transform(X_all_raw, train_idx, use_scaler, cols):
    """X_all_raw (n_rows, n_cols) transformed exactly as that fold's model
    would have seen its inputs (see eval_utils.run_cv), applied to every row
    (not just that fold's own train/test rows) so the resulting formula can
    be evaluated in-sample on the whole dataset."""
    X_all = X_all_raw
    if use_scaler:
        sc = StandardScaler().fit(X_all_raw[train_idx])
        X_all = sc.transform(X_all)
    return pd.DataFrame(X_all, columns=cols)


def _build_fold_transforms(X_full, use_scaler):
    """5 DataFrames (one per CV fold), each holding X_full's rows exactly as
    that fold's model would have seen them post scaler -- reconstructed with
    the same KFold(5, shuffle=True, random_state=42) used by
    eval_utils.run_cv (see cv_kwargs in test_*_bp_*.py; no groups or
    stratify_by are passed, so plain KFold applies)."""
    cols = list(X_full.columns)
    X_all_raw = X_full.values.astype(float)
    splits = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X_full))
    return [_fold_transform(X_all_raw, train_idx, use_scaler, cols)
            for train_idx, _ in splits]


def _best_and_interp_formula(config_fs_dir, family, fold_transforms, y_values, r2_cache, cache_key):
    """(best_formula, best_formula_r2, best_formula_complexity,
    interp_formula, interp_formula_r2, interp_formula_complexity).
    Formulas are mapped back to real feature names, or "" (r2/complexity NaN)
    if no candidate formula was found. Scans every candidate formula from
    every fold's relationships_fold*.csv/formulas_fold*.csv in this leaf,
    evaluated in-sample on the full dataset via that fold's reconstructed
    transform (fold_transforms[fold_idx]). The returned r2 is exactly the
    full-dataset r2 that formula was selected by, not re-derived from
    anything else; complexity is calculate_complexity() on that same
    formula string (the same measure used to gate the interp_formula pool),
    not the "complexity" column elsewhere in the row (highest-r2-row of
    relationships.csv / mean pysr complexity)."""
    prefix = 'relationships_fold' if family == 'deeppysr' else 'formulas_fold'
    model_type = family
    best = (-np.inf, "", [], np.nan)
    best_interp = (-np.inf, "", [], np.nan)

    for fold_idx in range(N_SPLITS):
        fold_file = os.path.join(config_fs_dir, f"{prefix}{fold_idx}.csv")
        if not os.path.exists(fold_file):
            continue
        df = pd.read_csv(fold_file)
        if 'formula' not in df.columns or df.empty:
            continue
        X_eval = fold_transforms[fold_idx]
        feature_names = list(X_eval.columns)

        for formula in df['formula'].astype(str).unique():
            if not formula or formula.lower() == 'nan':
                continue
            key = (*cache_key, fold_idx, formula)
            if key not in r2_cache:
                y_pred = evaluate_formula(formula, X_eval, model_type=model_type)
                r2, _, _, _ = calculate_metrics(y_values, y_pred)
                r2_cache[key] = r2
            r2 = r2_cache[key]
            complexity = calculate_complexity(formula)

            if r2 > best[0]:
                best = (r2, formula, feature_names, complexity)
            if complexity < INTERP_MAX_COMPLEXITY and r2 > best_interp[0]:
                best_interp = (r2, formula, feature_names, complexity)

    best_formula = map_variable_names(best[1], best[2], model_type=model_type) if best[1] else ""
    best_formula_r2 = best[0] if best[1] else np.nan
    best_formula_complexity = best[3]
    interp_formula = map_variable_names(best_interp[1], best_interp[2], model_type=model_type) if best_interp[1] else ""
    interp_formula_r2 = best_interp[0] if best_interp[1] else np.nan
    interp_formula_complexity = best_interp[3]
    return (best_formula, best_formula_r2, best_formula_complexity,
            interp_formula, interp_formula_r2, interp_formula_complexity)


def _process_sr_family(age_path, target, age, input_type, family, complexity_fn,
                        X_full, y_full, transforms_cache, formula_r2_cache, rows):
    family_dir = os.path.join(age_path, family)
    if not os.path.exists(family_dir):
        return
    use_scaler = _deeppysr_uses_scaler(input_type) if family == 'deeppysr' else False
    y_values = y_full.values if hasattr(y_full, 'values') else np.array(y_full)

    for config in sorted(os.listdir(family_dir)):
        config_dir = os.path.join(family_dir, config)
        if not os.path.isdir(config_dir):
            continue
        for fs in FS_SUBFOLDERS:
            fs_dir = os.path.join(config_dir, fs)
            pred_file = os.path.join(fs_dir, "predictions.csv")
            if not os.path.exists(pred_file):
                continue
            df_pred = pd.read_csv(pred_file)
            r2, rmse, mae, pearson_r = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
            f1_macro = _bp_f1_macro(target, df_pred['y_true'], df_pred['y_pred'])
            complexity = complexity_fn(fs_dir)

            cache_key = (family, fs)
            if cache_key not in transforms_cache:
                transforms_cache[cache_key] = _build_fold_transforms(X_full, use_scaler)
            (best_formula, best_formula_r2, best_formula_complexity,
             interp_formula, interp_formula_r2, interp_formula_complexity) = _best_and_interp_formula(
                fs_dir, family, transforms_cache[cache_key], y_values, formula_r2_cache, cache_key)

            rows.append({
                'target': target, 'age': age, 'input_type': input_type, 'model': family,
                'ftsl': FTSL_LABELS[fs], 'config': config, 'r2': r2, 'rmse': rmse, 'mae': mae,
                'pearson_r': pearson_r, 'f1_macro': f1_macro, 'complexity': complexity,
                'best_formula': best_formula, 'best_formula_r2': best_formula_r2,
                'best_formula_complexity': best_formula_complexity,
                'interp_formula': interp_formula, 'interp_formula_r2': interp_formula_r2,
                'interp_formula_complexity': interp_formula_complexity,
                'source_path': fs_dir,
            })


def process_results():
    rows = []
    for input_type in INPUT_TYPES:
        variant_dir = _variant_dir(input_type)
        if not os.path.exists(variant_dir):
            continue
        load_fn = INPUT_TYPE_LOADERS[input_type]
        for target in TARGETS:
            for age in AGES:
                age_path = os.path.join(variant_dir, f"age_{age}_{target}")
                if not os.path.exists(age_path):
                    continue
                _process_baselines(age_path, target, age, input_type, rows)

                has_sr = (os.path.exists(os.path.join(age_path, "deeppysr"))
                          or os.path.exists(os.path.join(age_path, "pysr")))
                if not has_sr:
                    continue
                print(f"  Loading {input_type}/{target}/age_{age} for formula evaluation...")
                _, X_full, y_full = load_fn(target, age)
                transforms_cache, formula_r2_cache = {}, {}
                _process_sr_family(age_path, target, age, input_type, "deeppysr", _deeppysr_complexity,
                                    X_full, y_full, transforms_cache, formula_r2_cache, rows)
                _process_sr_family(age_path, target, age, input_type, "pysr", _pysr_complexity,
                                    X_full, y_full, transforms_cache, formula_r2_cache, rows)

    df = pd.DataFrame(rows, columns=['target', 'age', 'input_type', 'model', 'ftsl', 'config',
                                      'r2', 'rmse', 'mae', 'pearson_r', 'f1_macro', 'complexity',
                                      'best_formula', 'best_formula_r2', 'best_formula_complexity',
                                      'interp_formula', 'interp_formula_r2', 'interp_formula_complexity',
                                      'source_path'])
    df['r2'] = df['r2'].clip(lower=0)
    return df


def select_best_models(df):
    """One row per (target, age, input_type, model): max f1_macro (clinical-
    bin classification performance) across config."""
    if df.empty:
        return df
    idx = df.groupby(['target', 'age', 'input_type', 'model'])['f1_macro'].idxmax()
    return df.loc[idx].sort_values(['target', 'age', 'input_type', 'model']).reset_index(drop=True)


def _collapse_best(df, keep_col):
    """Best (max f1_macro) row per (target, age, keep_col), collapsing every
    other grouping dimension (model/input_type/config)."""
    idx = df.groupby(['target', 'age', keep_col])['f1_macro'].idxmax()
    return df.loc[idx].reset_index(drop=True)


METRICS = ['r2', 'pearson_r', 'f1_macro']
METRIC_LABELS = {'r2': 'R2', 'pearson_r': 'Pearson r', 'rmse': 'RMSE', 'mae': 'MAE',
                  'f1_macro': 'F1 (macro, clinical bins)'}


def _plot_metric_vs_age(df, keep_col, legend_title, out_path, suptitle, palette_name="tab10"):
    """Grid of subplots: one column per target that actually has data
    (rather than a fixed 2 -- dia_bp is skipped until those runs exist), one
    row per metric (r2, pearson_r, f1_macro). Each cell's line is the same
    best-(max f1_macro) row from _collapse_best, so r2/pearson_r are read off
    the exact row f1_macro was maximized on, not maximized independently per
    metric. palette_name switches to e.g. "tab20" when keep_col has more
    series than tab10's 10 distinguishable hues."""
    plot_df = _collapse_best(df, keep_col)
    if plot_df.empty:
        print(f"No data to plot for {out_path}.")
        return

    targets_present = [t for t in TARGETS if t in plot_df['target'].unique()]
    if not targets_present:
        print(f"No target data available to plot for {out_path}.")
        return

    series = sorted(plot_df[keep_col].unique())
    palette = sns.color_palette(palette_name, n_colors=len(series))
    colors = dict(zip(series, palette))

    ncols, nrows = len(targets_present), len(METRICS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), sharex=True, squeeze=False)
    plt.rcParams.update({'font.size': 18})
    for row, metric in enumerate(METRICS):
        for col, target in enumerate(targets_present):
            ax = axes[row][col]
            sub = plot_df[plot_df['target'] == target]
            sns.lineplot(data=sub, x='age', y=metric, hue=keep_col, ax=ax, palette=colors,
                         linestyle='-', linewidth=2.5, marker='o', markersize=6)
            if row == 0:
                ax.set_title(target, fontsize=22, fontweight='bold')
            if row == nrows - 1:
                ax.set_xlabel('Age', fontsize=18)
            else:
                ax.set_xlabel('')
            ax.set_ylabel(METRIC_LABELS[metric] if col == 0 else '', fontsize=18)
            ax.set_ylim(0.0, 1.0)
            ax.tick_params(axis='both', labelsize=16)
            if ax.get_legend():
                ax.get_legend().remove()

    legend_elements = [Line2D([0], [0], color=colors[s], lw=3, marker='o', label=s) for s in series]
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.91, 0.5),
               fontsize=17, frameon=True, title=legend_title, title_fontsize=19, handlelength=3.5)
    plt.suptitle(suptitle, fontsize=26, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def _plot_models_per_input_type(df, out_dir):
    """One models-vs-age grid plot per input_type actually present in df
    (best f1_macro collapsed across config only -- not across input_type),
    so each plot shows how the models compare within that specific input
    type."""
    for input_type in INPUT_TYPES:
        sub = df[df['input_type'] == input_type]
        if sub.empty:
            continue
        _plot_metric_vs_age(
            sub, keep_col='model', legend_title='Model',
            out_path=os.path.join(out_dir, f"bp_models_vs_age_{input_type}.png"),
            suptitle=f'Blood Pressure Prediction ({input_type}): Best Model vs Age')


# The only 4 models that actually emit feature_importance.csv for bp (KAN's
# wrapper doesn't produce one here, unlike in diab_raine; MLP never does for
# any dataset).
FEATURE_IMPORTANCE_MODELS = ['ElasticNet', 'ExtraTrees', 'RandomForest', 'XGBoost']

MODEL_DISPLAY_ORDER = ['deeppysr', 'pysr', 'KAN', 'ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']
MODEL_COLORS = dict(zip(MODEL_DISPLAY_ORDER, sns.color_palette("tab10", n_colors=len(MODEL_DISPLAY_ORDER))))


def _collect_feature_importance(variant_dir, target):
    """Long-format feature importance for one (input_type, target) across
    every age found under variant_dir: columns age, model (base name, e.g.
    'ElasticNet'), ftsl (always 'all_feature'), variable, weight. Each
    (model, age) leaf's feature_importance.csv is normalized to
    percent-of-that-leaf's-total first (mirrors
    analysis_insulin_variant.py's aggregate_feature_importance)."""
    importance_data = []
    for age in AGES:
        baselines_dir = os.path.join(variant_dir, f"age_{age}_{target}", "baselines")
        if not os.path.exists(baselines_dir):
            continue
        for m in os.listdir(baselines_dir):
            if m not in FEATURE_IMPORTANCE_MODELS:
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
                        'age': age, 'model': m, 'ftsl': FTSL_LABELS[fs],
                        'variable': row['feature'],
                        'weight': (row['importance'] / total * 100) if total > 0 else 0,
                    })
    return pd.DataFrame(importance_data)


def _plot_feature_importance_by_model(imp_df, variant_dir, target, top_n=15):
    """4 subplots, one per FEATURE_IMPORTANCE_MODELS entry. Each subplot
    shows that model's own top-N features (by mean weight across ages) on
    the x-axis, with one grouped bar per age."""
    ages_present = sorted(imp_df['age'].unique()) if not imp_df.empty else []
    age_palette = sns.color_palette("viridis", n_colors=max(len(ages_present), 1))
    age_colors = dict(zip(ages_present, age_palette))

    fig, axes = plt.subplots(1, len(FEATURE_IMPORTANCE_MODELS), figsize=(30, 7), squeeze=False)
    axes = axes.reshape(-1)
    for i, model in enumerate(FEATURE_IMPORTANCE_MODELS):
        ax = axes[i]
        panel_df = imp_df[imp_df['model'] == model]
        if panel_df.empty:
            ax.set_visible(False)
            continue
        top_features = panel_df.groupby('variable')['weight'].mean().sort_values(ascending=False).head(top_n).index
        plot_df = panel_df[panel_df['variable'].isin(top_features)].copy()
        plot_df['variable'] = pd.Categorical(plot_df['variable'], categories=top_features, ordered=True)
        sns.barplot(data=plot_df, x='variable', y='weight', hue='age', palette=age_colors,
                    hue_order=ages_present, ax=ax)
        ax.set_title(model, fontsize=16, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Importance (%)', fontsize=12)
        ax.tick_params(axis='x', rotation=90, labelsize=9)
        if ax.get_legend():
            ax.get_legend().remove()

    legend_elements = [Line2D([0], [0], color=age_colors[a], lw=6, label=str(a)) for a in ages_present]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=len(ages_present), fontsize=12, frameon=True, title='Age')
    variant_name = os.path.basename(variant_dir)
    plt.suptitle(f'Top {top_n} Feature Importance by Model ({variant_name}, {target})',
                 fontsize=22, fontweight='bold', y=1.14)
    plt.tight_layout()
    plot_path = os.path.join(variant_dir, f"feature_importance_by_model_{target}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {plot_path}")


def _aggregate_feature_importance(variant_dir, target, top_n=15):
    """Collects, saves, and plots feature importance for one (input_type,
    target). See _collect_feature_importance for the CSV and
    _plot_feature_importance_by_model for the plot."""
    imp_df = _collect_feature_importance(variant_dir, target)
    if imp_df.empty:
        return imp_df

    csv_path = os.path.join(variant_dir, f"feature_importance_aggregated_{target}.csv")
    imp_df.to_csv(csv_path, index=False)
    print(f"Feature importance aggregated to {csv_path}")

    _plot_feature_importance_by_model(imp_df, variant_dir, target, top_n=top_n)
    return imp_df


def _feature_importance_per_input_type():
    for input_type in INPUT_TYPES:
        variant_dir = _variant_dir(input_type)
        if not os.path.exists(variant_dir):
            continue
        for target in TARGETS:
            _aggregate_feature_importance(variant_dir, target)


def _plot_predictions_scatter(best_df):
    """One true-vs-predicted scatter grid per (input_type, target, age), one
    subplot per model present in best_df at that combo (already the best,
    max-f1_macro row per model from select_best_models). y_true/y_pred are read
    straight from that row's winning predictions.csv (best_df's
    source_path) -- the exact predictions the row's r2/rmse/mae/pearson_r
    were computed from, not a re-evaluation. Saved under each input_type's
    own results folder as scatter_predictions/age_<age>_<target>_scatter.png."""
    for input_type in INPUT_TYPES:
        variant_dir = _variant_dir(input_type)
        if not os.path.exists(variant_dir):
            continue
        sub_it = best_df[best_df['input_type'] == input_type]
        if sub_it.empty:
            continue
        out_dir = os.path.join(variant_dir, "scatter_predictions")
        os.makedirs(out_dir, exist_ok=True)
        n_saved = 0

        for target in TARGETS:
            sub_t = sub_it[sub_it['target'] == target]
            if sub_t.empty:
                continue
            for age in sorted(sub_t['age'].unique()):
                age_df = sub_t[sub_t['age'] == age]
                models_present = [m for m in MODEL_DISPLAY_ORDER if m in age_df['model'].values]
                if not models_present:
                    continue

                ncols = 4
                nrows = int(np.ceil(len(models_present) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
                axes = axes.reshape(-1)

                for i, m in enumerate(models_present):
                    row = age_df[age_df['model'] == m].iloc[0]
                    pred_file = os.path.join(row['source_path'], 'predictions.csv')
                    ax = axes[i]
                    if not os.path.exists(pred_file):
                        ax.set_visible(False)
                        continue
                    df_pred = pd.read_csv(pred_file)
                    y_true = df_pred['y_true'].values
                    y_pred = df_pred['y_pred'].values

                    lo, hi = float(np.min(y_true)), float(np.max(y_true))
                    pad = (hi - lo) * 0.1 if hi > lo else 1.0
                    lo_d, hi_d = lo - pad, hi + pad
                    # Symbolic (deeppysr/pysr) predictions occasionally
                    # overflow to extreme magnitudes; clip only the display
                    # value so those points still show pinned at the edge
                    # instead of collapsing the axis scale for every model.
                    y_pred_display = np.clip(y_pred, lo_d, hi_d)

                    ax.plot([lo_d, hi_d], [lo_d, hi_d], 'k--', lw=1)
                    ax.scatter(y_true, y_pred_display, alpha=0.5, s=20, color=MODEL_COLORS.get(m, 'gray'))
                    ax.set_xlim(lo_d, hi_d)
                    ax.set_ylim(lo_d, hi_d)
                    ax.set_title(f"{m} (R2={row['r2']:.2f}, r={row['pearson_r']:.2f}, F1={row['f1_macro']:.2f})",
                                 fontsize=12, fontweight='bold')
                    ax.set_xlabel(f'True {target} (mmHg)', fontsize=10)
                    ax.set_ylabel(f'Predicted {target} (mmHg)', fontsize=10)

                for j in range(len(models_present), len(axes)):
                    axes[j].set_visible(False)

                plt.suptitle(f'{input_type} / {target} / age {age}: True vs Predicted',
                             fontsize=18, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                out_path = os.path.join(out_dir, f"age_{age}_{target}_scatter.png")
                plt.savefig(out_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                n_saved += 1

        if n_saved:
            print(f"{n_saved} scatter plot(s) saved under {out_dir}")


def _plot_confusion_matrices(best_df):
    """One confusion-matrix grid per (input_type, target, age), one subplot
    per model present in best_df at that combo -- same selection and
    predictions.csv source as _plot_predictions_scatter, but y_true/y_pred
    are binned into the target's 3 clinical classes (_bin_bp) before the
    confusion matrix is computed, so this reads directly as "how often does
    each model's continuous prediction land in the right clinical category."
    Saved under each input_type's own results folder as
    conf_matrix/age_<age>_<target>_confmat.png."""
    from sklearn.metrics import confusion_matrix

    for input_type in INPUT_TYPES:
        variant_dir = _variant_dir(input_type)
        if not os.path.exists(variant_dir):
            continue
        sub_it = best_df[best_df['input_type'] == input_type]
        if sub_it.empty:
            continue
        out_dir = os.path.join(variant_dir, "conf_matrix")
        os.makedirs(out_dir, exist_ok=True)
        n_saved = 0

        for target in TARGETS:
            sub_t = sub_it[sub_it['target'] == target]
            if sub_t.empty:
                continue
            class_labels = BP_CLASS_LABELS[target]
            for age in sorted(sub_t['age'].unique()):
                age_df = sub_t[sub_t['age'] == age]
                models_present = [m for m in MODEL_DISPLAY_ORDER if m in age_df['model'].values]
                if not models_present:
                    continue

                ncols = 4
                nrows = int(np.ceil(len(models_present) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
                axes = axes.reshape(-1)

                for i, m in enumerate(models_present):
                    row = age_df[age_df['model'] == m].iloc[0]
                    pred_file = os.path.join(row['source_path'], 'predictions.csv')
                    ax = axes[i]
                    if not os.path.exists(pred_file):
                        ax.set_visible(False)
                        continue
                    df_pred = pd.read_csv(pred_file)
                    y_true_bin = _bin_bp(target, df_pred['y_true'])
                    y_pred_bin = _bin_bp(target, df_pred['y_pred'])
                    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1, 2])

                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                                xticklabels=class_labels, yticklabels=class_labels,
                                annot_kws={'size': 11})
                    ax.set_title(f"{m} (F1={row['f1_macro']:.2f})", fontsize=12, fontweight='bold')
                    ax.set_xlabel('Predicted', fontsize=14)
                    ax.set_ylabel('True', fontsize=14)
                    ax.tick_params(axis='both', labelsize=12)

                for j in range(len(models_present), len(axes)):
                    axes[j].set_visible(False)

                plt.suptitle(f'{input_type} / {target} / age {age}: Confusion Matrix (clinical bins)',
                             fontsize=18, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                out_path = os.path.join(out_dir, f"age_{age}_{target}_confmat.png")
                plt.savefig(out_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                n_saved += 1

        if n_saved:
            print(f"{n_saved} confusion matrix plot(s) saved under {out_dir}")


HEATMAP_MODEL_ORDER = ['RandomForest', 'ExtraTrees', 'XGBoost', 'ElasticNet', 'MLP (SHAP)']

_SENS_INK = "#0b0b0b"
_SENS_BLUE_STEPS = ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                    "#256abf", "#184f95", "#104281", "#0d366b"]
_SENS_CMAP = LinearSegmentedColormap.from_list("bp_seq_blue", _SENS_BLUE_STEPS, N=256)


def _bp_mlp_shap_importance(X, y, task='regression', use_smote=False, epochs=80, batch_size=64,
                             max_train_rows=None, n_background=50, n_explain=200,
                             random_state=42):
    """Same as deeppysr_paper/codes/common.py::mlp_shap_importance, except the
    SHAP Permutation explainer is given an explicit max_evals sized to the
    feature count. common.py's version leaves max_evals at SHAP's internal
    default (500), which is too low for bp's PGSto5/to5/recent input types
    (hundreds to 1000+ features -- SHAP raises unless
    max_evals >= 2*num_features+1); the paper's original 5 datasets never hit
    this since they all have well under 250 features."""
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


def _bp_mlp_shap_importance_subprocess(input_type, target, age, random_state=42, n_explain=60, timeout=1800):
    """Runs _bp_mlp_shap_importance in a fresh subprocess (_mlp_shap_worker.py)
    instead of in-process. Importing model_utils (which pulls in torch AND,
    via its own module-level `from pysr import PySRRegressor`, juliacall)
    reliably segfaults once this long-running analysis process has already
    done enough prior matplotlib/numpy work -- reproduced independently of
    which specific combo triggers it (just running ~50 throwaway
    plt.savefig() calls before the same X/y/call is enough), even though the
    identical import + SHAP call succeeds in a fresh process every time.
    This is a Julia-embedding fragility, not a bug in the SHAP call itself
    -- isolating it into its own subprocess sidesteps it entirely by always
    running under the conditions where it works."""
    worker = os.path.join(current_dir, "_mlp_shap_worker.py")
    fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        result = subprocess.run(
            [sys.executable, worker,
             "--input_type", input_type, "--target", target, "--age", str(age),
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


def _bp_conventional_importance(row):
    """{feature: pct} for one FEATURE_IMPORTANCE_MODELS row of best_df, read
    from that row's own winning (ftsl-selected) feature_importance.csv --
    mirrors common.py's load_conventional_importance, but reads the specific
    leaf best_df already picked for this (target, age, input_type, model)
    rather than a fixed baselines_dir/<model>/feature_importance.csv."""
    imp_file = os.path.join(row['source_path'], 'feature_importance.csv')
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
    aggregate_permutation_sensitivity with input_type columns instead."""
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


DEEPPYSR_VARIANT_COLORS = {'Best': '#3987e5', 'Interpretable': '#e58939'}


def _plot_deeppysr_sensitivity(row, X_full, y_full, out_dir, target, age, input_type,
                                n_repeats=15, random_state=42):
    """Combined best-vs-interpretable DeepPySR sensitivity plot for one
    (target, age, input_type) combo: grouped horizontal bars over the union
    of every variable either formula references (no top-N truncation),
    annotated with each formula's own complexity and full-dataset
    r2/pearson_r/f1_macro. Those three metrics are evaluated fresh here (via
    evaluate_formula on X_full/y_full, same basis for both formulas) --
    NOT the row's own r2/pearson_r/f1_macro columns, which come from
    predictions.csv (leak-free-ish CV), nor best_formula_r2/interp_formula_r2
    (in-sample r2 only, no pearson_r/f1_macro stored per-formula).

    Saves under out_dir (the same age_<age>_<target> directory as
    bp_sensitivity.csv/.png):
      bp_deeppysr_sensitivity.csv
      bp_deeppysr_sensitivity.png
    """
    variants = []
    for label, formula_col, complexity_col in [
            ('Best', 'best_formula', 'best_formula_complexity'),
            ('Interpretable', 'interp_formula', 'interp_formula_complexity')]:
        formula = row[formula_col]
        if not formula or pd.isna(formula):
            continue
        formula = str(formula)
        try:
            y_pred = evaluate_formula(formula, X_full, model_type='deeppysr')
            r2, _, _, pearson_r = calculate_metrics(y_full, y_pred)
            f1 = _bp_f1_macro(target, y_full, y_pred)
        except Exception as e:
            print(f"    DeepPySR {label} eval failed for {input_type}/{target}/age_{age}: {e}")
            continue
        try:
            sens = formula_sensitivity(formula, X_full, n_repeats=n_repeats, random_state=random_state)
            sens = {k: v for k, v in sens.items() if v > 0}
        except Exception as e:
            print(f"    DeepPySR {label} sensitivity failed for {input_type}/{target}/age_{age}: {e}")
            continue
        if not sens:
            continue
        variants.append({
            'label': label, 'formula': formula, 'complexity': row[complexity_col],
            'r2': r2, 'pearson_r': pearson_r, 'f1_macro': f1, 'sens': sens,
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
    ax.set_title(f'{target} ({input_type}, age {age}): DeepPySR sensitivity — Best vs Interpretable',
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
    png_path = os.path.join(out_dir, 'bp_deeppysr_sensitivity.png')
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    csv_rows = [
        {'target': target, 'age': age, 'input_type': input_type, 'formula_type': v['label'],
         'variable': var, 'sensitivity_pct': pct, 'complexity': v['complexity'],
         'r2': v['r2'], 'pearson_r': v['pearson_r'], 'f1_macro': v['f1_macro']}
        for v in variants for var, pct in v['sens'].items()
    ]
    pd.DataFrame(csv_rows).to_csv(os.path.join(out_dir, 'bp_deeppysr_sensitivity.csv'), index=False)

    return png_path


def compute_bp_sensitivity(best_df, n_repeats=15, random_state=42, top_n=10):
    """Per-(target, input_type, age) feature-importance comparison across the
    conventional model families present in best_df (HEATMAP_MODEL_ORDER):
      - ElasticNet/ExtraTrees/RandomForest/XGBoost: that combo's own winning
        feature_importance.csv (_bp_conventional_importance).
      - MLP: SHAP importance from a fresh refit of the paper's MLP
        architecture on the combo's full dataset (common.mlp_shap_importance).
    PySR and DeepPySR are intentionally excluded from this heatmap -- their
    permutation-based formula sensitivity isn't comparable to native/SHAP
    feature importance, and DeepPySR gets its own dedicated comparison below.
    All vectors normalised to sum to 100 within their own model.

    Also calls _plot_deeppysr_sensitivity for every combo with a deeppysr
    row, saving a combined best-vs-interpretable DeepPySR permutation
    sensitivity plot alongside the files below (PySR itself isn't plotted
    anywhere here -- nothing downstream currently consumes it).

    Saves, under that combo's own results_bp_<input_type>/
    age_<age>_<target>/ directory:
      bp_sensitivity.csv             -- long format: variable, model, pct
      bp_sensitivity.png             -- heatmap, top_n features x models present
      bp_deeppysr_sensitivity.csv    -- DeepPySR best vs interpretable, all variables
      bp_deeppysr_sensitivity.png    -- grouped bar chart, best vs interpretable
    """
    combos = best_df[['target', 'age', 'input_type']].drop_duplicates()
    if combos.empty:
        print("  No combos to compute sensitivity for.")
        return pd.DataFrame()

    all_rows = []
    for _, combo in combos.iterrows():
        target, age, input_type = combo['target'], int(combo['age']), combo['input_type']
        age_dir = os.path.join(_variant_dir(input_type), f"age_{age}_{target}")
        if not os.path.exists(age_dir):
            continue

        sub = best_df[(best_df['target'] == target) & (best_df['age'] == age)
                       & (best_df['input_type'] == input_type)]

        importances = {}
        for model_name in FEATURE_IMPORTANCE_MODELS:
            row = sub[sub['model'] == model_name]
            if row.empty:
                continue
            imp = _bp_conventional_importance(row.iloc[0])
            if imp:
                importances[model_name] = imp

        deeppysr_row = sub[sub['model'] == 'deeppysr']
        needs_X = (not deeppysr_row.empty) or ('MLP' in sub['model'].values)

        X_full = y_full = None
        if needs_X:
            print(f"  Loading {input_type}/{target}/age_{age} for sensitivity analysis...")
            _, X_full, y_full = INPUT_TYPE_LOADERS[input_type](target, age)

        if 'MLP' in sub['model'].values:
            try:
                # n_explain trimmed from common.py's default (200) -- with
                # max_evals now scaled to feature count (up to ~1300 for
                # some input types), the full 200 would take hours across
                # the whole grid; 60 keeps the SHAP estimate stable enough
                # for a top-10-feature ranking at a fraction of the runtime.
                # Run out-of-process -- see _bp_mlp_shap_importance_subprocess.
                mlp_imp = _bp_mlp_shap_importance_subprocess(input_type, target, age,
                                                              random_state=random_state, n_explain=60)
                if mlp_imp:
                    importances['MLP (SHAP)'] = mlp_imp
            except Exception as e:
                print(f"    MLP SHAP failed for {input_type}/{target}/age_{age}: {e}")

        if not deeppysr_row.empty and X_full is not None:
            _plot_deeppysr_sensitivity(deeppysr_row.iloc[0], X_full, y_full, age_dir, target, age, input_type,
                                        n_repeats=n_repeats, random_state=random_state)

        if not importances:
            continue

        cols = [m for m in HEATMAP_MODEL_ORDER if m in importances]
        all_features = sorted(set().union(*[d.keys() for d in importances.values()]))
        table = pd.DataFrame({m: {f: importances[m].get(f, 0.0) for f in all_features} for m in cols})

        long_df = table.reset_index().melt(id_vars='index', var_name='model', value_name='pct')
        long_df = long_df.rename(columns={'index': 'variable'})
        long_df = long_df[long_df['pct'] > 0].copy()
        long_df.insert(0, 'input_type', input_type)
        long_df.insert(0, 'age', age)
        long_df.insert(0, 'target', target)
        csv_path = os.path.join(age_dir, "bp_sensitivity.csv")
        long_df.to_csv(csv_path, index=False)

        title = f'{target} ({input_type}, age {age}): feature importance across models'
        png_path = os.path.join(age_dir, "bp_sensitivity.png")
        _plot_sensitivity_heatmap(table, png_path, title, top_n=top_n)

        all_rows.append(long_df)
        print(f"  Sensitivity saved to {age_dir}")

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def aggregate_permutation_sensitivity(top_n=15, min_pct=1.0):
    """Combine every per-combo DeepPySR permutation-sensitivity CSV
    (bp_deeppysr_sensitivity.csv, written by _plot_deeppysr_sensitivity) into
    one overview: which variables drive DeepPySR's formulas, and does that
    change depending on which data source (input_type) the model was given?
    Only the 'Interpretable' formula_type is kept -- the complexity-capped
    formula a clinician could actually read, not the unconstrained 'Best'
    one.

    Ranking is by *how often* a variable is a driver, not its mean
    sensitivity_pct share -- a plain mean is dominated by combos whose
    formula happens to have very few terms (a variable alone in a 2-term
    formula gets ~90%+ of that formula's normalised share by construction,
    regardless of how rarely it shows up anywhere else). Instead, for each
    input_type, each variable's cell is "in what % of the (target, age)
    combos evaluated for that input_type does this variable appear as a
    driver (sensitivity_pct > min_pct)" -- top_n variables are chosen by
    total appearance count summed across every input_type.

    Note this view only covers combos where DeepPySR was itself the
    winning/best_df model with a positive-sensitivity Interpretable formula
    -- a minority of all (target, age, input_type) combos (see
    aggregated_results.csv row counts). For the fuller, non-DeepPySR-only
    picture see the per-combo bp_sensitivity.png heatmaps linked below.

    Saves, under RESULTS_BASE_DIR:
      bp_deeppysr_sensitivity_overview.csv  -- every kept combo's rows,
                                                concatenated (unaveraged)
      bp_deeppysr_sensitivity_overview.png  -- heatmap, top_n variables x
                                                input_type, % of that input
                                                type's combos where the
                                                variable is a driver
    """
    rows = []
    for input_type in INPUT_TYPES:
        variant_dir = _variant_dir(input_type)
        if not os.path.exists(variant_dir):
            continue
        for target in TARGETS:
            for age in AGES:
                f = os.path.join(variant_dir, f"age_{age}_{target}", "bp_deeppysr_sensitivity.csv")
                if not os.path.exists(f):
                    continue
                df = pd.read_csv(f)
                df = df[df['formula_type'] == 'Interpretable']
                if not df.empty:
                    rows.append(df)
    if not rows:
        print("No DeepPySR sensitivity data to aggregate.")
        return pd.DataFrame()

    long_df = pd.concat(rows, ignore_index=True)
    csv_path = os.path.join(RESULTS_BASE_DIR, "bp_deeppysr_sensitivity_overview.csv")
    long_df.to_csv(csv_path, index=False)
    print(f"DeepPySR sensitivity overview saved to {csv_path}")

    combos = long_df[['input_type', 'target', 'age']].drop_duplicates()
    n_combos = combos.groupby('input_type').size()
    present = (long_df[long_df['sensitivity_pct'] > min_pct]
               [['variable', 'input_type', 'target', 'age']].drop_duplicates())
    counts = present.groupby(['variable', 'input_type']).size().unstack(fill_value=0)
    it_order = [it for it in INPUT_TYPES if it in n_combos.index]
    counts = counts.reindex(columns=it_order, fill_value=0)
    pct_table = counts.div(n_combos.reindex(it_order), axis=1) * 100.0

    top_vars = counts.sum(axis=1).sort_values(ascending=False).head(top_n).index
    plot_table = pct_table.loc[top_vars]

    png_path = os.path.join(RESULTS_BASE_DIR, "bp_deeppysr_sensitivity_overview.png")
    _plot_sensitivity_heatmap(
        plot_table, png_path,
        title="Blood pressure: how often each variable drives DeepPySR's interpretable formula, by data source"
              f" (n combos: {', '.join(f'{it}={n}' for it, n in n_combos.items())})",
        top_n=top_n, cbar_label="% of that input type's combos where this variable is a driver")
    print(f"DeepPySR sensitivity overview plot saved to {png_path}")
    return long_df


def main():
    out_csv = os.path.join(RESULTS_BASE_DIR, "bp_aggregated_results.csv")
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv)
        print(f"Results loaded from {out_csv}")
    else:
        df = process_results()
        df.to_csv(out_csv, index=False)
        print(f"Results saved to {out_csv}")

    best_df = select_best_models(df)
    best_csv = os.path.join(RESULTS_BASE_DIR, "bp_best_models.csv")
    best_df.to_csv(best_csv, index=False)
    print(f"Best models saved to {best_csv}")

    _plot_metric_vs_age(
        df, keep_col='model', legend_title='Model',
        out_path=os.path.join(RESULTS_BASE_DIR, "bp_models_vs_age.png"),
        suptitle='Blood Pressure Prediction: Best Model vs Age')
    _plot_metric_vs_age(
        df, keep_col='input_type', legend_title='Input type',
        out_path=os.path.join(RESULTS_BASE_DIR, "bp_input_types_vs_age.png"),
        suptitle='Blood Pressure Prediction: Best Input Type vs Age')
    _plot_models_per_input_type(df, RESULTS_BASE_DIR)
    _feature_importance_per_input_type()
    _plot_predictions_scatter(best_df)
    _plot_confusion_matrices(best_df)

    print("=== Computing formula sensitivity ===")
    compute_bp_sensitivity(best_df)

    print("=== Aggregating DeepPySR permutation sensitivity overview ===")
    aggregate_permutation_sensitivity()


if __name__ == "__main__":
    main()
