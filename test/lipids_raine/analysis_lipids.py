"""Aggregate regression results for the lipids RAINE tests (cholesterol,
triglyceride, hdl, ldl) across all input-type variants under results_lipids/.

Unlike analysis_insulin_variant.py, the r2/rmse/mae/pearson_r columns are NOT
leak-free CV -- they're read straight off each leaf's predictions.csv
(y_true/y_pred columns), matching how the baseline models are already
scored. For deeppysr/pysr, "complexity" is a single representative number
per (variant config, feature-set) leaf:
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
Each fold's formulas were fit on that fold's own (scaler/feature-selection)
view of the data, so before evaluating we reconstruct that exact view --
same KFold(5, shuffle=True, random_state=42) split as eval_utils.run_cv, the
StandardScaler fit on that fold's train rows (deeppysr only, and only for
variants whose test_deeppysr_lipids_*.py call omits scaler=False -- see
_deeppysr_uses_scaler), and the SelectKBest(f_regression, k=100) top-100
subset fit on that fold's train rows (only for ftsl='top100') -- then apply
that same transform to the full dataset before plugging into the formula.

Produces, under this script's directory:
  lipids_aggregated_results.csv   -- every (target, age, input_type, model,
                                      ftsl, config) row found on disk.
  lipids_best_models.csv          -- best (max r2) row per
                                      (target, age, input_type, model),
                                      collapsing across ftsl/config.
  lipids_models_vs_age.png        -- 4 subplots (targets), lines=model,
                                      x=age, best r2 collapsed across
                                      input_type and ftsl.
  lipids_input_types_vs_age.png   -- 4 subplots (targets), lines=input_type,
                                      x=age, best r2 collapsed across model
                                      and ftsl.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

from analysis_v1_utils import calculate_metrics, calculate_complexity, evaluate_formula, map_variable_names
from data_utils import load_data_PGS_only, load_data_keepto8, load_data_PGSto8, load_data_recent, load_data_nblood

RESULTS_BASE_DIR = os.path.join(current_dir, "../../archive/results/results_lipids")

TARGETS = ["cholesterol", "triglyceride", "hdl", "ldl"]
AGES = [14, 17, 20, 22, 27, 28]
FS_SUBFOLDERS = ["all_features", "top100"]
FTSL_LABELS = {"all_features": "all_feature", "top100": "top100"}

INPUT_TYPES = ["PGS", "PGSto8", "to8", "recent", "nblood"]
BASELINE_MODELS = ["ElasticNet", "ExtraTrees", "KAN", "MLP", "RandomForest", "XGBoost"]

# load_fn per input_type -- mirrors the _LOAD_FN dicts in
# test_baselines_pysr_lipids_to8.py / test_deeppysr_lipids_to8.py (PGS, to8,
# PGSto8) and test_baselines_pysr_lipids_recent.py / test_deeppysr_lipids_recent.py
# (recent, nblood).
INPUT_TYPE_LOADERS = {
    'PGS':    load_data_PGS_only,
    'PGSto8': load_data_PGSto8,
    'to8':    load_data_keepto8,
    'recent': load_data_recent,
    'nblood': load_data_nblood,
}

N_SPLITS = 5
RANDOM_STATE = 42
N_TOP = 100
INTERP_MAX_COMPLEXITY = 25


def _variant_dir(input_type):
    return os.path.join(RESULTS_BASE_DIR, f"results_lipids_{input_type}")


def _deeppysr_uses_scaler(input_type):
    """test_deeppysr_lipids_recent.py passes scaler=False explicitly for
    both recent and nblood; the to8/PGS/PGSto8 script
    test_deeppysr_lipids_to8.py omits scaler=..., so eval_utils.run_cv's
    default (True) applies there. PySR always runs with scaler=False
    regardless of input_type."""
    return input_type not in ('recent', 'nblood')


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
            rows.append({
                'target': target, 'age': age, 'input_type': input_type, 'model': model_name,
                'ftsl': FTSL_LABELS[fs], 'config': '', 'r2': r2, 'rmse': rmse, 'mae': mae,
                'pearson_r': pearson_r, 'complexity': np.nan,
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


def _fold_transform(X_all_raw, y_values, train_idx, use_scaler, use_topk, cols):
    """X_all_raw (n_rows, n_cols) transformed exactly as that fold's model
    would have seen its inputs (see eval_utils.run_cv: scaler applied before
    feature_selection), applied to every row (not just that fold's own
    train/test rows) so the resulting formula can be evaluated in-sample on
    the whole dataset."""
    X_train = X_all_raw[train_idx]
    X_all = X_all_raw
    if use_scaler:
        sc = StandardScaler().fit(X_train)
        X_train = sc.transform(X_train)
        X_all = sc.transform(X_all)
    if use_topk:
        k = min(N_TOP, X_train.shape[1])
        selector = SelectKBest(f_regression, k=k).fit(X_train, y_values[train_idx])
        support = selector.get_support()
        sel_cols = [c for c, s in zip(cols, support) if s]
        return pd.DataFrame(X_all[:, support], columns=sel_cols)
    return pd.DataFrame(X_all, columns=cols)


def _build_fold_transforms(X_full, y_full, use_scaler, use_topk):
    """5 DataFrames (one per CV fold), each holding X_full's rows exactly as
    that fold's model would have seen them post scaler/feature-selection --
    reconstructed with the same KFold(5, shuffle=True, random_state=42) used
    by eval_utils.run_cv (see cv_kwargs in test_*_lipids_*.py; no groups or
    stratify_by are passed, so plain KFold applies)."""
    y_values = y_full.values if hasattr(y_full, 'values') else np.array(y_full)
    cols = list(X_full.columns)
    X_all_raw = X_full.values.astype(float)
    splits = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X_full))
    return [_fold_transform(X_all_raw, y_values, train_idx, use_scaler, use_topk, cols)
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
            complexity = complexity_fn(fs_dir)

            cache_key = (family, fs)
            if cache_key not in transforms_cache:
                transforms_cache[cache_key] = _build_fold_transforms(
                    X_full, y_full, use_scaler, use_topk=(fs == 'top100'))
            (best_formula, best_formula_r2, best_formula_complexity,
             interp_formula, interp_formula_r2, interp_formula_complexity) = _best_and_interp_formula(
                fs_dir, family, transforms_cache[cache_key], y_values, formula_r2_cache, cache_key)

            rows.append({
                'target': target, 'age': age, 'input_type': input_type, 'model': family,
                'ftsl': FTSL_LABELS[fs], 'config': config, 'r2': r2, 'rmse': rmse, 'mae': mae,
                'pearson_r': pearson_r, 'complexity': complexity,
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
                # TEMPORARY: age 27 cholesterol results on disk were trained
                # before the 27/28 merge + outlier fix in data_utils.py (see
                # _lipid_target_series) -- exclude until those runs are
                # regenerated against the corrected/merged age-28 target.
                # Remove this once age 27 is retrained away entirely.
                if target == 'cholesterol' and age == 27:
                    continue
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
                                      'r2', 'rmse', 'mae', 'pearson_r', 'complexity',
                                      'best_formula', 'best_formula_r2', 'best_formula_complexity',
                                      'interp_formula', 'interp_formula_r2', 'interp_formula_complexity',
                                      'source_path'])
    df['r2'] = df['r2'].clip(lower=0)
    return df


def select_best_models(df):
    """One row per (target, age, input_type, model): max r2 across ftsl/config."""
    if df.empty:
        return df
    idx = df.groupby(['target', 'age', 'input_type', 'model'])['r2'].idxmax()
    return df.loc[idx].sort_values(['target', 'age', 'input_type', 'model']).reset_index(drop=True)


def _collapse_best(df, keep_col):
    """Best (max r2) row per (target, age, keep_col), collapsing every other
    grouping dimension (model/input_type/ftsl/config)."""
    idx = df.groupby(['target', 'age', keep_col])['r2'].idxmax()
    return df.loc[idx].reset_index(drop=True)


METRICS = ['r2', 'pearson_r']
METRIC_LABELS = {'r2': 'R2', 'pearson_r': 'Pearson r', 'rmse': 'RMSE', 'mae': 'MAE'}


def _plot_metric_vs_age(df, keep_col, legend_title, out_path, suptitle, palette_name="tab10"):
    """Grid of subplots: one column per target that actually has data
    (rather than a fixed 4 -- hdl/triglyceride are skipped until those runs
    exist), one row per metric (r2, pearson_r, rmse, mae). Each cell's line
    is the same best-(max r2) row from _collapse_best, so pearson_r/rmse/mae
    are read off the exact row r2 was maximized on, not maximized
    independently per metric. palette_name switches to e.g. "tab20" when
    keep_col has more series than tab10's 10 distinguishable hues (e.g. the
    16 model x ftsl combinations in _plot_model_ftsl_per_input_type)."""
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5 * nrows), sharex=True, squeeze=False)
    plt.rcParams.update({'font.size': 13})
    for row, metric in enumerate(METRICS):
        for col, target in enumerate(targets_present):
            ax = axes[row][col]
            sub = plot_df[plot_df['target'] == target]
            sns.lineplot(data=sub, x='age', y=metric, hue=keep_col, ax=ax, palette=colors,
                         linestyle='--', linewidth=2.5, marker='o', markersize=6)
            if row == 0:
                ax.set_title(target, fontsize=16, fontweight='bold')
            if row == nrows - 1:
                ax.set_xlabel('Age', fontsize=13)
            else:
                ax.set_xlabel('')
            ax.set_ylabel(METRIC_LABELS[metric] if col == 0 else '', fontsize=13)
            if ax.get_legend():
                ax.get_legend().remove()

    legend_elements = [Line2D([0], [0], color=colors[s], lw=3, marker='o', label=s) for s in series]
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.91, 0.5),
               fontsize=12, frameon=True, title=legend_title, title_fontsize=14, handlelength=3.5)
    plt.suptitle(suptitle, fontsize=20, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def _plot_models_per_input_type(df, out_dir):
    """One models-vs-age grid plot per input_type actually present in df
    (best r2 collapsed across ftsl/config only -- not across input_type),
    so each plot shows how the models compare within that specific
    feature-set variant."""
    for input_type in INPUT_TYPES:
        sub = df[df['input_type'] == input_type]
        if sub.empty:
            continue
        _plot_metric_vs_age(
            sub, keep_col='model', legend_title='Model',
            out_path=os.path.join(out_dir, f"lipids_models_vs_age_{input_type}.png"),
            suptitle=f'Lipids Prediction ({input_type}): Best R2 by Model vs Age')


def _collapse_best_by_ftsl(df):
    """Best (max r2) row per (target, age, model, ftsl): collapses only
    across config (the SR hyperparameter grid point), NOT across ftsl --
    unlike select_best_models/_collapse_best, which fold all_feature and
    top100 into one family line. Adds a 'model_ftsl' column (e.g.
    'XGBoost (top100)', 'deeppysr (all_feature)') so both feature-set
    variants of every model/family show up as distinct series."""
    idx = df.groupby(['target', 'age', 'model', 'ftsl'])['r2'].idxmax()
    out = df.loc[idx].copy()
    out['model_ftsl'] = out['model'] + ' (' + out['ftsl'] + ')'
    return out


def _plot_model_ftsl_per_input_type(df, out_dir=None):
    """Per input_type: same R2/Pearson-r-vs-age grid as
    lipids_models_vs_age_<input_type>.png, but all_feature and top100 are
    kept as separate series per model/family instead of collapsed into one
    best-of-family line -- e.g. 'XGBoost (all_feature)' vs
    'XGBoost (top100)', 'deeppysr (all_feature)' vs 'deeppysr (top100)'.
    Saved under each input_type's own results folder (out_dir override is
    for testing; production calls always save alongside that variant)."""
    for input_type in INPUT_TYPES:
        variant_dir = _variant_dir(input_type)
        if not os.path.exists(variant_dir):
            continue
        sub = df[df['input_type'] == input_type]
        if sub.empty:
            continue
        plot_df = _collapse_best_by_ftsl(sub)
        save_dir = out_dir or variant_dir
        _plot_metric_vs_age(
            plot_df, keep_col='model_ftsl', legend_title='Model (feature set)',
            out_path=os.path.join(save_dir, "lipids_model_ftsl_vs_age.png"),
            suptitle=f'Lipids Prediction ({input_type}): R2 by Model x Feature-set vs Age',
            palette_name="tab20")


# The only 4 models that actually emit feature_importance.csv for lipids
# (KAN's wrapper doesn't produce one here, unlike in diab_raine; MLP never
# does for any dataset).
FEATURE_IMPORTANCE_MODELS = ['ElasticNet', 'ExtraTrees', 'RandomForest', 'XGBoost']

MODEL_DISPLAY_ORDER = ['deeppysr', 'pysr', 'KAN', 'ElasticNet', 'ExtraTrees', 'MLP', 'RandomForest', 'XGBoost']
MODEL_COLORS = dict(zip(MODEL_DISPLAY_ORDER, sns.color_palette("tab10", n_colors=len(MODEL_DISPLAY_ORDER))))


def _collect_feature_importance(variant_dir, target):
    """Long-format feature importance for one (input_type, target) across
    every age found under variant_dir: columns age, model (base name, e.g.
    'ElasticNet'), ftsl ('all_feature'/'top100'), variable, weight. Each
    (model, ftsl, age) leaf's feature_importance.csv is normalized to
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
    """8 subplots: the 4 FEATURE_IMPORTANCE_MODELS x 2 ftsl (all_feature,
    top100) -- the only combinations that actually have feature_importance
    data. Each subplot shows that (model, ftsl)'s own top-N features (by
    mean weight across ages) on the x-axis, with one grouped bar per age."""
    ages_present = sorted(imp_df['age'].unique()) if not imp_df.empty else []
    age_palette = sns.color_palette("viridis", n_colors=max(len(ages_present), 1))
    age_colors = dict(zip(ages_present, age_palette))

    panels = [(m, fs) for m in FEATURE_IMPORTANCE_MODELS for fs in ['all_feature', 'top100']]
    fig, axes = plt.subplots(2, 4, figsize=(30, 14), squeeze=False)
    axes = axes.reshape(-1)
    for i, (model, fs) in enumerate(panels):
        ax = axes[i]
        panel_df = imp_df[(imp_df['model'] == model) & (imp_df['ftsl'] == fs)]
        if panel_df.empty:
            ax.set_visible(False)
            continue
        top_features = panel_df.groupby('variable')['weight'].mean().sort_values(ascending=False).head(top_n).index
        plot_df = panel_df[panel_df['variable'].isin(top_features)].copy()
        plot_df['variable'] = pd.Categorical(plot_df['variable'], categories=top_features, ordered=True)
        sns.barplot(data=plot_df, x='variable', y='weight', hue='age', palette=age_colors,
                    hue_order=ages_present, ax=ax)
        ax.set_title(f'{model} ({fs})', fontsize=16, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Importance (%)', fontsize=12)
        ax.tick_params(axis='x', rotation=90, labelsize=9)
        if ax.get_legend():
            ax.get_legend().remove()

    legend_elements = [Line2D([0], [0], color=age_colors[a], lw=6, label=str(a)) for a in ages_present]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=len(ages_present), fontsize=12, frameon=True, title='Age')
    variant_name = os.path.basename(variant_dir)
    plt.suptitle(f'Top {top_n} Feature Importance by Model x Feature-set ({variant_name}, {target})',
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
    max-r2 row per model from select_best_models). y_true/y_pred are read
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
                    ax.set_title(f"{m} (R2={row['r2']:.2f}, r={row['pearson_r']:.2f})",
                                 fontsize=12, fontweight='bold')
                    ax.set_xlabel(f'True {target}', fontsize=10)
                    ax.set_ylabel(f'Predicted {target}', fontsize=10)

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


def main():
    out_csv = os.path.join(RESULTS_BASE_DIR, "lipids_aggregated_results.csv")
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv)
        print(f"Results loaded from {out_csv}")
    else:
        df = process_results()
        df.to_csv(out_csv, index=False)
        print(f"Results saved to {out_csv}")

    best_df = select_best_models(df)
    best_csv = os.path.join(RESULTS_BASE_DIR, "lipids_best_models.csv")
    best_df.to_csv(best_csv, index=False)
    print(f"Best models saved to {best_csv}")

    _plot_metric_vs_age(
        df, keep_col='model', legend_title='Model',
        out_path=os.path.join(RESULTS_BASE_DIR, "lipids_models_vs_age.png"),
        suptitle='Lipids Prediction: Best R2 by Model vs Age')
    _plot_metric_vs_age(
        df, keep_col='input_type', legend_title='Input type',
        out_path=os.path.join(RESULTS_BASE_DIR, "lipids_input_types_vs_age.png"),
        suptitle='Lipids Prediction: Best R2 by Input Type vs Age')
    _plot_models_per_input_type(df, RESULTS_BASE_DIR)
    _plot_model_ftsl_per_input_type(df)
    _feature_importance_per_input_type()
    _plot_predictions_scatter(best_df)


if __name__ == "__main__":
    main()