# Blood pressure (RAINE) results index — all-features only

Links to the plots produced by `analysis_bp.py`'s all-features-only pass. Run it to regenerate everything under `results_bp/`. **This document is restricted to `ftsl == 'all_feature'` results throughout — top100 (feature-selected) results are excluded from every table, plot, and "best" selection below.**

## Meta

- **Targets** (2): sys_bp, dia_bp — both in mmHg.
- **Input types** (4): `PGS`, `to5`, `PGSto5`, `recent`. (The longitudinal-feature-engineering variant was dropped from bp_raine entirely — the CV scripts and this analysis no longer support it.)
- **Ages**: 10, 14, 17, 20, 22.
- **Models**: DeepPySR, PySR, KAN, ElasticNet, ExtraTrees, MLP, RandomForest, XGBoost.
- **"Best" row selection**: max F1 (macro) on the clinical bins below, per (target, age, input_type, model), collapsed across config only — ftsl is fixed to `all_feature` throughout and never collapsed with top100.

**Clinical bin definitions** (mmHg, AHA/ACC 2017 adult categories collapsed to 3 tiers — see `BP_BIN_EDGES` in `analysis_bp.py`):

| Target | Class 0 | Class 1 | Class 2 |
|---|---|---|---|
| sys_bp | Normal: < 120 | Elevated: 120–139 | High: ≥ 140 |
| dia_bp | Normal: < 80 | Elevated: 80–89 | High: ≥ 90 |

No pediatric (age/sex/height-percentile) norms are applied even at ages 10/14 — this F1_macro is a rough clinical-relevance signal for model comparison, not a diagnostic label (same simplification `analysis_lipids.py` makes for its own adolescent ages).

## Key findings

Scoped to `PGS`, `PGSto5`, `to5`, `recent`, all_feature only (40 target/age/input_type combos, 320 best-model rows).

- **DeepPySR wins most often**: max-F1 model in 22/40 combos (ElasticNet 8, KAN 5, ExtraTrees 2, XGBoost 2, MLP 1, RandomForest 0, PySR 0). Mean F1 0.385 — best of any model — despite ExtraTrees/RandomForest having the highest mean R2 (0.144 vs DeepPySR's 0.094), the same F1-vs-R2 divergence seen in the lipids analysis.
- **DeepPySR's "Interpretable" formula generalizes better than its own unconstrained "Best" formula.** Evaluated fresh on the full dataset: the complexity-capped Interpretable variant scores mean R2 0.18 / F1 0.38 at complexity ≈23, while the unconstrained Best variant — chosen purely by in-sample r2 with no complexity limit — scores *worse* (R2 0.08 / F1 0.26) despite being over twice as complex (≈53). The least-constrained fit is not the best-generalizing one.
- **sys_bp is easier than dia_bp** (mean F1 0.385 vs 0.332) and its F1 climbs steadily with age (0.353 at 10 → 0.417 at 20, dipping to 0.404 at 22); dia_bp stays flat (~0.33) across every age.
- **`recent` is the strongest input type for sys_bp** specifically (mean F1 0.430, vs PGSto5 0.400, to5 0.390, PGS 0.321) — but the four input types are nearly indistinguishable for dia_bp (0.328–0.339), i.e. adding recency/PGS information helps predict systolic but not diastolic pressure.
- **Own-target childhood BP dominates conventional-model feature importance**, more so for diastolic than systolic. Across 200 (model × combo) top-feature picks in `bp_sensitivity_all_feature.csv` (RandomForest/ExtraTrees/XGBoost/ElasticNet/MLP-SHAP): **cbp2_yr5** (diastolic BP at year 5) is #1 in 46/100 dia_bp instances, **cbp1_yr5** (systolic BP at year 5) is #1 in 26/100 sys_bp instances. Secondary drivers: sex (`sex_x`/`sex01`, 31 combined, mostly sys_bp) and two polygenic scores, **PGS004831** (20, sys_bp) and **PGS004604** (17, dia_bp) — the same PGS004604 also recurs as DeepPySR's top sensitivity driver in its Best formulas.

## Overview plots

- [Best model vs age — all_feature (both targets)](results_bp/bp_models_vs_age_all_feature.png)
- [Best input type vs age — all_feature (both targets)](results_bp/bp_input_types_vs_age_all_feature.png)
- [Models vs age — PGS, all_feature](results_bp/bp_models_vs_age_PGS_all_feature.png)
- [Models vs age — PGSto5, all_feature](results_bp/bp_models_vs_age_PGSto5_all_feature.png)
- [Models vs age — to5, all_feature](results_bp/bp_models_vs_age_to5_all_feature.png)
- [Models vs age — recent, all_feature](results_bp/bp_models_vs_age_recent_all_feature.png)

## Per (target, age) plots

Scatter (true vs. predicted), confusion matrix (clinical bins), feature-importance sensitivity heatmap (ElasticNet/ExtraTrees/RandomForest/XGBoost + MLP SHAP), and DeepPySR best-vs-interpretable permutation sensitivity, for each input type / target / age, all_feature only.

### PGS

| Target | Age | Scatter | Confusion matrix | Sensitivity | DeepPySR sensitivity |
|---|---|---|---|---|---|
| sys_bp | 10 | [scatter](results_bp/results_bp_PGS/scatter_predictions_all_feature/age_10_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix_all_feature/age_10_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_10_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGS/age_10_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 14 | [scatter](results_bp/results_bp_PGS/scatter_predictions_all_feature/age_14_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix_all_feature/age_14_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_14_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGS/age_14_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 17 | [scatter](results_bp/results_bp_PGS/scatter_predictions_all_feature/age_17_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix_all_feature/age_17_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_17_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGS/age_17_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 20 | [scatter](results_bp/results_bp_PGS/scatter_predictions_all_feature/age_20_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix_all_feature/age_20_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_20_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGS/age_20_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 22 | [scatter](results_bp/results_bp_PGS/scatter_predictions_all_feature/age_22_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix_all_feature/age_22_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_22_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGS/age_22_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 10 | [scatter](results_bp/results_bp_PGS/scatter_predictions_all_feature/age_10_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix_all_feature/age_10_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_10_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGS/age_10_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 14 | [scatter](results_bp/results_bp_PGS/scatter_predictions_all_feature/age_14_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix_all_feature/age_14_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_14_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGS/age_14_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 17 | [scatter](results_bp/results_bp_PGS/scatter_predictions_all_feature/age_17_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix_all_feature/age_17_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_17_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGS/age_17_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 20 | [scatter](results_bp/results_bp_PGS/scatter_predictions_all_feature/age_20_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix_all_feature/age_20_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_20_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGS/age_20_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 22 | [scatter](results_bp/results_bp_PGS/scatter_predictions_all_feature/age_22_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix_all_feature/age_22_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_22_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGS/age_22_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |

### PGSto5

| Target | Age | Scatter | Confusion matrix | Sensitivity | DeepPySR sensitivity |
|---|---|---|---|---|---|
| sys_bp | 10 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions_all_feature/age_10_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix_all_feature/age_10_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_10_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGSto5/age_10_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 14 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions_all_feature/age_14_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix_all_feature/age_14_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_14_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGSto5/age_14_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 17 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions_all_feature/age_17_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix_all_feature/age_17_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_17_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGSto5/age_17_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 20 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions_all_feature/age_20_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix_all_feature/age_20_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_20_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGSto5/age_20_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 22 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions_all_feature/age_22_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix_all_feature/age_22_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_22_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGSto5/age_22_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 10 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions_all_feature/age_10_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix_all_feature/age_10_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_10_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGSto5/age_10_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 14 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions_all_feature/age_14_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix_all_feature/age_14_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_14_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGSto5/age_14_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 17 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions_all_feature/age_17_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix_all_feature/age_17_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_17_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGSto5/age_17_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 20 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions_all_feature/age_20_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix_all_feature/age_20_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_20_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGSto5/age_20_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 22 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions_all_feature/age_22_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix_all_feature/age_22_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_22_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_PGSto5/age_22_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |

### to5

| Target | Age | Scatter | Confusion matrix | Sensitivity | DeepPySR sensitivity |
|---|---|---|---|---|---|
| sys_bp | 10 | [scatter](results_bp/results_bp_to5/scatter_predictions_all_feature/age_10_sys_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix_all_feature/age_10_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_10_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_to5/age_10_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 14 | [scatter](results_bp/results_bp_to5/scatter_predictions_all_feature/age_14_sys_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix_all_feature/age_14_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_14_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_to5/age_14_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 17 | [scatter](results_bp/results_bp_to5/scatter_predictions_all_feature/age_17_sys_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix_all_feature/age_17_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_17_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_to5/age_17_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 20 | [scatter](results_bp/results_bp_to5/scatter_predictions_all_feature/age_20_sys_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix_all_feature/age_20_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_20_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_to5/age_20_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 22 | [scatter](results_bp/results_bp_to5/scatter_predictions_all_feature/age_22_sys_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix_all_feature/age_22_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_22_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_to5/age_22_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 10 | [scatter](results_bp/results_bp_to5/scatter_predictions_all_feature/age_10_dia_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix_all_feature/age_10_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_10_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_to5/age_10_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 14 | [scatter](results_bp/results_bp_to5/scatter_predictions_all_feature/age_14_dia_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix_all_feature/age_14_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_14_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_to5/age_14_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 17 | [scatter](results_bp/results_bp_to5/scatter_predictions_all_feature/age_17_dia_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix_all_feature/age_17_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_17_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_to5/age_17_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 20 | [scatter](results_bp/results_bp_to5/scatter_predictions_all_feature/age_20_dia_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix_all_feature/age_20_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_20_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_to5/age_20_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 22 | [scatter](results_bp/results_bp_to5/scatter_predictions_all_feature/age_22_dia_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix_all_feature/age_22_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_22_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_to5/age_22_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |

### recent

| Target | Age | Scatter | Confusion matrix | Sensitivity | DeepPySR sensitivity |
|---|---|---|---|---|---|
| sys_bp | 10 | [scatter](results_bp/results_bp_recent/scatter_predictions_all_feature/age_10_sys_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix_all_feature/age_10_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_10_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_recent/age_10_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 14 | [scatter](results_bp/results_bp_recent/scatter_predictions_all_feature/age_14_sys_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix_all_feature/age_14_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_14_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_recent/age_14_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 17 | [scatter](results_bp/results_bp_recent/scatter_predictions_all_feature/age_17_sys_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix_all_feature/age_17_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_17_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_recent/age_17_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 20 | [scatter](results_bp/results_bp_recent/scatter_predictions_all_feature/age_20_sys_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix_all_feature/age_20_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_20_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_recent/age_20_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| sys_bp | 22 | [scatter](results_bp/results_bp_recent/scatter_predictions_all_feature/age_22_sys_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix_all_feature/age_22_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_22_sys_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_recent/age_22_sys_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 10 | [scatter](results_bp/results_bp_recent/scatter_predictions_all_feature/age_10_dia_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix_all_feature/age_10_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_10_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_recent/age_10_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 14 | [scatter](results_bp/results_bp_recent/scatter_predictions_all_feature/age_14_dia_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix_all_feature/age_14_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_14_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_recent/age_14_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 17 | [scatter](results_bp/results_bp_recent/scatter_predictions_all_feature/age_17_dia_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix_all_feature/age_17_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_17_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_recent/age_17_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 20 | [scatter](results_bp/results_bp_recent/scatter_predictions_all_feature/age_20_dia_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix_all_feature/age_20_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_20_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_recent/age_20_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |
| dia_bp | 22 | [scatter](results_bp/results_bp_recent/scatter_predictions_all_feature/age_22_dia_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix_all_feature/age_22_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_22_dia_bp/bp_sensitivity_all_feature.png) | [sensitivity](results_bp/results_bp_recent/age_22_dia_bp/bp_deeppysr_sensitivity_all_feature.png) |

