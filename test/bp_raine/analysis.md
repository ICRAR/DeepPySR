# Blood pressure (RAINE) results index

Links to the plots produced by `analysis_bp.py`. Run it to regenerate everything under `results_bp/`. (Feature-selection is a single fixed step — the models are trained on the full feature set; there is no separate feature-selected variant to compare against.)

## Meta

- **Targets** (2): sys_bp, dia_bp — both in mmHg.
- **Input types** (4): `PGS`, `to5`, `PGSto5`, `recent`. (The longitudinal-feature-engineering variant was dropped from bp_raine entirely — the CV scripts and this analysis no longer support it.)
- **Ages**: 10, 14, 17, 20, 22.
- **Models**: DeepPySR, PySR, KAN, ElasticNet, ExtraTrees, MLP, RandomForest, XGBoost.
- **"Best" row selection**: max F1 (macro) on the clinical bins below, per (target, age, input_type, model), collapsed across config.

**Clinical bin definitions** (mmHg, AHA/ACC 2017 adult categories collapsed to 3 tiers — see `BP_BIN_EDGES` in `analysis_bp.py`):

| Target | Class 0 | Class 1 | Class 2 |
|---|---|---|---|
| sys_bp | Normal: < 120 | Elevated: 120–139 | High: ≥ 140 |
| dia_bp | Normal: < 80 | Elevated: 80–89 | High: ≥ 90 |

No pediatric (age/sex/height-percentile) norms are applied even at ages 10/14 — this F1_macro is a rough clinical-relevance signal for model comparison, not a diagnostic label (same simplification `analysis_lipids.py` makes for its own adolescent ages).

## Key findings

*F1 = how reliably a model sorts a child into the right BP category (Normal/Elevated/High), not just how close its number is. Four data sources compared: `PGS` (genetics only, available from birth), `to5` (growth/health data to age 5), `PGSto5` (both), `recent` (most recent measurements near the prediction age).*

- **DeepPySR predicts best overall** — top model in 22/40 age×target×source combinations (best average F1, 0.385), even though tree models (RandomForest/ExtraTrees) explain slightly more raw variance. DeepPySR also outputs an actual formula, not a black box.
- **Its simpler formula is the more trustworthy one**: the complexity-capped "Interpretable" formula generalises better (F1 0.38) than the unconstrained "Best" one (F1 0.26) — the more elaborate equation was overfit.
- **Systolic BP gets easier to predict with age; diastolic doesn't** (F1 ~0.35→0.42 for systolic vs. a flat ~0.33 for diastolic across ages 10–22).
- **Richer/recent data helps predict systolic BP, not diastolic**: `recent` is clearly the best source for systolic (F1 0.43 vs. 0.32 for genetics-only); for diastolic, all four sources score about the same (~0.33).
- **A child's own earlier BP is the single biggest predictor**, followed by sex and two genetic risk scores — reassuringly, each score matches the trait it's predicting (a systolic-BP score drives systolic predictions, a diastolic-BP score drives diastolic).

[**→ What drives DeepPySR's own formulas, by data source**](results_bp/bp_deeppysr_sensitivity_overview.png) ([data](results_bp/bp_deeppysr_sensitivity_overview.csv))

## Overview plots

- [Best model vs age (both targets)](results_bp/bp_models_vs_age.png)
- [Best input type vs age (both targets)](results_bp/bp_input_types_vs_age.png)
- [Models vs age — PGS](results_bp/bp_models_vs_age_PGS.png)
- [Models vs age — PGSto5](results_bp/bp_models_vs_age_PGSto5.png)
- [Models vs age — to5](results_bp/bp_models_vs_age_to5.png)
- [Models vs age — recent](results_bp/bp_models_vs_age_recent.png)
- [**Permutation sensitivity overview** — what drives DeepPySR's interpretable formulas, by data source](results_bp/bp_deeppysr_sensitivity_overview.png)

## Per (target, age) plots

Scatter (true vs. predicted), confusion matrix (clinical bins), feature-importance sensitivity heatmap (ElasticNet/ExtraTrees/RandomForest/XGBoost + MLP SHAP), and DeepPySR best-vs-interpretable permutation sensitivity, for each input type / target / age.

### PGS

| Target | Age | Scatter | Confusion matrix | Sensitivity | DeepPySR sensitivity |
|---|---|---|---|---|---|
| sys_bp | 10 | [scatter](results_bp/results_bp_PGS/scatter_predictions/age_10_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix/age_10_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_10_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGS/age_10_sys_bp/bp_deeppysr_sensitivity.png) |
| sys_bp | 14 | [scatter](results_bp/results_bp_PGS/scatter_predictions/age_14_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix/age_14_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_14_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGS/age_14_sys_bp/bp_deeppysr_sensitivity.png) |
| sys_bp | 17 | [scatter](results_bp/results_bp_PGS/scatter_predictions/age_17_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix/age_17_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_17_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGS/age_17_sys_bp/bp_deeppysr_sensitivity.png) |
| sys_bp | 20 | [scatter](results_bp/results_bp_PGS/scatter_predictions/age_20_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix/age_20_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_20_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGS/age_20_sys_bp/bp_deeppysr_sensitivity.png) |
| sys_bp | 22 | [scatter](results_bp/results_bp_PGS/scatter_predictions/age_22_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix/age_22_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_22_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGS/age_22_sys_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 10 | [scatter](results_bp/results_bp_PGS/scatter_predictions/age_10_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix/age_10_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_10_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGS/age_10_dia_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 14 | [scatter](results_bp/results_bp_PGS/scatter_predictions/age_14_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix/age_14_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_14_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGS/age_14_dia_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 17 | [scatter](results_bp/results_bp_PGS/scatter_predictions/age_17_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix/age_17_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_17_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGS/age_17_dia_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 20 | [scatter](results_bp/results_bp_PGS/scatter_predictions/age_20_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix/age_20_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_20_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGS/age_20_dia_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 22 | [scatter](results_bp/results_bp_PGS/scatter_predictions/age_22_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGS/conf_matrix/age_22_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGS/age_22_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGS/age_22_dia_bp/bp_deeppysr_sensitivity.png) |

### PGSto5

| Target | Age | Scatter | Confusion matrix | Sensitivity | DeepPySR sensitivity |
|---|---|---|---|---|---|
| sys_bp | 10 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions/age_10_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix/age_10_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_10_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGSto5/age_10_sys_bp/bp_deeppysr_sensitivity.png) |
| sys_bp | 14 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions/age_14_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix/age_14_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_14_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGSto5/age_14_sys_bp/bp_deeppysr_sensitivity.png) |
| sys_bp | 17 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions/age_17_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix/age_17_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_17_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGSto5/age_17_sys_bp/bp_deeppysr_sensitivity.png) |
| sys_bp | 20 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions/age_20_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix/age_20_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_20_sys_bp/bp_sensitivity.png) | — |
| sys_bp | 22 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions/age_22_sys_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix/age_22_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_22_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGSto5/age_22_sys_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 10 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions/age_10_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix/age_10_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_10_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGSto5/age_10_dia_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 14 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions/age_14_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix/age_14_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_14_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGSto5/age_14_dia_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 17 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions/age_17_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix/age_17_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_17_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGSto5/age_17_dia_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 20 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions/age_20_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix/age_20_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_20_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGSto5/age_20_dia_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 22 | [scatter](results_bp/results_bp_PGSto5/scatter_predictions/age_22_dia_bp_scatter.png) | [confmat](results_bp/results_bp_PGSto5/conf_matrix/age_22_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_PGSto5/age_22_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_PGSto5/age_22_dia_bp/bp_deeppysr_sensitivity.png) |

### to5

| Target | Age | Scatter | Confusion matrix | Sensitivity | DeepPySR sensitivity |
|---|---|---|---|---|---|
| sys_bp | 10 | [scatter](results_bp/results_bp_to5/scatter_predictions/age_10_sys_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix/age_10_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_10_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_to5/age_10_sys_bp/bp_deeppysr_sensitivity.png) |
| sys_bp | 14 | [scatter](results_bp/results_bp_to5/scatter_predictions/age_14_sys_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix/age_14_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_14_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_to5/age_14_sys_bp/bp_deeppysr_sensitivity.png) |
| sys_bp | 17 | [scatter](results_bp/results_bp_to5/scatter_predictions/age_17_sys_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix/age_17_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_17_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_to5/age_17_sys_bp/bp_deeppysr_sensitivity.png) |
| sys_bp | 20 | [scatter](results_bp/results_bp_to5/scatter_predictions/age_20_sys_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix/age_20_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_20_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_to5/age_20_sys_bp/bp_deeppysr_sensitivity.png) |
| sys_bp | 22 | [scatter](results_bp/results_bp_to5/scatter_predictions/age_22_sys_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix/age_22_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_22_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_to5/age_22_sys_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 10 | [scatter](results_bp/results_bp_to5/scatter_predictions/age_10_dia_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix/age_10_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_10_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_to5/age_10_dia_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 14 | [scatter](results_bp/results_bp_to5/scatter_predictions/age_14_dia_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix/age_14_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_14_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_to5/age_14_dia_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 17 | [scatter](results_bp/results_bp_to5/scatter_predictions/age_17_dia_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix/age_17_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_17_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_to5/age_17_dia_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 20 | [scatter](results_bp/results_bp_to5/scatter_predictions/age_20_dia_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix/age_20_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_20_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_to5/age_20_dia_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 22 | [scatter](results_bp/results_bp_to5/scatter_predictions/age_22_dia_bp_scatter.png) | [confmat](results_bp/results_bp_to5/conf_matrix/age_22_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_to5/age_22_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_to5/age_22_dia_bp/bp_deeppysr_sensitivity.png) |

### recent

| Target | Age | Scatter | Confusion matrix | Sensitivity | DeepPySR sensitivity |
|---|---|---|---|---|---|
| sys_bp | 10 | [scatter](results_bp/results_bp_recent/scatter_predictions/age_10_sys_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix/age_10_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_10_sys_bp/bp_sensitivity.png) | — |
| sys_bp | 14 | [scatter](results_bp/results_bp_recent/scatter_predictions/age_14_sys_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix/age_14_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_14_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_recent/age_14_sys_bp/bp_deeppysr_sensitivity.png) |
| sys_bp | 17 | [scatter](results_bp/results_bp_recent/scatter_predictions/age_17_sys_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix/age_17_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_17_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_recent/age_17_sys_bp/bp_deeppysr_sensitivity.png) |
| sys_bp | 20 | [scatter](results_bp/results_bp_recent/scatter_predictions/age_20_sys_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix/age_20_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_20_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_recent/age_20_sys_bp/bp_deeppysr_sensitivity.png) |
| sys_bp | 22 | [scatter](results_bp/results_bp_recent/scatter_predictions/age_22_sys_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix/age_22_sys_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_22_sys_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_recent/age_22_sys_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 10 | [scatter](results_bp/results_bp_recent/scatter_predictions/age_10_dia_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix/age_10_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_10_dia_bp/bp_sensitivity.png) | — |
| dia_bp | 14 | [scatter](results_bp/results_bp_recent/scatter_predictions/age_14_dia_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix/age_14_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_14_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_recent/age_14_dia_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 17 | [scatter](results_bp/results_bp_recent/scatter_predictions/age_17_dia_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix/age_17_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_17_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_recent/age_17_dia_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 20 | [scatter](results_bp/results_bp_recent/scatter_predictions/age_20_dia_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix/age_20_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_20_dia_bp/bp_sensitivity.png) | [sensitivity](results_bp/results_bp_recent/age_20_dia_bp/bp_deeppysr_sensitivity.png) |
| dia_bp | 22 | [scatter](results_bp/results_bp_recent/scatter_predictions/age_22_dia_bp_scatter.png) | [confmat](results_bp/results_bp_recent/conf_matrix/age_22_dia_bp_confmat.png) | [sensitivity](results_bp/results_bp_recent/age_22_dia_bp/bp_sensitivity.png) | — |

