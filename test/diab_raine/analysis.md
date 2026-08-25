# Insulin (RAINE) results index

Links to the plots produced by `analysis_insulin.py`. Run it to regenerate everything under `results_insulin/`.

## Meta

- **Target** (1): fasting insulin (`diab_raine`), mU/L — a continuous value, not a clinical category, so results here are scored by **R2** (% of variance explained) and **Pearson r** (correlation strength), not classification metrics.
- **Data sources** (4): `PGS` (genetics only), `to8` (growth/health data to age 8), `PGSto8` (both), `recent` (most recent available measurements near the prediction age).
- **Ages**: 14, 17, 20, 22, 28.
- **Models**: DeepPySR, PySR, KAN, ElasticNet, ExtraTrees, MLP, RandomForest, XGBoost.
- **"Best" row selection**: max R2 per (age, model), independently per data source.

## Key findings

- **Early-life data alone can't predict later insulin levels.** Using genetics (`PGS`), early growth/health data (`to8`), or both together (`PGSto8`), every model scores close to R2 = 0 at every age — none of them beat a naive "predict the average" baseline by any meaningful margin.
- **Recent measurements are a different story.** Once concurrent/recent data (`recent`) is used, accuracy climbs sharply and keeps improving with age — R2 rises from 0.11 at age 14 to 0.64 at age 28 (Pearson r 0.34 → 0.81) for the best model.
- **DeepPySR is the standout model overall** — highest average R2 across all 20 age×source combinations (0.088, vs RandomForest's 0.049), driven almost entirely by its strong performance on `recent`. By raw win count RandomForest edges ahead (8/20 vs 7/20), but its wins are small margins on the near-unpredictable `PGS`/`to8`/`PGSto8` sources, not on the one source where the task is actually learnable.
- **No accuracy trade-off for interpretability on `recent`**: DeepPySR's complexity-capped "Interpretable" formula is *identical* to its unconstrained "Best" one at every age — the best-performing formula already happened to be simple enough to read.
- **What drives the `recent` predictions changes with age, but the theme is consistent**: body-composition measures (BMI at 14/17/22, waist circumference at 28) and the person's own earlier insulin/diabetes-related values recur as top drivers across ages, even though the single most important variable differs age to age. For the early-life-only sources, a handful of genetic risk scores and early growth variables (e.g. `PGS001351` for `PGS`, `f_vege`/`m_headsz` for `to8`) recur most consistently — see the plot below.

[**→ What drives DeepPySR's own formulas, by data source**](results_insulin/insulin_deeppysr_sensitivity_overview.png) ([data](results_insulin/insulin_deeppysr_sensitivity_overview.csv)) — for the fuller picture across all models, see the per-age heatmaps linked below.

## Overview plots

- [Combined DeepPySR comparison across data sources](results_insulin/insulin_deeppysr_metrics_vs_age_combined.png) ([data](results_insulin/insulin_deeppysr_combined_metrics.csv))
- [Metrics vs age — PGS](results_insulin/results_insulin_PGS/insulin_metrics_vs_age.png)
- [Metrics vs age — to8](results_insulin/results_insulin_to8/insulin_metrics_vs_age.png)
- [Metrics vs age — PGSto8](results_insulin/results_insulin_PGSto8/insulin_metrics_vs_age.png)
- [Metrics vs age — recent](results_insulin/results_insulin_recent/insulin_metrics_vs_age.png)
- [Feature importance by model — PGS](results_insulin/results_insulin_PGS/feature_importance_by_model.png)
- [Feature importance by model — to8](results_insulin/results_insulin_to8/feature_importance_by_model.png)
- [Feature importance by model — PGSto8](results_insulin/results_insulin_PGSto8/feature_importance_by_model.png)
- [Feature importance by model — recent](results_insulin/results_insulin_recent/feature_importance_by_model.png)

## Per (data source, age) plots

Scatter (true vs. predicted, formula models), feature-importance sensitivity heatmap (ElasticNet/ExtraTrees/RandomForest/XGBoost + MLP SHAP), and DeepPySR best-vs-interpretable permutation sensitivity, for each data source / age.

### PGS

| Age | Scatter | Sensitivity | DeepPySR sensitivity |
|---|---|---|---|
| 14 | [scatter](results_insulin/results_insulin_PGS/formula_predictions/age_14_scatter.png) | [sensitivity](results_insulin/results_insulin_PGS/age_14_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_PGS/age_14_diab_raine/insulin_deeppysr_sensitivity.png) |
| 17 | [scatter](results_insulin/results_insulin_PGS/formula_predictions/age_17_scatter.png) | [sensitivity](results_insulin/results_insulin_PGS/age_17_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_PGS/age_17_diab_raine/insulin_deeppysr_sensitivity.png) |
| 20 | [scatter](results_insulin/results_insulin_PGS/formula_predictions/age_20_scatter.png) | [sensitivity](results_insulin/results_insulin_PGS/age_20_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_PGS/age_20_diab_raine/insulin_deeppysr_sensitivity.png) |
| 22 | [scatter](results_insulin/results_insulin_PGS/formula_predictions/age_22_scatter.png) | [sensitivity](results_insulin/results_insulin_PGS/age_22_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_PGS/age_22_diab_raine/insulin_deeppysr_sensitivity.png) |
| 28 | [scatter](results_insulin/results_insulin_PGS/formula_predictions/age_28_scatter.png) | [sensitivity](results_insulin/results_insulin_PGS/age_28_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_PGS/age_28_diab_raine/insulin_deeppysr_sensitivity.png) |

### to8

| Age | Scatter | Sensitivity | DeepPySR sensitivity |
|---|---|---|---|
| 14 | [scatter](results_insulin/results_insulin_to8/formula_predictions/age_14_scatter.png) | [sensitivity](results_insulin/results_insulin_to8/age_14_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_to8/age_14_diab_raine/insulin_deeppysr_sensitivity.png) |
| 17 | [scatter](results_insulin/results_insulin_to8/formula_predictions/age_17_scatter.png) | [sensitivity](results_insulin/results_insulin_to8/age_17_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_to8/age_17_diab_raine/insulin_deeppysr_sensitivity.png) |
| 20 | [scatter](results_insulin/results_insulin_to8/formula_predictions/age_20_scatter.png) | [sensitivity](results_insulin/results_insulin_to8/age_20_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_to8/age_20_diab_raine/insulin_deeppysr_sensitivity.png) |
| 22 | [scatter](results_insulin/results_insulin_to8/formula_predictions/age_22_scatter.png) | [sensitivity](results_insulin/results_insulin_to8/age_22_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_to8/age_22_diab_raine/insulin_deeppysr_sensitivity.png) |
| 28 | [scatter](results_insulin/results_insulin_to8/formula_predictions/age_28_scatter.png) | [sensitivity](results_insulin/results_insulin_to8/age_28_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_to8/age_28_diab_raine/insulin_deeppysr_sensitivity.png) |

### PGSto8

| Age | Scatter | Sensitivity | DeepPySR sensitivity |
|---|---|---|---|
| 14 | [scatter](results_insulin/results_insulin_PGSto8/formula_predictions/age_14_scatter.png) | [sensitivity](results_insulin/results_insulin_PGSto8/age_14_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_PGSto8/age_14_diab_raine/insulin_deeppysr_sensitivity.png) |
| 17 | [scatter](results_insulin/results_insulin_PGSto8/formula_predictions/age_17_scatter.png) | [sensitivity](results_insulin/results_insulin_PGSto8/age_17_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_PGSto8/age_17_diab_raine/insulin_deeppysr_sensitivity.png) |
| 20 | [scatter](results_insulin/results_insulin_PGSto8/formula_predictions/age_20_scatter.png) | [sensitivity](results_insulin/results_insulin_PGSto8/age_20_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_PGSto8/age_20_diab_raine/insulin_deeppysr_sensitivity.png) |
| 22 | [scatter](results_insulin/results_insulin_PGSto8/formula_predictions/age_22_scatter.png) | [sensitivity](results_insulin/results_insulin_PGSto8/age_22_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_PGSto8/age_22_diab_raine/insulin_deeppysr_sensitivity.png) |
| 28 | [scatter](results_insulin/results_insulin_PGSto8/formula_predictions/age_28_scatter.png) | [sensitivity](results_insulin/results_insulin_PGSto8/age_28_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_PGSto8/age_28_diab_raine/insulin_deeppysr_sensitivity.png) |

### recent

| Age | Scatter | Sensitivity | DeepPySR sensitivity |
|---|---|---|---|
| 14 | [scatter](results_insulin/results_insulin_recent/formula_predictions/age_14_scatter.png) | [sensitivity](results_insulin/results_insulin_recent/age_14_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_recent/age_14_diab_raine/insulin_deeppysr_sensitivity.png) |
| 17 | [scatter](results_insulin/results_insulin_recent/formula_predictions/age_17_scatter.png) | [sensitivity](results_insulin/results_insulin_recent/age_17_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_recent/age_17_diab_raine/insulin_deeppysr_sensitivity.png) |
| 20 | [scatter](results_insulin/results_insulin_recent/formula_predictions/age_20_scatter.png) | [sensitivity](results_insulin/results_insulin_recent/age_20_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_recent/age_20_diab_raine/insulin_deeppysr_sensitivity.png) |
| 22 | [scatter](results_insulin/results_insulin_recent/formula_predictions/age_22_scatter.png) | [sensitivity](results_insulin/results_insulin_recent/age_22_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_recent/age_22_diab_raine/insulin_deeppysr_sensitivity.png) |
| 28 | [scatter](results_insulin/results_insulin_recent/formula_predictions/age_28_scatter.png) | [sensitivity](results_insulin/results_insulin_recent/age_28_diab_raine/insulin_sensitivity.png) | [sensitivity](results_insulin/results_insulin_recent/age_28_diab_raine/insulin_deeppysr_sensitivity.png) |
