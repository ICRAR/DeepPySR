# DeepPySR — Experiments & Reproducibility

This repository contains the experiment scripts, cross-validation pipelines, and
result artifacts used to produce the results in our paper on **DeepPySR**, a
multi-layer symbolic regression method that extends symbolic regression with
dynamic variable pruning (DVPS), exponential Pareto selection (EPS), and
hierarchical (multi-layer) equation discovery.

The DeepPySR **method implementation** (the Julia-backed Python package) lives in a
separate repository: [`deeppysr.jl`](https://github.com/ICRAR/deeppysr.jl). This
repo depends on that package and focuses on applying it to benchmark datasets,
running baselines, and reproducing the paper's tables and figures.

If you're looking for the algorithm itself (`PySRRegressor`, `DeepPySR`, tuning
`vps`/`vpr`/`aps`, etc.), see the `deeppysr.jl` README. This repo's README covers
how the paper's experiments are organized and how to (re)run them.

---

## Repository layout

```
scripts/            SLURM/HPC job scripts used to launch CV runs on our cluster
  cv/<dataset>/        Grid-search / CV launch scripts per dataset (deeppysr, pysr, baselines)
  convergence/          Convergence-comparison job scripts (Feynman + BMI only)
  deep_stats/           Statistics-generation scripts for DeepPySR-specific analysis

test/                Python entry points, shared utilities, and result CSVs/figures
  __init__.py, analysis_utils.py, eval_utils.py,
  model_utils.py, convergence_utils.py,
  deep_analysis_utils.py            Shared helpers imported by every dataset's test scripts
  <dataset>/                        One folder per benchmark dataset (see below)

test_data/           Raw datasets (NOT tracked in git — see "Data availability")

setup.py / pyproject.toml   Legacy packaging files (superseded by deeppysr.jl; kept for reference)
```

### Datasets covered (`test/<dataset>/`)

| Folder | Task | Data source | Public? |
|---|---|---|---|
| `feynman` | Symbolic recovery (I.8.14, I.13.4, I.6.2a, I.9.18, I.32.17) | Feynman Symbolic Regression Database | Yes |
| `bodyfat` | Regression (R²) | UCI-style body fat dataset | Yes |
| `wineQuality` | Regression (R²), red & white | UCI Wine Quality | Yes |
| `studentPerformance` | Regression (R²), Math & Portuguese | UCI Student Performance | Yes |
| `heart` | Classification (F1) | UCI Heart Disease | Yes |
| `stroke` | Classification (F1) | Kaggle stroke dataset | Yes |
| `diabetes` | Classification (F1) | CDC BRFSS Diabetes Health Indicators (70k samples, 21 features) | Yes |
| `bmi`, `newbmiforecast` | Longitudinal/age-specific regression (R²) | RAINE Study Gen2 cohort (BMI, polygenic scores) | **No — restricted** |
| `diab_raine` | Insulin/glucose/HOMA-IR regression | RAINE Study Gen2 cohort | **No — restricted** |

Each dataset folder generally contains:
- `test_all_models_bmi.py`-style scripts (naming varies per dataset) — the main entry
  points that run DeepPySR, plain PySR, and ML baselines (ElasticNet, RandomForest,
  XGBoost, MLP, KAN, KANSym) under cross-validation.
- `*_utils.py` — dataset-specific loaders and preprocessing.
- `run_stats_analysis.py` / `deep_analysis.py` — post-hoc statistics (Wilcoxon
  signed-rank tests, feature importance aggregation, Pareto-front plots).
- `aggregated_results.csv`, `wilcoxon_results*.csv`, `interpretable_deeppysr_formulas.csv`,
  `*_best_models_metrics.csv` — the numeric results and discovered formulas reported
  in the paper. These contain **aggregate metrics and symbolic formulas only**, not
  raw records, and are safe to share even for restricted-data experiments.
- `results_*_all/`, `*.png` — generated figures (Pareto fronts, feature importance,
  convergence, ablations).

## Data availability

`test_data/` is excluded from version control (see `.gitignore`) because it mixes:
- **Public benchmark data** (UCI datasets, Feynman equations, CDC growth reference
  tables) that you can download yourself — see each dataset's loader in
  `test/<dataset>/*_utils.py` for the expected file layout and source.
- **Restricted-access data**: the RAINE Study Gen2 cohort (BMI trajectories,
  polygenic risk scores, insulin/glucose/HOMA-IR measures) used in `bmi`,
  `newbmiforecast`, and `diab_raine`. This data requires an approved data access
  application to the RAINE Study and is **not distributed here**. Scripts for these
  datasets are included for transparency and reproducibility of the *method*, but
  will not run without independently obtaining RAINE data and placing it under
  `test_data/Health/raine/` in the layout expected by `test/diab_raine/data_utils.py`
  and `test/bmi/bmi_utils.py`.

The CSV/PNG artifacts committed under `test/diab_raine/` and `test/bmi/` are
**aggregate outputs** (metrics, discovered formulas, summary plots) derived from
the restricted data, not the underlying patient-level records.

## Running the experiments

Each dataset's main script accepts CLI arguments for grid search over DeepPySR's
`vps` (variable pruning schedule), `vpr` (pruning ratio), and `aps` (adaptive
parsimony scaling) parameters, e.g.:

```bash
python -u test/bmi/test_all_models_bmi.py --setting age_specific --age 10 --vps 25
```

The `scripts/cv/<dataset>/` folders contain the exact SLURM scripts used to launch
these runs on the Pawsey Setonix HPC cluster. **They are cluster-specific**:
paths like `/scratch/pawsey0411/fchen1/...` and the `--account=pawsey0411` SLURM
directive belong to our allocation and must be edited for any other environment.
`generate_scripts.sh` shows the template used to generate them.

Typical pipeline order per dataset: **CV train → analysis → stats → deep analysis
→ convergence** (convergence comparisons were only run for Feynman and BMI).
Iteration counts differ by dataset: 500 PySR iterations for most datasets, 100 for
`stroke`, `diabetes`, and `diab_raine` (results for the latter are reported in the
paper's Supplementary Material).

## Installation

```bash
# 1. Install the DeepPySR method package (requires Julia >= 1.9)
git clone https://github.com/ICRAR/deeppysr.jl.git
cd deeppysr.jl && pip install -e .

# 2. Install this repo's experiment dependencies
cd ../DeepPySR
pip install -e .   # see setup.py for the full dependency list
```

Optional: [AI Feynman 2.0](https://github.com/SJ001/AI-Feynman) and
[pykan](https://github.com/KindXiaoming/pykan) are used as baselines in some
scripts and are imported defensively (skipped if unavailable).

## Citation

If you use this code, please cite our paper (citation details to be added upon
publication).