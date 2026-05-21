# github actions

eval "$(ssh-agent -s)"
ssh-add /scratch/pawsey0411/fchen1/.ssh/setonix
---------------------------------------

source /scratch/pawsey0411/fchen1/DeepPySR/.venv/bin/activate
export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
export PYTHON_JULIAPKG_OFFLINE=no
rm -rf /scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env
python -m juliapkg update

----------------------------------------
for file in *.sh; do sbatch "$file"; done

find . -name "overall_metrics.csv" -exec dirname {} \;
----------------------------------------
rm Manifest.toml
julia --project=. -e 'using Pkg; Pkg.update(); Pkg.precompile()'

----------------------------------------

# notes:
stroke, diabetes, diabetes130us are 100 nit, others are 500

| Dataset   | Machine | nit | cvtrain | analysis | stats | deep    | convergence  | 
|:----------|:--------|:----|:--------|:---------|-------|:--------|:-------------|
| BMI       | setonix | 500 | Yes     | Yes      | Yes   | Yes     | Yes          |
| feynman   | setonix | 500 | Yes     | Yes      | NA    | NA      | yes          |
| heart     | setonix | 500 | Yes     | Yes      | Yes   | Yes     | no           |
| stroke    | setonix | 100 | Yes     | Yes      | Yes   | **bad** | no           |
| bodyfat   | setonix | 500 | Yes     | Yes      | Yes   | Yes     | **unstable** |
| wine      | setonix | 500 | Yes     | Yes      | Yes   | Yes     | no           |
| diabetes  | setonix | 100 | Yes     | Yes      | Yes   | Yes     | no           |
|students   | setonix | 500 | Yes | Yes       | Yes    | Yes      | no           |
bmiforecast, on a400
rerun deep for bmi longitudinal and students, for plots

steps: cv train -> analysis, stats, deep, convergence

notes: ICC causes convergence unstable
notes: convergence needs to select the best model,
fix the bmiforecast problem in vscode
