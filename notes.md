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

| Dataset   | Machine | nit | cvtrain | analysis | deep | convergence | stats | 
|:----------|:--------|:----|:--------|:---------|:-----|:------------|:------|
| BMI       | setonix | 500 | finish  | analysis |      |             |       |
| feynman   | setonix | 500 | finish  | analysis |      |             |       |
| heart     | setonix | 500 | finish  | analysis |      |             |       |
| stroke    | setonix | 100 | finish  | analysis |      |             |       |
| bodyfat   | setonix | 500 | finish  | analysis |      |             |       |
| wine      | setonix | 500 | finish  | analysis |      |             |       |
| diabetes  | setonix | 100 | kan     |          |      |             |       |


steps: cv train -> analysis -> deep, convergence -> stats