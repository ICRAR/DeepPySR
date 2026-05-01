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

| Dataset          | Machine | nit | Status    | Analysis |
|:-----------------|:--------|:----|:----------|:---------|
| BMI              | setonix | 500 | running   |          |
| BMI warm         | setonix | 500 | running   |          |
| feynman          | setonix | 500 | running   |          |
| feynman warm     | setonix | 500 | running   |          |
| heart            | setonix | 500 | completed | pysr     |
| heart warm       | setonix | 500 | completed | ----     |
| stroke           | setonix | 100 | completed | wait     |
| stroke warm      | setonix | 100 | completed | ----     |
| bodyfat          | setonix | 500 | completed | pysr     |
| bodyfat warm     | setonix | 500 | completed | ----     |
| wine             | setonix | 500 | completed |          |
| wine warm        | setonix | 500 | running   |          |
| diabetes         | a400    | 100 | completed | wait     |
| diabetes warm    | setonix | 100 | completed | ----     |


# next step
run pysr, analysis, run convergence
run on gwas