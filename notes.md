# github actions

eval "$(ssh-agent -s)"
ssh-add /scratch/pawsey0411/fchen1/.ssh/setonix
---------------------------------------
if new environment

bash
JULIA_SRC="/scratch/pawsey0411/fchen1/deeppysr.jl/python/deeppysr/julia_src"

$JULIA --project=$JULIA_SRC -e '
import Pkg
Pkg.rm("DebugAdapter")
Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()
'
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

| Dataset   | Machine | nit | cvtrain | analysis | stats | deep    | convergence | 
|:----------|:--------|:----|:--------|:---------|-------|:--------|:------------|
| BMI       | setonix | 500 | Yes     | Yes      | Yes   | Yes     | Yes         |
| feynman   | setonix | 500 | Yes     | Yes      | NA    | NA      | yes         |
| heart     | setonix | 500 | Yes     | Yes      | Yes   | Yes     | no          |
| stroke    | setonix | 100 | Yes     | Yes      | Yes   | **bad** | no          |
| bodyfat   | setonix | 500 | Yes     | Yes      | Yes   | Yes     | no           |
| wine      | setonix | 500 | Yes     | Yes      | Yes   | Yes     | no          |
| diabetes  | setonix | 100 | Yes     | Yes      | Yes   | Yes     | no          |
|students   | setonix | 500 | Yes | Yes       | Yes    | Yes      | no          |

steps: cv train -> analysis, stats, deep, convergence

notes: ICC causes convergence unstable
notes: convergence needs to select the best model,

----------------------------------------
# bmiforecast

----------------------------------------
# diab_raine

glucose <6.1 normal, 6.1 - 6.9 prediabetes, >7 diabetes
homa-ir Below 1.0: Optimal diab_raine sensitivity.
1.0 to 1.9: Normal / borderline.
2.0 to 2.9: Early diab_raine resistance.
3.0 and above: Moderate to severe diab_raine resistance