#!/bin/bash
#SBATCH --job-name=bmi_baselines_age17
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/bmi_baselines_age17.log

export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export DEEPPYSR_PATH="/scratch/pawsey0411/fchen1/deeppysr.jl/python"
#export PYTHONPATH="$PROJECT_ROOT:$DEEPPYSR_PATH:$PYTHONPATH"

export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
export JULIA="$PYTHON_JULIAPKG_PROJECT/pyjuliapkg/install/bin/julia"
cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python -m juliapkg update
export PYTHON_JULIAPKG_OFFLINE=no

$JULIA --project=$PYTHON_JULIAPKG_PROJECT -e 'import Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()'
set -e

echo "Starting bmi_baselines_age17 at $(date)"
python -u test/bmi/test_baselines_pysr_bmi.py --setting age_specific --age 17
echo "Finished bmi_baselines_age17 at $(date)"

