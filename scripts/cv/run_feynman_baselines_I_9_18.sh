#!/bin/bash
#SBATCH --job-name=feynman_baselines_I_9_18
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/feynman_baselines_I_9_18.log

export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export DEEPPYSR_PATH="/scratch/pawsey0411/fchen1/deeppysr.jl/python"
#export PYTHONPATH="$PROJECT_ROOT:$DEEPPYSR_PATH:$PYTHONPATH"

export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
export JULIA="$PYTHON_JULIAPKG_PROJECT/pyjuliapkg/install/bin/julia"
$JULIA --project=$PYTHON_JULIAPKG_PROJECT -e 'using Pkg; Pkg.status()'
export PYTHON_JULIAPKG_OFFLINE=no

cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python -m juliapkg update
set -e

echo "Starting feynman_baselines_I_9_18 at $(date)"
python -u test/feynman/test_baselines_pysr_feynman.py --eq_name I.9.18
echo "Finished feynman_baselines_I_9_18 at $(date)"

