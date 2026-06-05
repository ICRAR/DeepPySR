#!/bin/bash
#SBATCH --job-name=bmiforecast_pysr_aps0.1
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/bmiforecast_pysr_aps0.1.log
export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export DEEPPYSR_PATH="/scratch/pawsey0411/fchen1/deeppysr.jl/python"
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
echo "Starting bmiforecast_pysr_aps0.1 at $(date)"
python -u test/bmiforecast/test_baselines_bmiforecast.py --aps 0.1
echo "Finished bmiforecast_pysr_aps0.1 at $(date)"
