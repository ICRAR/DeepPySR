#!/bin/bash
#SBATCH --job-name=newbmiforecast
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/newbmiforecast.log
export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export DEEPPYSR_PATH="/scratch/pawsey0411/fchen1/deeppysr.jl/python"
export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
export JULIA="$PYTHON_JULIAPKG_PROJECT/pyjuliapkg/install/bin/julia"
export PYTHON_JULIAPKG_OFFLINE=yes
cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
set -e
echo "Starting bmiforecast_baselines at $(date)"
python -u test/newbmiforecast/test_newbmiforecast.py
echo "Finished bmiforecast_baselines at $(date)"
