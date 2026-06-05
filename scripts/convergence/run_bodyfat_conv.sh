#!/bin/bash
#SBATCH --job-name=bodyfat_conv
#SBATCH --account=pawsey0411
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/bodyfat_conv.log

export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export DEEPPYSR_PATH="/scratch/pawsey0411/fchen1/deeppysr.jl/python"

export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"

cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Starting bodyfat Convergence analysis at $(date)"
python -u test/bodyfat/bodyfat_convergence.py
echo "Finished bodyfat Convergence analysis at $(date)"
