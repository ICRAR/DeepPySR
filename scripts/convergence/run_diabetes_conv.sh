#!/bin/bash
#SBATCH --job-name=diabetes_conv
#SBATCH --account=pawsey0411
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/diabetes_conv.log

export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export MYPYSR_PATH="/scratch/pawsey0411/fchen1/mypysr.jl/python"

export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"

cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Starting diabetes Convergence analysis at $(date)"
python -u test/diabetes/diabetes_convergence.py
echo "Finished diabetes Convergence analysis at $(date)"
