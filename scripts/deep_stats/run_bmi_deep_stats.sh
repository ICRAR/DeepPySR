#!/bin/bash
#SBATCH --job-name=bmi_ds
#SBATCH --account=pawsey0411
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/bmi_deep_stats.log

export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export MYPYSR_PATH="/scratch/pawsey0411/fchen1/mypysr.jl/python"

export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"

cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run Deep Analysis
echo "Starting bmi Deep analysis at $(date)"
python -u test/bmi/deep_analysis.py

# Run Stats Analysis
echo "Starting bmi Stats analysis at $(date)"
python -u test/bmi/run_stats_analysis.py
echo "Finished bmi Deep and Stats analysis at $(date)"
