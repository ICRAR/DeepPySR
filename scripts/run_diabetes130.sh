#!/bin/bash
#SBATCH --job-name=diabetes130
#SBATCH --account=pawsey0411
#SBATCH --time=4-00:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/diabetes130.log

# Locate the project root from this script's location
PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"

# Use the project root and virtual environment
cd $PROJECT_ROOT
if [ -e "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "ERROR: virtual environment not found at $PROJECT_ROOT/.venv"
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
set -e

echo "Starting diabetes130 test at $(date)"

python -u test/diabetes130us/test_all_models_diab130.py
STATUS=$?
echo "Finished diabetes130 test at $(date) with status $STATUS"
exit $STATUS