#!/bin/bash
#SBATCH --job-name=bodyfat
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/bodyfat.log

# Locate the project root from this script's location
PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"

# Force Julia to use the correct Python executable
export PYTHON_JL_RUNTIME_PYTHON="$PROJECT_ROOT/.venv/bin/python"
# Help juliacall find its own libraries
export LD_LIBRARY_PATH="$PROJECT_ROOT/.venv/julia_env/pyjuliapkg/install/lib:$LD_LIBRARY_PATH"
# Disable PyCall/PythonCall automatic updates during the run
export JULIA_PYTHONCALL_EXE="@PyCall"

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

echo "Starting bodyfat test at $(date)"
python -u test/bodyfat/test_all_models_bodyfat.py
STATUS=$?
echo "Finished bodyfat test at $(date) with status $STATUS"
exit $STATUS
