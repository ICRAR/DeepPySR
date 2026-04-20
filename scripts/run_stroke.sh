#!/bin/bash
#SBATCH --job-name=stroke
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/stroke.log

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
export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_OFFLINE=no
rm -rf /scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env
# Install the Julia dependencies again (this is usually fast)
python -m juliapkg update
set -e

echo "Starting stroke test at $(date)"
python -u test/stroke/test_all_models_stroke.py
STATUS=$?
echo "Finished stroke test at $(date) with status $STATUS"
exit $STATUS
