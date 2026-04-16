#!/bin/bash
#SBATCH --job-name=stroke
#SBATCH --account=pawsey0411
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --output=scripts/stroke.%j.log
#SBATCH --error=scripts/stroke.%j.err

# Locate the project root from this script's location
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use the project root and virtual environment
cd "$PROJECT_ROOT"
if [ -e "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "ERROR: virtual environment not found at $PROJECT_ROOT/.venv"
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
set -e

echo "Starting stroke test at $(date)"
python -u test/stroke/test_all_models_stroke.py
STATUS=$?
echo "Finished stroke test at $(date) with status $STATUS"
exit $STATUS
