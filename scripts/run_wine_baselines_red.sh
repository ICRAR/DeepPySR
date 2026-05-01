#!/bin/bash
#SBATCH --job-name=wine_baselines_red
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/wine_baselines_red.log

PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
export PYTHON_JULIAPKG_OFFLINE=yes
cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
set -e

echo "Starting wine_baselines_red at $(date)"
python -u test/wineQuality/test_baselines_pysr_wine.py --wine_type red
echo "Finished wine_baselines_red at $(date)"

