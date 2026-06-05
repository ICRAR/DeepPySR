#!/bin/bash
#SBATCH --job-name=stroke_vps75
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/stroke_vps75.log

export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export DEEPPYSR_PATH="/scratch/pawsey0411/fchen1/deeppysr.jl/python"
#export PYTHONPATH="$PROJECT_ROOT:$DEEPPYSR_PATH:$PYTHONPATH"

export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
python -m juliapkg exe -- -e 'using Pkg; Pkg.status()'
export PYTHON_JULIAPKG_OFFLINE=no

cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python -m juliapkg update
set -e

echo "Starting stroke_vps75 at $(date)"
python -u test/stroke/test_all_models_stroke.py --vps 75
echo "Finished stroke_vps75 at $(date)"

