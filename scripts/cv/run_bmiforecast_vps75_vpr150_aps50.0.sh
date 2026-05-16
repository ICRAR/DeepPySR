#!/bin/bash
#SBATCH --job-name=bmiforecast_vps75_vpr150_aps50.0
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/bmiforecast_vps75_vpr150_aps50.0.log
export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export MYPYSR_PATH="/scratch/pawsey0411/fchen1/mypysr.jl/python"
export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
python -m juliapkg exe -- -e 'using Pkg; Pkg.status()'
export PYTHON_JULIAPKG_OFFLINE=no
cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python -m juliapkg update
set -e
echo "Starting bmiforecast_vps75_vpr150_aps50.0 at $(date)"
python -u test/bmiforecast/test_deeppysr_bmiforecast.py --vps 75 --vpr 150 --aps 50.0
echo "Finished bmiforecast_vps75_vpr150_aps50.0 at $(date)"
