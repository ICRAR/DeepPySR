#!/bin/bash
#SBATCH --job-name=diabetes_vps25_vpr150_aps10.0
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/diabetes_vps25_vpr150_aps10.0.log

export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export MYPYSR_PATH="/scratch/pawsey0411/fchen1/mypysr.jl/python"
#export PYTHONPATH="$PROJECT_ROOT:$MYPYSR_PATH:$PYTHONPATH"

export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
python -m juliapkg exe -- -e 'using Pkg; Pkg.status()'
export PYTHON_JULIAPKG_OFFLINE=no

cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python -m juliapkg update
set -e

echo "Starting diabetes_vps25_vpr150_aps10.0 at $(date)"
python -u test/diabetes/test_all_models_diabetes.py --vps 25 --vpr 150 --aps 10.0
echo "Finished diabetes_vps25_vpr150_aps10.0 at $(date)"

