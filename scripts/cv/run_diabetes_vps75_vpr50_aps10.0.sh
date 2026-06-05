#!/bin/bash
#SBATCH --job-name=diabetes_vps75_vpr50_aps10.0
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/diabetes_vps75_vpr50_aps10.0.log

export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export DEEPPYSR_PATH="/scratch/pawsey0411/fchen1/deeppysr.jl/python"
#export PYTHONPATH="$PROJECT_ROOT:$DEEPPYSR_PATH:$PYTHONPATH"

export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
export JULIA="$PYTHON_JULIAPKG_PROJECT/pyjuliapkg/install/bin/julia"
cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python -m juliapkg update
python $PROJECT_ROOT/scripts/fix_julia_manifest.py
export PYTHON_JULIAPKG_OFFLINE=yes

$JULIA --project=$PYTHON_JULIAPKG_PROJECT -e 'import Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()'
set -e

echo "Starting diabetes_vps75_vpr50_aps10.0 at $(date)"
python -u test/diabetes/test_all_models_diabetes.py --vps 75 --vpr 50 --aps 10.0
echo "Finished diabetes_vps75_vpr50_aps10.0 at $(date)"

