#!/bin/bash
#SBATCH --job-name=insulin_baselines_age22_homa_ir
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/insulin_baselines_age22_homa_ir.log

export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export DEEPPYSR_PATH="/scratch/pawsey0411/fchen1/deeppysr.jl/python"

export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
export JULIA="$PYTHON_JULIAPKG_PROJECT/pyjuliapkg/install/bin/julia"
cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python -m juliapkg update
python $PROJECT_ROOT/scripts/fix_julia_manifest.py
export PYTHON_JULIAPKG_OFFLINE=no

$JULIA --project=$PYTHON_JULIAPKG_PROJECT -e 'import Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()'
set -e

echo "Starting insulin_baselines_age22_homa_ir at $(date)"
python -u test/insulin/test_baselines_pysr_insulin.py --age 22 --target homa_ir
echo "Finished insulin_baselines_age22_homa_ir at $(date)"
