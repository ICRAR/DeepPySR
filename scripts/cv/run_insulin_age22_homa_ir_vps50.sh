#!/bin/bash
#SBATCH --job-name=insulin_age22_homa_ir_vps50
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/insulin_age22_homa_ir_vps50.log

export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export DEEPPYSR_PATH="/scratch/pawsey0411/fchen1/deeppysr.jl/python"

export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
export JULIA="$PYTHON_JULIAPKG_PROJECT/pyjuliapkg/install/bin/julia"
$JULIA --project=$PYTHON_JULIAPKG_PROJECT -e 'import Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()'
export PYTHON_JULIAPKG_OFFLINE=no

cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
# python -m juliapkg update
set -e

echo "Starting insulin_age22_homa_ir_vps50 at $(date)"
python -u test/insulin/test_deeppysr_insulin.py --age 22 --target homa_ir --vps 50
echo "Finished insulin_age22_homa_ir_vps50 at $(date)"
