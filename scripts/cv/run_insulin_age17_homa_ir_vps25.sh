#!/bin/bash
#SBATCH --job-name=insulin_age17_homa_ir_vps25
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/insulin_age17_homa_ir_vps25.log

export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export DEEPPYSR_PATH="/scratch/pawsey0411/fchen1/deeppysr.jl/python"

export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
export JULIA="$PYTHON_JULIAPKG_PROJECT/pyjuliapkg/install/bin/julia"
cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHON_JULIAPKG_OFFLINE=yes

python $PROJECT_ROOT/scripts/fix_julia_manifest.py
set -e

echo "Starting insulin_age17_homa_ir_vps25 at $(date)"
python -u test/insulin/test_deeppysr_insulin.py --age 17 --target homa_ir --vps 25
echo "Finished insulin_age17_homa_ir_vps25 at $(date)"
