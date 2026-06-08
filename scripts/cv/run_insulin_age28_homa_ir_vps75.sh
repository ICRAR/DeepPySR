#!/bin/bash
#SBATCH --job-name=insulin_age28_homa_ir_vps75
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/insulin_age28_homa_ir_vps75.log

export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export DEEPPYSR_PATH="/scratch/pawsey0411/fchen1/deeppysr.jl/python"

export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
export JULIA="$PYTHON_JULIAPKG_PROJECT/pyjuliapkg/install/bin/julia"
export PYTHON_JULIAPKG_OFFLINE=yes

cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
set -e

echo "Starting insulin_age28_homa_ir_vps75 at $(date)"
python -u test/insulin/test_deeppysr_insulin.py --age 28 --vps 75
echo "Finished insulin_age28_homa_ir_vps75 at $(date)"
