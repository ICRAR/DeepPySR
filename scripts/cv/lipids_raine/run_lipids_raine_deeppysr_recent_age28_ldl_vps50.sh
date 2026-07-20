#!/bin/bash
#SBATCH --job-name=lipids_raine_deeppysr_recent_age28_ldl_vps50
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/lipids_raine_deeppysr_recent_age28_ldl_vps50.log

export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export DEEPPYSR_PATH="/scratch/pawsey0411/fchen1/deeppysr.jl/python"

export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
export JULIA="$PYTHON_JULIAPKG_PROJECT/pyjuliapkg/install/bin/julia"
cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHON_JULIAPKG_OFFLINE=yes

set -e

echo "Starting lipids_raine_deeppysr_recent_age28_ldl_vps50 at $(date)"
python -u test/lipids_raine/test_deeppysr_lipids_recent.py --target ldl --age 28 --vps 50
echo "Finished lipids_raine_deeppysr_recent_age28_ldl_vps50 at $(date)"

