#!/bin/bash
#SBATCH --job-name=bp_raine_deeppysr_pgsto5_age14_dia_bp_vps25
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/bp_raine_deeppysr_pgsto5_age14_dia_bp_vps25.log

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

echo "Starting bp_raine_deeppysr_pgsto5_age14_dia_bp_vps25 at $(date)"
python -u test/bp_raine/test_deeppysr_bp_to5.py --test PGSto5 --target dia_bp --age 14 --vps 25
echo "Finished bp_raine_deeppysr_pgsto5_age14_dia_bp_vps25 at $(date)"

