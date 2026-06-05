#!/bin/bash
#SBATCH --job-name=feynman_I_8_14_vps50
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/feynman_I_8_14_vps50.log

export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export DEEPPYSR_PATH="/scratch/pawsey0411/fchen1/deeppysr.jl/python"
#export PYTHONPATH="$PROJECT_ROOT:$DEEPPYSR_PATH:$PYTHONPATH"

export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
export JULIA="$PYTHON_JULIAPKG_PROJECT/pyjuliapkg/install/bin/julia"
cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHON_JULIAPKG_OFFLINE=yes

python $PROJECT_ROOT/scripts/fix_julia_manifest.py
set -e

echo "Starting feynman_I_8_14_vps50 at $(date)"
python -u test/feynman/test_all_models_feynman.py --eq_name I.8.14 --vps 50
echo "Finished feynman_I_8_14_vps50 at $(date)"

