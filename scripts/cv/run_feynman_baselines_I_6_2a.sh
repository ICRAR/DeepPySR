#!/bin/bash
#SBATCH --job-name=feynman_baselines_I_6_2a
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/feynman_baselines_I_6_2a.log

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

echo "Starting feynman_baselines_I_6_2a at $(date)"
python -u test/feynman/test_baselines_pysr_feynman.py --eq_name I.6.2a
echo "Finished feynman_baselines_I_6_2a at $(date)"

