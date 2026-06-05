#!/bin/bash
#SBATCH --job-name=bmi_age17_vps50
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/bmi_age17_vps50.log

export PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
export DEEPPYSR_PATH="/scratch/pawsey0411/fchen1/deeppysr.jl/python"
#export PYTHONPATH="$PROJECT_ROOT:$DEEPPYSR_PATH:$PYTHONPATH"

export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
python -m juliapkg exe -- -e 'using Pkg; Pkg.status()'
export PYTHON_JULIAPKG_OFFLINE=no

cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python -m juliapkg update
set -e

echo "Starting bmi_age17_vps50 at $(date)"
python -u test/bmi/test_all_models_bmi.py --setting age_specific --age 17 --vps 50
echo "Finished bmi_age17_vps50 at $(date)"

