#!/bin/bash
#SBATCH --job-name=bmiwarm
#SBATCH --output=/home/fchen/DeepPySR/scripts/bmiwarm.out
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1

# Change to the project root directory
cd /home/fchen/DeepPySR
. /home/fchen/DeepPySR/.venv/bin/activate

# Run the python script
export PYTHONPATH=$PYTHONPATH:.
python -u test/bmi/test_all_models_bmi_warm.py
