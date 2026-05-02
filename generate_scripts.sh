#!/bin/bash

# Configuration
PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR/"
JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"

TEMPLATE='#!/bin/bash
#SBATCH --job-name=JOB_NAME
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/LOG_NAME.log

export PROJECT_ROOT="'$PROJECT_ROOT'"
export MYPYSR_PATH="/scratch/pawsey0411/fchen1/mypysr.jl/python"
#export PYTHONPATH="$PROJECT_ROOT:$MYPYSR_PATH:$PYTHONPATH"

export JULIA_DEPOT_PATH="'$JULIA_DEPOT_PATH'"
export PYTHON_JULIAPKG_PROJECT="'$PYTHON_JULIAPKG_PROJECT'"
python -m juliapkg exe -- -e '\''using Pkg; Pkg.status()'\''
export PYTHON_JULIAPKG_OFFLINE=no

cd $PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python -m juliapkg update
set -e

echo "Starting JOB_NAME at $(date)"
python -u COMMAND
echo "Finished JOB_NAME at $(date)"
'

create_script() {
    local name=$1
    local command=$2
    local job_name=${name//\//_}
    local content="${TEMPLATE//JOB_NAME/$job_name}"
    content="${content//LOG_NAME/$job_name}"
    content="${content//COMMAND/$command}"
    echo "$content" > "scripts/run_${name}.sh"
    chmod +x "scripts/run_${name}.sh"
}

mkdir -p scripts

# --- DeepPySR Parallel Jobs ---

# BMI
for vps in 25 50 75; do
    create_script "bmi_long_vps${vps}" "test/bmi/test_all_models_bmi.py --setting longitudinal --vps ${vps}"
    for age in 8 10 14 17 20 23 27; do
        create_script "bmi_age${age}_vps${vps}" "test/bmi/test_all_models_bmi.py --setting age_specific --age ${age} --vps ${vps}"
    done
done

# Wine
for wine in red white; do
    for vps in 25 50 75; do
        create_script "wine_${wine}_vps${vps}" "test/wineQuality/test_all_models_wine.py --wine_type ${wine} --vps ${vps}"
    done
done

# Diabetes, Heart, Stroke, Bodyfat
for dataset in diabetes heart stroke bodyfat; do
    for vps in 25 50 75; do
        create_script "${dataset}_vps${vps}" "test/${dataset}/test_all_models_${dataset}.py --vps ${vps}"
    done
done

# Feynman
for eq in "I.6.2a" "I.8.14" "I.13.4" "I.9.18" "I.32.17"; do
    eq_clean=${eq//./_}
    for vps in 25 50 75; do
        create_script "feynman_${eq_clean}_vps${vps}" "test/feynman/test_all_models_feynman.py --eq_name ${eq} --vps ${vps}"
    done
done

# --- Baselines & PySR Jobs ---

# BMI Baselines
create_script "bmi_baselines_long" "test/bmi/test_baselines_pysr_bmi.py --setting longitudinal"
for age in 8 10 14 17 20 23 27; do
    create_script "bmi_baselines_age${age}" "test/bmi/test_baselines_pysr_bmi.py --setting age_specific --age ${age}"
done

# Wine Baselines
for wine in red white; do
    create_script "wine_baselines_${wine}" "test/wineQuality/test_baselines_pysr_wine.py --wine_type ${wine}"
done

# Diabetes, Heart, Stroke Baselines
for dataset in diabetes heart stroke; do
    create_script "${dataset}_baselines" "test/${dataset}/test_baselines_pysr_${dataset}.py"
done

# Feynman Baselines
for eq in "I.6.2a" "I.8.14" "I.13.4" "I.9.18" "I.32.17"; do
    eq_clean=${eq//./_}
    create_script "feynman_baselines_${eq_clean}" "test/feynman/test_baselines_pysr_feynman.py --eq_name ${eq}"
done
