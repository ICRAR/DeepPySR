#!/bin/bash
# Generates SLURM job scripts under scripts/cv/lipids_raine/ and scripts/cv/bp_raine/,
# mirroring the layout of scripts/cv/diab_raine/ but extended over multiple targets
# and with/without longitudinal feature engineering (--no_feateng).

set -e

TEMPLATE='#!/bin/bash
#SBATCH --job-name=JOB_NAME
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/fchen1/DeepPySR/scripts/JOB_NAME.log

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

echo "Starting JOB_NAME at $(date)"
python -u COMMAND
echo "Finished JOB_NAME at $(date)"
'

create_script() {
    local dataset=$1
    local job_name=$2
    local command=$3
    local content="${TEMPLATE//JOB_NAME/$job_name}"
    content="${content//COMMAND/$command}"
    local outdir="scripts/cv/${dataset}"
    mkdir -p "$outdir"
    echo "$content" > "${outdir}/run_${job_name}.sh"
    chmod +x "${outdir}/run_${job_name}.sh"
}

# Args: dataset short to8_variant pgsto8_variant targets_var ages_var
generate_dataset() {
    local dataset=$1 short=$2 to8_variant=$3 pgsto8_variant=$4
    local -n targets=$5
    local -n ages=$6

    local baselines_to8_script="test/${dataset}/test_baselines_pysr_${short}_${to8_variant}.py"
    local baselines_recent_script="test/${dataset}/test_baselines_pysr_${short}_recent.py"
    local deeppysr_to8_script="test/${dataset}/test_deeppysr_${short}_${to8_variant}.py"
    local deeppysr_recent_script="test/${dataset}/test_deeppysr_${short}_recent.py"

    for target in "${targets[@]}"; do
        for age in "${ages[@]}"; do
            # --- Baselines + PySR ---

            # PGS: feateng is forced off internally by the test script, no with/without split
            create_script "$dataset" "${dataset}_baselines_pgs_age${age}_${target}" \
                "${baselines_to8_script} --test PGS --target ${target} --age ${age}"

            for variant in "$to8_variant" "$pgsto8_variant"; do
                local variant_lc="${variant,,}"
                create_script "$dataset" "${dataset}_baselines_${variant_lc}_age${age}_${target}" \
                    "${baselines_to8_script} --test ${variant} --target ${target} --age ${age}"
                create_script "$dataset" "${dataset}_baselines_${variant_lc}_age${age}_${target}_nofeateng" \
                    "${baselines_to8_script} --test ${variant} --target ${target} --age ${age} --no_feateng"
            done

            create_script "$dataset" "${dataset}_baselines_recent_age${age}_${target}" \
                "${baselines_recent_script} --target ${target} --age ${age}"
            create_script "$dataset" "${dataset}_baselines_recent_age${age}_${target}_nofeateng" \
                "${baselines_recent_script} --target ${target} --age ${age} --no_feateng"

            # --- DeepPySR ---
            for vps in 25 50 75; do
                create_script "$dataset" "${dataset}_deeppysr_pgs_age${age}_${target}_vps${vps}" \
                    "${deeppysr_to8_script} --test PGS --target ${target} --age ${age} --vps ${vps}"

                for variant in "$to8_variant" "$pgsto8_variant"; do
                    local variant_lc="${variant,,}"
                    create_script "$dataset" "${dataset}_deeppysr_${variant_lc}_age${age}_${target}_vps${vps}" \
                        "${deeppysr_to8_script} --test ${variant} --target ${target} --age ${age} --vps ${vps}"
                    create_script "$dataset" "${dataset}_deeppysr_${variant_lc}_age${age}_${target}_vps${vps}_nofeateng" \
                        "${deeppysr_to8_script} --test ${variant} --target ${target} --age ${age} --vps ${vps} --no_feateng"
                done

                create_script "$dataset" "${dataset}_deeppysr_recent_age${age}_${target}_vps${vps}" \
                    "${deeppysr_recent_script} --target ${target} --age ${age} --vps ${vps}"
                create_script "$dataset" "${dataset}_deeppysr_recent_age${age}_${target}_vps${vps}_nofeateng" \
                    "${deeppysr_recent_script} --target ${target} --age ${age} --vps ${vps} --no_feateng"
            done
        done
    done
}

lipids_targets=(cholesterol triglyceride hdl ldl)
lipids_ages=(14 17 20 22 27 28)
generate_dataset "lipids_raine" "lipids" "to8" "PGSto8" lipids_targets lipids_ages

bp_targets=(sys_bp dia_bp)
bp_ages=(10 14 17 20 22)
generate_dataset "bp_raine" "bp" "to5" "PGSto5" bp_targets bp_ages

echo "Done."
