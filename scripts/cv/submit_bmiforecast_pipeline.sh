#!/bin/bash
# Master submission script for the BMI forecast rolling pipeline.
#
# For each forecast year (10, 13, 16, 20, 23, 26):
#   1. Submit 27 parallel DeepPySR grid-search jobs (all vps x vpr x aps combos).
#   2. Submit 4 parallel PySR jobs (one per aps value).
#   3. Submit 1 baseline job.
#   4. Submit a rolling step job that depends on ALL above jobs finishing.
#      This selects the best model/formula and fills missing BMI for that year.
#   5. The next year's jobs depend on the rolling step finishing.
#
# Usage:
#   bash scripts/cv/submit_bmiforecast_pipeline.sh
set -e

PROJECT_ROOT="/scratch/pawsey0411/fchen1/DeepPySR"
MYPYSR_PATH="/scratch/pawsey0411/fchen1/mypysr.jl/python"
JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
PYTHON_JULIAPKG_PROJECT="/scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env"
ACCOUNT="pawsey0411"
LOG_DIR="/scratch/pawsey0411/fchen1/DeepPySR/scripts"

YEARS=(10 13 16 20 23 26)
VPS_LIST=(25 50 75)
VPR_LIST=(50 100 150)
APS_LIST=(1.0 10.0 50.0)
PYSR_APS_LIST=(0.1 1.0 10.0 50.0)

# SLURM preamble for DeepPySR grid-search jobs
deeppysr_preamble() {
    local year=$1 vps=$2 vpr=$3 aps=$4
    cat <<EOF
#!/bin/bash
#SBATCH --job-name=bmi_y${year}_vps${vps}_vpr${vpr}_aps${aps}
#SBATCH --account=${ACCOUNT}
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=${LOG_DIR}/bmi_y${year}_vps${vps}_vpr${vpr}_aps${aps}.log
export PROJECT_ROOT="${PROJECT_ROOT}"
export MYPYSR_PATH="${MYPYSR_PATH}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH}"
export PYTHON_JULIAPKG_PROJECT="${PYTHON_JULIAPKG_PROJECT}"
python -m juliapkg exe -- -e 'using Pkg; Pkg.status()'
export PYTHON_JULIAPKG_OFFLINE=no
cd \$PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="\$PROJECT_ROOT:\$PYTHONPATH"
python -m juliapkg update
set -e
echo "Starting bmi_y${year}_vps${vps}_vpr${vpr}_aps${aps} at \$(date)"
python -u test/bmiforecast/test_deeppysr_bmiforecast.py --year ${year} --vps ${vps} --vpr ${vpr} --aps ${aps}
echo "Finished bmi_y${year}_vps${vps}_vpr${vpr}_aps${aps} at \$(date)"
EOF
}

# SLURM preamble for PySR jobs
pysr_preamble() {
    local year=$1 aps=$2
    cat <<EOF
#!/bin/bash
#SBATCH --job-name=bmi_pysr_y${year}_aps${aps}
#SBATCH --account=${ACCOUNT}
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=${LOG_DIR}/bmi_pysr_y${year}_aps${aps}.log
export PROJECT_ROOT="${PROJECT_ROOT}"
cd \$PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="\$PROJECT_ROOT:\$PYTHONPATH"
set -e
echo "Starting bmi_pysr_y${year}_aps${aps} at \$(date)"
python -u test/bmiforecast/test_baselines_bmiforecast.py --year ${year} --aps ${aps}
echo "Finished bmi_pysr_y${year}_aps${aps} at \$(date)"
EOF
}

# SLURM preamble for baseline jobs
baseline_preamble() {
    local year=$1
    cat <<EOF
#!/bin/bash
#SBATCH --job-name=bmi_baselines_y${year}
#SBATCH --account=${ACCOUNT}
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --output=${LOG_DIR}/bmi_baselines_y${year}.log
export PROJECT_ROOT="${PROJECT_ROOT}"
cd \$PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="\$PROJECT_ROOT:\$PYTHONPATH"
set -e
echo "Starting bmi_baselines_y${year} at \$(date)"
python -u test/bmiforecast/test_baselines_bmiforecast.py --year ${year}
echo "Finished bmi_baselines_y${year} at \$(date)"
EOF
}

# SLURM preamble for rolling step jobs
rolling_preamble() {
    local year=$1
    cat <<EOF
#!/bin/bash
#SBATCH --job-name=bmi_rolling_y${year}
#SBATCH --account=${ACCOUNT}
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=${LOG_DIR}/bmi_rolling_y${year}.log
export PROJECT_ROOT="${PROJECT_ROOT}"
cd \$PROJECT_ROOT
source ".venv/bin/activate"
export PYTHONPATH="\$PROJECT_ROOT:\$PYTHONPATH"
set -e
echo "Starting rolling step for y${year}bmi at \$(date)"
python -u test/bmiforecast/run_rolling_bmiforecast.py --year ${year}
echo "Finished rolling step for y${year}bmi at \$(date)"
EOF
}

prev_rolling_jid=""  # job ID of the previous year's rolling step

for year in "${YEARS[@]}"; do
    echo "=== Submitting jobs for year ${year} ==="
    all_jids=()

    # 1. DeepPySR grid-search jobs (27 parallel)
    for vps in "${VPS_LIST[@]}"; do
        for vpr in "${VPR_LIST[@]}"; do
            for aps in "${APS_LIST[@]}"; do
                dep_flag=""
                if [ -n "$prev_rolling_jid" ]; then
                    dep_flag="--dependency=afterok:${prev_rolling_jid}"
                fi
                jid=$(deeppysr_preamble $year $vps $vpr $aps | sbatch $dep_flag --parsable)
                all_jids+=($jid)
                echo "  DeepPySR y${year} vps${vps} vpr${vpr} aps${aps} -> JID ${jid}"
            done
        done
    done

    # 2. PySR jobs (4 parallel, one per aps)
    for aps in "${PYSR_APS_LIST[@]}"; do
        dep_flag=""
        if [ -n "$prev_rolling_jid" ]; then
            dep_flag="--dependency=afterok:${prev_rolling_jid}"
        fi
        jid=$(pysr_preamble $year $aps | sbatch $dep_flag --parsable)
        all_jids+=($jid)
        echo "  PySR y${year} aps${aps} -> JID ${jid}"
    done

    # 3. Baseline job (1 job, runs all baselines for this year)
    dep_flag=""
    if [ -n "$prev_rolling_jid" ]; then
        dep_flag="--dependency=afterok:${prev_rolling_jid}"
    fi
    jid=$(baseline_preamble $year | sbatch $dep_flag --parsable)
    all_jids+=($jid)
    echo "  Baselines y${year} -> JID ${jid}"

    # 4. Rolling step depends on ALL jobs above
    all_dep=$(IFS=:; echo "${all_jids[*]}")
    echo "  Submitting rolling step for year ${year} (depends on: ${all_dep})"
    rolling_jid=$(rolling_preamble $year | sbatch --dependency=afterok:${all_dep} --parsable)
    echo "  Rolling step y${year} -> JID ${rolling_jid}"

    prev_rolling_jid=$rolling_jid
done

echo ""
echo "All jobs submitted. The pipeline will run year by year:"
echo "  [DeepPySR x27 + PySR x4 + Baselines x1] (parallel) -> rolling step -> next year ..."
