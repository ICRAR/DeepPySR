#!/bin/bash
#SBATCH --account=pawsey0411
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --job-name=pgsc_calc
#SBATCH --output=/scratch/pawsey0411/fchen1/pgs/nextflowpgs.log

module load nextflow/25.04.6
module load singularity/4.1.0-slurm
export NXF_SINGULARITY_CACHEDIR=/scratch/pawsey0411/$USER/singularity_cache
export NXF_HOME=/scratch/pawsey0411/$USER/.nextflow

PGS_LIST=(
#LDL:
PGS004641 # – multi-ancestry, UKB, 1.8M
PGS003032 #– EU, 7.4M
PGS003785 #– EU, 1.5M
PGS004970 # – Multi-ancestry, 1.29M
PGS000892 #– Eu, 1.1M
PGS003033 #– ukb, 1.1M
PGS002337 #– ukb, 1.1M
PGS003873 #– eu, 1M
PGS002703 #– ukb, 0.97M
PGS003976 #– eu, 0.82M
PGS003029 #– ukb, 0.34M
PGS003978 #– multi-ancestry, 0.35M
PGS002654 #– ukb, 0.27M
#Total chole:
PGS003137 #– 7M
PGS003819 #- 1.5M
PGS003138 #- 1.1M
PGS002352 #- 1.1M
PGS002718 #- 0.98M
PGS003481 #- 0.84M
PGS003134 #- 0.59M
PGS002669 #- 0.31M
# HDL:
PGS002957 #- 7.4M
PGS004777 #- 4.55M
PGS003768 #- 1.49M
PGS002781 #- 1.24M
PGS003986 #- 1.14M
PGS002958 #- 1.1M
PGS002329 #- 1.1M
PGS004086 #- 1.1M
PGS003879 #- 1.1M
PGS004028 #- .99M
PGS002695 #- 0.97M
PGS004140 #- 0.87M
PGS002954 #- 0.82NM
# Triglyceride
PGS003147 #- 9M
PGS003802 #- 1.5M
PGS003148 #- 1.1
PGS002353 #- 1.1
PGS002719 #- 0.98
PGS003482 #- 0.84
PGS003144 #- 0.76
PGS002197
PGS003149
)

for PGS_ID in "${PGS_LIST[@]}"; do
nextflow run pgscatalog/pgsc_calc \
  -profile singularity \
  --input /scratch/pawsey0411/fchen1/pgs/samplesheet.csv \
  --target_build GRCh38 \
  --pgs_id ${PGS_ID} \
  --outdir /scratch/pawsey0411/fchen1/pgs/pgs_score/${PGS_ID}/
done