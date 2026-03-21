#!/bin/bash
#SBATCH --job-name=Euclideanizer_Pipeline
#SBATCH --partition=commons
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/amk19/Euclideanizer/slurm_outputs/%x.%j.out
#SBATCH --error=/scratch/amk19/Euclideanizer/slurm_outputs/%x.%j.err

# Resubmission logic
MAX_RESUBMIT=3
RESUBMIT_COUNT=${1:-0}

if [ "$RESUBMIT_COUNT" -lt "$MAX_RESUBMIT" ]; then
    echo "Resubmit $((RESUBMIT_COUNT + 1)) of $MAX_RESUBMIT queued, starting in 24h1m."
    sbatch --begin=now+1441minutes "$0" $((RESUBMIT_COUNT + 1))
else
    echo "Reached max resubmissions ($MAX_RESUBMIT), not requeuing."
fi

source /scratch/amk19/Euclideanizer/.venv/bin/activate

module load GCCcore/13.3.0 FFmpeg/7.0.2

python /scratch/amk19/Euclideanizer/Pipeline/run_hpo.py --config /scratch/amk19/Euclideanizer/Pipeline/dev/configs/hpo_config_chrom21.yaml