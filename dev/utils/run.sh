#!/bin/bash
#SBATCH --job-name=Euclideanizer_Pipeline
#SBATCH --partition=commons
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/amk19/slurm_outputs/%x.%j.out
#SBATCH --error=/scratch/amk19/slurm_outputs/%x.%j.err

source /scratch/amk19/Euclideanizer/.venv/bin/activate

module load GCCcore/13.3.0 FFmpeg/7.0.2

python /scratch/amk19/Euclideanizer/Pipeline/run_hpo.py --config /scratch/amk19/Euclideanizer/Pipeline/dev/configs/hpo_config.yaml