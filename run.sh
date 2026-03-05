#!/bin/bash
#SBATCH --job-name=ChromVAE_DistMap
#SBATCH --partition=commons
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/amk19/slurm_outputs/%x.%j.out
#SBATCH --error=/scratch/amk19/slurm_outputs/%x.%j.err

source /scratch/amk19/ChromVAE/.venv/bin/activate

module load GCCcore/13.3.0 FFmpeg/7.0.2

python Euclideanizer_Pipeline/run.py --config Euclideanizer_Pipeline/config_sample.yaml 