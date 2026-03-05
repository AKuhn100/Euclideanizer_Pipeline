#!/bin/bash
# Example SLURM job script for the Euclideanizer pipeline.
# Edit partition, resources, and paths for your cluster and environment.

#SBATCH --job-name=Euclideanizer_Pipeline
#SBATCH --partition=commons
#SBATCH --gres=gpu:1
# For multi-GPU: e.g. --gres=gpu:2; the pipeline will automatically run tasks in parallel (one process per GPU).
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x.%j.out
#SBATCH --error=slurm_logs/%x.%j.err

# Activate your Python environment (venv, conda, or module).
# source /path/to/your/venv/bin/activate
# module load Python/3.9  # example; adjust for your cluster

# Optional: load GCC, FFmpeg (needed for training videos).
# module load GCCcore/13.3.0 FFmpeg/7.0.2

# Run from the pipeline directory so relative paths in config resolve correctly.
PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PIPELINE_DIR"
mkdir -p slurm_logs

# Overwrite existing run: add --no-resume --yes-overwrite
python run.py --config config_sample.yaml
