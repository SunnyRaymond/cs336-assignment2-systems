#!/bin/bash
#SBATCH --job-name=naive_ddp_verify
#SBATCH --partition=fatq
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-00:10:00
#SBATCH --output=log/%x_%j.log
#SBATCH --error=log/%x_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

module load cuda12.6/toolkit/12.6

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

source /var/scratch/dpp2567/miniconda3/etc/profile.d/conda.sh
conda activate base

uv run python -m cs336_systems.naive_ddp_verify \
  --world-size 2 \
  --steps 20 \
  --local-batch-size 16
