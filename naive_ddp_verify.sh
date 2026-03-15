#!/bin/bash
#SBATCH --job-name=naive_ddp_verify
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:10:00
#SBATCH --output=log/%x_%j.log
#SBATCH --error=log/%x_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

module load cuda12.6/toolkit/12.6

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

echo "change uv cache dir to /var/scratch/dpp2567/.uv_cache"
source /var/scratch/dpp2567/miniconda3/etc/profile.d/conda.sh
export UV_CACHE_DIR=/var/scratch/dpp2567/.uv_cache
export XDG_CACHE_HOME=/var/scratch/dpp2567/.cache
export UV_PROJECT_ENVIRONMENT=/var/scratch/dpp2567/.venvs/cs336-assignment2-systems

conda activate base

uv run python -m cs336_systems.naive_ddp_verify \
  --world-size 2 \
  --steps 20 \
  --local-batch-size 16
