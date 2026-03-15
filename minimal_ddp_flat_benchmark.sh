#!/bin/bash
#SBATCH --job-name=minimal_ddp_flat
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:30:00
#SBATCH --output=log/%x_%j.log
#SBATCH --error=log/%x_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cuda12.6/toolkit/12.6

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

echo "change uv cache dir to /var/scratch/dpp2567/.uv_cache"
source /var/scratch/dpp2567/miniconda3/etc/profile.d/conda.sh
export UV_CACHE_DIR=/var/scratch/dpp2567/.uv_cache
export XDG_CACHE_HOME=/var/scratch/dpp2567/.cache
export UV_PROJECT_ENVIRONMENT=/var/scratch/dpp2567/.venvs/cs336-assignment2-systems

conda activate base

mkdir -p profiles/ddp

uv run python -m cs336_systems.minimal_ddp_flat_benchmark \
  --world-size 2 \
  --model-size small \
  --batch-size-global 2 \
  --context-length 128 \
  --warmup-steps 3 \
  --measure-steps 10 \
  --optimizer adamw \
  > profiles/ddp/minimal_ddp_flat_benchmark_large_2gpu.json
