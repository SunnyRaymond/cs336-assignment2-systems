#!/bin/bash
#SBATCH --job-name=naive_ddp_bench
#SBATCH --partition=fatq
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:30:00
#SBATCH --output=log/%x_%j.log
#SBATCH --error=log/%x_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cuda12.6/toolkit/12.6

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

source /var/scratch/dpp2567/miniconda3/etc/profile.d/conda.sh
conda activate base

mkdir -p profiles/ddp

uv run python -m cs336_systems.naive_ddp_benchmark \
  --world-size 2 \
  --batch-size-global 4 \
  --context-length 128 \
  --warmup-steps 3 \
  --measure-steps 10 \
  > profiles/ddp/naive_ddp_benchmark_xl_2gpu.json
