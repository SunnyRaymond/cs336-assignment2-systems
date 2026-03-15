#!/bin/bash
#SBATCH --job-name=allreduce_bench
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
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

mkdir -p profiles/allreduce

uv run python -m cs336_systems.benchmark_allreduce \
  --backends gloo_cpu nccl_gpu \
  --process-counts 2 4 \
  --sizes-mb 1 10 100 1024 \
  --adaptive-iters \
  --csv-out profiles/allreduce/allreduce_benchmark.csv \
  --json-out profiles/allreduce/allreduce_benchmark.json
