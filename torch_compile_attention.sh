#!/bin/bash
#SBATCH --job-name=torch_compile_attn
#SBATCH --partition=fatq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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

mkdir -p profiles/attention

echo "Running attention benchmark with torch.compile optimizations..."

# 1.3(a): compare uncompiled vs compiled attention under the same sweep config.
uv run python -m cs336_systems.benchmark_attention \
  --batch-size 8 \
  --dmodels 16 32 64 128 \
  --seq-lens 256 1024 4096 8192 16384 \
  --forward-steps 100 \
  --backward-steps 100 \
  --warmup-steps 10 \
  --device cuda \
  --dtype float32 \
  --implementations uncompiled compiled \
  --csv-out profiles/attention/attention_benchmark_compile.csv \
  --json-out profiles/attention/attention_benchmark_compile.json
