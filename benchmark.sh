#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=fatq
#SBATCH --constraint=TitanRTX
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --output=log/%x_%j.log
#SBATCH --error=log/%x_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Job info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "PWD:  $(pwd)"
echo "Date: $(date)"
echo

module load cuda12.6/toolkit/12.6

echo "=== GPU info ==="
nvidia-smi
echo

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

# Conda setup
source /var/scratch/dpp2567/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "=== Python / uv info ==="
which python || true
which uv || true
python --version || true
echo

echo "=== 1.1.3 Benchmark start ==="
echo "=== a ==="

# (a) Single run example (forward+backward)
uv run python -m cs336_systems.benchmark \
  --size small \
  --context-length 128 \
  --warmup-steps 5 \
  --measure-steps 10 \
  --mode forward-backward \
  --device cuda

echo "=== b ==="

# (b) Table 1 sizes, 5 warmup + 10 measured (forward+backward)
for s in small medium large xl 2.7b; do
  echo "===== $s ====="
  uv run python -m cs336_systems.benchmark \
    --size "$s" \
    --context-length 128 \
    --warmup-steps 5 \
    --measure-steps 10 \
    --mode forward-backward \
    --device cuda
done

echo "=== c ==="
# (c) Warmup sensitivity: 0, 1, 2, 5 warmup steps
for w in 0 1 2 5; do
  echo "===== warmup=$w ====="
  uv run python -m cs336_systems.benchmark \
    --size small \
    --context-length 128 \
    --warmup-steps "$w" \
    --measure-steps 10 \
    --mode forward-backward \
    --device cuda
done

echo "=== Done ==="
date
