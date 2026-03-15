#!/bin/bash
#SBATCH --job-name=nsys_profile
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
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
echo "Date: $(date)"
echo

module load cuda12.6/toolkit/12.6

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

source /var/scratch/dpp2567/miniconda3/etc/profile.d/conda.sh
conda activate base

mkdir -p profiles/nsys

run_profile() {
  local size="$1"
  local ctx="$2"
  local mode="$3"
  local out="profiles/nsys/${size}_ctx${ctx}_${mode}"

  echo "=== Profiling ${size} ctx=${ctx} mode=${mode} ==="
  set +e
  uv run nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    -o "${out}" \
    python -m cs336_systems.benchmark \
      --size "${size}" \
      --context-length "${ctx}" \
      --warmup-steps 5 \
      --measure-steps 10 \
      --mode "${mode}" \
      --device cuda \
      --annotate-attention
  local rc=$?
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "WARNING: failed for size=${size}, ctx=${ctx}, mode=${mode} (likely OOM). Continuing."
    return
  fi

  # Export text summaries for quick grep without GUI.
  uv run nsys stats --report cuda_gpu_kern_sum --format csv "${out}.nsys-rep" > "${out}_kernels.csv" || true
  uv run nsys stats --report nvtx_sum --format csv "${out}.nsys-rep" > "${out}_nvtx.csv" || true
}

echo "=== Forward-only profiles (for a,b,c,e) ==="
# Based on log/nsys_profile_93046.log OOM results, keep only non-OOM configs:
# forward: small(128,256,512,1024), medium(128,256,512), large(128,256)
run_profile "small" "128" "forward"
run_profile "small" "256" "forward"
run_profile "small" "512" "forward"
run_profile "small" "1024" "forward"
run_profile "medium" "128" "forward"
run_profile "medium" "256" "forward"
run_profile "medium" "512" "forward"
run_profile "large" "128" "forward"
run_profile "large" "256" "forward"

echo "=== Full train-step profiles (for d) ==="
# train_step: small(128,256,512), medium(128,256)
run_profile "small" "128" "train_step"
run_profile "small" "256" "train_step"
run_profile "small" "512" "train_step"
run_profile "medium" "128" "train_step"
run_profile "medium" "256" "train_step"

echo "=== Done ==="
date
