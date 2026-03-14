#!/bin/bash
#SBATCH --job-name=memory_profile
#SBATCH --partition=fatq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
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

mkdir -p profiles/memory

# Assignment 1.1.6 target setup:
# - model: 2.7B
# - context lengths: 128, 256, 512
# - modes: forward and full training step
SIZE="2.7b"
CONTEXTS=(128 256 512)
MODES=(forward train_step)

run_mem_profile() {
  local ctx="$1"
  local mode="$2"
  local amp="$3"
  local out="profiles/memory/${SIZE}_ctx${ctx}_${mode}_amp-${amp}.json"

  echo "=== Memory profiling size=${SIZE} ctx=${ctx} mode=${mode} amp=${amp} ==="
  set +e
  uv run python -m cs336_systems.benchmark \
    --size "${SIZE}" \
    --context-length "${ctx}" \
    --warmup-steps 5 \
    --measure-steps 1 \
    --mode "${mode}" \
    --loss cross_entropy \
    --device cuda \
    --dtype float32 \
    --amp "${amp}" \
    --memory-profile \
    --memory-snapshot-prefix "profiles/memory/memory_snapshot" \
    > "${out}"
  local rc=$?
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "WARNING: run failed (likely OOM): ctx=${ctx}, mode=${mode}, amp=${amp}"
  fi
}

echo "=== FP32 memory profiles (parts a,b) ==="
for c in "${CONTEXTS[@]}"; do
  for m in "${MODES[@]}"; do
    run_mem_profile "${c}" "${m}" "none"
  done
done

echo "=== Mixed precision memory profiles (part c) ==="
for c in "${CONTEXTS[@]}"; do
  for m in "${MODES[@]}"; do
    run_mem_profile "${c}" "${m}" "float16"
  done
done

echo "=== Done ==="
date
