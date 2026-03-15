#!/bin/bash
#SBATCH --job-name=torch_compile_model
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
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

mkdir -p profiles/compile_model

echo "Running full model benchmark with torch.compile optimizations..."
echo "Using torch.compile backend=aot_eager (inductor/triton toolchain unavailable on this cluster)."

# 1.3(b): compare vanilla vs compiled full model.
# Keep to sizes likely to fit on constrained GPUs.
SIZES=(small medium large)
MODES=(forward train_step)

for s in "${SIZES[@]}"; do
  for m in "${MODES[@]}"; do
    echo "=== vanilla size=${s} mode=${m} ==="
    uv run python -m cs336_systems.benchmark \
      --size "${s}" \
      --context-length 128 \
      --warmup-steps 5 \
      --measure-steps 10 \
      --mode "${m}" \
      --device cuda \
      > "profiles/compile_model/${s}_${m}_vanilla.json"

    echo "=== compiled size=${s} mode=${m} ==="
    uv run python -m cs336_systems.benchmark \
      --size "${s}" \
      --context-length 128 \
      --warmup-steps 5 \
      --measure-steps 10 \
      --mode "${m}" \
      --device cuda \
      --compile-model \
      --compile-backend aot_eager \
      > "profiles/compile_model/${s}_${m}_compiled.json"
  done
done
