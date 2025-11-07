#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$HOME/projects/def-jic823/universal_ner"

if [ ! -d "$REPO_DIR" ]; then
  echo "Repo not found at $REPO_DIR" >&2
  exit 1
fi

cd "$REPO_DIR"
echo "Pulling latest..."
git pull --ff-only || true

echo "Submitting setup job..."
SETUP_JOB=$(sbatch cluster/nibi/slurm/setup_env.sbatch | awk '{print $NF}')
echo "Setup job: $SETUP_JOB"

echo "Submitting HF inference job..."
INFER_JOB=$(sbatch --dependency=afterok:$SETUP_JOB cluster/nibi/slurm/infer_gpu_hf.sbatch | awk '{print $NF}')
echo "Inference job (after setup): $INFER_JOB"

echo "Use: squeue -j $SETUP_JOB,$INFER_JOB and tail -f uniner-hf-$INFER_JOB.out"
