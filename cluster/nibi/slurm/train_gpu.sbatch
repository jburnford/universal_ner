#!/bin/bash
#SBATCH --job-name=uniner-train
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
echo "[INFO] Job $SLURM_JOBID on $(hostname) started at $(date)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Optional: load site-specific modules (edit as needed)
module purge || true
# module load cuda/12.1 cudnn/8.9  # example; adjust to Nibi
# module load python/3.10           # if using system python modules

# Activate conda/mamba environment (choose one)
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate uniner
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate uniner
else
  echo "[ERROR] No conda/micromamba found in PATH." >&2
  exit 1
fi

# Put caches and temp on fast local storage
export HF_HOME=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/hf
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export MPLCONFIGDIR=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/mpl
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$MPLCONFIGDIR"

# Move to repo root (this script is typically launched from repo root)
cd "$SLURM_SUBMIT_DIR"

# Sanity: show GPU/driver
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

echo "[INFO] Python: $(python --version)"
echo "[INFO] Torch:  $(python -c 'import torch,sys; print(torch.__version__, torch.cuda.is_available())' || echo 'not installed')"

# TODO: Replace the following placeholder command with the actual training entry point.
# Examples (edit to match this repo):
#   python scripts/train.py --config configs/base.yaml
#   python -m universal_ner.train --config configs/train.yaml
#
echo "[ERROR] Edit cluster/nibi/slurm/train_gpu.sbatch to run the correct training command." >&2
exit 2

