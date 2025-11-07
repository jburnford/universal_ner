#!/bin/bash
#SBATCH --job-name=uniner-infer
#SBATCH --account=def-jic823
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
echo "[INFO] Job $SLURM_JOBID on $(hostname) started at $(date)"

module purge || true
module load python/3.12
module load cuda/12.2
module load gcc arrow/18.1.0

if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate uniner-infer
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate uniner-infer
elif [ -f "$HOME/projects/def-jic823/universal-ner/.venv/bin/activate" ]; then
  # Fallback: use a virtualenv similar to your Chandra setup
  source "$HOME/projects/def-jic823/universal-ner/.venv/bin/activate"
else
  echo "[ERROR] No conda/micromamba found in PATH." >&2
  exit 1
fi

export HF_HOME=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/hf
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# Ensure we run from the repo root
REPO_DIR="$HOME/projects/def-jic823/universal_ner"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Show GPU info
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

# vLLM CLI inference (GPU)
python -m src.serve.cli \
  --model_path Universal-NER/UniNER-7B-type \
  --tensor_parallel_size 1 \
  --max_input_length 512

echo "[INFO] Inference completed at $(date)"
