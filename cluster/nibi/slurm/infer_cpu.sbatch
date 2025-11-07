#!/bin/bash
#SBATCH --job-name=uniner-infer
#SBATCH --account=def-jic823
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
echo "[INFO] Job $SLURM_JOBID on $(hostname) started at $(date)"

module purge || true
module load python/3.12

if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate uniner
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate uniner
elif [ -f "$HOME/projects/def-jic823/universal-ner/.venv/bin/activate" ]; then
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

# TODO: Replace with the actual inference entry point and args.
# Examples:
#   python scripts/infer.py --input data/dev.txt --model checkpoints/best
#   python -m universal_ner.predict --input input.jsonl --output output.jsonl --model <path>
#
echo "[ERROR] Edit cluster/nibi/slurm/infer_cpu.sbatch to run the correct inference command." >&2
exit 2
