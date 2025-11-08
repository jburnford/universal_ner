#!/bin/bash
#SBATCH --job-name=uniner-test
#SBATCH --account=def-jic823
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -e

module load python/3.12
module load cuda/12.2
module load gcc arrow/18.1.0
module load opencv/4.10.0

# Prefer your venv similar to Chandra; fall back to conda/mamba
if [ -f "$HOME/projects/def-jic823/universal-ner/.venv/bin/activate" ]; then
  source "$HOME/projects/def-jic823/universal-ner/.venv/bin/activate"
elif command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"; micromamba activate uniner-infer
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"; conda activate uniner-infer
else
  echo "[ERROR] No venv/conda/mamba found." >&2; exit 1
fi

export HF_HOME=$HOME/projects/def-jic823/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

# Ensure we run from the repo root
REPO_DIR="$HOME/projects/def-jic823/universal_ner"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

python - << 'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
PY

echo ""
echo "========================================="
echo "Running Universal-NER Test Script"
echo "========================================="
echo ""

# Run non-interactive test script
python -m src.serve.hf_test \
  --model_path Universal-NER/UniNER-7B-type \
  --max_new_tokens 256

echo ""
echo "[DONE] Test completed at $(date)"
