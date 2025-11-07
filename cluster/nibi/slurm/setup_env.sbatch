#!/bin/bash
#SBATCH --job-name=uniner-setup
#SBATCH --account=def-jic823
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module load python/3.12
module load cuda/12.2
module load gcc arrow/18.1.0

VENV="$HOME/projects/def-jic823/universal-ner/.venv"
REPO_DIR="$HOME/projects/def-jic823/universal_ner"

python3 -m venv "$VENV"
source "$VENV/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

cd "$REPO_DIR"
# Install dependencies from requirements.txt
python -m pip install -r requirements.txt

# Ensure HF inference deps are present (common crash point: missing sentencepiece)
python -m pip install sentencepiece

echo "[SETUP DONE] Venv: $VENV"
