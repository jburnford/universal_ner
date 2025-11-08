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
module load opencv/4.10.0

VENV="$HOME/projects/def-jic823/universal-ner/.venv"
REPO_DIR="$HOME/projects/def-jic823/universal_ner"

python3 -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

cd "$REPO_DIR"
# Install only dependencies needed for HF inference (skip vllm to avoid opencv issues)
# Use numpy<2 to match system scipy
python -m pip install "numpy<2" transformers>=4.30.1 tokenizers>=0.13.1 fire sentencepiece accelerate protobuf

echo "[SETUP DONE] Venv: $VENV"
