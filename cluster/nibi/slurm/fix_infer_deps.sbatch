#!/bin/bash
#SBATCH --job-name=uniner-fixdeps
#SBATCH --account=def-jic823
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module load python/3.12

VENV="$HOME/projects/def-jic823/universal-ner/.venv"
source "$VENV/bin/activate"

python -m pip install -U pip
python -m pip install transformers>=4.30.1 tokenizers>=0.13.1 sentencepiece fire

echo "[FIX DONE] Ensured HF inference deps are installed."
