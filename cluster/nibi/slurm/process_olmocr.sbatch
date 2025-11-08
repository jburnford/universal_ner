#!/bin/bash
#SBATCH --job-name=olmocr-ner
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

# Activate venv
source "$HOME/projects/def-jic823/universal-ner/.venv/bin/activate"

export HF_HOME=$HOME/projects/def-jic823/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

# Set paths
REPO_DIR="$HOME/projects/def-jic823/universal_ner"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Default input/output paths (can be overridden via command line)
INPUT_FILE="${INPUT_FILE:-$HOME/projects/def-jic823/olmocr_test/P000837.json}"
OUTPUT_FILE="${OUTPUT_FILE:-$HOME/projects/def-jic823/olmocr_test/P000837.ner.json}"
ENTITY_TYPES="${ENTITY_TYPES:-person,location,organization,date}"
MAX_PAGES="${MAX_PAGES:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-3000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

echo ""
echo "========================================="
echo "OLMoCR → Universal-NER Pipeline"
echo "========================================="
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Entity types: $ENTITY_TYPES"
echo "Max pages: $MAX_PAGES"
echo "Chunk size: $CHUNK_SIZE chars"
echo "========================================="
echo ""

# Run processing script
python -m src.serve.process_olmocr_json \
  --input_file "$INPUT_FILE" \
  --output_file "$OUTPUT_FILE" \
  --model_path Universal-NER/UniNER-7B-type \
  --entity_types "$ENTITY_TYPES" \
  --max_pages "$MAX_PAGES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --chunk_size "$CHUNK_SIZE"

echo ""
echo "[DONE] Processing completed at $(date)"
