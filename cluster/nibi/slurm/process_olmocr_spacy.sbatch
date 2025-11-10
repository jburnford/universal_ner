#!/bin/bash
#SBATCH --job-name=spacy-ner
#SBATCH --account=def-jic823
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out

set -e

module load python/3.12

# Activate venv
source "$HOME/projects/def-jic823/universal-ner/.venv/bin/activate"

# Set paths
REPO_DIR="$HOME/projects/def-jic823/universal_ner"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Default input/output paths
INPUT_FILE="${INPUT_FILE:-$HOME/projects/def-jic823/olmocr_test/P000837.json}"
OUTPUT_FILE="${OUTPUT_FILE:-$HOME/projects/def-jic823/olmocr_test/P000837.spacy.json}"
SPACY_MODEL="${SPACY_MODEL:-en_core_web_lg}"
MAX_PAGES="${MAX_PAGES:-1}"

echo ""
echo "========================================="
echo "OLMoCR → spaCy NER Pipeline"
echo "========================================="
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Model: $SPACY_MODEL"
echo "Max pages: $MAX_PAGES"
echo "========================================="
echo ""

# Run processing script
python -m src.serve.process_olmocr_spacy \
  --input_file "$INPUT_FILE" \
  --output_file "$OUTPUT_FILE" \
  --model "$SPACY_MODEL" \
  --max_pages "$MAX_PAGES"

echo ""
echo "[DONE] Processing completed at $(date)"
