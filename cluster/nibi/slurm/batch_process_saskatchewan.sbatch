#!/bin/bash
#SBATCH --job-name=ner-batch
#SBATCH --account=def-jic823
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --array=0-586

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

# Directories
INPUT_DIR="$HOME/projects/def-jic823/saskatchewan_ocr"
OUTPUT_DIR="$HOME/projects/def-jic823/saskatchewan_ner"
mkdir -p "$OUTPUT_DIR"

# Get list of files and select one for this array task
FILES=($INPUT_DIR/P*.json)
INPUT_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"
BASENAME=$(basename "$INPUT_FILE" .json)
OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}.ner.json"

# Skip if already processed
if [ -f "$OUTPUT_FILE" ]; then
    echo "[SKIP] $BASENAME already processed"
    exit 0
fi

echo ""
echo "========================================="
echo "Saskatchewan NER Batch Processing"
echo "========================================="
echo "Array Task: $SLURM_ARRAY_TASK_ID / 586"
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Model: Universal-NER/UniNER-7B-type"
echo "Settings: 2K chunks, baseline (no enhanced descriptions)"
echo "========================================="
echo ""

# Run processing script with baseline settings
python -m src.serve.process_olmocr_json \
  --input_file "$INPUT_FILE" \
  --output_file "$OUTPUT_FILE" \
  --model_path Universal-NER/UniNER-7B-type \
  --entity_types "person,location,organization,date" \
  --max_pages 1 \
  --max_new_tokens 512 \
  --chunk_size 2000

echo ""
echo "[DONE] $BASENAME completed at $(date)"
