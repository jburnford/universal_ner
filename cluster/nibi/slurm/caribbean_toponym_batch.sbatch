#!/bin/bash
#SBATCH --job-name=caribbean-toponym
#SBATCH --account=def-jic823
#SBATCH --array=0-99
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=%x-%A_%a.out

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

# Input/output directories
INPUT_DIR="$HOME/projects/def-jic823/caribbean_ner_100/input"
OUTPUT_DIR="$HOME/projects/def-jic823/caribbean_toponym"
mkdir -p "$OUTPUT_DIR"

# Get list of input files (sorted)
FILES=($INPUT_DIR/*.json)
TOTAL_FILES=${#FILES[@]}

# Get file for this array task
INPUT_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"
BASENAME=$(basename "$INPUT_FILE" .json)

echo "========================================="
echo "Caribbean Toponym NER Batch Processing"
echo "========================================="
echo "Task: $SLURM_ARRAY_TASK_ID / $TOTAL_FILES"
echo "Document: $BASENAME"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR/${BASENAME}.toponym.ner.json"
echo "========================================="
echo ""

# Run toponym extraction
python -m src.serve.process_olmocr_json_toponyms \
  --input_file "$INPUT_FILE" \
  --output_file "$OUTPUT_DIR/${BASENAME}.toponym.ner.json" \
  --model_path Universal-NER/UniNER-7B-type \
  --entity_types "person,toponym,water_body,landform,administrative_region,route,organization,date" \
  --max_new_tokens 512 \
  --chunk_size 2000 \
  --use_enhanced_descriptions True

echo ""
echo "✓ $BASENAME completed at $(date)"
echo ""
