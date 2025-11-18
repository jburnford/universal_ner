#!/bin/bash
#SBATCH --job-name=colonial-ner
#SBATCH --account=def-jic823
#SBATCH --array=0-369%50
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
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
INPUT_BASE="/home/jic823/colonialofficelist/output_3"
OUTPUT_DIR="$HOME/projects/def-jic823/colonial_output3_ner"
mkdir -p "$OUTPUT_DIR"

# Get all text/markdown files from all year directories
FILES=($(find "$INPUT_BASE" -type d -name "*_manual_parsed" -exec find {} \( -name "*.txt" -o -name "*.md" \) \; | sort))
TOTAL_FILES=${#FILES[@]}

echo "========================================="
echo "Colonial Office List NER - output_3"
echo "========================================="
echo "Task: $SLURM_ARRAY_TASK_ID / $TOTAL_FILES"
echo "Total files to process: $TOTAL_FILES"
echo "========================================="
echo ""

# Get file for this array task
INPUT_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"
# Remove extension (.txt or .md)
BASENAME=$(basename "$INPUT_FILE")
BASENAME="${BASENAME%.*}"

# Extract year from path (e.g., 1896_manual_parsed)
YEAR_DIR=$(basename "$(dirname "$INPUT_FILE")")
YEAR=${YEAR_DIR%%_*}

# Create output filename with year prefix
OUTPUT_FILE="$OUTPUT_DIR/${YEAR}_${BASENAME}.ner.json"

# Skip if already processed
if [ -f "$OUTPUT_FILE" ]; then
    echo "[SKIP] $OUTPUT_FILE already exists"
    exit 0
fi

echo "Processing:"
echo "  Year: $YEAR"
echo "  Territory: $BASENAME"
echo "  Input: $INPUT_FILE"
echo "  Output: $OUTPUT_FILE"
echo ""

# Run NER extraction
python -m src.serve.process_markdown_ner \
  --input_file "$INPUT_FILE" \
  --output_file "$OUTPUT_FILE" \
  --model_path Universal-NER/UniNER-7B-type \
  --entity_types "person,toponym,water_body,landform,administrative_region,route,organization,date" \
  --max_new_tokens 512 \
  --chunk_size 2000 \
  --use_enhanced_descriptions True

echo ""
echo "[DONE] $YEAR/$BASENAME completed at $(date)"
