#!/bin/bash
#SBATCH --job-name=sask-full-toponym
#SBATCH --account=def-jic823
#SBATCH --array=0-9999%100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
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
OCR_DIR="$HOME/projects/def-jic823/pdfs_sask_test/results/json"
OUTPUT_DIR="$HOME/projects/def-jic823/saskatchewan_full_toponym"
XML_DIR="$HOME/projects/def-jic823/saskatchewan_full_toponym_xml"
METADATA_FILE="$HOME/projects/def-jic823/saskatchewan_metadata.json"
UNPROCESSED_LIST="$HOME/projects/def-jic823/saskatchewan_unprocessed_files.txt"
mkdir -p "$OUTPUT_DIR" "$XML_DIR"

# Read from pre-generated list of unprocessed files
if [ ! -f "$UNPROCESSED_LIST" ]; then
    echo "ERROR: Unprocessed file list not found at $UNPROCESSED_LIST"
    echo "Run generate_unprocessed_list.sh first"
    exit 1
fi

# Get the file for this task
INPUT_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$UNPROCESSED_LIST")

if [ -z "$INPUT_FILE" ]; then
    echo "ERROR: No file found for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

BASENAME=$(basename "$INPUT_FILE" .json)

echo "========================================="
echo "Saskatchewan Full Collection - Toponym NER"
echo "========================================="
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Document: $BASENAME"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR/${BASENAME}.toponym.ner.json"
echo "========================================="
echo ""

# Double-check not already processed (race condition protection)
if [ -f "$XML_DIR/${BASENAME}.toponym.xml" ]; then
    echo "✓ Already processed (XML exists), skipping."
    exit 0
fi

# Run toponym extraction directly on OLMoCR format
python -m src.serve.process_olmocr_json_toponyms \
  --input_file "$INPUT_FILE" \
  --output_file "$OUTPUT_DIR/${BASENAME}.toponym.ner.json" \
  --model_path Universal-NER/UniNER-7B-type \
  --entity_types "person,toponym,water_body,landform,administrative_region,route,organization,date" \
  --max_new_tokens 512 \
  --chunk_size 2000 \
  --use_enhanced_descriptions True

echo ""
echo "✓ NER completed, converting to XML with metadata..."

# Convert to XML with metadata
python3 << PYEOF
import json
from pathlib import Path
from src.utils.convert_ner_to_xml import process_toponym_file

# Load metadata
metadata_file = Path("$METADATA_FILE")
metadata_db = {}
if metadata_file.exists():
    with open(metadata_file) as f:
        metadata_db = json.load(f)

# Find metadata for this file
json_path_key = "$INPUT_FILE"
basename = "$BASENAME"

metadata = None
for key in metadata_db.keys():
    if basename in key or key in basename:
        metadata = metadata_db[key]
        break

# Convert to XML with metadata
process_toponym_file(
    "$OUTPUT_DIR/${BASENAME}.toponym.ner.json",
    "$XML_DIR/${BASENAME}.toponym.xml",
    metadata
)
PYEOF

echo "✓ Processing complete for $BASENAME"
