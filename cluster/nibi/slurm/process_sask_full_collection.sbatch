#!/bin/bash
#SBATCH --job-name=sask-full-toponym
#SBATCH --account=def-jic823
#SBATCH --array=0-13885%100
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
mkdir -p "$OUTPUT_DIR" "$XML_DIR"

# Get list of OCR files (sorted)
FILES=($OCR_DIR/*.json)
TOTAL_FILES=${#FILES[@]}

# Get file for this array task
INPUT_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"
BASENAME=$(basename "$INPUT_FILE" .json)

echo "========================================="
echo "Saskatchewan Full Collection - Toponym NER"
echo "========================================="
echo "Task: $SLURM_ARRAY_TASK_ID / $TOTAL_FILES"
echo "Document: $BASENAME"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR/${BASENAME}.toponym.ner.json"
echo "========================================="
echo ""

# Check if already processed
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
metadata = None
if json_path_key in metadata_db:
    metadata = metadata_db[json_path_key]
else:
    # Try alternative path formats
    for key in metadata_db:
        if key.endswith("/${BASENAME}.json"):
            metadata = metadata_db[key]
            break

if metadata:
    print(f"Found metadata: {metadata.get('title', 'Unknown')[:60]}")
else:
    print("Warning: No metadata found for this document")

# Process with metadata
ner_file = Path("$OUTPUT_DIR/${BASENAME}.toponym.ner.json")
output_file, entities, mentions = process_toponym_file(
    ner_file,
    "$XML_DIR",
    metadata=metadata
)

print(f"✓ Created: {output_file}")
print(f"  Entities: {entities}")
print(f"  Mentions: {mentions}")
PYEOF

echo ""
echo "✓ $BASENAME completed at $(date)"
echo ""
