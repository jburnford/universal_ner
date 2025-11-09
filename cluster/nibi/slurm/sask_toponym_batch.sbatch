#!/bin/bash
#SBATCH --job-name=sask-toponym-ner
#SBATCH --account=def-jic823
#SBATCH --array=0-586
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
SASK_DIR="$HOME/projects/def-jic823/saskatchewan_ner"
OUTPUT_DIR="$HOME/projects/def-jic823/saskatchewan_toponym"
mkdir -p "$OUTPUT_DIR"

# Get list of NER files (sorted)
FILES=($SASK_DIR/*.ner.json)
TOTAL_FILES=${#FILES[@]}

# Get file for this array task
INPUT_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"
BASENAME=$(basename "$INPUT_FILE" .ner.json)

echo "========================================="
echo "Saskatchewan Toponym NER Batch Processing"
echo "========================================="
echo "Task: $SLURM_ARRAY_TASK_ID / $TOTAL_FILES"
echo "Document: $BASENAME"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR/${BASENAME}.toponym.ner.json"
echo "========================================="
echo ""

# Create OLMoCR-format JSON from NER file
python3 << PYEOF
import json
from pathlib import Path

ner_file = Path("$INPUT_FILE")
output_dir = Path("$OUTPUT_DIR")
doc_id = "$BASENAME"

# Read NER file
with open(ner_file) as f:
    ner_data = json.load(f)

# Create OLMoCR-format JSON (just text, no entities)
olmocr_data = []
for page in ner_data:
    olmocr_data.append({
        "text": page.get("text", "")
    })

# Save as OLMoCR format for toponym processing
olmocr_file = output_dir / f"{doc_id}.olmocr.json"
with open(olmocr_file, 'w') as f:
    json.dump(olmocr_data, f)

print(f"Created: {olmocr_file}")
PYEOF

# Run toponym extraction
python -m src.serve.process_olmocr_json_toponyms \
  --input_file "$OUTPUT_DIR/${BASENAME}.olmocr.json" \
  --output_file "$OUTPUT_DIR/${BASENAME}.toponym.ner.json" \
  --model_path Universal-NER/UniNER-7B-type \
  --entity_types "person,toponym,water_body,landform,administrative_region,route,organization,date" \
  --max_new_tokens 512 \
  --chunk_size 2000 \
  --use_enhanced_descriptions True

# Clean up intermediate OLMoCR file to save space
rm -f "$OUTPUT_DIR/${BASENAME}.olmocr.json"

echo ""
echo "✓ $BASENAME completed at $(date)"
echo ""
