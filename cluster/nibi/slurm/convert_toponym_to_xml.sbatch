#!/bin/bash
#SBATCH --job-name=toponym-to-xml
#SBATCH --account=def-jic823
#SBATCH --array=0-585
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:10:00
#SBATCH --output=%x-%A_%a.out

set -e

# Set paths
REPO_DIR="$HOME/projects/def-jic823/universal_ner"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Get input and output directories from arguments
INPUT_DIR="$1"
OUTPUT_DIR="$2"
mkdir -p "$OUTPUT_DIR"

# Get list of NER files (sorted)
FILES=($INPUT_DIR/*.toponym.ner.json)
TOTAL_FILES=${#FILES[@]}

# Get file for this array task
INPUT_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"
BASENAME=$(basename "$INPUT_FILE" .toponym.ner.json)

echo "========================================="
echo "Converting NER to XML"
echo "========================================="
echo "Task: $SLURM_ARRAY_TASK_ID / $TOTAL_FILES"
echo "Document: $BASENAME"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR/${BASENAME}.toponym.xml"
echo "========================================="
echo ""

# Convert single file
python3 << PYEOF
from src.utils.convert_ner_to_xml import process_toponym_file
from pathlib import Path

input_file = Path("$INPUT_FILE")
output_dir = Path("$OUTPUT_DIR")

output_file, entities, mentions = process_toponym_file(input_file, output_dir)
print(f"✓ Created: {output_file}")
print(f"  Entities: {entities}")
print(f"  Mentions: {mentions}")
PYEOF

echo ""
echo "✓ $BASENAME completed at $(date)"
echo ""
