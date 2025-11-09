#!/bin/bash
#SBATCH --job-name=toponym-xml-retry
#SBATCH --account=def-jic823
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out

set -e

# Set paths
REPO_DIR="$HOME/projects/def-jic823/universal_ner"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Get input and output directories from arguments
INPUT_DIR="$1"
OUTPUT_DIR="$2"
FILENAME="$3"

mkdir -p "$OUTPUT_DIR"

INPUT_FILE="$INPUT_DIR/${FILENAME}.toponym.ner.json"

echo "========================================="
echo "Converting NER to XML (Retry)"
echo "========================================="
echo "Document: $FILENAME"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR/${FILENAME}.toponym.xml"
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
echo "✓ $FILENAME completed at $(date)"
echo ""
