#!/bin/bash
#SBATCH --job-name=test-toponym-ner
#SBATCH --account=def-jic823
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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

# Test directories
TEST_DIR="$HOME/projects/def-jic823/toponym_test"
mkdir -p "$TEST_DIR"

# Pick first Caribbean document for testing
INPUT_DIR="$HOME/projects/def-jic823/caribbean_ner_100/input"
TEST_FILE=$(ls "$INPUT_DIR"/*.json | head -1)
BASENAME=$(basename "$TEST_FILE" .json)

echo ""
echo "========================================="
echo "Toponym NER Test"
echo "========================================="
echo "Input:  $TEST_FILE"
echo "Output: $TEST_DIR/${BASENAME}.toponym.ner.json"
echo "Model: Universal-NER/UniNER-7B-type"
echo ""
echo "Entity Types:"
echo "  - toponym (named settlements)"
echo "  - water_body (rivers, bays, etc.)"
echo "  - landform (mountains, islands, etc.)"
echo "  - administrative_region (colonies, provinces, etc.)"
echo "  - route (roads, paths, etc.)"
echo "  - person"
echo "  - organization"
echo "  - date"
echo "========================================="
echo ""

# Run toponym extraction
python -m src.serve.process_olmocr_json_toponyms \
  --input_file "$TEST_FILE" \
  --output_file "$TEST_DIR/${BASENAME}.toponym.ner.json" \
  --model_path Universal-NER/UniNER-7B-type \
  --entity_types "person,toponym,water_body,landform,administrative_region,route,organization,date" \
  --max_new_tokens 512 \
  --chunk_size 2000 \
  --use_enhanced_descriptions True

echo ""
echo "========================================="
echo "Toponym extraction complete!"
echo "========================================="
echo ""

# Show results summary
python3 << 'PYEOF'
import json
from pathlib import Path
import sys

test_dir = Path("/home/jic823/projects/def-jic823/toponym_test")
result_file = list(test_dir.glob("*.toponym.ner.json"))[0]

with open(result_file) as f:
    data = json.load(f)

print("Entity Extraction Summary:")
print("=" * 60)

entity_types = ["person", "toponym", "water_body", "landform",
                "administrative_region", "route", "organization", "date"]

for et in entity_types:
    all_entities = set()
    for page in data:
        entities = page.get("entities", {}).get(et, [])
        all_entities.update(entities)

    print(f"{et:25} : {len(all_entities):5} unique entities")

print("=" * 60)

# Show sample toponyms
print("\nSample Toponyms (first 20):")
toponyms = set()
for page in data:
    toponyms.update(page.get("entities", {}).get("toponym", []))

for i, t in enumerate(sorted(list(toponyms))[:20], 1):
    print(f"  {i:2}. {t}")

PYEOF

echo ""
echo "[DONE] Test completed at $(date)"
