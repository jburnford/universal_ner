#!/bin/bash
#SBATCH --job-name=sask-toponym-test
#SBATCH --account=def-jic823
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
TEST_DIR="$HOME/projects/def-jic823/toponym_sask_test"
mkdir -p "$TEST_DIR"

# Select 1 Saskatchewan document for testing (P000045 - a medium-sized document)
SASK_DIR="$HOME/projects/def-jic823/saskatchewan_ner"
DOCS=("$SASK_DIR/P000045.ner.json")

echo ""
echo "========================================="
echo "Saskatchewan Toponym NER Test"
echo "========================================="
echo "Testing granular geographic entity extraction"
echo ""
echo "Test document: P000045 (Alexander Henry's Travels)"
echo "  Standard 'location' count: 525 entities"
echo ""
echo "Entity Types:"
echo "  Standard:  location (all geographic entities)"
echo "  Granular:  toponym, water_body, landform,"
echo "             administrative_region, route"
echo "========================================="
echo ""

# Process each test document
for ner_file in "${DOCS[@]}"; do
    basename_ner=$(basename "$ner_file")
    doc_id="${basename_ner%.ner.json}"

    echo "Processing $doc_id..."

    # Find original OLMoCR JSON (need to reconstruct path)
    # Saskatchewan files are named like P000045.ner.json
    # Original OLMoCR files should be in saskatchewan_raw or similar

    # For now, extract text from the NER file and create a temporary OLMoCR-format JSON
    python3 << PYEOF
import json
from pathlib import Path

ner_file = Path("$ner_file")
test_dir = Path("$TEST_DIR")
doc_id = "$doc_id"

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
olmocr_file = test_dir / f"{doc_id}.olmocr.json"
with open(olmocr_file, 'w') as f:
    json.dump(olmocr_data, f)

print(f"Created: {olmocr_file}")
PYEOF

    # Run toponym extraction
    python -m src.serve.process_olmocr_json_toponyms \
      --input_file "$TEST_DIR/${doc_id}.olmocr.json" \
      --output_file "$TEST_DIR/${doc_id}.toponym.ner.json" \
      --model_path Universal-NER/UniNER-7B-type \
      --entity_types "toponym,water_body,landform,administrative_region,route" \
      --max_new_tokens 512 \
      --chunk_size 2000 \
      --use_enhanced_descriptions True

    echo "  ✓ $doc_id completed"
    echo ""
done

echo "========================================="
echo "Toponym extraction complete!"
echo "========================================="
echo ""

# Create comparison report
python3 << 'PYEOF'
import json
from pathlib import Path
from collections import defaultdict

test_dir = Path("/home/jic823/projects/def-jic823/toponym_sask_test")
sask_dir = Path("/home/jic823/projects/def-jic823/saskatchewan_ner")

print("=" * 80)
print("COMPARISON: Standard Location vs Granular Geographic Entities")
print("=" * 80)
print()

# Process each test document
for toponym_file in sorted(test_dir.glob("*.toponym.ner.json")):
    doc_id = toponym_file.stem.replace('.toponym.ner', '')
    standard_file = sask_dir / f"{doc_id}.ner.json"

    print(f"Document: {doc_id}")
    print("-" * 80)

    # Load both versions
    with open(toponym_file) as f:
        toponym_data = json.load(f)

    with open(standard_file) as f:
        standard_data = json.load(f)

    # Count standard locations
    standard_locations = set()
    for page in standard_data:
        locs = page.get("entities", {}).get("location", [])
        standard_locations.update(locs)

    # Count granular geographic entities
    granular_counts = defaultdict(set)
    for page in toponym_data:
        entities = page.get("entities", {})
        for entity_type in ["toponym", "water_body", "landform", "administrative_region", "route"]:
            items = entities.get(entity_type, [])
            granular_counts[entity_type].update(items)

    total_granular = sum(len(v) for v in granular_counts.values())

    print(f"  Standard 'location' entities:  {len(standard_locations):4}")
    print(f"  Granular entities total:       {total_granular:4}")
    print()
    print("  Breakdown of granular entities:")
    for entity_type in ["toponym", "water_body", "landform", "administrative_region", "route"]:
        count = len(granular_counts[entity_type])
        print(f"    {entity_type:25} : {count:4}")
    print()

    # Show sample entities from each category
    print("  Sample Toponyms (settlements):")
    for i, t in enumerate(sorted(list(granular_counts["toponym"]))[:10], 1):
        print(f"    {i:2}. {t}")

    if granular_counts["water_body"]:
        print()
        print("  Sample Water Bodies:")
        for i, t in enumerate(sorted(list(granular_counts["water_body"]))[:5], 1):
            print(f"    {i:2}. {t}")

    if granular_counts["landform"]:
        print()
        print("  Sample Landforms:")
        for i, t in enumerate(sorted(list(granular_counts["landform"]))[:5], 1):
            print(f"    {i:2}. {t}")

    print()
    print("=" * 80)
    print()

print()
print("Analysis:")
print("- Toponyms: Named inhabited places (cities, towns, forts, missions)")
print("- Water Bodies: Rivers, lakes, bays, harbors, seas")
print("- Landforms: Mountains, islands, valleys, capes")
print("- Admin Regions: Provinces, territories, districts, colonies")
print("- Routes: Roads, paths, trails, passes")
print()
print("This granular approach separates 'Saskatchewan' (admin region)")
print("from 'Saskatoon' (toponym) from 'Saskatchewan River' (water body)")
print("=" * 80)

PYEOF

echo ""
echo "[DONE] Test completed at $(date)"
echo "Results in: $TEST_DIR"
