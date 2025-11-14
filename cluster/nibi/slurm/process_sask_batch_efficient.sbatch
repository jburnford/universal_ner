#!/bin/bash
#SBATCH --job-name=sask-full-toponym
#SBATCH --account=def-jic823
#SBATCH --array=0-99%100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=%x-b${BATCH_NUM}-%A_%a.out

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
UNPROCESSED_LIST="$HOME/projects/def-jic823/saskatchewan_unprocessed_files.txt"
mkdir -p "$OUTPUT_DIR"

# Batch configuration (passed via environment or defaults)
BATCH_OFFSET=${BATCH_OFFSET:-0}
FILES_PER_TASK=${FILES_PER_TASK:-25}
BATCH_NUM=${BATCH_NUM:-0}

# Calculate this worker's file range
START_INDEX=$((BATCH_OFFSET + (SLURM_ARRAY_TASK_ID * FILES_PER_TASK)))
END_INDEX=$((START_INDEX + FILES_PER_TASK))

echo "========================================="
echo "Saskatchewan Full Collection - Batch $BATCH_NUM"
echo "========================================="
echo "Task: $SLURM_ARRAY_TASK_ID / 100"
echo "File range: $START_INDEX to $((END_INDEX - 1))"
echo "Max runtime: 55 minutes"
echo "========================================="
echo ""

# Track start time
START_TIME=$(date +%s)
MAX_RUNTIME=$((55 * 60))  # 55 minutes in seconds
FILES_PROCESSED=0
FILES_SKIPPED=0

# Process files in this worker's range
for ((i=$START_INDEX; i<$END_INDEX; i++)); do
    # Check if we're approaching time limit
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -gt $MAX_RUNTIME ]; then
        echo ""
        echo "⏰ Approaching time limit ($ELAPSED seconds elapsed)"
        echo "✓ Processed: $FILES_PROCESSED files"
        echo "⊘ Skipped: $FILES_SKIPPED files (already done)"
        echo "Exiting gracefully..."
        exit 0
    fi

    # Get the file for this index (sed is 1-indexed)
    INPUT_FILE=$(sed -n "$((i + 1))p" "$UNPROCESSED_LIST")

    if [ -z "$INPUT_FILE" ]; then
        # Past end of file list
        break
    fi

    BASENAME=$(basename "$INPUT_FILE" .json)
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}.toponym.ner.json"

    # Skip if already processed
    if [ -f "$OUTPUT_FILE" ]; then
        FILES_SKIPPED=$((FILES_SKIPPED + 1))
        if [ $((FILES_SKIPPED % 10)) -eq 0 ]; then
            echo "[$i] Skipped $FILES_SKIPPED files... (elapsed: ${ELAPSED}s)"
        fi
        continue
    fi

    echo ""
    echo "========================================="
    echo "[$i] Processing: $BASENAME"
    echo "Elapsed: ${ELAPSED}s / ${MAX_RUNTIME}s"
    echo "Completed: $FILES_PROCESSED | Skipped: $FILES_SKIPPED"
    echo "========================================="

    # Run toponym extraction
    python -m src.serve.process_olmocr_json_toponyms \
      --input_file "$INPUT_FILE" \
      --output_file "$OUTPUT_FILE" \
      --model_path Universal-NER/UniNER-7B-type \
      --entity_types "person,toponym,water_body,landform,administrative_region,route,organization,date" \
      --max_new_tokens 512 \
      --chunk_size 2000 \
      --use_enhanced_descriptions True

    FILES_PROCESSED=$((FILES_PROCESSED + 1))
    echo "✓ Completed: $BASENAME"
done

echo ""
echo "========================================="
echo "✓ Batch $BATCH_NUM Task $SLURM_ARRAY_TASK_ID FINISHED"
echo "Files processed: $FILES_PROCESSED"
echo "Files skipped: $FILES_SKIPPED"
FINAL_TIME=$(date +%s)
TOTAL_ELAPSED=$((FINAL_TIME - START_TIME))
echo "Total time: ${TOTAL_ELAPSED}s"
echo "========================================="
