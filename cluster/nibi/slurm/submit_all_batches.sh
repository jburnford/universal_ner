#!/bin/bash
# Submit multiple batches to process Saskatchewan collection efficiently
# Each batch handles a different file range, all submitted at once to hold queue positions

REPO_DIR="$HOME/projects/def-jic823/universal_ner"
cd "$REPO_DIR"

# Cancel existing job if needed
if [ "$1" = "--cancel" ]; then
    echo "Cancelling job 4368548..."
    scancel 4368548
    shift
fi

# Configuration
FILES_PER_TASK=25
TASKS_PER_BATCH=100
FILES_PER_BATCH=$((FILES_PER_TASK * TASKS_PER_BATCH))

echo "========================================="
echo "Submitting Saskatchewan NER Processing Batches"
echo "========================================="
echo "Files per task: $FILES_PER_TASK"
echo "Tasks per batch: $TASKS_PER_BATCH"
echo "Files per batch: $FILES_PER_BATCH"
echo ""

# Submit 5 batches covering different file ranges
for batch_num in 0 1 2 3 4; do
    offset=$((batch_num * FILES_PER_BATCH))

    echo "Batch $batch_num: files $offset - $((offset + FILES_PER_BATCH - 1))"

    job_id=$(BATCH_OFFSET=$offset BATCH_NUM=$batch_num FILES_PER_TASK=$FILES_PER_TASK \
        sbatch --parsable cluster/nibi/slurm/process_sask_batch_efficient.sbatch)

    echo "  → Submitted job $job_id (100 tasks)"
    echo ""
done

echo "========================================="
echo "✓ All batches submitted!"
echo "Total: 500 concurrent jobs in queue"
echo "Expected coverage: 0-12,499 files"
echo ""
echo "Monitor with: squeue -u jic823"
echo "========================================="
