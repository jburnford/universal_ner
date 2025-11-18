#!/bin/bash
# Create batch file lists for efficient processing

INPUT_BASE="/home/jic823/colonialofficelist/output_3"
BATCH_DIR="$HOME/projects/def-jic823/colonial_batches"
FILES_PER_BATCH=20

mkdir -p "$BATCH_DIR"

echo "Creating batch file lists..."
echo ""

# Get all files
FILES=($(find "$INPUT_BASE" -type d -name "*_manual_parsed" -exec find {} \( -name "*.txt" -o -name "*.md" \) \; | sort))
TOTAL=${#FILES[@]}

echo "Total files: $TOTAL"
echo "Files per batch: $FILES_PER_BATCH"

# Calculate number of batches
NUM_BATCHES=$(( (TOTAL + FILES_PER_BATCH - 1) / FILES_PER_BATCH ))
echo "Number of batches: $NUM_BATCHES"
echo ""

# Create batch files
for ((batch=0; batch<NUM_BATCHES; batch++)); do
    batch_file="$BATCH_DIR/batch_${batch}.txt"
    > "$batch_file"  # Clear file

    start=$((batch * FILES_PER_BATCH))
    end=$((start + FILES_PER_BATCH))

    if [ $end -gt $TOTAL ]; then
        end=$TOTAL
    fi

    count=0
    for ((i=start; i<end; i++)); do
        echo "${FILES[$i]}" >> "$batch_file"
        ((count++))
    done

    echo "Batch $batch: $count files"
done

echo ""
echo "✓ Batch files created in: $BATCH_DIR"
echo "  Array size for SLURM: --array=0-$((NUM_BATCHES-1))"
