#!/bin/bash
#SBATCH --job-name=colonial-ner-batch
#SBATCH --account=def-jic823
#SBATCH --array=0-18%10
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
INPUT_BASE="/home/jic823/colonialofficelist/output_3"
OUTPUT_DIR="$HOME/projects/def-jic823/colonial_output3_ner"
BATCH_DIR="$HOME/projects/def-jic823/colonial_batches"
mkdir -p "$OUTPUT_DIR" "$BATCH_DIR"

echo "========================================="
echo "Colonial Office List NER - Efficient Batch"
echo "========================================="
echo "Batch task: $SLURM_ARRAY_TASK_ID / 19"
echo "Processing ~20 files per batch"
echo "========================================="
echo ""

# Get the file list for this batch
BATCH_FILE="$BATCH_DIR/batch_${SLURM_ARRAY_TASK_ID}.txt"

if [ ! -f "$BATCH_FILE" ]; then
    echo "ERROR: Batch file not found: $BATCH_FILE"
    exit 1
fi

FILE_COUNT=$(wc -l < "$BATCH_FILE")
echo "Files in this batch: $FILE_COUNT"
echo ""

# Process entire batch with model loaded once
python -m src.serve.process_markdown_batch \
  --file_list "$BATCH_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --model_path Universal-NER/UniNER-7B-type \
  --entity_types "person,toponym,water_body,landform,administrative_region,route,organization,date" \
  --max_new_tokens 512 \
  --chunk_size 2000 \
  --use_enhanced_descriptions True

echo ""
echo "[DONE] Batch $SLURM_ARRAY_TASK_ID completed at $(date)"
