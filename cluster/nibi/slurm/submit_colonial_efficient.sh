#!/bin/bash
# Submit efficient batch processing for Colonial Office List output_3

cd ~/projects/def-jic823/universal_ner

echo "========================================="
echo "Colonial Office List NER - Efficient Batch"
echo "========================================="
echo ""
echo "Creating batch file lists (20 files per batch)..."
bash cluster/nibi/slurm/create_batches.sh
echo ""

NUM_BATCHES=$(ls ~/projects/def-jic823/colonial_batches/batch_*.txt 2>/dev/null | wc -l)

if [ $NUM_BATCHES -eq 0 ]; then
    echo "ERROR: No batch files created"
    exit 1
fi

echo "========================================="
echo "Ready to submit:"
echo "  Batches: $NUM_BATCHES"
echo "  Files per batch: ~20"
echo "  Total files: ~370"
echo "  Max concurrent: 10 batches"
echo "  Time per batch: ~10-15 minutes"
echo "  Total runtime: ~30-45 minutes"
echo "========================================="
echo ""
echo "Submitting batch job..."

JOB_ID=$(sbatch --parsable cluster/nibi/slurm/process_colonial_output3_efficient.sbatch)

echo "✓ Job submitted: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -u jic823"
echo "  squeue -j $JOB_ID"
echo ""
echo "Check progress:"
echo "  ls ~/projects/def-jic823/colonial_output3_ner/*.ner.json | wc -l"
echo ""
echo "View logs:"
echo "  tail -f ~/projects/def-jic823/universal_ner/colonial-ner-batch-${JOB_ID}_*.out"
echo ""
