#!/bin/bash
# Submit Colonial Office List NER processing for output_3

cd ~/projects/def-jic823/universal_ner

echo "========================================="
echo "Colonial Office List NER - output_3"
echo "========================================="
echo ""
echo "Files to process: 370 colonial territories (1896-1906)"
echo "Array jobs: 370 tasks, max 50 concurrent"
echo "Time per file: ~30 minutes (conservative estimate)"
echo "Output directory: ~/projects/def-jic823/colonial_output3_ner"
echo ""
echo "Submitting batch job..."
echo ""

JOB_ID=$(sbatch --parsable cluster/nibi/slurm/process_colonial_output3.sbatch)

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
echo "  tail -f ~/projects/def-jic823/universal_ner/colonial-ner-${JOB_ID}_*.out"
echo ""
