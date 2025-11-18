#!/bin/bash
# Count files in output_3 for batch job sizing

INPUT_BASE="/home/jic823/colonialofficelist/output_3"

echo "Counting text files in output_3..."
echo ""

# Get all text/markdown files from year directories
FILES=($(find "$INPUT_BASE" -type d -name "*_manual_parsed" -exec find {} \( -name "*.txt" -o -name "*.md" \) \; | sort))
TOTAL=${#FILES[@]}

echo "Total files found: $TOTAL"
echo ""
echo "Breakdown by year:"
for year_dir in "$INPUT_BASE"/*_manual_parsed/; do
    year=$(basename "$year_dir" | cut -d'_' -f1)
    txt_count=$(find "$year_dir" -name "*.txt" | wc -l)
    md_count=$(find "$year_dir" -name "*.md" | wc -l)
    total_count=$((txt_count + md_count))
    echo "  $year: $total_count files ($txt_count .txt, $md_count .md)"
done

echo ""
echo "Array size for SLURM: --array=0-$((TOTAL-1))%50"
echo ""
echo "Sample files:"
for i in 0 50 100 150 200 250 300 350; do
    if [ $i -lt $TOTAL ]; then
        file="${FILES[$i]}"
        year_dir=$(basename "$(dirname "$file")")
        year=${year_dir%%_*}
        basename=$(basename "$file")
        basename="${basename%.*}"
        ext="${file##*.}"
        echo "  [$i] $year/$basename (.$ext)"
    fi
done
