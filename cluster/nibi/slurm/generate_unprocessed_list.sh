#!/bin/bash
# Generate list of unprocessed Saskatchewan files

OCR_DIR="$HOME/projects/def-jic823/pdfs_sask_test/results/json"
XML_DIR="$HOME/projects/def-jic823/saskatchewan_full_toponym_xml"
OUTPUT_LIST="$HOME/projects/def-jic823/saskatchewan_unprocessed_files.txt"

echo "Scanning for unprocessed files..."
echo "OCR directory: $OCR_DIR"
echo "XML directory: $XML_DIR"
echo "Output list: $OUTPUT_LIST"

# Clear existing list
> "$OUTPUT_LIST"

# Find all JSON files that don't have corresponding XML
count=0
for json_file in "$OCR_DIR"/*.json; do
    basename=$(basename "$json_file" .json)
    xml_file="$XML_DIR/${basename}.toponym.xml"

    if [ ! -f "$xml_file" ]; then
        echo "$json_file" >> "$OUTPUT_LIST"
        ((count++))
    fi
done

echo "Found $count unprocessed files"
echo "List written to: $OUTPUT_LIST"
echo ""
echo "To submit job, update array size to 0-$((count-1))%100 and run:"
echo "sbatch cluster/nibi/slurm/process_sask_full_collection_smart.sbatch"
