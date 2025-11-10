# Saskatchewan Full Collection Processing

## Overview

Complete pipeline for processing 13,886 Saskatchewan historical documents (through 1945) from Internet Archive with NER and metadata embedding.

## Data Source

**Location**: `~/projects/def-jic823/pdfs_sask_test/` (Nibi cluster)
- **OCR Files**: 13,887 JSON files in `results/json/`
- **Metadata**: `~/projects/def-jic823/saskatchewan_metadata.json`
- **Collections**: Peel Newspapers (11,047), Peel Print (1,582), plus government publications and microfilm

## Complete Pipeline

### 1. Metadata Extraction ✅ COMPLETE

Already extracted from backup database (`archive_tracking.db.backup-20251009-212014`):

```bash
# Location on Nibi
~/projects/def-jic823/saskatchewan_metadata.json  # 13,886 entries
~/projects/def-jic823/saskatchewan_metadata_mapping.csv  # Raw CSV
```

**Metadata fields**: identifier, title, creator, publisher, date, year, language, subject, collection, description

### 2. Batch NER Processing with Metadata Embedding

**Script**: `cluster/nibi/slurm/process_sask_full_collection.sbatch`

```bash
# On Nibi cluster
cd ~/projects/def-jic823/universal_ner
git pull
sbatch cluster/nibi/slurm/process_sask_full_collection.sbatch
```

**What it does**:
1. Processes all 13,886 OCR JSON files through Universal-NER
2. Extracts toponyms and 7 other entity types (person, water_body, landform, administrative_region, route, organization, date)
3. Converts to XML with embedded Internet Archive metadata
4. Skips already-processed files (idempotent - safe to rerun)
5. Limits to 100 concurrent GPU jobs

**Outputs**:
- **NER Results**: `~/projects/def-jic823/saskatchewan_full_toponym/*.toponym.ner.json`
- **XML with Metadata**: `~/projects/def-jic823/saskatchewan_full_toponym_xml/*.toponym.xml`

**Job parameters**:
- Array size: 0-13885 (max 100 concurrent)
- GPU: 1x H100
- Memory: 32GB
- Time: 1 hour per document
- Estimated total time: ~140 hours of GPU time (distributed across 100 GPUs = ~1.4 hours wall time)

### 3. XML Output Format

```xml
<document id="P000045" source_file="P000045.toponym.ner.json"
          total_entity_count="156" total_mention_count="892">
  <metadata>
    <identifier>P000045</identifier>
    <title>Travels and adventures in Canada and the Indian territories...</title>
    <creator>Henry, Alexander elder, 1739-1824</creator>
    <publisher>Printed and published by I. Riley (New York)</publisher>
    <date>1809</date>
    <year>1809</year>
    <language>English</language>
    <subject>Saskatchewan--Description and travel; 1776</subject>
    <collection>peel_print; peel; university_of_alberta_libraries</collection>
  </metadata>
  <text paragraph_count="142">
    <paragraph id="p0" char_start="0" char_end="1234">Full paragraph text...</paragraph>
  </text>
  <entities>
    <toponyms unique_count="45" mention_count="234">
      <toponym name="Saskatchewan" mention_count="23">
        <mention paragraph_id="p0" char_start="123" char_end="135"/>
      </toponym>
    </toponyms>
  </entities>
</document>
```

## Monitoring Progress

```bash
# Check running jobs
squeue -u jic823 | grep sask-full

# Count completed files
ls ~/projects/def-jic823/saskatchewan_full_toponym_xml/*.xml | wc -l

# View recent log
ls -t sask-full-toponym-*.out | head -1 | xargs tail -20

# Check for errors
grep -l "Error\|Failed" sask-full-toponym-*.out | wc -l
```

## Downloading Results

```bash
# Download XML files (locally)
mkdir -p ~/saskatchewan_full_toponym_xml
rsync -avz --progress nibi:projects/def-jic823/saskatchewan_full_toponym_xml/ ~/saskatchewan_full_toponym_xml/

# Download NER files (if needed)
rsync -avz --progress nibi:projects/def-jic823/saskatchewan_full_toponym/ ~/saskatchewan_full_toponym/
```

## Collection Statistics

| Metric | Value |
|--------|-------|
| Total Documents | 13,886 |
| OCR Files | 13,887 JSON files (15 GB) |
| Metadata Coverage | 100% (all have linked metadata) |
| Collections | Peel Newspapers, Peel Print, Gov't Publications, Microfilm |
| Date Range | ~1800s - 1945 |
| Previous Coverage | 587 documents (4.2%) |
| New Coverage | 13,886 documents (100%) |

## Comparison to Previous Collection

| Aspect | Previous (587 files) | Full Collection (13,886 files) |
|--------|---------------------|-------------------------------|
| **Source** | saskatchewan_ner/ | pdfs_sask_test/results/json/ |
| **Coverage** | Peel collection subset | All Saskatchewan through 1945 |
| **Metadata** | Minimal | Full Internet Archive metadata |
| **Size** | 405 MB XML | ~9.5 GB XML (estimated) |
| **Processing** | Already complete | Ready to run |

## Cost Estimate

- **GPU Hours**: ~140 hours of H100 time
- **Wall Time**: ~1.4 hours with 100 concurrent jobs
- **Storage**: ~10 GB for XML output (compressed ~2 GB)

## Notes

1. **Idempotent**: Script checks for existing XML files and skips them - safe to rerun if interrupted
2. **Metadata Fallback**: If metadata not found for a file, processing continues without it
3. **Entity Types**: Extracts 8 types (person, toponym, water_body, landform, administrative_region, route, organization, date)
4. **Existing Infrastructure**: Uses the same proven pipeline as Caribbean and initial Saskatchewan collections

## Related Files

- **Batch Script**: `cluster/nibi/slurm/process_sask_full_collection.sbatch`
- **XML Converter**: `src/utils/convert_ner_to_xml.py` (with metadata support)
- **NER Processor**: `src/serve/process_olmocr_json_toponyms.py`
- **Documentation**: `docs/README.md`, `docs/XML_CONVERSION_OPTIMIZATION.md`

## Future Work

1. Create co-occurrence analysis scripts for the full collection
2. Build temporal analysis tools (tracking place names over time 1800s-1945)
3. Integrate with Saskatchewan Knowledge Graph (Neo4j)
4. Create visualization dashboard for geographic entity distribution

## Troubleshooting

### Job fails with OOM
Increase memory: `#SBATCH --mem=64G`

### Metadata not loading
Check path in script matches: `$HOME/projects/def-jic823/saskatchewan_metadata.json`

### Too many concurrent jobs
Reduce limit: `#SBATCH --array=0-13885%50` (50 instead of 100)

### Missing input files
Verify OCR directory: `ls ~/projects/def-jic823/pdfs_sask_test/results/json/ | wc -l`
