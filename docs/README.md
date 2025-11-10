# UniversalNER Documentation

This directory contains documentation for the UniversalNER toponym extraction project.

## Documents

### [XML_CONVERSION_OPTIMIZATION.md](XML_CONVERSION_OPTIMIZATION.md)
**November 2025 - Major Performance Improvement**

Documents the optimization of the NER-to-XML conversion process that achieved:
- **80-97% faster** processing (from 30+ min timeouts to 1-7 min)
- **85-91% smaller** file sizes (4.7GB → 405MB for Saskatchewan collection)
- **100% success rate** (687/687 documents processed successfully)
- **Zero memory errors** (eliminated OOM kills)

Key change: Removed co-occurrence computation from XML generation, enabling flexible downstream analysis with any proximity metric.

## Project Overview

**UniversalNER Toponym Extraction Pipeline**

A scalable system for extracting geographic entities (toponyms) from historical documents using:
- OLMoCR for OCR (optical character recognition)
- Universal-NER for named entity recognition
- Structured XML output with character offsets

### Collections Processed

1. **Saskatchewan Collection**: 587 historical documents
2. **Caribbean Collection**: 99 historical documents

### Output Format

Simplified XML structure with:
- Full text preservation (paragraphs)
- Character offsets for each entity mention
- Entity type categorization
- Metadata (document ID, entity/mention counts)

Example:
```xml
<document id="doc_id" total_entity_count="2964" total_mention_count="133115">
  <text paragraph_count="142">
    <paragraph id="p0" char_start="0" char_end="1234">...</paragraph>
  </text>
  <entities>
    <toponyms unique_count="456" mention_count="12345">
      <toponym name="Saskatchewan" mention_count="42">
        <mention paragraph_id="p0" char_start="123" char_end="135"/>
      </toponym>
    </toponyms>
  </entities>
</document>
```

## Usage

### Converting NER JSON to XML

```bash
# Single file
python -m src.utils.convert_ner_to_xml \
  --input_dir /path/to/ner_json \
  --output_dir /path/to/xml_output

# Batch processing with SLURM
sbatch --array=0-N cluster/nibi/slurm/convert_toponym_to_xml.sbatch \
  /path/to/ner_json \
  /path/to/xml_output
```

### Computing Co-occurrence (Downstream Analysis)

See [XML_CONVERSION_OPTIMIZATION.md](XML_CONVERSION_OPTIMIZATION.md#computing-co-occurrence-in-downstream-analysis) for Python code to compute entity co-occurrence from the simplified XML format.

## Performance Metrics

| Collection | Documents | Total Size | Avg File Size | Processing Time |
|------------|-----------|------------|---------------|-----------------|
| Saskatchewan | 587 | 405 MB | ~690 KB | ~2-3 min (parallel) |
| Caribbean | 99 | 308 MB | ~3.1 MB | ~2-3 min (parallel) |

Largest documents processed successfully:
- philippicluverii00clve_5: 34,829 entities, 615,295 mentions → 51MB XML (7 min)
- bim_eighteenth-century dictionary: 10,203 entities → 19MB XML (30 min)

## Contributing

When making changes to the XML conversion pipeline:
1. Update this documentation
2. Add performance benchmarks
3. Document breaking changes clearly
4. Include migration examples

## History

- **November 2025**: Major optimization - removed co-occurrence computation
- **Earlier 2025**: Initial implementation with nested nearby entities structure
