#!/usr/bin/env python3
"""
Process multiple markdown/text files in batch with Universal-NER.
Loads model once, processes many files efficiently.
"""
import fire
import torch
import json
import sys
from pathlib import Path
from transformers import pipeline

sys.path.insert(0, '/project/6080182/universal_ner')
from src.utils import preprocess_instance


def chunk_text_by_paragraphs(text, target_size=2000, overlap=200):
    """Chunk text at paragraph boundaries."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        if current_size + para_size > target_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))

            if overlap > 0 and len(current_chunk) > 0:
                current_chunk = [current_chunk[-1]]
                current_size = len(current_chunk[-1])
            else:
                current_chunk = []
                current_size = 0

        current_chunk.append(para)
        current_size += para_size

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def extract_entities(generator, text, entity_types, max_new_tokens=256, use_enhanced_descriptions=False):
    """Extract entities of specified types from text."""

    enhanced_descriptions = {
        "person": "individuals, historical figures, officials, traders, explorers, captains, or named persons including titles and honorifics",
        "organization": "companies, trading corporations, governmental bodies, colonial organizations, administrative offices, institutional entities, ships, missionary stations, or chartered companies",
        "date": "specific dates, years, months, or time periods with numerical indicators",
        "toponym": "named inhabited places and settlements including cities, towns, villages, hamlets, forts, trading posts, missions, plantations, estates, or any named location where people live or lived",
        "water_body": "named bodies of water including rivers, streams, creeks, lakes, ponds, seas, oceans, bays, harbors, straits, channels, or maritime features",
        "landform": "named natural land features including mountains, hills, valleys, plains, islands, peninsulas, capes, points, or other terrain features",
        "administrative_region": "named administrative or political divisions including countries, colonies, provinces, territories, districts, counties, parishes, or governmental jurisdictions",
        "route": "named routes, roads, paths, trails, passes, or transportation corridors",
        "location": "places, settlements, geographic locations, forts, trading posts, rivers, bays, straits, territories, or regions"
    }

    results = {}

    for entity_type in entity_types:
        if use_enhanced_descriptions and entity_type in enhanced_descriptions:
            type_description = enhanced_descriptions[entity_type]
        else:
            type_description = entity_type

        example = {
            "conversations": [
                {"from": "human", "value": f"Text: {text}"},
                {"from": "gpt", "value": "I've read this text."},
                {"from": "human", "value": f"What describes {type_description} in the text?"},
                {"from": "gpt", "value": "[]"}
            ]
        }

        prompt = preprocess_instance(example['conversations'])

        try:
            output = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                return_full_text=False,
                temperature=0.0,
                do_sample=False
            )

            response = output[0]['generated_text'].strip()

            try:
                entities = json.loads(response)
                if isinstance(entities, list):
                    results[entity_type] = entities
                else:
                    results[entity_type] = []
            except json.JSONDecodeError:
                results[entity_type] = []

        except Exception as e:
            print(f"Error extracting {entity_type}: {e}")
            results[entity_type] = []

    return results


def process_file_batch(
    file_list: str,
    output_dir: str,
    model_path: str = "Universal-NER/UniNER-7B-type",
    entity_types: str = "person,toponym,water_body,landform,administrative_region,route,organization,date",
    max_new_tokens: int = 512,
    chunk_size: int = 2000,
    use_enhanced_descriptions: bool = True
):
    """
    Process a batch of markdown/text files with NER (loads model once).

    Args:
        file_list: Path to text file containing list of input files (one per line)
        output_dir: Directory for output JSON files
        model_path: HuggingFace model path
        entity_types: Comma-separated entity types
        max_new_tokens: Maximum tokens for generation
        chunk_size: Size of text chunks for processing
        use_enhanced_descriptions: Use detailed entity type descriptions
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse entity types
    if isinstance(entity_types, str):
        entity_types_list = [t.strip() for t in entity_types.split(',')]
    elif isinstance(entity_types, (list, tuple)):
        entity_types_list = list(entity_types)
    else:
        entity_types_list = [entity_types]

    # Read file list
    with open(file_list) as f:
        input_files = [line.strip() for line in f if line.strip()]

    print(f"========================================")
    print(f"Batch NER Processing")
    print(f"========================================")
    print(f"Files in batch: {len(input_files)}")
    print(f"Entity types: {', '.join(entity_types_list)}")
    print(f"Enhanced descriptions: {use_enhanced_descriptions}")
    print(f"Output directory: {output_dir}")
    print(f"========================================")
    print()

    # Load model ONCE for entire batch
    print(f"Loading model: {model_path}")
    import time
    load_start = time.time()
    generator = pipeline(
        'text-generation',
        model=model_path,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    load_time = time.time() - load_start
    print(f"✓ Model loaded in {load_time:.1f}s")
    print()

    # Process each file
    processed = 0
    skipped = 0
    errors = 0

    for idx, input_file in enumerate(input_files, 1):
        input_path = Path(input_file)

        if not input_path.exists():
            print(f"[{idx}/{len(input_files)}] ⚠ File not found: {input_file}")
            errors += 1
            continue

        # Create output filename
        basename = input_path.stem
        # Extract year from path if present (e.g., 1896_manual_parsed)
        year_match = None
        for parent in input_path.parents:
            if '_manual_parsed' in parent.name:
                year_match = parent.name.split('_')[0]
                break

        if year_match:
            output_file = output_path / f"{year_match}_{basename}.ner.json"
        else:
            output_file = output_path / f"{basename}.ner.json"

        # Skip if already processed
        if output_file.exists():
            print(f"[{idx}/{len(input_files)}] ⊘ Already processed: {basename}")
            skipped += 1
            continue

        print(f"[{idx}/{len(input_files)}] Processing: {basename}")

        try:
            # Read file
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # Chunk text
            chunks = chunk_text_by_paragraphs(text, target_size=chunk_size)
            print(f"  → {len(chunks)} chunks")

            # Collect entities from all chunks
            all_entities = {et: set() for et in entity_types_list}

            for chunk_idx, chunk in enumerate(chunks):
                chunk_entities = extract_entities(
                    generator,
                    chunk,
                    entity_types_list,
                    max_new_tokens=max_new_tokens,
                    use_enhanced_descriptions=use_enhanced_descriptions
                )

                # Merge entities
                for et in entity_types_list:
                    if et in chunk_entities:
                        all_entities[et].update(chunk_entities[et])

            # Create output
            result = {
                "file": str(input_path.name),
                "source_file": str(input_file),
                "chunks_processed": len(chunks),
                "entities": {et: sorted(list(all_entities[et])) for et in entity_types_list}
            }

            # Write output
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Print summary
            total_entities = sum(len(result['entities'][et]) for et in entity_types_list)
            print(f"  ✓ Extracted {total_entities} unique entities")
            processed += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            errors += 1

    print()
    print(f"========================================")
    print(f"Batch Complete")
    print(f"========================================")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"Total time: {time.time() - load_start:.1f}s")
    print(f"========================================")


if __name__ == '__main__':
    fire.Fire(process_file_batch)
