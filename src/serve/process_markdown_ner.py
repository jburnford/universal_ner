#!/usr/bin/env python3
"""
Process markdown files with Universal-NER to extract entities.

Input: Markdown file (.md)
Output: JSON file with entity annotations
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
    """
    Chunk text at paragraph boundaries.

    Args:
        text: Input text to chunk
        target_size: Target size for each chunk (characters)
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    # Split on double newlines (paragraph breaks)
    paragraphs = text.split('\n\n')

    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        # If adding this paragraph would exceed target, save current chunk
        if current_size + para_size > target_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))

            # Start new chunk with overlap (last paragraph from previous chunk)
            if overlap > 0 and len(current_chunk) > 0:
                current_chunk = [current_chunk[-1]]
                current_size = len(current_chunk[-1])
            else:
                current_chunk = []
                current_size = 0

        current_chunk.append(para)
        current_size += para_size

    # Add final chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def extract_entities(generator, text, entity_types, max_new_tokens=256, use_enhanced_descriptions=False):
    """Extract entities of specified types from text."""

    # Enhanced entity type descriptions
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
        # Use enhanced description if enabled, otherwise use simple type name
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

            # Parse the output (expecting JSON array format)
            try:
                entities = json.loads(response)
                if isinstance(entities, list):
                    results[entity_type] = entities
                else:
                    results[entity_type] = []
            except json.JSONDecodeError:
                # If not valid JSON, try to extract entities from text
                results[entity_type] = []

        except Exception as e:
            print(f"Error extracting {entity_type}: {e}")
            results[entity_type] = []

    return results


def process_file(
    input_file: str,
    output_file: str = None,
    model_path: str = "Universal-NER/UniNER-7B-type",
    entity_types: str = "person,toponym,water_body,landform,administrative_region,organization,date",
    max_new_tokens: int = 512,
    chunk_size: int = 2000,
    use_enhanced_descriptions: bool = True
):
    """
    Process a markdown file and extract entities.

    Args:
        input_file: Path to markdown file
        output_file: Path for output JSON (default: input_file.ner.json)
        model_path: HuggingFace model path
        entity_types: Comma-separated entity types
        max_new_tokens: Maximum tokens for generation
        chunk_size: Size of text chunks for processing
        use_enhanced_descriptions: Use detailed entity type descriptions
    """
    input_path = Path(input_file)

    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}.ner.json"

    print(f"Loading model: {model_path}")
    generator = pipeline(
        'text-generation',
        model=model_path,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print(f"Reading input: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Handle entity_types as either string or tuple (Fire parsing)
    if isinstance(entity_types, str):
        entity_types_list = [t.strip() for t in entity_types.split(',')]
    elif isinstance(entity_types, (list, tuple)):
        entity_types_list = list(entity_types)
    else:
        entity_types_list = [entity_types]

    print(f"Processing markdown file...")
    print(f"Entity types: {', '.join(entity_types_list)}")
    print(f"Enhanced descriptions: {use_enhanced_descriptions}")

    # Chunk text
    chunks = chunk_text_by_paragraphs(text, target_size=chunk_size)
    print(f"Split into {len(chunks)} chunks")

    # Collect entities from all chunks
    all_entities = {et: set() for et in entity_types_list}

    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}")

        chunk_entities = extract_entities(
            generator,
            chunk,
            entity_types_list,
            max_new_tokens=max_new_tokens,
            use_enhanced_descriptions=use_enhanced_descriptions
        )

        # Merge entities from this chunk
        for et in entity_types_list:
            if et in chunk_entities:
                all_entities[et].update(chunk_entities[et])

    # Convert sets to sorted lists for output
    result = {
        "file": str(input_path.name),
        "source_file": str(input_file),
        "chunks_processed": len(chunks),
        "entities": {et: sorted(list(all_entities[et])) for et in entity_types_list}
    }

    print(f"\nWriting output: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\nEntity Extraction Summary:")
    print("=" * 60)
    for et in entity_types_list:
        print(f"{et:25} : {len(result['entities'][et]):5} unique entities")
    print("=" * 60)

    return output_file


if __name__ == '__main__':
    fire.Fire(process_file)
