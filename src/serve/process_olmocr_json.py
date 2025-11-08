#!/usr/bin/env python3
"""
Process OLMoCR JSON output with Universal-NER to extract entities.

Input: OLMoCR JSON file with structure: [{"id": "...", "text": "..."}]
Output: Enriched JSON with entity annotations
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
        if current_size > 0 and current_size + para_size > target_size:
            chunks.append('\n\n'.join(current_chunk))

            # Start new chunk with overlap (keep last paragraph for context)
            if overlap > 0 and current_chunk:
                current_chunk = [current_chunk[-1]]
                current_size = len(current_chunk[0])
            else:
                current_chunk = []
                current_size = 0

        current_chunk.append(para)
        current_size += para_size + 2  # +2 for \n\n

    # Add final chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def extract_entities(generator, text, entity_types, max_new_tokens=256):
    """Extract entities of specified types from text."""
    results = {}

    for entity_type in entity_types:
        example = {
            "conversations": [
                {"from": "human", "value": f"Text: {text}"},
                {"from": "gpt", "value": "I've read this text."},
                {"from": "human", "value": f"What describes {entity_type} in the text?"},
                {"from": "gpt", "value": "[]"}
            ]
        }

        prompt = preprocess_instance(example['conversations'])
        outputs = generator(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
        result = outputs[0]['generated_text'].strip()

        # Parse the result (should be a JSON array like ["entity1", "entity2"])
        try:
            entities = eval(result)  # Safe here since it's model output
            if entities and isinstance(entities, list):
                results[entity_type] = entities
        except:
            # If parsing fails, store raw result
            results[entity_type] = result

    return results


def merge_entity_results(entity_lists):
    """
    Merge entity results from multiple chunks, removing duplicates.

    Args:
        entity_lists: List of entity dictionaries

    Returns:
        Merged entity dictionary with unique entities
    """
    merged = {}

    for entities in entity_lists:
        for entity_type, entity_list in entities.items():
            if entity_type not in merged:
                merged[entity_type] = []

            if isinstance(entity_list, list):
                # Add unique entities only
                for entity in entity_list:
                    if entity not in merged[entity_type]:
                        merged[entity_type].append(entity)

    return merged


def main(
    input_file: str,
    output_file: str = None,
    model_path: str = "Universal-NER/UniNER-7B-type",
    entity_types: str = "person,location,organization,date",
    max_new_tokens: int = 256,
    max_pages: int = None,
    chunk_size: int = 2000,
):
    """
    Process OLMoCR JSON file with Universal-NER.

    Args:
        input_file: Path to OLMoCR JSON file
        output_file: Path to output JSON file (default: input_file + .ner.json)
        model_path: HuggingFace model path
        entity_types: Comma-separated list of entity types to extract
        max_new_tokens: Max tokens for generation
        max_pages: Maximum number of pages to process (default: all)
        chunk_size: Maximum characters per text chunk for NER
    """
    # Parse entity types
    if isinstance(entity_types, (list, tuple)):
        entity_types_list = list(entity_types)
    else:
        entity_types_list = [et.strip() for et in entity_types.split(',')]

    # Set output file
    if output_file is None:
        output_file = str(Path(input_file).with_suffix('')) + '.ner.json'

    print(f"[INFO] Loading OLMoCR JSON from: {input_file}")
    with open(input_file, 'r') as f:
        pages = json.load(f)

    print(f"[INFO] Found {len(pages)} pages in OLMoCR output")

    if max_pages:
        pages = pages[:max_pages]
        print(f"[INFO] Processing first {max_pages} pages only")

    print(f"[INFO] Loading Universal-NER model: {model_path}")
    generator = pipeline('text-generation', model=model_path, dtype=torch.float16, device=0)
    print(f"[INFO] Model loaded successfully")

    print(f"[INFO] Entity types: {entity_types_list}")
    print(f"[INFO] Processing pages...")
    print("=" * 80)

    results = []
    for i, page in enumerate(pages):
        page_id = page.get('id', f'page_{i}')
        text = page.get('text', '')

        print(f"\n[PAGE {i+1}/{len(pages)}] ID: {page_id}")
        print(f"Text length: {len(text)} characters")

        # Chunk text at paragraph boundaries
        chunks = chunk_text_by_paragraphs(text, target_size=chunk_size, overlap=200)
        print(f"Split into {len(chunks)} chunks (target: {chunk_size} chars/chunk)")

        # Process each chunk
        chunk_entities = []
        for chunk_idx, chunk in enumerate(chunks):
            print(f"  Processing chunk {chunk_idx+1}/{len(chunks)} ({len(chunk)} chars)...", end=' ')
            entities = extract_entities(generator, chunk, entity_types_list, max_new_tokens)
            chunk_entities.append(entities)

            # Print chunk results
            total = sum(len(v) if isinstance(v, list) else 0 for v in entities.values())
            print(f"{total} entities")

        # Merge entities from all chunks
        merged_entities = merge_entity_results(chunk_entities)

        # Print final results
        total_entities = sum(len(v) if isinstance(v, list) else 0 for v in merged_entities.values())
        print(f"Total unique entities: {total_entities}")
        for etype, ents in merged_entities.items():
            if isinstance(ents, list) and ents:
                print(f"  {etype}: {len(ents)} - {ents[:5]}{'...' if len(ents) > 5 else ''}")

        # Store result
        results.append({
            'id': page_id,
            'text': page.get('text', ''),
            'entities': merged_entities,
            'metadata': page.get('metadata', {}),
            'num_chunks': len(chunks)
        })

    # Save results
    print(f"\n{'=' * 80}")
    print(f"[INFO] Saving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[DONE] Processed {len(results)} pages")
    print(f"[DONE] Output saved to: {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
