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

        # If text is too long, chunk it
        if len(text) > chunk_size:
            print(f"[WARN] Text too long ({len(text)} chars), processing first {chunk_size} chars only")
            text = text[:chunk_size]

        # Extract entities
        entities = extract_entities(generator, text, entity_types_list, max_new_tokens)

        # Print results
        total_entities = sum(len(v) if isinstance(v, list) else 0 for v in entities.values())
        print(f"Extracted {total_entities} entities:")
        for etype, ents in entities.items():
            if isinstance(ents, list) and ents:
                print(f"  {etype}: {len(ents)} - {ents[:3]}{'...' if len(ents) > 3 else ''}")

        # Store result
        results.append({
            'id': page_id,
            'text': page.get('text', ''),
            'entities': entities,
            'metadata': page.get('metadata', {})
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
