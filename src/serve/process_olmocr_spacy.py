#!/usr/bin/env python3
"""
Process OLMoCR JSON output with spaCy NER to extract entities.

Input: OLMoCR JSON file with structure: [{"id": "...", "text": "..."}]
Output: Enriched JSON with entity annotations
"""
import fire
import spacy
import json
from pathlib import Path
from collections import defaultdict


def process_with_spacy(text, nlp):
    """Extract entities using spaCy."""
    doc = nlp(text)

    entities = defaultdict(list)

    # Map spaCy labels to our schema
    label_map = {
        'PERSON': 'person',
        'GPE': 'location',      # Geopolitical entity
        'LOC': 'location',      # Non-GPE locations
        'ORG': 'organization',
        'DATE': 'date',
    }

    for ent in doc.ents:
        entity_type = label_map.get(ent.label_)
        if entity_type:
            entity_text = ent.text.strip()
            if entity_text and entity_text not in entities[entity_type]:
                entities[entity_type].append(entity_text)

    return dict(entities)


def main(
    input_file: str,
    output_file: str = None,
    model: str = "en_core_web_lg",
    max_pages: int = None,
):
    """
    Process OLMoCR JSON file with spaCy NER.

    Args:
        input_file: Path to OLMoCR JSON file
        output_file: Path to output JSON file (default: input_file + .spacy.json)
        model: spaCy model to use (en_core_web_lg or en_core_web_trf)
        max_pages: Maximum number of pages to process (default: all)
    """
    # Set output file
    if output_file is None:
        output_file = str(Path(input_file).with_suffix('')) + '.spacy.json'

    print(f"[INFO] Loading OLMoCR JSON from: {input_file}")
    with open(input_file, 'r') as f:
        pages = json.load(f)

    print(f"[INFO] Found {len(pages)} pages in OLMoCR output")

    if max_pages:
        pages = pages[:max_pages]
        print(f"[INFO] Processing first {max_pages} pages only")

    print(f"[INFO] Loading spaCy model: {model}")
    nlp = spacy.load(model)
    print(f"[INFO] Model loaded successfully")

    print(f"[INFO] Processing pages...")
    print("=" * 80)

    results = []
    for i, page in enumerate(pages):
        page_id = page.get('id', f'page_{i}')
        text = page.get('text', '')

        print(f"\n[PAGE {i+1}/{len(pages)}] ID: {page_id}")
        print(f"Text length: {len(text)} characters")

        # Process entire page at once (spaCy handles long texts well)
        entities = process_with_spacy(text, nlp)

        # Print results
        total_entities = sum(len(v) for v in entities.values())
        print(f"Total entities: {total_entities}")
        for etype, ents in entities.items():
            if ents:
                print(f"  {etype}: {len(ents)} - {ents[:5]}{'...' if len(ents) > 5 else ''}")

        # Store result
        results.append({
            'id': page_id,
            'text': page.get('text', ''),
            'entities': entities,
            'metadata': page.get('metadata', {}),
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
