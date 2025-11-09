#!/usr/bin/env python3
"""
Convert NER JSON files to structured XML with full text preservation and character offsets.

This module converts Universal-NER toponym extraction results to a structured XML format
that enables:
- Full text preservation with paragraph structure
- Precise character offsets for each entity mention
- Co-occurrence analysis via nearby entity tracking
- Multi-referent detection support

Usage:
    python -m src.utils.convert_ner_to_xml --input_dir <path> --output_dir <path>

Or import as a module:
    from src.utils.convert_ner_to_xml import process_toponym_file, batch_convert
"""

import fire
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import defaultdict


def find_entity_mentions(text, entity):
    """
    Find all mentions of an entity with character offsets.

    Args:
        text: Full document text
        entity: Entity name to search for

    Returns:
        List of dicts with 'start', 'end', 'text' keys
    """
    entity_escaped = re.escape(entity)
    mentions = []

    for match in re.finditer(entity_escaped, text, re.IGNORECASE):
        mentions.append({
            'start': match.start(),
            'end': match.end(),
            'text': match.group()
        })

    return mentions


def get_nearby_entities(all_entities, char_offset, text, window=500):
    """
    Find other entities mentioned within window characters of this mention.

    Args:
        all_entities: Dict of entity_type -> entity_name -> mentions
        char_offset: Character position of current mention
        text: Full document text
        window: Window size in characters (default 500)

    Returns:
        Dict of entity_type -> list of entity names
    """
    nearby = defaultdict(list)
    window_start = max(0, char_offset - window)
    window_end = min(len(text), char_offset + window)

    for entity_type, entities_dict in all_entities.items():
        for entity_name, mentions in entities_dict.items():
            for mention in mentions:
                if window_start <= mention['start'] <= window_end:
                    if mention['start'] != char_offset:
                        nearby[entity_type].append(entity_name)
                        break

    return nearby


def split_into_paragraphs(text):
    """
    Split text into paragraphs and track character offsets.

    Args:
        text: Full document text

    Returns:
        List of dicts with 'text', 'start', 'end' keys
    """
    paragraphs = []
    current_offset = 0

    for para_text in text.split('\n\n'):
        if para_text.strip():
            paragraphs.append({
                'text': para_text.strip(),
                'start': current_offset,
                'end': current_offset + len(para_text)
            })
        current_offset += len(para_text) + 2

    return paragraphs


def get_paragraph_id(char_offset, paragraphs):
    """Find which paragraph contains this character offset."""
    for i, para in enumerate(paragraphs):
        if para['start'] <= char_offset <= para['end']:
            return i
    return -1


def process_toponym_file(ner_file, output_dir, entity_types=None):
    """
    Convert a single toponym NER file to structured XML.

    Args:
        ner_file: Path to .toponym.ner.json file
        output_dir: Directory for output XML files
        entity_types: List of entity types to process (default: all toponym types)

    Returns:
        Tuple of (output_file_path, total_entities, total_mentions)
    """
    if entity_types is None:
        entity_types = ["toponym", "water_body", "landform", "administrative_region",
                       "route", "person", "organization", "date"]

    with open(ner_file) as f:
        data = json.load(f)

    doc_id = ner_file.stem.replace('.toponym.ner', '').replace('.ner', '')

    # Build full text and collect all entity mentions by type
    full_text = ""
    all_entities = defaultdict(lambda: defaultdict(list))

    for page in data:
        page_text = page.get("text", "")
        page_start_offset = len(full_text)

        entities = page.get("entities", {})

        for entity_type in entity_types:
            entity_list = entities.get(entity_type, [])
            for entity_name in entity_list:
                if isinstance(entity_name, str):
                    mentions = find_entity_mentions(page_text, entity_name)
                    for mention in mentions:
                        all_entities[entity_type][entity_name].append({
                            'start': page_start_offset + mention['start'],
                            'end': page_start_offset + mention['end'],
                            'text': mention['text']
                        })

        full_text += page_text + "\n\n"

    paragraphs = split_into_paragraphs(full_text)

    # Build XML
    root = ET.Element("document")
    root.set("id", doc_id)
    root.set("source_file", ner_file.name)

    total_entities = sum(len(entities_dict) for entities_dict in all_entities.values())
    total_mentions = sum(len(mentions) for entities_dict in all_entities.values()
                         for mentions in entities_dict.values())

    root.set("total_entity_count", str(total_entities))
    root.set("total_mention_count", str(total_mentions))

    # Add text element with paragraph structure
    text_elem = ET.SubElement(root, "text")
    text_elem.set("paragraph_count", str(len(paragraphs)))

    for para_id, para in enumerate(paragraphs):
        para_elem = ET.SubElement(text_elem, "paragraph")
        para_elem.set("id", f"p{para_id}")
        para_elem.set("char_start", str(para['start']))
        para_elem.set("char_end", str(para['end']))
        para_elem.text = para['text']

    # Add entities by type
    entities_elem = ET.SubElement(root, "entities")

    for entity_type in entity_types:
        if entity_type not in all_entities or not all_entities[entity_type]:
            continue

        type_elem = ET.SubElement(entities_elem, entity_type + "s")
        type_elem.set("unique_count", str(len(all_entities[entity_type])))
        type_elem.set("mention_count", str(sum(len(mentions) for mentions in all_entities[entity_type].values())))

        for entity_name in sorted(all_entities[entity_type].keys()):
            mentions = all_entities[entity_type][entity_name]

            entity_elem = ET.SubElement(type_elem, entity_type)
            entity_elem.set("name", entity_name)
            entity_elem.set("mention_count", str(len(mentions)))

            for mention in mentions:
                mention_elem = ET.SubElement(entity_elem, "mention")

                para_id = get_paragraph_id(mention['start'], paragraphs)
                mention_elem.set("paragraph_id", f"p{para_id}" if para_id >= 0 else "unknown")
                mention_elem.set("char_start", str(mention['start']))
                mention_elem.set("char_end", str(mention['end']))

                nearby = get_nearby_entities(all_entities, mention['start'], full_text, window=500)
                if nearby:
                    nearby_elem = ET.SubElement(mention_elem, "nearby_entities")
                    nearby_elem.set("window_chars", "500")

                    for nearby_type, nearby_names in sorted(nearby.items()):
                        if nearby_names:
                            type_group = ET.SubElement(nearby_elem, nearby_type + "s")
                            for nearby_name in sorted(set(nearby_names)):
                                entity_ref = ET.SubElement(type_group, nearby_type)
                                entity_ref.text = nearby_name

    # Pretty print XML
    xml_str = minidom.parseString(ET.tostring(root, encoding='utf-8')).toprettyxml(indent="  ")
    lines = [line for line in xml_str.split('\n') if line.strip()]
    xml_str = '\n'.join(lines)

    # Save XML file
    output_file = Path(output_dir) / f"{doc_id}.toponym.xml"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(xml_str)

    return output_file, total_entities, total_mentions


def batch_convert(input_dir, output_dir, entity_types=None):
    """
    Convert all NER JSON files in a directory to structured XML.

    Args:
        input_dir: Directory containing .toponym.ner.json files
        output_dir: Directory for output XML files
        entity_types: List of entity types to process (default: all toponym types)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    ner_files = sorted(input_path.glob("*.toponym.ner.json"))
    if not ner_files:
        # Try regular .ner.json files
        ner_files = sorted(input_path.glob("*.ner.json"))

    total_files = len(ner_files)

    print(f"Converting {total_files} NER files to structured XML...")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    total_entities = 0
    total_mentions = 0

    for i, ner_file in enumerate(ner_files):
        if i % 50 == 0:
            print(f"  Processing {i}/{total_files}...")

        try:
            output_file, entities, mentions = process_toponym_file(ner_file, output_path, entity_types)
            total_entities += entities
            total_mentions += mentions
        except Exception as e:
            print(f"  ERROR processing {ner_file.name}: {e}")
            continue

    print(f"\nCompleted: {total_files} files")
    print(f"Total unique entities: {total_entities}")
    print(f"Total mentions: {total_mentions}")
    print(f"Output directory: {output_path}")


if __name__ == "__main__":
    fire.Fire(batch_convert)
