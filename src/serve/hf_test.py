import fire
import torch
from transformers import pipeline

import sys
sys.path.insert(0, '/project/6080182/universal_ner')
from src.utils import preprocess_instance

def main(
    model_path: str = "Universal-NER/UniNER-7B-type",
    max_new_tokens: int = 256,
):
    print("[TEST] Loading model...")
    generator = pipeline('text-generation', model=model_path, dtype=torch.float16, device=0)
    print("[TEST] Model loaded successfully!")

    # Test cases
    test_cases = [
        {
            "text": "Apple Inc. is planning to open a new store in San Francisco next month.",
            "entity_type": "organization"
        },
        {
            "text": "Apple Inc. is planning to open a new store in San Francisco next month.",
            "entity_type": "location"
        },
        {
            "text": "Dr. Sarah Johnson published her research at Stanford University in 2023.",
            "entity_type": "person"
        }
    ]

    print("\n" + "="*80)
    print("Running Universal-NER Test Cases")
    print("="*80 + "\n")

    for i, test in enumerate(test_cases, 1):
        text = test["text"]
        entity_type = test["entity_type"]

        print(f"\n[TEST {i}]")
        print(f"Text: {text}")
        print(f"Entity type: {entity_type}")

        example = {
            "conversations": [
                {"from": "human", "value": f"Text: {text}"},
                {"from": "gpt", "value": "I've read this text."},
                {"from": "human", "value": f"What describes {entity_type} in the text?"},
                {"from": "gpt", "value": "[]"}
            ]
        }

        prompt = preprocess_instance(example['conversations'])
        print(f"\nPrompt length: {len(prompt)} chars")

        outputs = generator(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
        result = outputs[0]['generated_text']

        print(f"Result: {result}")
        print("-" * 80)

    print("\n[TEST] All test cases completed successfully!")
    print("="*80)

if __name__ == "__main__":
    fire.Fire(main)
