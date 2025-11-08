# Universal-NER Optimization Roadmap
## For Historical Document Processing (Hudson's Bay Company Documents)

**Date:** November 7, 2025
**Current Performance:** F1 49.79% (vs Gemini baseline)

---

## Executive Summary

Based on extensive testing and research, Universal-NER 2K chunks significantly outperforms spaCy (F1: 49.79% vs 26.90%) for historical document NER. This roadmap outlines immediate optimizations and long-term fine-tuning strategies to achieve 70%+ F1 score.

---

## Current Performance Baseline

### Chunk Size Analysis (Completed)
| Chunk Size | Total Entities | Person | Location | Organization | Date |
|------------|---------------|--------|----------|--------------|------|
| **2K** ✓   | **642**       | 105    | 345      | 76           | 116  |
| 3K         | 546           | 88     | 302      | 63           | 93   |
| 4K         | 514           | 74     | 283      | 53           | 104  |

**Conclusion:** 2K character chunks (~450-500 tokens) are optimal.

### Entity Type Performance (vs Gemini Gold Standard)
| Entity Type  | Recall | Issues |
|--------------|--------|--------|
| Person       | 73.7%  | Missing complex titles, single surnames |
| Location     | 70.0%  | Missing compound locations |
| **Organization** | **45.2%** | **Major weakness** - Missing historical companies, ships, government bodies |
| Date         | 58.2%* | *Actual dates (excluding temporal phrases) |

**Overall F1:** 49.79% (Precision: 46.8%, Recall: 53.3%)

---

## Phase 1: Immediate Optimizations (No Fine-Tuning)

### 1.1 Enhanced Entity Type Descriptions

**Current:**
```python
entity_types = ["person", "location", "organization", "date"]
```

**Optimized (Implement This Week):**
```python
entity_descriptions = {
    "person": "individuals, historical figures, officials, traders, explorers, captains, or named persons including titles and honorifics",

    "location": "places, settlements, geographic locations, forts, trading posts, rivers, bays, straits, territories, or regions",

    "organization": "companies, trading corporations, governmental bodies, colonial organizations, administrative offices, institutional entities, ships, missionary stations, or chartered companies",

    "date": "specific dates, years, months, or time periods with numerical indicators"
}

# In query:
f"What describes {entity_descriptions['organization']} in the text?"
```

**Expected Impact:** +5-10% recall on organizations

---

### 1.2 Switch to UniNER-7B-all Model

**Current:** `Universal-NER/UniNER-7B-type`
**Recommended:** `Universal-NER/UniNER-7B-all`

**Rationale:**
- Trained on combination of type + definition + 40 supervised datasets
- More robust to entity type paraphrasing
- Better generalization for archaic/historical terminology

**Implementation:**
```bash
# In process_olmocr.sbatch
--model_path Universal-NER/UniNER-7B-all
```

**Expected Impact:** +3-5% F1 across all types

---

### 1.3 Improved Chunking Strategy

**Current:** Paragraph-based chunking with 200 char overlap

**Optimized:** Sentence-based chunking with token-aware overlap

```python
def chunk_document_optimized(text, max_chars=2000, overlap_tokens=100):
    """
    Sentence-based chunking with token-counted overlap
    """
    from transformers import LlamaTokenizer
    import re

    tokenizer = LlamaTokenizer.from_pretrained("Universal-NER/UniNER-7B-all")

    # Split into sentences (preserve abbreviations)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_length + sentence_len > max_chars and current_chunk:
            # Save chunk
            chunks.append(' '.join(current_chunk))

            # Calculate overlap in tokens
            overlap_sentences = []
            overlap_size = 0
            for s in reversed(current_chunk):
                s_tokens = len(tokenizer(s)['input_ids'])
                if overlap_size + s_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_size += s_tokens
                else:
                    break

            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_len + 1  # +1 for space

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
```

**Expected Impact:** +2-3% recall (fewer boundary-split entities)

---

### 1.4 Post-Processing Rules

**For Organization Entities:**

```python
def enhance_organizations(entities, text):
    """
    Post-process to capture common patterns missed by model
    """
    import re

    enhanced = set(entities)

    # Pattern 1: Ship names - H.M.S. "ShipName" or H.B. Co.'s ship "ShipName"
    ship_pattern = r'(?:H\.M\.S\.|H\.B\.\s+Co\.\'s\s+ship)\s+"([^"]+)"'
    for match in re.finditer(ship_pattern, text):
        enhanced.add(match.group(0))

    # Pattern 2: Possessive companies - "Company's" forms
    possessive_pattern = r"([A-Z][A-Za-z'\s]+(?:Company|Corporation|Society|Association))'s"
    for match in re.finditer(possessive_pattern, text):
        enhanced.add(match.group(1))

    # Pattern 3: Government bodies - "Committee of..."
    gov_pattern = r'(?:Committee|Department|Office|Bureau|Board)\s+of\s+(?:the\s+)?[A-Z][A-Za-z\s]+'
    for match in re.finditer(gov_pattern, text):
        if len(match.group(0)) < 80:  # Avoid descriptive phrases
            enhanced.add(match.group(0))

    return list(enhanced)
```

**Expected Impact:** +5-8% recall on organizations

---

## Phase 2: Fine-Tuning Preparation (Weeks 2-3)

### 2.1 Annotate Historical Training Data

**Requirement:** 500-1000 annotated historical document examples

**Data Sources:**
1. Hudson's Bay Company documents (already processing)
2. Colonial Office correspondence
3. Saskatchewan settlement records
4. Railway company documents

**Annotation Format:**
```json
[
  {
    "id": "hbc_001",
    "conversations": [
      {
        "from": "human",
        "value": "Text: The Hudson's Bay Company established Fort Garry on the Red River in 1822 under Governor George Simpson."
      },
      {
        "from": "gpt",
        "value": "I've read this text."
      },
      {
        "from": "human",
        "value": "What describes companies, trading corporations, governmental bodies, colonial organizations, administrative offices, or institutional entities in the text?"
      },
      {
        "from": "gpt",
        "value": "[\"Hudson's Bay Company\"]"
      },
      {
        "from": "human",
        "value": "What describes individuals, historical figures, officials, traders, or named persons in the text?"
      },
      {
        "from": "gpt",
        "value": "[\"George Simpson\"]"
      },
      {
        "from": "human",
        "value": "What describes places, settlements, geographic locations, forts, or territories in the text?"
      },
      {
        "from": "gpt",
        "value": "[\"Fort Garry\", \"Red River\"]"
      },
      {
        "from": "human",
        "value": "What describes specific dates, years, or time periods in the text?"
      },
      {
        "from": "gpt",
        "value": "[\"1822\"]"
      }
    ]
  }
]
```

**Tools:**
- Use current Universal-NER predictions as starting point
- Manual correction/augmentation
- Use Gemini for bootstrapping additional examples
- Store in: `data/historical_ner_train.json`

---

### 2.2 Fine-Tuning Strategy: LoRA (Recommended)

**Why LoRA:**
- Requires only 2x A100 40GB (vs 8x A100 for full fine-tuning)
- Faster training (5 epochs in ~6 hours)
- Smaller output model (~100MB adapter vs 13GB full model)
- Can be merged back to full model if needed

**Hardware Requirements:**
- **Minimum:** 2x A100 40GB GPUs
- **Optimal:** 1x A100 80GB GPU
- Available on Nibi cluster

**Training Script Modifications:**

```bash
# Create: cluster/nibi/slurm/finetune_lora.sbatch

#!/bin/bash
#SBATCH --job-name=uniner-lora
#SBATCH --account=def-jic823
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

module load python/3.12
module load cuda/12.2

source "$HOME/projects/def-jic823/universal-ner/.venv/bin/activate"

cd "$HOME/projects/def-jic823/universal_ner/src/train"

# Install PEFT for LoRA
pip install peft==0.7.1

# Modified training script
torchrun --nproc_per_node=2 --master_port=20001 \
    fastchat/train/train_mem.py \
    --model_name_or_path Universal-NER/UniNER-7B-all \
    --data_path ../../data/historical_ner_train.json \
    --bf16 True \
    --output_dir ../../models/uniner-historical-lora \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 3e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --tf32 True \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05
```

**Expected Training Time:** 6-8 hours (500 examples, 5 epochs)

---

### 2.3 Validation Strategy

**Test Sets:**
1. Hold out 20% of historical documents for testing
2. Use Gemini-annotated P000837.json as benchmark
3. Measure improvement on organization recall specifically

**Metrics to Track:**
- Overall F1 (target: 70%+)
- Organization recall (target: 70%+ from current 45%)
- Person/Location recall (maintain 70%+)
- Inference speed (should remain ~3 min per document)

---

## Phase 3: Production Deployment (Week 4)

### 3.1 Model Serving Setup

**Option A: vLLM Server (Recommended for Batch Processing)**

```bash
# Start vLLM server with fine-tuned model
python -m vllm.entrypoints.api_server \
    --model models/uniner-historical-lora-merged \
    --tensor-parallel-size 1 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.9
```

**Option B: Transformers Pipeline (Current Approach)**

```python
# Load fine-tuned model with LoRA adapter
from transformers import pipeline
from peft import PeftModel

base_model = pipeline('text-generation',
                      model='Universal-NER/UniNER-7B-all',
                      device=0)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model.model,
    'models/uniner-historical-lora'
)
```

---

### 3.2 Integration with OLMoCR Pipeline

**Updated Pipeline:**

```
1. OLMoCR (OCR) → JSON with text
2. Text Preprocessing → Sentence segmentation
3. Chunking (2K chars, 100 token overlap)
4. Universal-NER (fine-tuned) → Entities per chunk
5. Post-processing → Deduplicate + enhance organizations
6. Neo4j Integration → Knowledge graph
```

**Performance Targets:**
- Throughput: 10-15 pages/minute (single H100)
- F1 Score: 70%+ overall
- Organization recall: 70%+

---

## Expected Outcomes

### After Phase 1 (Immediate Optimizations)
- **Current:** F1 49.79%, Org recall 45.2%
- **Target:** F1 58-62%, Org recall 55-60%
- **Timeline:** 1 week

### After Phase 2 (Fine-Tuning)
- **Current:** F1 49.79%, Org recall 45.2%
- **Target:** F1 70-75%, Org recall 70-75%
- **Timeline:** 3-4 weeks

### After Phase 3 (Production)
- Scalable pipeline for 60M+ pages
- Integration with Neo4j knowledge graph
- Automated NER for incoming documents

---

## Resource Requirements

### Immediate (Phase 1)
- Developer time: 1 week
- No additional hardware

### Fine-Tuning (Phase 2)
- Annotation time: 2 weeks (500-1000 examples)
- Compute: 2x A100 40GB for 8 hours (~$50-100 on Nibi)
- Storage: 200MB for adapter weights

### Production (Phase 3)
- Infrastructure: 1x H100 GPU for serving
- Cost: ~$190/M pages (OLMoCR estimate)

---

## Next Steps (Priority Order)

1. **✓ COMPLETED:** Chunk size optimization → 2K chars optimal
2. **✓ COMPLETED:** Performance analysis vs spaCy and Gemini
3. **NEXT:** Implement enhanced entity type descriptions
4. **NEXT:** Test UniNER-7B-all model variant
5. **NEXT:** Add post-processing rules for organizations
6. **NEXT:** Begin annotation of historical training data
7. **LATER:** Fine-tune with LoRA on Nibi cluster
8. **LATER:** Deploy fine-tuned model to production pipeline

---

## Key Research Findings

### Model Architecture
- **Base:** LLaMA 2 7B
- **Variants:** UniNER-7B-type (current), UniNER-7B-all (recommended), UniNER-7B-definition
- **Input Limit:** 512 tokens (includes text + entity type description)
- **Output:** JSON array format, temperature=0 for deterministic results

### Training Details
- **Official Setup:** 8x A100 40GB, 3 epochs, batch size 256
- **LoRA Alternative:** 2x A100 40GB, 5 epochs, LoRA rank 16
- **Data Format:** Multi-turn conversations with entity type queries
- **Training Data:** 48.7MB (ChatGPT Pile-NER + 40 supervised datasets)

### Performance Benchmarks
- **Zero-shot:** 41.7% average F1 on 43 datasets
- **vs ChatGPT:** 7-9 F1 points higher
- **vs spaCy (our test):** 85% better F1 (49.79% vs 26.90%)

---

## References

- **Paper:** UniversalNER (arXiv:2308.03279)
- **Code:** https://github.com/universal-ner/universal-ner
- **Models:** https://huggingface.co/Universal-NER
- **Local Repo:** `/home/jic823/universalNER`
- **Research:** Task agent analysis (November 7, 2025)
