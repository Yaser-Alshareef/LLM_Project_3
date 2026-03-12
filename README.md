# Project 3: 

## Overview
This project evaluates three different language model approaches for natural language generation (NLG) on the Databricks Dolly dataset. The goal is to compare cloud-based API models, instruction-tuned fine-tuned models, and base model fine-tuning to understand tradeoffs in cost, quality, and compute.

| Model | Type | Parameters | Environment |
|-------|------|-----------|-------------|
| DeepSeek-R1 | API (cloud) | ~67B | Cloud API |
| TinyLlama + LoRA | Local fine-tuned (instruction-tuned base) | 1.1B | Local |
| OPT-1.3B + LoRA | Local fine-tuned (base model) | 1.3B | Local  |

---

## Project Structure
```
Project_3/
├── data/
│   ├── clean_dolly_train.json          # Training split (JSONL)
│   └── clean_dolly_test.json           # Test split (JSONL)
├── model/
│   ├── opt_model/                      # OPT-1.3B LoRA adapter
│   │   ├── adapter_model.safetensors
│   │   └── adapter_config.json
│   └── opt-1.3b-base/                 # OPT-1.3B base model (see setup)
├── notebooks/
│   ├── data_cleaning.ipynb             # Data preprocessing & cleaning
│   ├── deepseek_eval.ipynb             # DeepSeek zero-shot & few-shot evaluation
│   ├── tinyllama_train.ipynb           # TinyLlama LoRA fine-tuning
│   ├── tinyllama_eval.ipynb            # TinyLlama evaluation
│   ├── opt_train.ipynb                 # OPT-1.3B LoRA fine-tuning
│   ├── opt_eval.ipynb                  # OPT-1.3B evaluation
│   └── zero_vs_few_comparison.csv      # DeepSeek results for cross-model comparison
└── README.md
```

---

## Data Pipeline

### Dataset
- **Source**: Databricks Dolly 15K — a human-generated instruction-following dataset
- **Format**: JSONL with fields: `instruction`, `context`, `response`, `category`
- **Categories**: 8 task types — brainstorming, classification, closed_qa, creative_writing, general_qa, information_extraction, open_qa, summarization
- **Splits**: Train / Test

### Preprocessing (`data_cleaning.ipynb`)
- Removed duplicates and empty responses
- Standardized field names and formatting
- Split into train/val/test sets
- Saved as clean JSONL files

---

## Models

### 1. DeepSeek-R1 (API-based, ~67B)
- **Approach**: Zero-shot and few-shot prompting via cloud API
- **No fine-tuning** — relies entirely on pre-trained knowledge
- **Evaluation**: Compared zero-shot vs few-shot to measure in-context learning

### 2. TinyLlama-1.1B + LoRA
- **Base model**: TinyLlama-1.1B-Chat (already instruction-tuned)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) on Dolly training set
- **Key insight**: Starts from an instruction-tuned checkpoint — LoRA refines an already capable model

### 3. OPT-1.3B + LoRA
- **Base model**: facebook/opt-1.3b (base language model, NOT instruction-tuned)
- **Fine-tuning**: LoRA on Dolly training set
- **Key insight**: Starts from a raw pre-trained model — tests whether LoRA alone can teach instruction-following from scratch

---

## Evaluation Strategy

### Balanced Sampling
- Stratified sampling: equal examples per category
- Up to 5 examples per category for fair comparison
- Ensures no single category dominates the aggregate metrics

### Metrics
| Metric | What it measures |
|--------|-----------------|
| **BLEU** | N-gram precision — exact word overlap with reference |
| **ROUGE-1** | Unigram recall — word-level coverage |
| **ROUGE-2** | Bigram recall — phrase-level coverage |
| **ROUGE-L** | Longest common subsequence — structural similarity |
| **BERTScore F1** | Semantic similarity using contextual embeddings (roberta-large) |

### Per-Category Breakdown
Each model is evaluated both overall and per-category to identify strengths and weaknesses across different task types.

---

## Results & Findings

### Key Findings

1. **DeepSeek dominates all metrics** — As expected, a ~67B parameter cloud model significantly outperforms 1B-class local models.

2. **TinyLlama + LoRA** achieves strong results considering its small size.

3. **BERTScore is more forgiving than BLEU** — Semantic similarity (BERTScore) scores are consistently higher than lexical overlap (BLEU), suggesting models capture meaning even when exact wording differs.

4. **Few-shot** prompting provides no benefit for already instruction-tuned models on diverse tasks.


### Cross-Model Comparison Summary

| Metric | DeepSeek Zero-Shot | DeepSeek Few-Shot | TinyLlama + LoRA | OPT-1.3B + LoRA |
|--------|-------------------|-------------------|------------------|------------------|
| BLEU | Medium | Higher | Low–Medium | Low |
| ROUGE-L | Medium | Higher | Low–Medium | Low |
| BERTScore | High | Higher | Medium | Medium–Low |

### OPT-1.3B + LoRA Metrics (20-sample test set)

| Metric    | Value   |
|-----------|---------|
| BLEU      | 0.0174  |
| ROUGE-1   | 0.2597  |
| ROUGE-2   | 0.0877  |
| ROUGE-L   | 0.1873  |

---

## Issues Encountered

### 1. OPT-1.3B Base Model Download Failure (Local)
**Problem**: HuggingFace Hub download failed repeatedly:

---

## Future Work

### 1. Scale Up with QLoRA on 8B+ Models
Fine-tune larger models (e.g., LLaMA-3-8B, Mistral-7B) using **QLoRA** (4-bit quantization + LoRA). This enables training 7–8B parameter models on consumer GPUs (16GB VRAM) while approaching cloud-model quality.

### 2. Instruction-Tuned vs Base Model Study
Systematically compare LoRA on instruction-tuned bases (e.g., LLaMA-3-8B-Instruct) vs raw bases (e.g., LLaMA-3-8B) across multiple model families to quantify the instruction-tuning advantage.

### 3. DPO / RLHF Alignment
Apply Direct Preference Optimization (DPO) or RLHF after LoRA fine-tuning to improve response quality beyond supervised fine-tuning — especially for creative and open-ended categories where current models underperform.

### 4. Human Evaluation
Add human evaluation scores alongside automated metrics. BLEU/ROUGE often undervalue correct but differently-worded responses. Human ratings on fluency, relevance, and completeness would provide a more reliable comparison.

### 5. Multi-Turn & RAG Integration
Extend evaluation to multi-turn conversations and Retrieval-Augmented Generation (RAG) pipelines, where smaller fine-tuned models combined with retrieval may close the gap with large cloud models.

### 6. Efficient Deployment
Explore model compression (GPTQ, AWQ quantization) and inference optimization (vLLM, GGUF) for deploying fine-tuned models in production with minimal latency and hardware requirements.

---

## References

- https://www.promptingguide.ai/
- https://huggingface.co/learn/llm-course/en/
- https://www.youtube.com/watch?v=uikZs6y0qgI
- https://arxiv.org/abs/2106.09685
- https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
- https://medium.com/@pur4v/understanding-llm-evaluation-metrics-bleu-rouge-exact-match-and-bertscore-716487e40bdd
- https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- https://www.deepseek.com/en/




