# 🚀 Project Portfolio

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS%2FCUDA-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1-000000?style=flat-square&logo=apple&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20UI-000000?style=flat-square&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📍 Table of Contents

- [🤖 AI & LLM Orchestration — GPT from Scratch](#-ai--llm-orchestration)
- [📱 iPad LiDAR: Proof of Concept](#-ipad-lidar-proof-of-concept)
- [🧪 Learning & Random Projects](#-learning--random-projects)

---

## 🤖 AI & LLM Orchestration

> **Building a GPT-style LLM from scratch on Apple Silicon M1**

A complete, hands-on implementation of a GPT-style transformer LLM — including environment setup, bug fixes, training, and a web-based chat interface. Built as a proof of concept to understand how large language models work under the hood.

### What Was Built

- ✅ GPT-style transformer trained from scratch
- ✅ Custom BPE tokenizer trained on the TinyStories dataset
- ✅ All bugs in the `BuildYourLLM` pipeline identified and fixed
- ✅ Web-based chat interface with conversation logging
- ✅ Fine-tuning script for continuous learning from chat history

### Hardware & Environment

| Component | Details |
|-----------|---------|
| Machine | MacBook Pro M1 (Apple Silicon) |
| GPU | Apple Silicon MPS (Metal Performance Shaders) |
| Python | 3.12 (system framework) |
| Key Libraries | PyTorch, Transformers, Tokenizers, Datasets, Flask |
| Dataset | TinyStories (~2M children's stories via HuggingFace) |
| Model Size | 30,142,848 parameters (small/tiny config) |

---

### 1. Environment Setup

Installing dependencies failed immediately because the `yaml` module was missing. The PyPI package is `pyyaml` but imports as `yaml` — a common gotcha.

```
ModuleNotFoundError: No module named 'yaml'
```

**Fix:**
```bash
pip3 install pyyaml
# Note: the package is 'pyyaml' but you import it as 'import yaml'
```

After installing `pyyaml`, the pipeline launched and automatically installed all remaining dependencies:

```bash
python3 -m pip install -q --break-system-packages \
  torch transformers datasets tokenizers sentencepiece tqdm pyyaml numpy safetensors
```

> 💡 **Tip:** Your terminal's working directory points to an inode. Moving or trashing a folder in Finder does **not** change where your terminal thinks it is — it stays in the old directory until you `cd` elsewhere.

---

### 2. Bugs Found & Fixed

A full audit of all pipeline scripts revealed **4 bugs**. All were fixed before training began.

---

#### Bug 1 — `actual_vocab_size` Undefined (`train.py`, line 392)

**Error:**
```
NameError: name 'actual_vocab_size' is not defined
```

**Root cause:** The variable `actual_vocab_size` was referenced in the `GPTConfig` constructor before it was assigned. The tokenizer needed to be loaded first.

**Fix — added to `train.py` before `GPTConfig` creation:**
```python
# Load tokenizer to get the real vocab size
from tokenizers import Tokenizer as _Tokenizer
_tok_dir = Path(args.data_dir) / 'tokenizer'
_tok = _Tokenizer.from_file(str(_tok_dir / 'tokenizer.json'))
actual_vocab_size = _tok.get_vocab_size()
print(f'Tokenizer vocab size: {actual_vocab_size}')

config = GPTConfig(
    vocab_size=actual_vocab_size,  # ✅ now defined
    ...
)
```

---

#### Bug 2 — OpenWebText Dataset Not Implemented (`prepare_data.py`)

**Root cause:** `openwebtext` was listed as a valid CLI option but had no implementation — it would raise `NotImplementedError` if selected.

**Fix — added `download_openwebtext()` function:**
```python
def download_openwebtext(output_dir: Path, max_chars: int = 50_000_000):
    from datasets import load_dataset

    print('Downloading OpenWebText (streaming)...')
    ds = load_dataset('Skylion007/openwebtext', split='train',
                      streaming=True, trust_remote_code=True)
    chunks, total = [], 0
    for row in ds:
        t = row['text'].strip()
        if not t:
            continue
        chunks.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return '\n'.join(chunks)
```

---

#### Bug 3 — Hardcoded Token IDs (`export_hf.py`)

**Root cause:** `bos_token_id` and `eos_token_id` were hardcoded to `50256` (GPT-2's value). Our custom tokenizer has a completely different vocab size, producing a broken HuggingFace config.

**Fix:**
```python
actual_vocab_size = cfg.get('vocab_size', 50257)
eos_token_id = actual_vocab_size - 1  # last token = <endoftext>

hf_config = {
    'vocab_size':   actual_vocab_size,  # was hardcoded 50257 ❌
    'bos_token_id': eos_token_id,       # was hardcoded 50256 ❌
    'eos_token_id': eos_token_id,       # was hardcoded 50256 ❌
    ...
}
```

---

#### Bug 4 — Broken Fallback Imports (`train_tokenizer.py`)

**Root cause:** The `try` block imported `processors` and `RobertaProcessing`, but these were never used. On the fallback path, these imports were missing, causing an `ImportError` on first install.

**Fix — removed unused imports:**
```python
# BEFORE ❌
from tokenizers import (Tokenizer, models, trainers,
                        pre_tokenizers, decoders, processors)
from tokenizers.processors import RobertaProcessing  # never used

# AFTER ✅
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
# RobertaProcessing removed entirely
```

---

### 3. Training

#### Configuration Used

```bash
# tiny.yaml — much faster than the default small config
python3 run_pipeline.py --config configs/tiny.yaml
```

```yaml
# Model architecture (tiny config)
n_layer: 4       # transformer blocks
n_head: 4        # attention heads
n_embd: 256      # embedding dimension
epochs: 2
batch_size: 8
context_len: 512
```

#### Dataset — TinyStories

| Split | Size |
|-------|------|
| Training set | 47,500,000 characters |
| Validation set | 2,500,000 characters |
| Tokenizer vocab | 20,712 tokens (custom BPE) |
| Total parameters | 30,142,848 |

The `tiny.yaml` config streams TinyStories directly from HuggingFace to keep RAM usage low.

#### Training Metrics

| Metric | Meaning |
|--------|---------|
| `loss` | Cross-entropy loss — lower is better; < 2.0 means coherent text |
| `ppl` | Perplexity — how "surprised" the model is by the next token |
| `lr` | Current learning rate (decays over time via scheduler) |
| `step/s` | Training throughput in steps per second |
| `eta` | Estimated time remaining |

> 💡 A loss of **2.45** after only 10% of training is healthy — the model is learning fast. Loss below 2.0 generally means the model can generate coherent text.

#### Apple Silicon MPS

The M1's GPU (Metal Performance Shaders) is used automatically — confirmed by `Using Apple Silicon MPS` in the output. No extra configuration needed.

#### Running Individual Steps

```bash
# Run only step 3 (tokenizer training)
python3 run_pipeline.py --config configs/tiny.yaml --only-step 3

# Run steps 4–11 sequentially
for step in 4 5 6 7 8 9 10 11; do
  python3 run_pipeline.py --config configs/tiny.yaml --only-step $step
done
```

---

### 4. Chat Interface & Conversation Logging

A complete web-based chat interface was built with three components:

| File | Purpose |
|------|---------|
| `server.py` | Flask server — loads the model and serves the UI |
| `templates/index.html` | Dark terminal-styled chat interface |
| `finetune_from_logs.py` | Retrains the model on logged conversations |

#### Quick Start

```bash
# 1. Install Flask
pip3 install flask --break-system-packages

# 2. Start the server
cd llm_chat
python3 server.py \
    --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/best.pt

# 3. Open in browser → http://localhost:5000
```

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the chat UI |
| `POST` | `/generate` | Generates a response for a given prompt |
| `GET` | `/logs` | Returns all logged conversations as JSON |

#### Conversation Log Format

Every message is automatically saved to `logs/conversations.jsonl`:

```json
{
  "timestamp":  "2026-03-06T14:30:00",
  "session_id": "sess_abc123",
  "prompt":     "Once upon a time there was",
  "response":   "a little fox who lived in the forest..."
}
```

---

### 5. Continuous Learning from Conversations

> **Can the model learn while you talk to it?**
> No — once training is complete, the weights are frozen. Inference does not update the model. This is true of all LLMs, including ChatGPT.

> 💡 Production systems like ChatGPT periodically fine-tune on collected conversation data — they do **not** update in real-time from individual chats.

#### The Practical Workflow

```
1. Chat with the model → conversations auto-save to logs/conversations.jsonl
2. Review and curate the log (delete bad/repetitive responses)
3. Run finetune_from_logs.py periodically (daily or weekly)
4. Restart the server with the updated checkpoint
5. Repeat — the model gradually improves on your usage patterns
```

#### Running Fine-tuning on Logs

```bash
cd llm_chat

python3 finetune_from_logs.py \
    --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/best.pt \
    --logs logs/conversations.jsonl \
    --epochs 1

# Restart the server with the fine-tuned checkpoint
python3 server.py \
    --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/finetuned_chat/finetuned_best.pt
```

#### LoRA Fine-tuning

Fine-tuning uses **LoRA (Low-Rank Adaptation)** rather than updating all weights:

| Property | Detail |
|----------|--------|
| Weight updates | ~1% of total weights |
| Base weights | Frozen — no catastrophic forgetting |
| Speed | Much faster than full fine-tuning |
| Quality | Close to full fine-tuning for most tasks |

#### Three Approaches to Continuous Learning

| Approach | How It Works | Best For |
|----------|-------------|----------|
| **Periodic LoRA Fine-tuning** | Fine-tune on conversation logs every N days | Most practical; low cost |
| **Instruction Fine-tuning** | Train on curated prompt/response JSONL pairs | Teaching Q&A behaviour |
| **Full Fine-tuning** | Update all weights on new data | Maximum quality; high cost |

---

### 6. Capabilities & Limitations

> This is a **base language model** trained on children's stories. It learned to predict the next word given previous context. It generates story-style continuations — not answers to questions.

| ✅ Can Do | ❌ Cannot Do |
|-----------|-------------|
| Continue a story from a prompt | Answer factual questions accurately |
| Generate fluent, grammatical text | Follow complex instructions |
| Produce children's story-style prose | Hold a coherent multi-turn conversation |
| Run entirely on-device (M1 Mac) | Reason or plan like a large model |

#### To Get Proper Q&A Behaviour

Use instruction fine-tuning with a JSONL file of prompt/response pairs:

```bash
python3 scripts/finetune.py \
    --checkpoint output/tiny/checkpoints/best.pt \
    --data data/samples/instructions.jsonl \
    --method instruct \
    --epochs 2
```

```jsonl
{"instruction": "What is 2+2?", "input": "", "output": "4"}
{"instruction": "Write a haiku about autumn", "input": "", "output": "Leaves fall silently..."}
```

---

### 7. Quick Reference

#### Full Pipeline

```bash
# Run the complete pipeline
python3 run_pipeline.py --config configs/tiny.yaml

# Skip training, use an existing checkpoint
python3 run_pipeline.py --config configs/tiny.yaml --skip-train

# Run a single step only
python3 run_pipeline.py --config configs/tiny.yaml --only-step 4
```

#### Chat Server

```bash
cd llm_chat
pip3 install flask --break-system-packages

# Start with base checkpoint
python3 server.py \
    --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/best.pt

# Start with fine-tuned checkpoint
python3 server.py \
    --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/finetuned_chat/finetuned_best.pt
```

#### Fine-tuning

```bash
cd llm_chat

python3 finetune_from_logs.py \
    --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/best.pt \
    --logs logs/conversations.jsonl \
    --epochs 1
```

---

[↑ Back to top](#-project-portfolio)

---

<div align="center">
<sub>Built with 🧠 curiosity and ☕ coffee on an M1 MacBook Pro</sub>
</div>
