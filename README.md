# 🚀 My Project Portfolio

### 📍 Quick Links
* [🤖 AI & LLM Orchestration](#-ai--llm-orchestration)
* [📱 iPad LiDAR: Proof of Concept](#-ipad-lidar-proof-of-concept)
* [🧪 Learning & Random Projects](#-learning--random-projects)

---

### 🤖 AI & LLM Orchestration Creation a small model llm for Proof of Concept 

1. Project Overview
This document captures a complete hands-on session building a GPT-style Large Language Model from scratch on an Apple Silicon M1 MacBook Pro. It covers environment setup, bug fixes, training, and building a chat interface — including all code, errors encountered, and how they were resolved.

What Was Built
A GPT-style transformer LLM trained from scratch
Custom BPE tokenizer on TinyStories dataset
All bugs in the BuildYourLLM pipeline fixed
A web-based chat interface with conversation logging
A fine-tuning script for continuous learning from conversations

Hardware & Environment

Component	Details
Machine	MacBook Pro M1 (Apple Silicon)
GPU	: Apple Silicon MPS (Metal Performance Shaders)
Python	: Python 3.12 (system framework)
Key Libraries	: PyTorch, transformers, tokenizers, datasets, Flask
Dataset	: TinyStories (~2M children's stories from HuggingFace)
Model Size	: 30,142,848 parameters (small config)

2. Environment Setup & First Errors

Installing Dependencies
The pipeline failed immediately on launch because the yaml module was missing. The package name on PyPI is pyyaml but it imports as yaml in Python — a common gotcha.
Error Encountered
$ python3 run_pipeline.py
ModuleNotFoundError: No module named 'yaml'
Fix
pip3 install pyyaml
 
# Note: the package is called 'pyyaml' but imports as 'import yaml'
💡 The terminal's working directory is a file descriptor pointing to an inode. Moving or trashing a folder in Finder does NOT change where your terminal thinks it is — it stays in the old (now trashed) directory until you cd elsewhere.

Pipeline Successfully Started
After installing pyyaml, the pipeline launched and began Step 1 — installing all Python dependencies automatically:
STEP 1 — Installing Python dependencies
▶ python3 -m pip install -q --break-system-packages torch transformers
  datasets tokenizers sentencepiece tqdm pyyaml numpy safetensors

3. Bugs Found & Fixed
A thorough audit of all Python scripts in the project revealed 4 bugs. All were fixed and a corrected.

Bug 1 — actual_vocab_size Undefined (train.py line 392)
Error
NameError: name 'actual_vocab_size' is not defined
File scripts/train.py, line 392, in train
    vocab_size = actual_vocab_size,
Root Cause
The variable actual_vocab_size was referenced in the GPTConfig constructor before it was ever assigned. The tokenizer needed to be loaded first to get the real vocab size.
Fix — Added to train.py before GPTConfig creation
# ── Get actual vocab size from tokenizer ──────────────────────
from tokenizers import Tokenizer as _Tokenizer
_tok_dir = Path(args.data_dir) / 'tokenizer'
_tok = _Tokenizer.from_file(str(_tok_dir / 'tokenizer.json'))
actual_vocab_size = _tok.get_vocab_size()
print(f'Tokenizer vocab size: {actual_vocab_size}')
 
# ── Config ────────────────────────────────────────────────────
config = GPTConfig(
    vocab_size = actual_vocab_size,   # now defined ✓
    ...
)
Bug 2 — OpenWebText Dataset Not Implemented (prepare_data.py)
Root Cause
The openwebtext option was listed in the CLI choices but had no implementation — it would raise NotImplementedError if selected.
Fix — Added download_openwebtext() function
def download_openwebtext(output_dir: Path, max_chars: int = 50_000_000):
    from datasets import load_dataset

print('Downloading OpenWebText (streaming)...')
    ds = load_dataset('Skylion007/openwebtext', split='train',
                      streaming=True, trust_remote_code=True)
    chunks, total = [], 0
    for row in ds:
        t = row['text'].strip()
        if not t: continue
        chunks.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return '\n'.join(chunks)

Bug 3 — Hardcoded Token IDs (export_hf.py)
Root Cause
The bos_token_id and eos_token_id were hardcoded to 50256 (GPT-2's value) but our custom tokenizer may have a completely different vocab size. This would produce a broken HuggingFace config.
Fix
actual_vocab_size = cfg.get('vocab_size', 50257)
# Derive eos/bos from actual vocab (last token = endoftext)
eos_token_id = actual_vocab_size - 1
 
hf_config = {
    'vocab_size':   actual_vocab_size,   # was hardcoded 50257
    'bos_token_id': eos_token_id,        # was hardcoded 50256
    'eos_token_id': eos_token_id,        # was hardcoded 50256
    ...
}

Bug 4 — Broken Fallback Imports (train_tokenizer.py)
Root Cause
The try block imported processors and RobertaProcessing but these were never used. On the except/fallback path these imports were missing, causing an ImportError on first install.

Fix — Removed unused imports
# BEFORE (broken):
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.processors import RobertaProcessing  # never used — causes ImportError
 
# AFTER (fixed):
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
# RobertaProcessing removed entirely

4. Training the LLM
Understanding Training Output

During training the following metrics were displayed every 100 steps. Here is what each one means:


<img width="603" height="265" alt="image" src="https://github.com/user-attachments/assets/90940a88-ed07-4083-a2d2-b09ca4f9381a" />





 
💡 Loss of 2.45 after only 10% of training is healthy — the model is learning fast. Loss below 2.0 generally means the model can generate coherent text.



Training Configuration Used
# tiny.yaml config — much faster than default small config
python3 run_pipeline.py --config configs/tiny.yaml
 
# Model architecture:
n_layer = 4     # transformer blocks
n_head  = 4     # attention heads
n_embd  = 256   # embedding dimension
epochs  = 2
batch_size = 8
context_len = 512

Apple Silicon MPS GPU
The M1's GPU (Metal Performance Shaders) was used automatically — confirmed by 'Using Apple Silicon MPS' in output. This is as fast as PyTorch can go on Mac. No extra configuration was needed.


Dataset — TinyStories
The tiny.yaml config uses TinyStories from HuggingFace — ~2 million short children's stories. It was streamed directly rather than downloading the full dataset to save RAM.
Training set: 47,500,000 characters
Validation set: 2,500,000 characters
Tokenizer vocabulary: 20,712 tokens (custom BPE trained on the data)
Total model parameters: 30,142,848


Running Individual Pipeline Steps
If the pipeline gets stuck (e.g. on a HuggingFace retry), individual steps can be run in isolation:

# Run only step 3 (tokenizer training)
python3 run_pipeline.py --config configs/tiny.yaml --only-step 3
 
# Run steps 4 through 11 in sequence
for step in 4 5 6 7 8 9 10 11; do
  python3 run_pipeline.py --config configs/tiny.yaml --only-step $step
done

5. Chat Interface & Conversation Logging

What Was Built

A complete web-based chat interface (llm_chat project) was built with three components:
server.py — Flask server that loads the model and serves the UI
templates/index.html — Dark terminal-styled chat interface
finetune_from_logs.py — Script to retrain on logged conversations

Quick Start
# 1. Install Flask

pip3 install flask --break-system-packages
 
# 2. Start the server (point to your trained checkpoint)

cd llm_chat
python3 server.py \
    --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/best.pt
 
# 3. Open browser

# → http://localhost:5000

Conversation Log Format
Every message sent through the chat UI is automatically saved to logs/conversations.jsonl. Each line is a JSON object:
{
  "timestamp":  "2026-03-06T14:30:00",
  "session_id": "sess_abc123",
  "prompt":     "Once upon a time there was",
  "response":   "a little fox who lived in the forest..."
}

API Endpoints

<img width="664" height="140" alt="image" src="https://github.com/user-attachments/assets/36dbdaa6-04f4-491d-9289-c6f137c04d2a" />





6. Continuous Learning from Conversations
 
Can the Model Learn While Talking?
Not in real-time — once training is done the weights are frozen. Talking to the model is inference only; nothing is saved back. This is true of all LLMs including ChatGPT.

💡 The live model you chat with in ChatGPT is NOT updating from your conversations in real-time. Periodic fine-tuning on collected data is what production systems actually do.

The Practical Workflow:
Chat with the model — conversations auto-save to logs/conversations.jsonl
Review and curate the log — delete bad/repetitive responses
Run finetune_from_logs.py periodically (daily or weekly)
Restart the server with the updated checkpoint
Repeat — model gradually improves on your usage patterns
Running Fine-tuning on Logs
cd llm_chat
 
python3 finetune_from_logs.py \
    --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/best.pt \
    --logs logs/conversations.jsonl \
    --epochs 1
 
# After fine-tuning, restart server with the new checkpoint:
python3 server.py \
    --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/finetuned_chat/finetuned_best.pt

LoRA Fine-tuning Explained
The fine-tuning uses LoRA (Low-Rank Adaptation) rather than updating all weights. This is important because:

Only ~1% of weights are updated — fast and memory efficient
Base weights are frozen — the model won't forget what it already learned (no catastrophic forgetting)
Quality close to full fine-tuning for most tasks
Standard approach used by the industry for updating large models


Three Approaches to Continuous Learning

<img width="664" height="140" alt="image" src="https://github.com/user-attachments/assets/01664daf-e65f-4cbd-a918-030472a50752" />



7. What the Model Can & Cannot Do :
What It Does
This is a base language model trained on children's stories. It learned to predict the next word given previous words. When you give it a prompt, it continues the text in a story-like style. It is not GROK :)

💡 This model is closer to an autocomplete engine trained on TinyStories than a conversational assistant. It generates story-style continuations, not answers to questions. Dont expect alot, this was me to learn. 

Capabilities vs Limitations

<img width="664" height="140" alt="image" src="https://github.com/user-attachments/assets/e7ba93f7-b9da-4972-bea8-aa765bbb104e" />

 
To Get Proper Q&A Behaviour
The finetune.py script supports instruction fine-tuning. Create a JSONL file of prompt/response pairs and run:

python3 scripts/finetune.py \
    --checkpoint output/tiny/checkpoints/best.pt \
    --data data/samples/instructions.jsonl \
    --method instruct \
    --epochs 2
 
# instructions.jsonl format:
{ "instruction": "What is 2+2?", "input": "", "output": "4" }
{ "instruction": "Write a haiku", "input": "", "output": "Leaves fall..." }

8. Quick Reference — All Key Commands :
Full Pipeline
# Run the complete pipeline with tiny (fast) config
python3 run_pipeline.py --config configs/tiny.yaml
 
# Skip training, use existing checkpoint
python3 run_pipeline.py --config configs/tiny.yaml --skip-train
 
# Run a single step only
python3 run_pipeline.py --config configs/tiny.yaml --only-step 4

Chat Server
cd llm_chat
pip3 install flask --break-system-packages
 
# Start server
python3 server.py --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/best.pt
 
# Start with fine-tuned checkpoint
python3 server.py --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/finetuned_chat/finetuned_best.pt

Fine-tuning on Conversations
cd llm_chat
 
# Fine-tune on logged conversations (run periodically)
python3 finetune_from_logs.py \
    --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/best.pt \
    --logs logs/conversations.jsonl \
    --epochs 1




[↑ Back to Menu](#-quick-links)

---




