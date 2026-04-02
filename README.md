# 🚀 AI Project Portfolio

<div align="center">

![Swift](https://img.shields.io/badge/Swift-5.9-FA7343?style=flat-square&logo=swift&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-000000?style=flat-square&logo=apple&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-macOS-lightgrey?style=flat-square&logo=apple)

*Two personal AI projects built from scratch on Apple Silicon*

</div>

---

## 📍 Table of Contents

- [🤖 Project 1 — George AI (Floating Mac Assistant)](#-project-1--george-ai)
- [🧠 Project 2 — Building a GPT-style LLM from Scratch](#-project-2--building-a-gpt-style-llm-from-scratch)
- [🧪 Learning & Random Projects](#-learning--random-projects)

---

<br>

# 🤖 Project 1 — George AI

> **A floating AI orb for macOS — voice-first, privacy-focused, and built entirely in Swift + Python**

George is an ambient AI assistant that floats above all your windows as an animated orb. It listens for your voice, routes questions intelligently between a local on-device model and cloud APIs, sees your screen on demand, plays chess, plays Arkanoid autonomously using reinforcement learning, and learns your preferences over time — all running natively on Apple Silicon.

### Tech Stack

| Layer | Technology |
|-------|-----------|
| UI & App | Swift / SwiftUI / AppKit / SpriteKit |
| Local AI | Apple MLX (`mlx_lm`) — Llama 3.2 3B Instruct 4-bit |
| Cloud AI | OpenRouter → OpenAI → Gemini (cascade fallback) |
| Backend | Python Flask server on `localhost:8765` |
| Speech In | Apple `SFSpeechRecognizer` + `AVAudioEngine` |
| Speech Out | Apple `AVSpeechSynthesizer` (premium voices) |
| Memory | Local JSON (`~/.george/memory.json`) — never sent to cloud |

### Source Files Documented

`GeorgeController.swift` · `server.py` · `Agent.swift` · `Memory.swift` · `OrbView.swift` · `ChessEngine.swift` · `ChessBoardView.swift` · `GamePlayer.swift` · `ArkanoidMac.swift` · `main.swift` · `AppDelegate.swift` · `setup.sh` · `george_mobile.html`

---

## Chapter 1 — System Architecture & AI Routing

### 1.1 Overview

George uses a two-tier intelligence model: a **local on-device language model** (running on Apple Silicon GPU via `mlx_lm`) for fast, private answers, and a **cascade of cloud APIs** for questions that require real-time information or richer reasoning. A Python backend server acts as the central router and bridge between the Swift UI layer and all AI providers.

**Key components:**

| File | Role |
|------|------|
| `GeorgeController.swift` | Main Swift controller — mic, speech recognition, screen capture, TTS, all AI calls |
| `server.py` | Python HTTP server on `localhost:8765` — LLM routing, local inference, cloud API calls |
| `Agent.swift` | Reinforcement learning layer — learns preferences and improves routing decisions over time |
| `Memory.swift` | Persistent memory of facts, conversation history, and user profile |

### 1.2 Smart 3-Step Routing System

George does not blindly send every question to the cloud. It uses a 3-step system to decide whether to answer locally or make an outbound internet request. The goal is maximum privacy and speed — only questions that genuinely require real-time data go online.

**Step 1 — Fast Pattern Matching (No LLM Needed)**

Before invoking any language model, `server.py` runs the query through compiled regex patterns. This is instant and uses zero tokens or API calls.

```python
# server.py — fast_route() function
def fast_route(user_input, has_cloud):
    if PERSONAL_LOCAL_PATTERNS.search(user_input): return 'LOCAL'
    if not has_cloud: return 'LOCAL'
    if FORCE_CLOUD_PATTERNS.search(user_input): return 'CLOUD'
    if CLOUD_PATTERNS.search(user_input): return 'CLOUD'
    if LOCAL_PATTERNS.search(user_input): return 'LOCAL'
    return None  # ambiguous — go to Step 2
```

| Pattern Category | Triggers |
|-----------------|---------|
| `PERSONAL_LOCAL` | Questions about you, your family, your health, your schedule — always answered from George's memory, never sent to cloud |
| `FORCE_CLOUD` | Phrases like "use the internet", "search online", "google it" — bypasses local model entirely |
| `CLOUD_PATTERNS` | News, scores, stock prices, current events, today's weather, release dates |
| `LOCAL_PATTERNS` | Definitions, jokes, stories, math, general knowledge, anything timeless |

**Step 2 — LLM Router (Ambiguous Queries)**

If pattern matching returns `None`, George uses its own local LLM to classify the query. A tiny 4-token inference call asks the model to output only the word `LOCAL` or `CLOUD`. Uses `temperature=0` for deterministic results. If the router errors, the fallback is always `LOCAL` to preserve privacy.

> 💡 The routing decision itself never leaves your Mac — even for queries that will ultimately go to the cloud.

**Step 3 — Cloud Cascade**

When a query needs the internet, George tries providers in order, falling back if a call fails:

| Priority | Provider | Notes |
|----------|---------|-------|
| 0 | `wttr.in` | Weather queries bypass all LLMs — free, no API key |
| 1 | OpenRouter | Tries `gpt-4o-mini`, `gemma-3`, `llama-3` in order |
| 2 | OpenAI | Direct fallback if OpenRouter fails |
| 3 | Gemini | Google Gemini 2.0 Flash as final fallback |
| 4 | Local fallback | If all cloud providers fail, answers locally with a caveat |

> 🟡 When any cloud provider answers, the orb displays a **yellow ring** — you always know when data left your Mac.

### 1.3 Cloud API Endpoints

| Service | Endpoint |
|---------|---------|
| OpenRouter | `https://openrouter.ai/api/v1/chat/completions` |
| OpenAI | `https://api.openai.com/v1/chat/completions` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent` |
| Weather | `https://wttr.in/{city}?format=j1` — free, no key required |
| News | `https://news.google.com/rss` — parsed locally |

### 1.4 Server Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/infer` | Main inference — runs the router, returns streaming JSON |
| `GET` | `/ping` | Health check — returns `{cloud: true/false}` |
| `GET` | `/` | Serves the mobile web UI |
| `POST` | `/reload_config` | Hot-reloads `~/.george/config.json` without restart |

### 1.5 Streaming Response Format

The server uses HTTP chunked transfer encoding to stream responses sentence-by-sentence. George starts speaking before it has finished generating the full answer.

```json
{"sentence": "Here is the first sentence.", "source": "local"}
{"sentence": "And here is the second.", "source": "cloud"}
{"done": true, "full": "Complete text", "source": "cloud"}
```

---

## Chapter 2 — Screen Vision

George can look at your screen on demand. When you ask *"Hey George, what's on my screen?"*, George takes a screenshot, encodes it as an image, and sends it to a cloud vision model.

**Trigger phrases:** "what's on my screen", "describe my screen", "what do you see", "look at my screen", "read the screen", "summarize this page"

### How It Works

George uses macOS's native `screencapture` CLI tool. The `-x` flag suppresses the shutter sound:

```swift
// GeorgeController.swift
let path = "/tmp/george_screen.png"
_ = await shell("screencapture -x " + path)
```

The screenshot is saved to `/tmp/george_screen.png` — a temporary file overwritten each use. It is **not** stored persistently on disk.

After capture, George base64-encodes the PNG and embeds it directly in the API request body — no intermediate file upload or third-party storage:

```swift
{ "type": "image_url",
  "image_url": { "url": "data:image/png;base64,..." } }
```

George sends a voice-friendly vision prompt that produces 2–3 natural spoken sentences. The response is streamed back and spoken aloud via `AVSpeechSynthesizer`.

> ⚠️ George **never** passively monitors your screen. Screen vision is triggered only when you explicitly ask for it — no background process, no timer, no event listener initiates it automatically.

Screen Recording permission must be granted in **System Settings → Privacy & Security → Screen Recording**.

---

## Chapter 3 — Hardware Integration

George integrates with macOS hardware through official Apple APIs and standard Unix tools — no kernel extensions, no custom drivers.

### Microphone

| Framework | Role |
|-----------|------|
| `AVAudioEngine` | Captures raw audio from the default mic at 44.1kHz, streams buffers to the speech recognizer |
| `SFSpeechRecognizer` | Apple's on-device speech recognition, locale `en-US`. A second instance (`interruptRecognizer`) listens for "stop" / "shut up" while George is speaking |

The mic is turned off while George is speaking to prevent it from hearing its own TTS output.

### GPU — Local AI Inference

The local language model runs on Apple Silicon GPU (M1/M2/M3/M4) using Apple's MLX framework via `mlx_lm`. Loaded into GPU memory at startup — the orb shows a "particles orbit" animation while local GPU inference is running.

### Hardware Stats

When you ask "check my hardware", George reads from standard macOS CLI tools:

| Data | Command |
|------|---------|
| CPU usage | `ps -A -o %cpu \| awk` |
| RAM total | `sysctl -n hw.memsize` |
| RAM free | `vm_stat` |
| Battery | `pmset -g batt` |
| Disk usage | `df -h /` |
| Network interface | `route get default` |
| Fan speed | `istats fan speed` (optional) |
| CPU temperature | `istats cpu temp` (optional) |
| Uptime | `uptime` |

All hardware data is **read-only**.

---

## Chapter 4 — Memory Engine

George's memory system is what makes it feel like a real companion rather than a stateless chatbot. Everything lives in `~/.george/memory.json` — never sent to the cloud.

### Storage Structure

```json
{
  "persona": { "name": "George", "traits": ["curious","warm","witty","direct","honest"] },
  "history": [ /* last 80 conversation turns */ ],
  "facts":   [ /* up to 300 plain-English facts about you */ ]
}
```

### Proactive Fact Mining

Every user message is scanned by a regex-based fact extractor **before** it reaches the AI. George does not rely on the LLM to notice personal facts — it mines them proactively. Categories detected:

Name · Spouse/Partner · Children · Extended family · Job · Location · Age · Pets · Hobbies (21 types) · Music genres (27 types)

Extracted facts are deduplicated and capped at 300 entries. Oldest facts are trimmed if the cap is exceeded.

### Correction System

George listens for corrections. If you say "correction", "remember I...", "actually I...", or "I don't listen to X", George removes the conflicting fact and stores the corrected version.

### System Prompt Injection

Every inference call receives a system prompt built from George's memory, injecting up to 40 recent facts plus recent conversation history. Key rules baked into every prompt:

- Voice-only output — no markdown, no asterisks, no bullet points
- Keep replies to 1–3 sentences unless asked for detail
- Naturally weave in past facts when they enrich the answer
- Never invent facts, news, weather, or current events without real data
- Never mention Claude, Anthropic, LLaMA, or that you're an AI model — you ARE George

### What Memory Never Does

- Zero network calls — all reads/writes are local JSON on your disk
- Facts are never uploaded separately — they travel only as part of the system prompt
- No expiry or TTL — facts persist until you correct them or edit the JSON manually

---

## Chapter 5 — Orb Visual System

The floating orb is a real-time status display that communicates George's internal state through animation, color, and behavior. Built in pure SwiftUI with **11 distinct visual styles**.

### The Five Orb States

| State | Visual Behavior |
|-------|----------------|
| `.booting` | Scale pulls in (0.92x), three animated dots cycle below to show loading progress |
| `.idle` | Slow breathing — scale oscillates 1.0 ↔ 1.07 over 3 seconds |
| `.listening` | Orb shrinks to 82%, ripple ring expands and fades, live speech transcription displayed |
| `.thinking` | Subtle nudge to 1.02x, 6 particles orbit the perimeter |
| `.speaking` | Rapid pulse between 1.0 and 1.26 scale — fast heartbeat effect |

### The 11 Orb Styles

| # | Style | Description |
|---|-------|-------------|
| 0 | Classic Blue | Standard radial gradient sphere with blue glow and orbiting particles when thinking |
| 1 | Blue Flower | 8 rotating ellipse petals orbit the outside of the orb |
| 2 | Plasma Arc | 3 partial circle arcs rotate at 120° offsets — electric plasma effect |
| 3 | Crystal Gem | 6 rotating diamond shapes behind the globe with ice-white facets |
| 4 | Deep Nebula | 4 rotating gradient nebula clouds + 12 star dots on the perimeter |
| 5 | Matrix Rain | Green globe with orbiting `0`, `1`, `▓`, `░` characters |
| 6 | Lava Lamp | 5 blurred orange/red blobs float and merge inside the orb |
| 7 | Ghost Light | Three concentric rings with low opacity breathe slowly around a white core |
| 8 | Aurora | 5 blurred aurora bands in gradient colors rotate slowly |
| 9 | Neon Pulse | Hot pink globe with 3 concentric neon rings pulsing outward |
| 10 | Vortex | 6 arc segments colored by hue rotation spin together |

When a cloud provider answers, the orb shows a **yellow ring** + "via cloud" label that fades in and out with a smooth 0.5-second transition.

**Double-tap the orb** to quit George. Single taps are ignored — voice is the primary input.

---

## Chapter 6 — Chess Engine

George includes a complete, self-contained chess implementation in Swift with no external libraries. It handles the full rules of chess and plays using minimax search with alpha-beta pruning.

### Board Representation

Pieces are encoded as raw integers 0–12 (white pieces 1–6, black 7–12, empty 0), stored as a flat 64-element array. `ChessBoard` is a Swift **struct** (value type) — every speculative move in the AI search creates an independent copy with no risk of corrupting the real game state. No "make/unmake" logic needed.

### Move Generation

Two-phase: `pseudoMoves()` generates all moves ignoring check, then `legalMoves()` filters by simulating each move and verifying the king is not left in check. All special rules implemented: en passant, castling (with rights revocation), pawn promotion (all 4 pieces), double pawn push.

### Board Evaluation

| Piece | Value | Reasoning |
|-------|-------|----------|
| Pawn | 100 | Baseline unit |
| Knight | 320 | Slightly less than bishop |
| Bishop | 330 | Bishop pair advantage |
| Rook | 500 | Worth ~5 pawns |
| Queen | 900 | Worth ~9 pawns |
| King | 20,000 | Sentinel — game ends before capture |

Each piece also has an 8×8 positional bonus table from classical chess programming literature — penalizing knights on the rim, rewarding advanced pawns, rewarding castled kings, discouraging early queen development. For Black pieces, the table is mirrored so both sides are evaluated symmetrically.

### Minimax with Alpha-Beta Pruning

Searches 4 moves deep. Without pruning, depth-4 requires evaluating ~810,000 positions (30⁴). Alpha-beta reduces the effective branching factor from ~30 to ~7, making responses under one second on Apple Silicon.

```swift
private static func alphabeta(_ board: ChessBoard, depth: Int,
                                alpha: Int, beta: Int, maximising: Bool) -> Int {
    if depth == 0 { return board.evaluate() }
    // ...
    if b <= a { break }  // beta cutoff — prune remaining branches
}
```

Sentinel values use ±10,000,000 instead of `Int.min`/`Int.max` to avoid overflow when evaluation adds position bonuses.

### Voice Move Input

Handles multiple formats: coordinate notation (`e2e4`, `e2 to e4`), piece + destination (`knight f3`), and castling (`castle kingside`). After each move George picks natural-language commentary based on which piece moved. If check or checkmate follows: *" Check."* or *" Checkmate. I win this time."*

---

## Chapter 7 — Reinforcement Learning Game Player (Arkanoid)

George can play Arkanoid autonomously using a Q-learning agent. It watches the game screen via screenshot, infers state, chooses actions, and sends real keyboard inputs to control the paddle — all in real time.

Say *"Hey George, play Arkanoid"* to launch it.

### Q-Learning Algorithm

Uses the Bellman equation to update a state→action value table after every move:

```
updated = current + α × (reward + γ × bestNextValue − current)
```

| Hyperparameter | Value | Reasoning |
|---------------|-------|----------|
| Learning rate (α) | 0.25 | Conservative for noisy screen-based state |
| Discount factor (γ) | 0.95 | Longer horizon — brick hits are delayed by many frames |
| Epsilon (ε) | 0.3 → 0.1 | Explore freely at first, then mostly exploit |
| Frame rate | 50ms / loop | 20 decisions per second |

### 8-Dimensional State Space

| Dimension | Description |
|-----------|-------------|
| Ball X | 10 horizontal buckets |
| Ball Y | 8 vertical buckets |
| Ball velocity X | Left / center / right (derived from position delta) |
| Ball velocity Y | Up / center / down |
| Paddle X | 10 horizontal buckets |
| Relative position | Ball X minus paddle X |
| Brick density | Approximate remaining brick count |
| Ball direction | Moving toward or away from paddle |

Velocity tracking was a key v7 improvement — without it, the agent can only react to current position. With velocity, George leads the ball and positions the paddle in advance.

### Reward Function

| Reward | Trigger |
|--------|---------|
| +2.0 | Ball moving toward paddle AND paddle within 15% of ball — active chase |
| +1.5 | A brick was destroyed this frame |
| +1.0 | Ball velocity moving toward paddle (correct anticipation) |
| -0.5 | Ball moving toward paddle but paddle is far away |
| -2.0 | Ball reached the bottom edge (ball lost) |

### Action Execution

George sends real `CoreGraphics` keyboard events to macOS. It plays the game exactly as a human would, through the input system. Virtual key codes: `123` = left arrow, `124` = right arrow.

### Persistence

The Q-table is saved to `~/.george/qtable.json`. George gets smarter across sessions and never forgets. Auto-save every 1,000 moves. On startup: *"Starting Arkanoid. I have 847 states learned."*

### The Arkanoid Game Itself

A full standalone SpriteKit Arkanoid clone embedded in George. Features: authentic 7-color brick palette by row, all 5 classic power-ups (Slow, Catch, Expand, Multi-ball, Disrupt), procedurally synthesized sound effects via `AVAudioEngine` (no audio files needed), fully resizable window, silver/gold indestructible bricks, multi-level progression.

---

## Chapter 8 — App Entry Point

`main.swift` is 46 lines and `AppDelegate.swift` is 42 lines — small files with enormous consequences. Every decision here affects how George behaves at the OS level.

### Crash Handlers

The very first thing George does — before any UI — is register crash signal handlers for `SIGBUS`, `SIGSEGV`, `SIGABRT`, `SIGILL`, and `SIGFPE`. On crash, the handler writes a timestamped entry to `~/Documents/George/george_debug.log`, then re-raises the signal so macOS can generate a proper `.crash` file with a full stack trace.

### Activation Policy

```swift
app.setActivationPolicy(.accessory)
```

`.accessory` means: no Dock icon, not shown in Cmd+Tab, no menu bar. George is ambient — always present, never intrusive.

### NSPanel Configuration

| Setting | Value | Why |
|---------|-------|-----|
| Style mask | `.borderless` + `.nonactivatingPanel` | No window chrome; clicking the orb doesn't steal focus |
| `panel.level` | `.floating` | Sits above all normal app windows |
| `backgroundColor` | `.clear` + `isOpaque = false` | Transparent background for the "floating in space" effect |
| `hasShadow` | `false` | Prevents a rectangular shadow artifact behind the transparent panel |
| `collectionBehavior` | `.canJoinAllSpaces` + `.fullScreenAuxiliary` | George appears on every Space and above full-screen apps |
| `isMovableByWindowBackground` | `true` | Drag the orb anywhere — no title bar to grab |

### Boot Sequence

```
1.  OS loads George binary → main.swift executes
2.  installCrashHandlers() registers signal handlers
3.  NSApplication.shared creates the app singleton
4.  AppDelegate() instantiated; wired to app
5.  setActivationPolicy(.accessory) — hidden from Dock
6.  app.run() — main run loop starts (never returns)
7.  applicationDidFinishLaunching fires
8.  NSPanel created and configured
9.  Panel positioned bottom-right of screen
10. GeorgeController() created — Memory + Agent loaded from disk
11. OrbView SwiftUI view tree created
12. NSHostingView bridges SwiftUI into AppKit
13. panel.makeKeyAndOrderFront — orb appears on screen
14. Task { await george.boot() } scheduled async
15. Boot: requests mic + speech recognition permissions
16. Boot: launches server.py as child process
17. Boot: pings localhost:8765 until server responds
18. Boot: server.py loads local LLM into Apple Silicon GPU
19. Boot: state → .idle — breathing animation begins
20. Boot: AVAudioEngine starts streaming audio
21. George speaks a random greeting
22. Fully operational — listening for "Hey George"
```

---

## Chapter 9 — Setup & Mobile UI

### setup.sh — One-Command Installation

A bash script that installs everything on a fresh Mac. Uses `set -e` — a failed step is loud and obvious rather than silently broken.

**8 steps:** Verify prerequisites → Create `~/.george/` → Copy `server.py` → Enter API keys → Install `mlx-lm` → Install `yt-dlp` → `swift build -c release` → Download Llama 3.2 3B (~1.8 GB)

Creates a `George.command` launcher on the Desktop. The launcher kills any existing process on port 8765 first to ensure a clean start.

### Mobile Web UI (george_mobile.html)

A self-contained single-file web app served by `server.py` at `GET /`. Open it on any phone, tablet, or computer — no install required.

- Dark space theme matching the Mac orb (CSS custom properties, randomly generated starfield)
- Animated CSS orb with 5 state classes — no Canvas, no WebGL
- Streaming text — sentences appear as they are generated
- Voice input via Web Speech API
- Mac TTS voice selector — responses are spoken on your Mac, not your phone

> You can be in another room, send George a query from your phone, and hear the response come out of your Mac speakers — the phone is just a remote control.

---

<br>
<br>

---

# 🧠 Project 2 — Building a GPT-style LLM from Scratch

> **A complete hands-on implementation of a GPT-style transformer, trained on Apple Silicon M1**

Built as a learning project to understand how large language models work under the hood. Covers environment setup, bug fixes, training, and a web-based chat interface — including all code, errors encountered, and how they were resolved.

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
| Model Size | 30,142,848 parameters (tiny config) |

---

## Chapter 1 — Environment Setup

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

> 💡 **Tip:** Your terminal's working directory points to an inode. Moving or trashing a folder in Finder does **not** change where your terminal thinks it is — stay in the same directory until you `cd` elsewhere.

---

## Chapter 2 — Bugs Found & Fixed

A full audit of all pipeline scripts revealed **4 bugs**. All were fixed before training began.

### Bug 1 — `actual_vocab_size` Undefined (`train.py`, line 392)

**Error:**
```
NameError: name 'actual_vocab_size' is not defined
```

**Root cause:** The variable was referenced in the `GPTConfig` constructor before it was assigned. The tokenizer needed to be loaded first.

**Fix:**
```python
from tokenizers import Tokenizer as _Tokenizer
_tok_dir = Path(args.data_dir) / 'tokenizer'
_tok = _Tokenizer.from_file(str(_tok_dir / 'tokenizer.json'))
actual_vocab_size = _tok.get_vocab_size()

config = GPTConfig(
    vocab_size=actual_vocab_size,  # ✅ now defined
    ...
)
```

### Bug 2 — OpenWebText Dataset Not Implemented (`prepare_data.py`)

**Root cause:** `openwebtext` was a valid CLI option with no implementation — it would raise `NotImplementedError` if selected.

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

### Bug 3 — Hardcoded Token IDs (`export_hf.py`)

**Root cause:** `bos_token_id` and `eos_token_id` were hardcoded to `50256` (GPT-2's value). Our custom tokenizer has a different vocab size, producing a broken HuggingFace config.

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

### Bug 4 — Broken Fallback Imports (`train_tokenizer.py`)

**Root cause:** The `try` block imported `RobertaProcessing` which was never used. On the fallback path, this import was missing, causing an `ImportError` on first install.

**Fix:**
```python
# BEFORE ❌
from tokenizers import (Tokenizer, models, trainers,
                        pre_tokenizers, decoders, processors)
from tokenizers.processors import RobertaProcessing  # never used

# AFTER ✅
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
```

---

## Chapter 3 — Training

### Configuration Used

```bash
# tiny.yaml — much faster than the default small config
python3 run_pipeline.py --config configs/tiny.yaml
```

```yaml
n_layer: 4       # transformer blocks
n_head: 4        # attention heads
n_embd: 256      # embedding dimension
epochs: 2
batch_size: 8
context_len: 512
```

### Dataset — TinyStories

| Split | Size |
|-------|------|
| Training set | 47,500,000 characters |
| Validation set | 2,500,000 characters |
| Tokenizer vocab | 20,712 tokens (custom BPE) |
| Total parameters | 30,142,848 |

The config streams TinyStories directly from HuggingFace to keep RAM usage low.

### Training Metrics Explained

| Metric | Meaning |
|--------|---------|
| `loss` | Cross-entropy loss — lower is better; below 2.0 means coherent text |
| `ppl` | Perplexity — how "surprised" the model is by the next token |
| `lr` | Current learning rate (decays over time via scheduler) |
| `step/s` | Training throughput in steps per second |
| `eta` | Estimated time remaining |

> 💡 A loss of **2.45** after only 10% of training is healthy — the model is learning fast.

Apple Silicon MPS is used automatically — confirmed by `Using Apple Silicon MPS` in output. No extra configuration needed.

### Running Individual Steps

```bash
# Run only step 3 (tokenizer training)
python3 run_pipeline.py --config configs/tiny.yaml --only-step 3

# Run steps 4–11 sequentially
for step in 4 5 6 7 8 9 10 11; do
  python3 run_pipeline.py --config configs/tiny.yaml --only-step $step
done
```

---

## Chapter 4 — Chat Interface & Conversation Logging

| File | Purpose |
|------|---------|
| `server.py` | Flask server — loads the model and serves the UI |
| `templates/index.html` | Dark terminal-styled chat interface |
| `finetune_from_logs.py` | Retrains the model on logged conversations |

### Quick Start

```bash
pip3 install flask --break-system-packages

cd llm_chat
python3 server.py \
    --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/best.pt

# Open → http://localhost:5000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET` | `/` | Serves the chat UI |
| `POST` | `/generate` | Generates a response for a given prompt |
| `GET` | `/logs` | Returns all logged conversations as JSON |

### Conversation Log Format

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

## Chapter 5 — Continuous Learning

> **Can the model learn while you talk to it?** No — once training is complete, weights are frozen. This is true of all LLMs including ChatGPT. Production systems fine-tune periodically on collected data — they do not update in real-time.

### The Workflow

```
1. Chat with the model → conversations auto-save to logs/conversations.jsonl
2. Review and curate the log (delete bad/repetitive responses)
3. Run finetune_from_logs.py periodically (daily or weekly)
4. Restart the server with the updated checkpoint
5. Repeat — the model gradually improves on your usage patterns
```

### Running Fine-tuning

```bash
cd llm_chat

python3 finetune_from_logs.py \
    --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/best.pt \
    --logs logs/conversations.jsonl \
    --epochs 1

# Restart with the fine-tuned checkpoint
python3 server.py \
    --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/finetuned_chat/finetuned_best.pt
```

### LoRA Fine-tuning

Fine-tuning uses **LoRA (Low-Rank Adaptation)** rather than updating all weights — only ~1% of weights are updated, base weights are frozen (no catastrophic forgetting), and quality is close to full fine-tuning for most tasks.

### Three Approaches

| Approach | How It Works | Best For |
|----------|-------------|----------|
| Periodic LoRA Fine-tuning | Fine-tune on conversation logs every N days | Most practical; low cost |
| Instruction Fine-tuning | Train on curated prompt/response JSONL pairs | Teaching Q&A behaviour |
| Full Fine-tuning | Update all weights on new data | Maximum quality; high cost |

---

## Chapter 6 — Capabilities & Limitations

> This is a **base language model** trained on children's stories. It generates story-style text continuations — not answers to questions. It's not GROK — it was built to learn.

| ✅ Can Do | ❌ Cannot Do |
|-----------|------------|
| Continue a story from a prompt | Answer factual questions accurately |
| Generate fluent, grammatical text | Follow complex instructions |
| Produce children's story-style prose | Hold a coherent multi-turn conversation |
| Run entirely on-device (M1 Mac) | Reason or plan like a large model |

### To Get Proper Q&A Behaviour

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

## Chapter 7 — Quick Reference

```bash
# Full pipeline
python3 run_pipeline.py --config configs/tiny.yaml

# Skip training, use existing checkpoint
python3 run_pipeline.py --config configs/tiny.yaml --skip-train

# Single step only
python3 run_pipeline.py --config configs/tiny.yaml --only-step 4

# Start chat server
cd llm_chat && python3 server.py \
    --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/best.pt

# Fine-tune on logged conversations
python3 finetune_from_logs.py \
    --checkpoint ../BuildYourLLM_fixed/output/tiny/checkpoints/best.pt \
    --logs logs/conversations.jsonl --epochs 1
```

---

<br>

# 🧪 Learning & Random Projects

*Coming soon — smaller experiments and one-off builds.*

---

<div align="center">
<sub>Built with 🧠 curiosity on an M1 MacBook Pro · All projects run entirely on Apple Silicon</sub>
</div>
