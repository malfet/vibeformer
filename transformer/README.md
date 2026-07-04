# vibeformer / transformer

A from-scratch, character-level **decoder-only transformer** (the architecture
from *Attention Is All You Need*), small enough to train on a laptop GPU (MPS).
Beyond the base model, this directory holds an experiment series on **fine-tuning
a tiny poetry model** — teaching a model Konstantin Balmont's style while
borrowing vocabulary from his Silver-Age contemporaries.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # torch, safetensors, sentencepiece, matplotlib, numpy
```

Training auto-selects CUDA → MPS → CPU. Master weights are fp32; forward/backward
run in bf16 via autocast.

## Model

`model.py` — decoder-only transformer with multi-head causal self-attention,
sinusoidal position encoding, weight-tied embeddings. Size is configurable
(`--d-model/--n-heads/--n-layers/--d-ff`) and stored in each checkpoint so
`generate`/`eval` rebuild the right shape automatically.

| Config | d_model | layers | heads | params |
|---|---:|---:|---:|---:|
| default (small) | 128 | 4 | 4 | ~0.8M |
| large | 256 | 6 | 8 | ~4.8M |

## Data (`data/`)

| File | What |
|---|---|
| `tiny_shakespeare.txt` | the classic char-LM dataset |
| `tiny_balmont.txt` | Balmont's poems, `✦`/`✧` poem markers (~2.1M chars) |
| `russian_silver_age.txt` | scraped Silver-Age corpus — Blok, Sologub, Bryusov, Annensky, Bely, Gippius, Voloshin, Khodasevich, Vyach. Ivanov, Tsvetaeva (Balmont **excluded**), ~6.9M chars |
| `fetch_russian_silver_age.py` | scraper (public-domain Lib.ru/Классика) |
| `build_vocab.py` → `russian_silver_age_vocab.json` | shared 163-char vocab (pretrain + fine-tune must share it) |
| `build_bpe.py` → `russian_silver_age_bpe.model` | SentencePiece unigram tokenizer (verse-aware: atomic `<nl>`/`✦`/`✧`, byte fallback) |

## Usage

```bash
# Train (char tokenizer). Checkpoint prefix comes from --name or the data file.
.venv/bin/python train.py --data data/russian_silver_age.txt --vocab data/russian_silver_age_vocab.json

# Fine-tune from a pretrained checkpoint, with replay to retain vocabulary
.venv/bin/python train.py --data data/tiny_balmont.txt --vocab data/russian_silver_age_vocab.json \
    --init-from russian_silver_age_best.safetensors --name tiny_balmont_ft \
    --mix data/russian_silver_age.txt --mix-frac 0.3 --lr 1e-4 --max-iters 40000

# Subword instead of char: pass --bpe data/russian_silver_age_bpe.model (no --vocab)
# Bigger model: add --d-model 256 --n-layers 6 --n-heads 8 --d-ff 1024

.venv/bin/python generate.py --checkpoint tiny_balmont_ft_best.safetensors
.venv/bin/python eval_val.py tiny_balmont_ft_best.safetensors           # honest bits/char
.venv/bin/python expressivity_probe.py tiny_balmont_ft_best.safetensors # vocabulary analysis
```

## Experiment: fine-tuning Balmont with Silver-Age poetry

**Goal:** train a model in Balmont's style, but give it more expressivity —
words and rhymes learned from other poets of his era.

**Approach:** pretrain on the Silver-Age corpus, then fine-tune on Balmont alone,
sharing one tokenizer. Evaluate on two axes: **bits/char** on held-out Balmont
(fidelity — lower is better, tokenizer-independent) and **expressivity** via a
generation probe (real-word rate; count of distinct generated words that appear
in *other* poets but not in Balmont — "era-only words").

### Results (all fine-tunes use replay mix 0.3, on held-out Balmont)

| Model | tokenizer | params | bits/char | real-word rate | era-only words |
|---|---|---:|---:|---:|---:|
| from-scratch baseline | char | 0.8M | 1.937 | 79% | 99 |
| pretrain → fine-tune | char | 0.8M | 1.907 | 72% | 198 |
| pretrain → fine-tune | BPE 4k | 1.3M | 2.121 | 71% | 150 |
| pretrain → fine-tune | BPE 2k | 1.0M | 2.166 | 69% | 165 |
| **pretrain → fine-tune** | **char, large** | **4.8M** | **1.838** | **86%** | **219** |

### What we learned

1. **Pretraining + fine-tuning beats from-scratch** — but only when the
   pretraining corpus *matches the target's distribution*. Aggressively cleaning
   the corpus (stripping numbers, foreign epigraphs, archaic spelling) made
   results **worse**, because Balmont's own text contains those same features.
   Distribution-match > tidiness.
2. **Fine-tuning on Balmont alone forgets the era vocabulary** — "catastrophic
   forgetting". A pretrained-only model produced ~319 era-only words; plain
   fine-tuning collapsed that to ~95 within ~2k steps.
3. **Replay fixes it** (`--mix`, `--mix-frac`): mixing a fraction of era data
   back into fine-tuning *both* fits Balmont better (regularization) *and*
   retains vocabulary. Sweep: era-only words rise 95 → 152 → 198 → 241 as
   mix-frac goes 0 → 0.15 → 0.3 → 0.5; bits/char is best around 0.3.
4. **BPE did not help at small scale** — the tiny model can't afford a 4k-way
   prediction, and rare subword embeddings are undertrained on a small corpus.
   Smaller (2k) vocab didn't rescue it. Character-level won at 0.8M params.
5. **Model size was the biggest lever.** Scaling the char model ~6× (0.8M → 4.8M)
   improved *everything at once* — lower bits/char, real-word rate 72% → 86%
   (far less invented junk), and the most era vocabulary. Capacity, not
   tokenization, was the bottleneck; the extra capacity let the model be both
   fluent *and* expressive instead of trading one for the other.

## How the metrics are computed

**bits/char** (`eval_val.py`) — the model's per-character cross-entropy on the
held-out Balmont split, in bits. For each next-token prediction we take the
negative log-probability the model assigned to the *correct* token, sum over the
whole val split, divide by the number of **characters**, and convert nats → bits:

```
bits/char = ( Σ −ln p(correct token) ) / (num characters) / ln 2
```

Dividing by characters (not tokens) is what makes it **comparable across
tokenizers** — a BPE token spans ~2 chars, so normalizing per character puts
char and subword models on the same scale (it's the standard bits-per-character
/ bits-per-byte metric). A single deterministic full-pass sweep is used, not the
noisy in-training estimate (which only samples the first ~13k val chars).

**real-word rate / era-only words** (`expressivity_probe.py`) — generate a batch
of poems (truncated to an equal character budget), extract words, and classify
them against the Balmont and Silver-Age vocabularies: real-word rate = fraction
that exist in either corpus; era-only = distinct words in *other* poets but not
in Balmont.
