# Transformers

Reference: https://jalammar.github.io/illustrated-transformer/

## Original Transformer (2017 - "Attention Is All You Need")

- Designed for **translation** (e.g., French → English)
- Has two halves: **Encoder** (reads input) and **Decoder** (generates output)
- Key innovation: **self-attention** — lets the model look at all positions in a sequence to understand context

## Three Branches of the Transformer

| Architecture | What it does | Examples |
|---|---|---|
| Encoder-only | Understanding/classifying text | BERT, RoBERTa |
| Encoder-Decoder | Seq-to-seq (translation, summarization) | T5, BART, original Transformer |
| Decoder-only | Text generation (predict next token) | GPT, Claude/Opus, LLaMA |

## Decoder-Only Models (GPT, Claude)

- Use only the decoder half of the original transformer
- Training objective: given a sequence of tokens, predict the next one
- Trained on trillions of tokens → learns language, reasoning, code, etc.
- The self-attention mechanism is what makes it all work

## How Decoder-Only Models Get Trained

No input/output pairs needed — the training data is just raw text.

Example with the sentence "The cat sat on the mat":

| Input (context)        | Target (predict this) |
|------------------------|-----------------------|
| The                    | cat                   |
| The cat                | sat                   |
| The cat sat            | on                    |
| The cat sat on         | the                   |
| The cat sat on the     | mat                   |

The text itself is both the input and the supervision signal, shifted by one position.

### Training pipeline

1. **Pre-training** — Next-token prediction over trillions of tokens (books, web, code). Forces the model to build deep internal representations of language, logic, and world knowledge.
2. **Fine-tuning / RLHF** — Train it to be helpful, follow instructions, refuse harmful requests. This is where "text predictor" becomes "assistant."

### Why it works

To predict the next word accurately, the model is forced to learn grammar, facts, reasoning, and patterns. A deceptively simple objective that produces emergent capabilities.

### Key difference from encoder-decoder

- **Encoder-decoder**: needs explicit input→output pairs (e.g., translation pairs)
- **Decoder-only**: just needs raw text — no pairing required

## Questions

- How exactly does self-attention differ between encoder and decoder? (decoder uses **masked** self-attention so it can't peek at future tokens)
