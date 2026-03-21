# BERT — Bidirectional Encoder Representations from Transformers

Reference: Google, 2018

## Architecture

A **stack of Transformer encoder layers** — that's it.

- BERT-base: 12 layers, 768-dim hidden states, 12 attention heads, 110M parameters
- BERT-large: 24 layers, 1024-dim hidden states, 16 attention heads, 340M parameters

Each layer has two sub-components:
1. **Multi-head self-attention** — every token attends to every other token (no masking)
2. **Feed-forward network** — processes each token's representation independently

## Input Representation

Each token gets **three embeddings summed together**:

| Embedding | What it encodes |
|---|---|
| **Token embedding** | Identity of the word/subword |
| **Position embedding** | Where it sits in the sequence (0, 1, 2...) |
| **Segment embedding** | Which sentence it belongs to (A vs B) |

Special tokens:
- `[CLS]` at the start — its final hidden state = whole-sequence representation (used for classification)
- `[SEP]` between/after sentences — marks sentence boundaries

Example input: `[CLS] the cat sat [SEP] it was tired [SEP]`

## Self-Attention: Encoder vs Decoder

- **Decoder (GPT)**: token 3 can only see tokens 0, 1, 2 — future is masked (needed for generation)
- **Encoder (BERT)**: token 3 sees **all tokens** — no mask (BERT understands text, doesn't generate it)

This is what "deeply bidirectional" means — every layer, every token, sees everything.

## Pre-Training Objectives

BERT can't use next-token prediction (that would require masking future tokens, killing bidirectionality). Instead it uses two objectives:

### 1. Masked Language Modeling (MLM)

- Randomly mask 15% of input tokens (replace with `[MASK]`)
- Model predicts the original token at each masked position
- Example: `The [MASK] sat on the mat` → predict "cat"
- Forces the model to build deep contextual representations using both left and right context

### 2. Next Sentence Prediction (NSP)

- Give the model two sentences: are they consecutive in the original text, or random?
- `[CLS] the cat sat [SEP] it was tired [SEP]` → **IsNext**
- `[CLS] the cat sat [SEP] penguins live in Antarctica [SEP]` → **NotNext**
- Uses the `[CLS]` token output for this binary classification
- Note: later research (RoBERTa) showed NSP isn't very useful and dropped it

## Fine-Tuning

After pre-training, add a small task-specific layer on top:

- **Classification** (sentiment, spam) → `[CLS]` output → linear layer → softmax
- **Token-level tasks** (NER, POS tagging) → each token's output → classify each one
- **Question answering** → token outputs → predict start/end positions of the answer span

The entire model gets fine-tuned end-to-end on labeled data. Pre-training already learned language, so you need relatively little task-specific data.

## BERT's Output

BERT outputs a **flat vector (768-dim) per token**. For a sentence, that's a 2D matrix:

```
Input:  [CLS]  The   cat   sat  [SEP]     →  5 tokens

         ┌─────────────────────────────┐
[CLS]    │  0.12  -0.45  0.78  ...     │  ← (768,)
The      │  0.33   0.11 -0.22  ...     │  ← (768,)
cat      │ -0.05   0.67  0.41  ...     │  ← (768,)
sat      │  0.19  -0.33  0.55  ...     │  ← (768,)
[SEP]    │  0.08   0.22 -0.17  ...     │  ← (768,)
         └─────────────────────────────┘
              shape: (5, 768)
```

"Encoder only" means no text generation — NOT no output. Every token gets a rich vector representation.

## Using BERT for Cosine Similarity

To compare sentences, you need **one vector per sentence**. Pool the token vectors:

| Method | How | Quality |
|---|---|---|
| **CLS token** | Take `h₀` | Easy, but mediocre for similarity |
| **Mean pooling** | Average all token vectors | Usually better than CLS |
| **Max pooling** | Element-wise max across tokens | Sometimes works |

**Problem**: vanilla BERT wasn't trained for sentence similarity. All sentences end up with similar vectors (the "anisotropy" problem).

**Solution**: **Sentence-BERT (SBERT)** — fine-tunes BERT with a siamese architecture so that similar sentences have high cosine similarity and dissimilar ones don't.

## Chunking Long Documents

BERT has a **512 token limit**. For longer text, split into chunks and encode separately.

How to link chunk vectors depends on the goal:

| Strategy | When to use |
|---|---|
| **Don't link — search per chunk** | Retrieval / RAG (most common) |
| **Average chunk vectors** | Rough document-level similarity |
| **Weighted average** | Weight by chunk length or position |
| **Keep both levels** | Chunk vectors for retrieval + averaged doc vector for comparison |

Use **overlapping chunks** so ideas aren't split across boundaries:
```
Chunk 1: tokens 0–450
Chunk 2: tokens 400–850      ← 50 token overlap
Chunk 3: tokens 800–1250
```

## Key Insight

The MLM trick is what makes BERT work. By corrupting input and asking the model to reconstruct it (instead of predicting left-to-right), BERT gets true bidirectional context at every layer. GPT trades this for generation ability. BERT trades generation for deeper understanding.
