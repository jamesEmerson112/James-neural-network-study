# How LLMs Process Text

The full pipeline from raw text to generated output in decoder-only models (GPT, Claude).

---

## The pipeline

```
YOUR PROMPT (raw text)
"Why did Bellman name it dynamic programming?"
═══════════════════════════════════════════════
                    │
                    ▼
         ┌──────────────────┐
         │   TOKENIZER      │
         │ (not a neural net │
         │  — just a lookup) │
         └──────────────────┘
                    │
    "Why" "did" "Bell" "man" "name" "it" "dynamic" "program" "ming" "?"
      │     │     │      │     │     │      │        │        │     │
      ▼     ▼     ▼      ▼     ▼     ▼      ▼        ▼        ▼     ▼
    [8912] [1550] [42863] [805] [836] [433] [14223]  [10920]  [5765] [30]

         10 token IDs (integers, no meaning yet)
                    │
                    ▼
         ┌──────────────────┐
         │ EMBEDDING TABLE  │    ◄── Word2Vec's legacy
         │                  │        Each token ID → a learned vector
         │ 8912  → [0.02, -0.41, 0.87, ... 4093 more dims]
         │ 1550  → [0.15,  0.33, 0.12, ... 4093 more dims]
         │ 42863 → [-0.08, 0.71, 0.45, ... 4093 more dims]
         │ ...                                              │
         └──────────────────┘
                    │
                    ▼
    10 vectors, each 4096-dimensional (for a model like Claude)
    ┌────────────────────────────────────────────────┐
    │ [0.02, -0.41, 0.87, ...]  ← "Why"             │
    │ [0.15,  0.33, 0.12, ...]  ← "did"             │
    │ [-0.08, 0.71, 0.45, ...]  ← "Bell"            │
    │ [0.44, -0.19, 0.63, ...]  ← "man"             │
    │ ...6 more vectors...                           │
    └────────────────────────────────────────────────┘
                    │
                    ▼
    ╔════════════════════════════════════════════════╗
    ║         TRANSFORMER LAYERS (×96 or so)        ║
    ║                                               ║
    ║  Each layer does two things:                  ║
    ║                                               ║
    ║  ┌─────────────────────────────────┐          ║
    ║  │ SELF-ATTENTION                  │          ║
    ║  │                                 │          ║
    ║  │ Every token looks at every      │          ║
    ║  │ token BEFORE it (causal mask):  │          ║
    ║  │                                 │          ║
    ║  │ "ming" attends to:             │          ║
    ║  │   "program" ████████ (strong)   │          ║
    ║  │   "dynamic" ██████ (strong)     │          ║
    ║  │   "Bell"    █████ (medium)      │          ║
    ║  │   "man"     ████ (medium)       │          ║
    ║  │   "name"    ███ (medium)        │          ║
    ║  │   "Why"     ██ (weaker)         │          ║
    ║  │   "did"     █ (weak)            │          ║
    ║  │   "it"      █ (weak)            │          ║
    ║  │                                 │          ║
    ║  │ Each vector gets REWRITTEN      │          ║
    ║  │ based on what it attended to.   │          ║
    ║  │ This is WHERE context happens.  │          ║
    ║  └─────────────────────────────────┘          ║
    ║                    │                          ║
    ║                    ▼                          ║
    ║  ┌─────────────────────────────────┐          ║
    ║  │ FEED-FORWARD (MLP)             │          ║
    ║  │                                 │          ║
    ║  │ Each vector gets transformed    │          ║
    ║  │ independently. This is where    │          ║
    ║  │ "knowledge" lives — facts the   │          ║
    ║  │ model memorized during training.│          ║
    ║  └─────────────────────────────────┘          ║
    ║                    │                          ║
    ║            repeat × 96 layers                 ║
    ╚════════════════════════════════════════════════╝
                    │
                    ▼
    FINAL VECTOR for the LAST token position ("?")
    [0.73, -0.22, 0.91, 0.04, ... 4093 more dims]

    This single vector now encodes the MEANING of the
    entire question, shaped by 96 layers of attention.
                    │
                    ▼
         ┌──────────────────┐
         │ OUTPUT HEAD       │
         │ (linear + softmax)│
         │                  │
         │ Maps 4096D vector │
         │ → probability over│
         │ ~100K tokens      │
         │                  │
         │ "He"     → 0.12  │
         │ "Bell"   → 0.08  │
         │ "Because"→ 0.07  │
         │ "The"    → 0.06  │
         │ "In"     → 0.04  │
         │ ...              │
         └──────────────────┘
                    │
              sample "He"
                    │
                    ▼
    APPEND "He" to the sequence, run the WHOLE thing again
    to predict the NEXT token... and again... and again...

    "He" → "chose" → "the" → "name" → "'" → "dynamic" → ...
```

---

## The compressed version

```
Text → tokens → embeddings → transformer layers → next token probability → sample → repeat
```

The transformer layers aren't "decoding" the embeddings in one shot. They **rewrite** those vectors ~96 times through attention. By the final layer, the vector for the last token has absorbed the meaning of every token before it. That final vector maps to a probability distribution over the vocabulary, one token gets picked, and the whole process repeats.

---

## Decoder-only vs encoder-decoder

Modern LLMs (GPT, Claude) are **decoder-only**. There is no separate encoder module — the input embeddings and the output generation share the same transformer stack.

```
BERT-style (encoder-decoder):          GPT / Claude (decoder-only):

Input → [Encoder] → fixed             Input → [same stack that
         representation →                       also generates output]
         [Decoder] → output                     (no separate encoder)

Two separate modules.                  One continuous pipeline where
Encoder "understands,"                 "understanding" and "generating"
Decoder "generates."                   are the same computation.
```

The "encoding" of the input happens implicitly as the decoder layers process it. Each layer of self-attention lets every token absorb context from all previous tokens. By layer 96, the model has "understood" the input well enough to predict the next token.

---

## Component details

### Tokenizer

Not a neural network. A deterministic algorithm (BPE — Byte Pair Encoding) that maps text to integer IDs. "Bellman" might become ["Bell", "man"] = [42863, 805]. The vocabulary is fixed at training time (~100K tokens for modern models).

### Embedding table

A lookup table: token ID → learned vector. Conceptually identical to Word2Vec, but trained jointly with the rest of the model instead of separately. The table is a matrix of shape `(vocab_size, embedding_dim)` — for a 100K vocabulary with 4096D embeddings, that's ~400M parameters just for the embedding layer.

### Self-attention

The mechanism that gives transformers their power. For each token position, attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where $Q$ (query), $K$ (key), $V$ (value) are linear projections of the input vectors. The $\frac{1}{\sqrt{d_k}}$ scaling prevents the dot products from getting too large. The **causal mask** ensures each token can only attend to tokens before it (not future tokens), which is what makes it autoregressive.

### Feed-forward (MLP)

Applied independently to each token's vector after attention. Two linear layers with a nonlinearity:

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

Research suggests this is where factual knowledge gets stored — attention routes information, the MLP stores and retrieves it.

### Output head

A linear projection from embedding dimension to vocabulary size, followed by softmax. The result is a probability distribution over the entire vocabulary. Sampling strategies (temperature, top-k, top-p) determine how the next token is selected from this distribution.

---

## Takeaway

- LLMs convert text → token IDs → embedding vectors, then run those vectors through ~96 transformer layers
- Each layer rewrites every vector via self-attention (context from other tokens) + feed-forward (stored knowledge)
- The final vector for the last position → probability over vocabulary → sample one token → repeat
- There is no separate "encoder" in GPT/Claude — the same stack handles both understanding and generation
- The embedding table at the start is the direct descendant of Word2Vec, trained jointly with the model

---

*See also:* [27_embedding_spaces_and_retrieval.md](27_embedding_spaces_and_retrieval.md) · [quiz5/quiz_5_10_word2vec_deep_dive.md](quiz5/quiz_5_10_word2vec_deep_dive.md)
