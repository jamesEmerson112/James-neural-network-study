# From Phrase Tables to Seq2Seq: How Translation Went Neural

> **For 20 years, translation was a lookup problem. Then in 2014, two LSTMs replaced the entire pipeline.**

---

## Part 1: Statistical Machine Translation (SMT) — The Old World

Before 2014, machine translation used **zero neural networks**. It was pure statistics + engineering.

### How SMT worked

```
"The cat sat on the mat"
         ↓
   Break into phrases: ["The cat", "sat on", "the mat"]
         ↓
   Look up phrase translations from huge bilingual corpora
   (millions of aligned sentence pairs)
         ↓
   Score every possible combination by probability
   P("Le chat") × P("assis sur") × P("le tapis") × P(this word order in French)
         ↓
   Pick the highest-scoring combination
         ↓
"Le chat assis sur le tapis"
```

No understanding of meaning. Just: "these words appeared near those words in a million documents."

### The evolution within SMT

```
Word-by-word (IBM, early 1990s):
  For each English word, what's the most probable French word?
  Problem: "kick the bucket" → "donner le seau" (literal nonsense)

Phrase-by-phrase (Koehn et al., 2003):
  Let statistics find chunks that translate together:
  "kick the bucket" → (one phrase) → "mourir"
  "in spite of"     → (one phrase) → "malgré"
  Better, but still just lookup + reordering rules
```

### The data that made SMT possible

Governments doing paperwork in multiple languages accidentally built the world's translation training data:

| Corpus | Source | Size |
|--------|--------|------|
| **Europarl** | EU Parliament debates | 60 million words per language, 21 languages |
| **UN Parallel Corpus** | UN documents | 6 languages, millions of aligned sentences |
| **Canadian Hansard** | Parliament (EN/FR) | ~8 million sentence pairs |

Per Franz Josef Och (who led Google Translate): building a *usable* SMT system for a single language pair required **150–200 million words** of bilingual text and **1 billion+ words** of monolingual text per language.

The phrase tables themselves grew to **hundreds of millions of entries**, 2+ GB per language pair.

**Google Translate ran SMT from 2006 to 2016** — ten years. It was serviceable but you could always tell it was a machine.

---

## Part 2: The 14-Year Gap (2000–2014)

LSTM was invented in 1997. Seq2Seq appeared in 2014. Why the gap?

**The field's attention went to images, not sequences:**

- AlexNet (2012) proved "deep learning + GPUs + data" works → everyone pivoted to CNNs on images
- LSTMs existed but were niche (handwriting recognition, speech)
- GPUs weren't powerful enough to train deep RNNs on large text corpora until ~2013
- SMT *worked well enough* — no urgency to replace it

Then Sutskever (Hinton's student, AlexNet co-author) turned GPU insights from vision toward sequences → Seq2Seq.

---

## Part 3: The Key Clarification — RNN, LSTM, and Seq2Seq

This is the relationship that's easy to confuse:

```
RNN (1986)  →  LSTM (1997)  →  Seq2Seq uses either one (2014)
 component       component       architecture
```

### RNN and LSTM are components, not architectures

They're just boxes that read a sequence and produce hidden states:

```
RNN:   reads sequence, but forgets early words (vanishing gradient)
LSTM:  reads sequence, remembers long-range (cell state highway)
```

Neither is inherently an encoder or decoder. They're **sequence processors** that can be used for anything:

```
Same LSTM, different jobs:

  As classifier:    sentence → LSTM → h_t → "positive/negative"
  As encoder:       sentence → LSTM → h_t → pass to decoder
  As decoder:       h_t + previous word → LSTM → next word
  As language model: "The cat" → LSTM → "sat"
```

### Seq2Seq is the wiring diagram

Seq2Seq is an **architecture** — it takes two LSTMs (or RNNs) and gives each a job:

```
LSTM alone:
  Sentence in → LSTM → (h_t, c_t)     ...now what?
  Good for classification, next-word prediction.
  But input and output are the SAME language/domain.

Seq2Seq:
  English in → LSTM (encoder) → (h_t, c_t) → LSTM (decoder) → French out
  TWO LSTMs chained together. One reads, one writes.
  Input and output can be COMPLETELY different:
    English → French
    Question → Answer
    Code → Documentation
```

The LSTM doesn't know it's an encoder or decoder. Seq2Seq is what assigns the roles.

---

## Part 4: The Bottleneck — Why Attention Was Invented Immediately

Seq2Seq's bold claim: compress an entire sentence into **one fixed-size vector**.

```
"Hi"                                    → [1000 floats]
"The cat sat on the mat"                → [1000 floats]  (same size!)
"Despite the unfavorable weather        → [1000 floats]  (SAME SIZE!)
 conditions, the delegation proceeded
 to ratify the treaty amendments"
```

This is the **information bottleneck**. Short sentences? Fine. Long sentences? Information gets crushed.

Bahdanau's fix (also 2014): **attention**. Instead of forcing everything through one vector, let the decoder look back at every encoder hidden state:

```
Seq2Seq (no attention):
  Decoder sees:  [one final vector]
  "I forgot what the beginning said"

Seq2Seq + Attention:
  Decoder sees:  [one vector] + can peek at EVERY encoder hidden state
  "Let me check... word 3 is relevant to what I'm translating now"
```

Both papers came out the same year (2014) — that's how fast people realized the bottleneck was the weak link.

---

## Part 5: The Paradigm Shift

```
SMT (1990s–2014):
  - Probability tables from counting co-occurrences
  - No neurons, no weights, no backprop
  - Phrase tables: hundreds of millions of entries, 2+ GB
  - Dozens of hand-engineered components
  - "How often did phrase X appear as translation of phrase Y?"

Seq2Seq (2014):
  - Two LSTMs: one reads, one writes
  - Trained end-to-end with backprop
  - Model: ~1.5 GB (mostly vocabulary embeddings)
  - Zero hand-engineering
  - "What does this sentence MEAN? Now say it in French."
```

In November 2016, Google switched from SMT to GNMT (Google Neural Machine Translation — basically Seq2Seq + Attention, scaled up). Translation quality jumped so noticeably that users could tell the difference overnight.

---

## The One-Sentence Story

**For 20 years, translation was done by looking up phrases in giant probability tables; then Sutskever showed that two LSTMs wired together — one to read, one to write — could replace the entire pipeline, and attention fixed the only thing wrong with it.**

---

## Cross-References

| Topic | See |
|-------|-----|
| LSTM mechanics (gates, cell state, gradient highway) | Note 11 |
| Vanishing gradient (why RNNs fail, why LSTMs work) | Note 10, Part 8; Note 13 |
| Assignment Phase 2 implementation details | Note 12 (battle plan) |
| Full historical timeline | Note 00 |
