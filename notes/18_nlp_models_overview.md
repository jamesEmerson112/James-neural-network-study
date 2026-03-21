# NLP Models Overview

## Evolution of Word Representations

1. **Word2Vec / GloVe** — static embeddings (each word gets one fixed vector regardless of context)
2. **ELMo** — contextualized embeddings via bidirectional LSTMs
3. **BERT** — contextualized embeddings via Transformer encoder
4. **GPT family** — Transformer decoder (autoregressive, left-to-right)

---

## ELMo (Embeddings from Language Models) — 2018, AllenNLP

- Uses **bidirectional LSTMs** (two separate LSTMs: left-to-right + right-to-left, then concatenated)
- "Shallow" bidirectionality — each direction trained independently, combined after
- Produces contextualized word embeddings used as **input features** to other models
- Not fine-tuned end-to-end; it's a feature extractor

## BERT (Bidirectional Encoder Representations from Transformers) — 2018, Google

- Uses the **Transformer encoder** with self-attention (NO LSTMs)
- **Deeply bidirectional** — masked language modeling (MLM) lets every layer see both directions simultaneously
- Designed to be **fine-tuned** end-to-end on downstream tasks
- Larger and more powerful than ELMo, not a smaller version of it

### Key distinction: BERT is NOT a small version of ELMo
- Different architecture (Transformer vs LSTM)
- Different training approach (MLM vs language modeling)
- Different usage pattern (fine-tuning vs feature extraction)

---

## Prerequisites for Understanding BERT

### Need to know well:
- Basic neural network concepts (layers, activations, loss, backpropagation)
- Word embeddings (Word2Vec/GloVe — words as vectors)
- **Attention mechanism** — the critical prerequisite
- Encoder-decoder concept (BERT uses only the encoder)

### Helpful but not required:
- RNN — processes sequences one token at a time (slow, struggles with long sequences)
- LSTM — solved RNN's vanishing gradient problem (remembers longer context, but still sequential)
- Understanding RNN/LSTM limitations helps appreciate *why* Transformers were invented

### Learning path:
Skim RNN/LSTM conceptually → focus on **attention and Transformers** → then BERT
