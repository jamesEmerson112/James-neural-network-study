# Timeline: Neural Networks → Sequence Models → Transformers & Beyond

> **This timeline is the table of contents for the entire course journey.**
> From Gödel proving math is incomplete to Claude answering your questions — 95 years of one idea building on the last.

## Era Files

| Era | File |
|-----|------|
| Foundations, Pioneers & AI Winter (1847-1985) | [00a](00a_timeline_1847-1985_foundations_pioneers_winter.md) |
| Resurrection & Deep Learning Revolution (1986-2017) | [00b](00b_timeline_1986-2017_resurrection_and_revolution.md) |
| The LLM Era (2018-Now) | [00c](00c_timeline_2018-now_llm_era.md) |
| Pre-1962 Visual Map | [00d](00d_timeline_pre1962_visual_map.md) |
| Study Guide — Deep Dives & Detailed Notes | [00e](00e_timeline_study_guide.md) |

## The Eras at a Glance

```
1931–1948   THE FOUNDATIONS      Gödel, Turing, McCulloch-Pitts, Shannon
1949–1962   THE PIONEERS         Hebb, Nash, Minsky (SNARC), Rosenblatt, Widrow & Hoff
1969–1985   THE AI WINTER        Minsky & Papert kill funding
1986–2008   THE RESURRECTION     Backpropagation, autoencoders, CNNs, LSTMs, CTC, deep belief nets
2012–2017   THE DEEP LEARNING    Word2Vec, GANs, Batch Norm, Attention, Transformers, PPO,
            REVOLUTION           bias & fairness awakening
2018–NOW    THE LLM ERA          ELMo → BERT → GPT, self-supervised learning, diffusion,
                                 scalable training, AI regulation — scale wins
```

## The One-Sentence Story

**Incompleteness (Gödel) → computation (Turing) → information as math (Shannon) → artificial neuron (McCulloch-Pitts) → learnable neuron (Perceptron) → gradient descent / adaptive filtering (ADALINE) → AI winter (Minsky & Papert) → multi-layer + backprop (MLP) → compress and reconstruct (autoencoders) → recurrence (RNN) → memory gates (LSTM) → deep networks revived (deep belief nets) → words as vectors (Word2Vec/GloVe) → learn by playing (RL: TD-Gammon → DQN → A3C → PPO) → attention → "attention is all you need" (Transformer) → context-aware embeddings (ELMo → BERT) → scale it up (GPT) → train it at scale (Batch Norm → mixed precision → TPUs → Megatron/DeepSpeed) → learn to generate (VAEs/GANs/diffusion) → learn without labels (SimCLR/DINO) → hear and speak (CTC → DeepSpeech → wav2vec → Whisper) → confront bias (COMPAS → Gender Shades → Model Cards → EU AI Act) → scale it WAY up (GPT-3/4) → align it with humans (RLHF) → make it multimodal & teach it to reason.**

## What Actually Changed in the 21st Century

The neural network **ideas** were all there by the 1980s–90s. What was missing:

1. **Compute** — GPUs didn't become available for training until the late 2000s
2. **Data** — the internet created massive labeled datasets (ImageNet, etc.)
3. **Scale** — earlier networks had dozens or thousands of neurons; modern ones have billions of parameters

The 21st century didn't invent neural networks. It gave them the **fuel** (data) and the **engine** (GPUs) to fulfill promises Rosenblatt made in 1957.

## The Evolution of Learning

```
Perceptron (1957):    "Wrong."              →  fixed-size nudge
ADALINE (1960):       "0.73 wrong."         →  proportional nudge (gradient descent)
Backprop (1986):      "Layer 3 is 0.73      →  proportional nudge for EVERY
                       wrong because          layer, traced back through
                       Layer 2 was 0.41       the whole network
                       wrong because
                       Layer 1 was 0.22
                       wrong."
```

**The irony: we teach machines intelligence by obsessively measuring their mistakes.**
