# Timeline: Neural Networks → Sequence Models → Transformers & Beyond

> **This timeline is the table of contents for the entire course journey.**
> From Gödel proving math is incomplete to Claude answering your questions — 95 years of one idea building on the last.

## The Foundations (1931–1943)

| Year | What | Who | Key Idea |
|------|------|-----|----------|
| 1931 | **Incompleteness theorems** | Kurt Gödel | Math cannot prove all truths about itself — there are limits to formal systems |
| 1847 | **Boolean algebra** | George Boole ("Laws of Thought") | All logical reasoning = AND/OR/NOT. Provably correct but too rigid for how humans think |
| 1936 | **Turing machine** | Alan Turing | Defined "what is computable" — abstract machine that can compute anything computable. The goalpost for everything after |
| 1943 | **McCulloch-Pitts neuron** | McCulloch & Pitts (18-yr-old logician) | Fused neuroscience + Boolean logic + Turing: networks of binary neurons are Turing-complete. Brain = computer. But can't learn — weights set by hand |

## The Pioneers (1949–1962)

| Year | What | Who | Key Idea |
|------|------|-----|----------|
| 1949 | **Hebbian learning** | Donald Hebb | "Neurons that fire together wire together" — first learning principle, but no algorithm |
| 1950 | **Turing Test** | Turing | Flipped the question: not "can machines compute?" but "can machines think?" Kicked off AI as a field |
| 1951 | **SNARC** | Minsky & Edmonds | First physical neural net machine — 40 neurons, solved mazes. Trained on reward (reinforcement), not error |
| 1956 | **"Artificial Intelligence" coined** | Dartmouth Workshop | McCarthy, Minsky, Shannon, Rochester — the field gets a name |
| 1957 | **Perceptron** | Rosenblatt | First *learnable* neuron — weights adjust from data. Error measured AFTER threshold (binary: right/wrong). `w ← w + η(y - ŷ) · x` — the seed of all training |
| 1959 | **"Machine learning" coined** | Arthur Samuel (IBM) | Checkers program that learned from self-play (built since 1952). The 1959 paper is where the term "machine learning" first appears |
| 1959 | **Visual cortex mapping** | Hubel & Wiesel | Discovered receptive fields in cat visual cortex that detect edges/orientations (simple/complex cell taxonomy formalized 1962). Nobel Prize 1981. Directly inspired CNNs decades later |
| 1960 | **ADALINE** | Widrow & Hoff (Stanford) | ADAptive LInear NEuron — error measured BEFORE threshold (continuous: "how far off?"). Introduced LMS (Least Mean Squares) — first use of gradient descent in a neural network. Essentially an adaptive noise filter; lasting use in telephone echo cancellation. Ted Hoff later co-invented the Intel 4004 microprocessor |
| 1960 | **Backprop precursor** | Henry Kelley | First derivation of backpropagation-like gradient computation, but in control theory (optimal flight paths) — not applied to neural networks |
| 1960 | **MINOS I** | Stanford Research Institute (Rosen & Brain) | Neural network machine for pattern classification — classifying symbols on army maps. MINOS II (1963) scaled to 6,600 adjustable weights |
| 1962 | **Samuel's Checkers** | Arthur Samuel (IBM) | Checkers program beat a former Connecticut champion — one of the first demos of ML winning a real game. Started in 1950s on IBM 701; demoed on live TV in 1956. Thomas Watson Sr. predicted a 15-point IBM stock rise from the demo — and it happened. One of the first times AI was used to sell stock |
| 1962 | **MADALINE** | Widrow & Hoff | Many ADALINEs — one of the first multi-layer neural networks. Used MADALINE Rule I: each ADALINE trained independently with LMS, but hidden layers kept fixed (no way to train them yet). The missing piece was backprop — 24 years away |
| 1962 | ***Principles of Neurodynamics*** | Rosenblatt | 616-page treatise — the theoretical framework for perceptrons. **Coined the term "back-propagating errors"** but didn't know how to implement it. Named the solution 24 years before Rumelhart/Hinton/Williams made it work |

## The AI Winter (1969–1985)

| Year | What | Who | Key Idea |
|------|------|-----|----------|
| 1965 | **GMDH** | Ivakhnenko & Lapa | Group Method of Data Handling — sometimes called the first deep learning. Trained networks layer by layer. By 1971 had 8-layer networks. Largely forgotten in the West |
| 1966 | **ELIZA** | Weizenbaum (MIT) | First chatbot — pattern-matched text to simulate a therapist. No learning at all, but fooled people. Early proof that "seeming intelligent" ≠ "being intelligent" |
| 1969 | **Apollo 11 Moon landing** | NASA | Humanity lands on the Moon — proof that massive government R&D investment works. But funding pivots to space; AI loses priority. The same year neural nets get killed below |
| 1969 | **Perceptron limitations** | Minsky & Papert | Proved single-layer perceptrons can't learn XOR. Triggered ~15 years of reduced funding. McCulloch & Pitts both died the same year |
| 1971 | Rosenblatt dies | — | Drowned in sailing accident on his 43rd birthday. Never saw backprop vindicate his ideas |
| 1974 | **Backprop for neural nets** | Paul Werbos (Harvard) | PhD thesis: first to explicitly apply the chain rule backward through layers to train neural networks. The right answer, 12 years too early — almost nobody read it |

## The Resurrection (1986–1997)

| Year | What | Who | Key Idea |
|------|------|-----|----------|
| 1986 | **MLP + Backprop** | Rumelhart, Hinton, Williams | "Learning representations by back-propagating errors" in *Nature* 323. Multi-layer networks trained with backpropagation. Solved XOR. Neural nets revived |
| 1986 | **PDP Book** | Rumelhart, McClelland (eds.) | *Parallel Distributed Processing* (MIT Press, 2 vols). Chapter 8 = full backprop treatment. Final section introduces training recurrent nets via unrolling. 30,000+ citations |
| 1986 | **Jordan Network** | Michael Jordan (UCSD) | "Serial Order" — first RNN variant. Output fed back to context units. Network remembers what it *said* |
| 1986 | **RNN** | Rumelhart, Hinton, Williams | Recurrent connections described in PDP Ch. 8 — share weights across time, feed hidden state back. Networks process sequences with memory |
| 1989 | **CNN** | Yann LeCun | Convolutional neural networks — local filters for image recognition |
| 1990 | **Elman Network** | Jeffrey Elman (UCSD) | "Finding Structure in Time" — hidden state (not output) fed back to context units. Became the standard "vanilla RNN" architecture |
| 1990 | **BPTT formalized** | Paul Werbos | "Backpropagation Through Time: What It Does and How to Do It" — unroll the RNN, run standard backprop on the unrolled graph |
| 1991 | Vanishing gradient proved | Hochreiter | Diploma thesis *"Untersuchungen zu dynamischen neuronalen Netzen"* (in German, TU Munich, advisor: Schmidhuber). Proved gradients decay exponentially through time steps |
| 1994 | Vanishing gradient confirmed | Bengio, Simard, Frasconi | "Learning Long-Term Dependencies with Gradient Descent is Difficult" — confirmed Hochreiter's result in English, widely read |
| 1997 | **LSTM** | Hochreiter & Schmidhuber | "Long Short-Term Memory" in *Neural Computation* 9(8). Cell state ("constant error carousel") + input/output gates solve vanishing gradients. Original had NO forget gate **(Assignment Phase 1)** |
| 2000 | **Forget gate added to LSTM** | Gers, Schmidhuber, Cummins | "Learning to Forget: Continual Prediction with LSTM" — without forget gate, cell state grows forever. This completed the modern LSTM architecture used today |

## The Deep Learning Revolution (2012–2017)

| Year | What | Who | Key Idea |
|------|------|-----|----------|
| 2012 | **AlexNet** | Krizhevsky, Sutskever, Hinton | CNN wins ImageNet by a landslide — deep learning revolution begins. GPUs + data + scale |
| 2014 | **GANs** | Goodfellow | Generator vs discriminator — adversarial training for image generation |
| 2014 | **Seq2Seq** | Sutskever, Vinyals, Le | Encoder → fixed vector → Decoder. First serious neural machine translation |
| 2014 | **Attention** | Bahdanau, Cho, Bengio | Decoder "looks back" at all encoder states instead of one fixed vector **(Assignment Phase 2)** |
| 2014 | GRU | Cho et al. | Simplified LSTM — 2 gates instead of 3, no separate cell state |
| 2015 | Attention variants | Luong et al. | Different scoring: dot product, cosine similarity, etc. |
| 2017 | **Transformer** | Vaswani et al. (Google) | "Attention Is All You Need" — drop recurrence. Self-attention over all positions in parallel **(Assignment Phases 3 & 4)** |

## The LLM Era (2018–Now)

| Year | What | Who | Key Idea |
|------|------|-----|----------|
| 2018 | **BERT** | Devlin et al. (Google) | Encoder-only transformer, pretrained with masked language modeling. `TransformerTranslator` is this style |
| 2018 | **GPT-1** | Radford et al. (OpenAI) | Decoder-only transformer, autoregressive pretraining on BookCorpus |
| 2019 | **GPT-2** | OpenAI | 1.5B params. Showed scaling up decoder-only transformers yields strong zero-shot performance |
| 2019 | **T5** | Google | "Text-to-Text Transfer Transformer" — frames every NLP task as text-in, text-out |
| 2020 | **GPT-3** | OpenAI | 175B params. In-context learning — few-shot via prompting, no fine-tuning needed |
| 2020 | **Vision Transformer (ViT)** | Dosovitskiy et al. (Google) | Applied transformers to images by splitting into patches. Transformers escape NLP |
| 2021 | **DALL-E** | OpenAI | Transformer generates images from text descriptions |
| 2022 | **ChatGPT / InstructGPT** | OpenAI | RLHF (reinforcement learning from human feedback) aligns GPT-3.5 to follow instructions |
| 2022 | **Chinchilla** | Hoffmann et al. (DeepMind) | Scaling laws — models were over-parameterized & under-trained. Train longer on more data |
| 2023 | **GPT-4** | OpenAI | Multimodal (text + vision), massive leap in reasoning |
| 2023 | **LLaMA** | Meta | Open-weight LLMs. Kicked off the open-source LLM explosion |
| 2023 | **Mixture of Experts (MoE)** | Mistral, Google | Only activate a subset of parameters per token — scale without scaling compute linearly |
| 2024 | **State-space models (Mamba)** | Gu & Dao | Alternative to attention — linear-time sequence modeling, challenging transformer dominance for long sequences |
| 2024-25 | **Reasoning models** | OpenAI (o1/o3), Anthropic (Claude), DeepSeek (R1) | Chain-of-thought at inference time. Models "think" before answering |

## World History ←→ Neural Network History (Pre-1962 Visual Map)

```
══════════════════════════════════════════════════════════════════════════════════════════
                    WORLD HISTORY  ←――――→  NEURAL NETWORK HISTORY
                         before 1962: how the world built the brain machine
══════════════════════════════════════════════════════════════════════════════════════════

1847 ·····································│···········································
                                          │    ┌───────────────────────────────┐
                                          │    │  BOOLEAN ALGEBRA (Boole)      │
                                          │    │  Logic = AND / OR / NOT       │
                                          │    └──────────────┬────────────────┘
                                          │                   │
              ~ 84 years pass ~           │         "logic can be mechanized"
                                          │                   │
1931 ·····································│···················│·····················
                                          │    ┌──────────────▼────────────────┐
                                          │    │  GÖDEL'S INCOMPLETENESS       │
                                          │    │  "Math has limits — some      │
                                          │    │   truths can't be proven"     │
                                          │    └──────────────┬────────────────┘
                                          │                   │ provokes: what CAN
                                          │                   │ be computed then?
1936 ·····································│···················│·····················
                                          │    ┌──────────────▼────────────────┐
                                          │    │  TURING MACHINE               │
                                          │    │  Defines computation itself   │
                                          │    └──────────────┬────────────────┘
                                          │                   │
1939 ┌────────────────────────────┐       │                   │
     │  WORLD WAR II BEGINS      │       │                   │
     │                            │       │                   │
     │  Turing breaks Enigma ─────────────────────────────────┤ computation
     │  at Bletchley Park         │       │                   │ becomes
     │                            │       │                   │ practical
     │  von Neumann: Manhattan ───────────────────┐           │
     │  Project + computing       │       │       │           │
     │                            │       │       │           │
1943 │                            │       │    ┌──▼───────────▼────────────────┐
     │                            │       │    │  McCULLOCH-PITTS NEURON      │
     │                   neuroscience ─────────│  Boolean logic + brain model  │
     │                     (how neurons   │    │  + Turing = Turing-complete   │
     │                      actually work)│    │  CAN'T LEARN (weights by hand)│
1945 │  WWII ENDS                 │       │    └──────────────┬────────────────┘
     │  ┌──────────────────────┐  │       │                   │
     │  │ ENIAC (1945)         │◄─┘       │                   │
     │  │ von Neumann arch.    │          │                   │
     │  │ = stored-program     │──── hardware substrate ──────┤
     │  │   computer           │          │                   │
     │  └──────────────────────┘          │                   │
     └────────────────────────────┘       │                   │
                                          │                   │
1947 ┌────────────────────────────┐       │                   │
     │  TRANSISTOR (Bell Labs)   │── smaller, faster ─────────┤
     │  replaces vacuum tubes    │   computers possible       │
     └────────────────────────────┘       │                   │
                                          │                   │
1949 ┌────────────────────────────┐       │    ┌──────────────▼────────────────┐
     │  NATO FOUNDED             │       │    │  HEBBIAN LEARNING (Hebb)      │
     │  Cold War → R&D money ─────────────────│  "Fire together, wire together"│
     │  starts flowing           │       │    │  First learning PRINCIPLE      │
     └────────────────────────────┘       │    │  (but no algorithm)           │
                                          │    └──────────────┬────────────────┘
                                          │                   │
1950 ┌────────────────────────────┐       │    ┌──────────────▼────────────────┐
     │  KOREAN WAR               │       │    │  TURING TEST                  │
     │  military wants machine ───────────────│  "Can machines THINK?"        │
     │  intelligence             │       │    │  Reframes the whole question   │
     └────────────────────────────┘       │    └──────────────┬────────────────┘
                                          │                   │
1951 ·····································│····┌──────────────▼────────────────┐
                                          │    │  SNARC (Minsky & Edmonds)    │
                                          │    │  40 hardware neurons          │
                                          │    │  Solved mazes via reward      │
                                          │    └──────────────┬────────────────┘
                                          │                   │
1952 ┌────────────────────────────┐       │                   │
     │  H-BOMB TEST              │       │                   │
     │  (von Neumann involved)   │       │                   │
     └────────────────────────────┘       │                   │
                                          │                   │
1953 ┌────────────────────────────┐       │                   │
     │  DNA DISCOVERED           │       │                   │
     │  (Watson & Crick)         │       │                   │
     │  biology = information    │       │                   │
     └────────────────────────────┘       │                   │
                                          │                   │
1956 ┌────────────────────────────┐       │    ┌──────────────▼────────────────┐
     │  EISENHOWER ERA           │       │    │  DARTMOUTH WORKSHOP           │
     │  Peak Cold War R&D ────────────────────│  "Artificial Intelligence"    │
     │  Rockefeller $$  ─────────────────────→│   coined — field gets a name  │
     │  ONR / Navy funding ──────────┐    │    │  McCarthy, Minsky, Shannon   │
     └────────────────────────────┘  │    │    └──────────────┬────────────────┘
                                     │    │                   │
1957 ┌────────────────────────────┐  │    │    ┌──────────────▼────────────────┐
     │  ★ SPUTNIK LAUNCHED ★     │  │    │    │  ★ PERCEPTRON (Rosenblatt) ★ │
     │  Space Race begins        │  │    │    │  First LEARNABLE neuron!      │
     │  USA panics → floods $$ ──────┴────────│  w ← w + η(y-ŷ)·x           │
     │  into science & defense   │       │    │  Navy-funded at Cornell       │
     └────────────────────────────┘       │    └────┬─────────┬───────────────┘
                                          │         │         │
1958 ┌────────────────────────────┐       │         │         │
     │  NASA FOUNDED             │       │         │         │
     │  DARPA FOUNDED ─── more AI funding─┘         │         │
     └────────────────────────────┘       │         │         │
                                          │         │         │
1959 ·····································│·········│·········│·····················
                                          │    ┌────▼─────────│───────────────┐
                                          │    │ "MACHINE LEARNING" COINED    │
                                          │    │  (Samuel, IBM — checkers)    │
                                          │    └──────────────│───────────────┘
                                          │                   │
                                          │    ┌──────────────│───────────────┐
                              neuroscience─────│ VISUAL CORTEX MAPPING        │
                                          │    │ (Hubel & Wiesel) — cat       │
                                          │    │ receptive fields → CNNs later│
                                          │    └──────────────│───────────────┘
                                          │                   │
1960 ┌────────────────────────────┐       │    ┌──────────────▼───────────────┐
     │  JFK ELECTED              │       │    │  ADALINE (Widrow & Hoff)     │
     │                            │       │    │  Perceptron → but CONTINUOUS │
     │  U-2 SPY PLANE shot down  │       │    │  error! "How far off?" not   │
     │  (Cold War espionage)     │       │    │  just "wrong." FIRST gradient│
     │                            │       │    │  descent in a neural network │
     └────────────────────────────┘       │    └────┬────────────────────────┘
                                          │         │
     ┌────────────────────────────┐       │    ┌────▼────────────────────────┐
     │  AEROSPACE / CONTROL      │       │    │  BACKPROP PRECURSOR (Kelley)│
     │  THEORY (flight paths) ────────────────│  Same math, but for rockets │
     │  Military-funded          │       │    │  not neurons — 26 yr wait   │
     └────────────────────────────┘       │    └─────────────────────────────┘
                                          │         │
     ┌────────────────────────────┐       │    ┌────▼────────────────────────┐
     │  ARMY needs map reading ───────────────│  MINOS I (SRI)              │
     └────────────────────────────┘       │    │  Classify army map symbols  │
                                          │    └─────────────────────────────┘
                                          │         │
1961 ┌────────────────────────────┐       │         │
     │  GAGARIN IN SPACE         │       │         │
     │  BAY OF PIGS              │       │         │
     └────────────────────────────┘       │         │
                                          │         │
1962 ┌────────────────────────────┐       │    ┌────▼────────────────────────┐
     │  ★ CUBAN MISSILE CRISIS ★ │       │    │  MADALINE (Widrow & Hoff)  │
     │                            │       │    │  Many ADALINEs = first      │
     │  "world nearly ends"      │       │    │  multi-layer network        │
     │                            │       │    │  BUT: can't train hidden    │
     │  meanwhile, quietly...    │       │    │  layers. Missing piece =    │
     │                            │       │    │  backprop (24 years away)   │
     └────────────────────────────┘       │    └─────────────┬───────────────┘
                                          │                  │
                                          │    ┌─────────────▼───────────────┐
                                          │    │  SAMUEL'S CHECKERS          │
                                          │    │  Beat human on LIVE TV      │
                                          │    │  IBM stock rose 15 points   │
                                          │    │  First time AI sold stock   │
                                          │    └─────────────┬───────────────┘
                                          │                  │
                                          │    ┌─────────────▼───────────────┐
                                          │    │  PRINCIPLES OF              │
                                          │    │  NEURODYNAMICS (Rosenblatt) │
                                          │    │  616 pages. Coins "back-    │
                                          │    │  propagating errors" —      │
                                          │    │  names it 24 years before   │
                                          │    │  anyone makes it work       │
                                          │    └─────────────────────────────┘
                                          │
══════════════════════════════════════════════════════════════════════════════════════════


THE THREE RIVERS THAT BUILT NEURAL NETWORKS:

    MATH/LOGIC                    NEUROSCIENCE                  GEOPOLITICS
    ──────────                    ────────────                  ───────────
    Boole (1847)                  Brain anatomy                 WWII
         ↓                             ↓                          ↓
    Gödel (1931)                  McCulloch-Pitts ←────── Turing's wartime
         ↓                        neuron (1943)            codebreaking
    Turing (1936)                      ↓                          ↓
         ↓                        Hebb (1949) ←──────── Cold War R&D $$$
         ↓                             ↓                          ↓
         └──────────→ Perceptron (1957) ←──────────── Sputnik panic $$$
                           ↓                           Navy funding
                      ADALINE (1960)                          ↓
                           ↓                           DARPA founded
                      MADALINE (1962)                         ↓
                           ↓                           Army → MINOS I
                      "back-propagating errors"
                       named but unsolved
                           ↓
                     ┌─────────────┐
                     │  24 YEARS   │
                     │  OF WAITING │
                     └──────┬──────┘
                            ↓
                      Backprop (1986)


KEY INSIGHT: Every neural network breakthrough before 1962 was funded by
the military or enabled by wartime computing. Sputnik (1957) and the
Perceptron (1957) are the same year — not a coincidence. The Cold War
built the brain machine.
```

## The Eras at a Glance

```
1931–1943   THE FOUNDATIONS      Gödel, Turing, McCulloch-Pitts
1949–1962   THE PIONEERS         Hebb, Minsky (SNARC), Rosenblatt, Widrow & Hoff
1969–1985   THE AI WINTER        Minsky & Papert kill funding
1986–1997   THE RESURRECTION     Backpropagation, CNNs, LSTMs
2012–2017   THE DEEP LEARNING    AlexNet, GANs, Attention, Transformers
            REVOLUTION
2018–NOW    THE LLM ERA          BERT, GPT, Claude — scale wins
```

## The One-Sentence Story

**Incompleteness (Gödel) → computation (Turing) → artificial neuron (McCulloch-Pitts) → learnable neuron (Perceptron) → gradient descent / adaptive filtering (ADALINE) → AI winter (Minsky & Papert) → multi-layer + backprop (MLP) → recurrence (RNN) → memory gates (LSTM) → attention → "attention is all you need" (Transformer) → scale it up (GPT/BERT) → scale it WAY up (GPT-3/4) → make it multimodal & teach it to reason.**

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

---

## Study Guide — Deep Dives & Detailed Notes

All notes in this repository, organized by narrative arc.

### Foundations & Historical Figures

- [Kurt Gödel — Incompleteness & the Limits of Formal Systems](godel/00_overview.md)
- [Alan Turing — Computation, Codebreaking & Morphogenesis](turing/00_overview.md)
- [John von Neumann — The Bridge Between Brains and Machines](von_neumann/00_overview.md)
- [McCulloch-Pitts & the 1943 Scene](01_mcculloch_pitts_and_the_1943_scene.md) — the neuron that fused Boolean logic, neuroscience, and Turing

### Pioneers & AI Winter

- [Rosenblatt and the Perceptron](02_rosenblatt_and_the_perceptron.md) — the first learnable neuron
- [ADALINE as Noise Filter](03_adaline_as_noise_filter.md) — continuous error and the birth of gradient descent
- [John McCarthy](04_john_mccarthy.md) — the man who named AI and invented LISP
- [Why LISP Matters](05_lisp_why_it_matters.md) — code as data, garbage collection, and AI's first language
- [LISP Deep Dive](05b_lisp_deep_dive.md) — technical deep dive into LISP internals
- [Minsky, Perceptrons — Death and Resurrection](06_minsky_perceptrons_death_and_resurrection.md) — how one book killed neural nets for 15 years

### Backprop, RNNs & LSTMs

- [Backprop: The Wiggle Ratio](09_backprop_the_wiggle_ratio.md) — chain rule through layers, intuitively
- [MLP, Backprop, and the Birth of RNNs](10_mlp_backprop_and_the_birth_of_rnns.md) — multi-layer networks learn
- [Vanishing Gradient and tanh](13_vanishing_gradient_and_tanh.md) — why deep nets forget
- [LSTM vs Seq2Seq vs Transformer](07_lstm_vs_seq2seq_vs_transformer.md) — three architectures compared
- [Phase 1: Naive LSTM](08_phase1_naive_lstm.md) — building LSTM from scratch
- [LSTM: The Memory Machine](11_lstm_the_memory_machine.md) — gates, cell state, and constant error carousel
- [NN LSTM Inputs and Outputs](15_nn_lstm_inputs_and_outputs.md) — practical tensor shapes and data flow

### Seq2Seq, Attention & Transformers

- [From Phrase Tables to Seq2Seq](14_from_phrase_tables_to_seq2seq.md) — machine translation's evolution
- [Assignment 3 Battle Plan](12_assignment3_battle_plan.md) — hands-on implementation strategy
- [Transformers](19_transformers.md) — encoder/decoder, self-attention, positional encoding
- [NLP Models Overview](18_nlp_models_overview.md) — ELMo → BERT → GPT word representation evolution
- [BERT](20_bert.md) — masked language modeling, encoder-only Transformer

### Vision

- [CNN Vision Tasks](17_cnn_vision_tasks.md) — convolutional neural networks for image understanding
- [Do Vision Transformers See Like CNNs?](16_do_vision_transformers_see_like_cnns.md) — ViT vs CNN comparison

### LLM Inference & Benchmarks

- [LLM Inference Stack](llm-inference-stack/) — how LLMs actually run in production
- [LLM Benchmarks](21_llm_benchmarks.md) — benchmark descriptions and what they measure

### Programming Languages

- [OCaml vs Haskell vs Erlang](programming-languages/ocaml-vs-haskell-vs-erlang.md) — three functional programming traditions compared. Connected to the LISP lineage: [McCarthy](04_john_mccarthy.md) → [Why LISP Matters](05_lisp_why_it_matters.md) → Lambda Calculus → ML → OCaml/Haskell
