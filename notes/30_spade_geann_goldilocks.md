# SPADE, GEANN, and Goldilocks — Vincent's Amazon Forecasting Trilogy

> Part of the [Neural Network Study Timeline](00_timeline.md). See also: [Vincent Quenneville-Belair](researchers/vincent_quenneville_belair.md), [Scaled Dot-Product Attention](23_scaled_dot_product_attention.md), [Reinforcement Learning](25_reinforcement_learning_overview.md).

Three papers from Vincent Quenneville-Belair's Amazon supply chain forecasting work (2023-2025). Each solves a different failure mode of neural forecasters, and together they form a coherent story about what goes wrong when you try to predict demand at Amazon scale.

---

## The Setup: Why Demand Forecasting Is Hard

Amazon sells hundreds of millions of products. For each one, they need to predict: **how many units will sell, at what time, in what region?**

```
Why this is harder than it sounds:

  Product A: sold 50 units/day for 6 months
             → Prime Day hits → sells 5,000 in one day
             → next day: back to 50
             → model thinks "demand is rising!" — overorders for weeks

  Product B: brand new listing, zero sales history
             → how many to stock? No data to learn from

  Product C: was selling well, then went out-of-stock for 3 months
             → recent sales = all zeros
             → but demand wasn't zero — SUPPLY was zero
             → model sees zeros and predicts zero
```

Standard neural forecasters (LSTMs, Transformers, DeepAR) struggle with all three. Vincent's papers attack each one.

---

## Paper 1: GEANN — Cold-Start Forecasting via Graphs (KDD 2023)

**Full title:** *Scalable Graph Augmentations for Multi-Horizon Time Series Forecasting*
**Authors:** Sitan Yang, Malcolm Wolff, Shankar Ramasubramanian, Vincent Quenneville-Belair, Ronak Mehta, Michael W. Mahoney
**arXiv:** https://arxiv.org/abs/2307.03595

### The Cold-Start Problem

A new product has **zero sales history**. An out-of-stock (OOS) product has corrupted history (zeros that represent supply failure, not demand). Traditional time series models need historical data — no data, no forecast.

```
Standard forecasting (works well):

  Product A:  [223, 198, 245, 210, 231, ...]  →  predict next 52 weeks
              Rich historical signal


Cold-start forecasting (fails):

  New product: [0, 0, 0, 0, 0, ...]  →  predict next 52 weeks
               No signal at all

  OOS product: [245, 210, 0, 0, 0, ...]  →  predict next 52 weeks
               Corrupted signal (supply = 0, not demand = 0)
```

### The Insight: Products Are Not Islands

A new bluetooth speaker might have zero history, but there are hundreds of *similar* bluetooth speakers with rich history. If you can figure out which products are "neighbors," you can borrow their demand patterns.

### Two Types of Graphs

GEANN constructs **two** separate product graphs, each capturing different kinds of similarity:

```
Graph 1: Browse Node Graph (domain knowledge)
──────────────────────────────────────────────
  Amazon organizes products into a category tree:
    Electronics > Computers > Mice > Wireless Mice

  Products sharing a browse node are connected.
  Edge weights = co-browsing frequency
    (how often customers browse both products in one session)

  Property: STABLE. The catalog hierarchy is deterministic.
  A new wireless mouse is immediately connected to existing
  wireless mice — even before a single sale.


Graph 2: k-NN Embedding Graph (data-driven)
────────────────────────────────────────────
  1. Train a base MQ-CNN forecaster on all products
  2. Extract internal embeddings for each product
  3. Compute similarity: |Corr(embedding_i, embedding_j)|
  4. Connect each product to its k=10 nearest neighbors

  Property: UNSTABLE. Different training runs produce
  different embeddings → different neighbors.
  ~50% of neighbors change across training runs.
```

**Key finding the paper discovers:** At scale (2M products), the browse node graph consistently wins. The k-NN graph actually *hurts* performance because the learned similarities are too noisy. **Structured domain knowledge beats data-driven graphs at industrial scale.**

### The Architecture: Graph Ensemble Module (GEM)

GEANN is **not a standalone model**. It's an augmentation module that plugs into an existing Seq2Seq forecaster (Amazon's MQ-CNN — Multi-Quantile Convolutional Neural Network).

```
Architecture — what changes vs. base MQ-CNN:

  Step 1: Temporal Encoding (same as before)
          Dilated causal convolutions process each product's
          time series into a fixed-size embedding

  Step 2: Static Encoding (same as before)
          Category, brand, price tier → separate embedding

  Step 3: Graph Ensemble Module (NEW)
          Takes temporal encodings of ALL products
          and passes them through GNNs:

          For each graph (browse_node, k-NN):
              g = GNN(all_temporal_encodings, graph)

          Weighted ensemble (weights learned during training):
              g_final = w_browse × g_browse + w_kNN × g_kNN

          Each GCN layer:
              H^(l+1) = activation(A_normalized · H^(l) · W^(l))

  Step 4: Decoding
          Input = concat(temporal_encoding, graph_augmentation, static_encoding)
          Output = multi-horizon quantile forecasts (P50, P90 for next 52 weeks)


  Additional parameters: ~50-65K added to an 850K base model
  Training: end-to-end (graph module trains with forecaster)
```

For a cold-start product with a weak/empty encoding, the GCN replaces its signal with aggregated signal from neighbors that DO have history.

### Why "Scalable" Matters

With 2 million nodes and k=10 neighbors each, you can't load the full graph into GPU memory. GEANN uses **L-hop subgraph sampling**:

```
Scalability via subgraph sampling:

  Full graph: 2,000,000 nodes
  Mini-batch: m seed nodes

  For L=2 GCN layers, extract only the local subgraph:
    Layer 2 needs → seed nodes' neighbors
    Layer 1 needs → those neighbors' neighbors
    Total: at most m × (1 + k + k²) = m × 111 nodes

  Critical property: GNN outputs for seed nodes are IDENTICAL
  whether computed on the subgraph or full graph.
  No approximation. Exact computation, just localized.

  Training time: ~30 min/epoch (single graph), ~1 hr/epoch (dual)
  on 8 NVIDIA V100 GPUs
```

### Results

| Scenario | P50 Improvement | P90 Improvement | Overall |
|---|---|---|---|
| Full catalog (100K products) | ~2.4% | ~2.4% | 0.976 (vs 1.000 baseline) |
| Full catalog (2M products) | ~1% | ~1% | 0.991 |
| **Newly launched products** | 2.5% | **9.0%** | **0.943** |
| **Recently out-of-stock** | 2.6% | 4.3% | **0.965** |

The improvement on cold-start products is **3-6x larger** than the full catalog improvement. The **9% P90 improvement for new products** is the standout — P90 directly drives safety stock decisions, so this means substantially better inventory for new launches.

### Connection to Embeddings and Transfer Learning

```
word2vec analogy:

  Rare word → meaning from surrounding words in corpus
  New product → demand pattern from neighboring products in catalog graph

  Both solve: "I've never seen this thing before,
               but I know what it's NEAR."


Transfer learning analogy:

  Classical:   Pre-train BERT on corpus → Fine-tune on task
  GEANN:       Products WITH history → (via graph) → Products WITHOUT history

  Unlike classical transfer, there's no separate pre-training phase.
  The GNN learns to transfer demand end-to-end.


RAG analogy:

  RAG:    LLM lacks knowledge → retrieve relevant documents → augment input
  GEANN:  Forecaster lacks history → retrieve similar products via graph → augment encoding

  Both solve: "My internal knowledge isn't enough, so I'll
               pull in relevant external information."
```

---

## Paper 2: SPADE — Handling Peak Events (NeurIPS Workshop 2024)

**Full title:** *Split Peak Attention DEcomposition*
**Authors:** Malcolm Wolff, Kin G. Olivares, Boris Oreshkin, Sunny Ruan, Sitan Yang, Abhinav Katoch, Shankar Ramasubramanian, Youxin Zhang, Michael W. Mahoney, Dmitry Efimov, Vincent Quenneville-Belair
**arXiv:** https://arxiv.org/abs/2411.05852

### The Peak Problem

Prime Day, Black Friday, Lightning Deals — these events cause massive demand spikes that distort everything around them.

```
Normal demand pattern:

  units
  100 |
   80 |     ___________          ___________
   60 |    /           \        /           \
   40 |___/             \______/             \___
      |________________________________________________ time


Same product on Prime Day:

  units
  5000|         *
  4000|        ***
  3000|       *****
  2000|      *******
  1000|     *********
   100|____***********_______________________________ time
            ^ ONE DAY

  The spike is 50x normal. The model has to learn:
  1. The spike is coming (peak detection)
  2. How big it will be (peak magnitude)
  3. That it will END (post-peak recovery)

  #3 is the hardest. Models see the spike and think "trend is up."
```

### The Fundamental Tension: Why You Can't Just "Train Harder"

The paper compares two production baselines that reveal a dilemma:

```
MQCNN (Multi-Quantile Convolutional Neural Network):
  - Dilated causal convolutions encode demand history
  - Cannot distinguish peak-driven demand from baseline demand
  - Result: UNDER-predicts during peaks
  - Paradoxically, post-peak accuracy is OK (it never lifted the forecast)
  - Trade-off: bad PE accuracy, decent PPE accuracy

MQT (Multi-Quantile Transformer):
  - Self-attention over demand history
  - Successfully captures peak patterns (attends to recent spikes)
  - But this is EXACTLY the problem: attention-weighted embeddings
    encode the elevated demand level, and that signal BLEEDS
    into post-peak forecasts
  - Trade-off: good PE accuracy, BAD PPE accuracy


THE DILEMMA:

  If you learn to predict peaks well  →  your embeddings carry peak
                                          signal into post-peak period
  If you ignore peaks                 →  post-peak is fine, but peak
                                          accuracy suffers

  You can't have both with a single pathway.
```

This isn't a model capacity issue. At Amazon's scale, both CNN-based and Transformer-based architectures exhibit this carry-over effect. The inductive bias of processing peaks and non-peaks through the **same encoder pathway** is too strong for data alone to overcome.

### The SPADE Solution: Two Parallel Pathways

SPADE resolves the dilemma by refusing to treat it as a single task.

```
Final Forecast = BaselineForecast + PeakAdjustment
```

Three input categories feed the architecture:
1. **Past time series (x^p):** Historical demand observations
2. **Static product info (x^s):** Category, brand, etc.
3. **Known future info (x^f):** Promotion schedules, holiday calendars — retailers decide these in advance

### Pathway 1: RobustConvolution (Baseline)

This is where SPADE prevents carry-over. Two sub-components:

**Forward-Fill Operation** — the paper's most elegant move:

```
Before demand enters the convolutional encoder, SPADE uses
the peak mask to REMOVE all peak demand values and replace
them with the most recent non-peak value:

  If m_t = 1 (peak period):
      x̃_t = x_{t*}   where t* = most recent non-peak time step

  Example:
    Raw:        [50, 52, 48, 5000, 4800, 51, 49, ...]
    Peak mask:  [ 0,  0,  0,    1,    1,  0,  0, ...]
    Forward-fill: [50, 52, 48,   48,   48, 51, 49, ...]
                                ^^^^  ^^^^
                        last non-peak value fills in

Rather than trying to TEACH the encoder to ignore peaks,
SPADE simply removes peaks from the encoder's input entirely.
The convolutions never see elevated demand.
They CANNOT carry it over.
```

**Dilated Convolutions** — standard dilated causal convolutions (same as MQ-CNN). But now operating on a history where peaks have been surgically removed.

### Pathway 2: PeakAttention

A specialized attention module for modeling **only** the peak incremental demand.

```
How PeakAttention differs from standard Transformer attention:

  Standard:   Attention = softmax(Q · K^T / √d) · V
                          ↑ attends to ALL positions

  SPADE:      Attention = softmax(q · k^T × m) · V
                                         ↑ mask applied INSIDE softmax
                          ↑ can ONLY attend to historical peak events

  The mask m zeros out all non-peak positions before the softmax.
  The model literally cannot look at non-peak history through
  this pathway. It can only find patterns among historical peaks.

  "What did demand look like during LAST year's Prime Day?"
  "What happened during the Lightning Deal 6 months ago?"

  Only those questions. Nothing else gets through.
```

**Query-Key-Value construction:**
- **Queries (q):** from encoded features + known future info (the forecast horizon)
- **Keys (k):** from historical encoded features
- **Values (v):** from historical peak demand information

The output is an adjustment term:
```
PeakForecast = Δ[t,h] + MLPDecoder(e_t^(p), e^(s))
```

During non-peak periods, PeakAdjustment ≈ 0 (defaults to baseline). During peaks, it adds the expected demand lift.

### Putting It Together

```
SPADE full architecture:

  Raw demand history + peak mask (from promotion calendar)
       │
       ├────────────────────────────────┐
       ▼                                ▼
  ┌──────────────────────┐    ┌────────────────────┐
  │  RobustConvolution   │    │   PeakAttention     │
  │                      │    │                      │
  │  1. Forward-fill:    │    │  Sparse masked       │
  │     remove peaks,    │    │  attention over      │
  │     replace with     │    │  ONLY historical     │
  │     last non-peak    │    │  peak events         │
  │                      │    │                      │
  │  2. Dilated conv     │    │  Mask applied inside │
  │     on clean history │    │  softmax — non-peak  │
  │                      │    │  positions zeroed    │
  └─────────┬────────────┘    └──────────┬───────────┘
            │                            │
            ▼                            ▼
     BaselineForecast            PeakAdjustment
            │                            │
            └────────────┬───────────────┘
                         ▼
                  Final Forecast
                  = Baseline + Peak
```

### Results

Weighted Quantile Loss relative to MQT baseline (lower = better, 1.0 = MQT):

| Metric | SPADE | MQCNN | MQT (baseline) |
|--------|-------|-------|----------------|
| P50 overall | 0.991 | 1.184 | 1.000 |
| P90 overall | 0.994 | 1.199 | 1.000 |
| P50 peak events | **0.967** | 0.974 | 1.000 |
| P50 post-peak | **0.958** | 0.939 | 1.000 |

- **Peak event accuracy:** 3.9% improvement over MQT
- **Post-peak accuracy:** 4.5% improvement over MQT
- **Worst-affected post-peak forecasts:** ~**30% improvement** (the tail of the distribution where carry-over causes the most damage)
- Trained on **40 p3dn.24xlarge machines** (8 V100 GPUs each). Full pipeline: ~20 hours. This is production-scale infrastructure.

### Ablation: Both Components Are Necessary

On a public tourism dataset with simulated peak contamination:
- Masked convolution alone: **10-17% improvement** on post-peak
- Peak attention alone: also improves
- **Both combined:** best results — the components are **complementary**, not redundant

### Connection to Transformer Attention

```
                Standard Transformer       SPADE PeakAttention
                ───────────────────       ──────────────────────
What Q         All positions in           Only historical peak
attends to     the sequence               event positions

Density        Dense (attends             Sparse (masked to
               everywhere)                peaks only)

Source of      Learned (model             Prescribed (mask from
sparsity       discovers what             known causal information —
               to attend to)             promotion calendars)

Purpose        General sequence           Specifically modeling
               modeling                   peak demand deltas

Info flow      Self-attention over        Cross-attention: future
               same sequence              info queries historical peaks
```

SPADE's sparse masking is related to Longformer/BigBird, but those use sparsity for computational efficiency. SPADE uses sparsity as an **inductive bias** — constraining what the model can attend to based on domain knowledge.

### Follow-up: SPADE-S (July 2025)

The team extended SPADE into "SPADE-S: A Sparsity-Robust Foundational Forecaster" (arXiv: 2507.21155), addressing systematic biases in low-magnitude and sparse time series, achieving up to 15% improvement. The decomposition approach generalizes beyond peak events.

---

## Paper 3: Goldilocks — Training Sample Selection via Bandits (KDD 2025)

**Full title:** *Goldilocks: An Active Sampling Bandit That's Just Right for Multi-Task Forecasting*
**Authors:** Vincent Quenneville-Belair et al.
**Venue:** KDD 2025 Workshop on AI for Supply Chain (AI4SupplyChain)
**Published via:** Amazon Science (no arXiv preprint available)

### The Training Problem

Amazon has millions of products. Training a forecasting model on all of them is expensive. But more importantly, **not all training examples are equally useful.**

```
Three types of training examples:

  TOO EASY:
    Product sells exactly 100 units every day, no variance
    The model already predicts this perfectly
    Training on it = wasted compute, teaches nothing

  TOO HARD:
    Product has erratic, unpredictable demand (one-off viral moments)
    The model can't learn a pattern because there isn't one
    Training on it = adds noise, confuses the model

  JUST RIGHT:
    Product has learnable patterns the model hasn't mastered yet
    Training on it = actual improvement in model quality

    "Not too hard, not too easy" — Goldilocks
```

### Connection to Curriculum Learning

This is related to **curriculum learning** (Bengio et al., 2009): the idea that neural networks learn better when training examples are presented in a meaningful order, from easy to hard, rather than randomly.

But Goldilocks goes further. Curriculum learning uses a fixed schedule (easy → hard). Goldilocks **adapts** — it continuously evaluates which examples are most useful *right now* and adjusts the sampling distribution on the fly.

```
Curriculum learning (static):

  Epoch 1-5:    easy examples
  Epoch 6-10:   medium examples
  Epoch 11-15:  hard examples

  Fixed schedule. Doesn't adapt to model state.


Goldilocks (adaptive):

  Epoch 1:   samples based on initial uncertainty
  Epoch 2:   re-evaluates — "model learned the easy ones,
             shift toward medium"
  Epoch 5:   "model is overfitting on category X,
             sample more from category Y"
  Epoch 10:  "these examples are permanently too noisy,
             stop sampling them"

  Continuously adapts. Responds to model state.
```

### Multi-Armed Bandit Formulation

This is where it connects to **reinforcement learning**. The training example selection is framed as a multi-armed bandit problem:

```
Multi-armed bandit:

  You're in a casino with K slot machines (arms).
  Each pull of an arm gives a random reward.
  Goal: maximize total reward over T pulls.

  Exploration: try arms you haven't pulled much (learn their rewards)
  Exploitation: pull the arm with highest known reward (maximize)


In Goldilocks:

  Arms      = groups of training examples (by product category,
              demand level, volatility, etc.)
  Pulling   = sampling a batch from that group for training
  Reward    = how much the model improved from training on that batch
              (measured by validation loss reduction)

  Exploration: try training on product groups you haven't seen much
  Exploitation: train more on groups where you got biggest improvements
```

### What Are the "Arms"?

The products are grouped into arms based on properties like:

- **Demand volume** — high sellers vs. low sellers
- **Volatility** — stable vs. erratic demand
- **Category** — electronics vs. grocery vs. apparel
- **Seasonality** — seasonal vs. year-round products

Each arm represents a cluster of similar forecasting tasks. The bandit learns which clusters provide the most learning signal.

### The Temperature Hyperparameter

Goldilocks introduces a **temperature** parameter controlling how aggressively sampling concentrates on harder examples — the same concept as softmax temperature in attention:

```
Temperature in attention (your note 23):

  High temp → flat distribution → attend to everything equally
  Low temp  → peaked distribution → attend almost exclusively to top match

Temperature in Goldilocks:

  High temp → sample uniformly across all product groups
  Low temp  → concentrate sampling on the "most useful" groups

  Too cold: exploits too hard, ignores potentially useful groups
  Too hot:  explores too much, wastes compute on easy/impossible groups
  Just right: the Goldilocks zone
```

### Dynamic Importance Sampling (DIS)

The paper introduces **Dynamic Importance Sampling** as an extension — adjusting the sampling distribution not just based on bandit rewards but also correcting for the distribution shift that non-uniform sampling introduces. When you oversample difficult examples, your gradient estimates become biased; DIS corrects for this with importance weights.

### "Multi-Task" Forecasting

"Multi-task" means the model forecasts multiple products simultaneously with shared parameters. This is standard in Amazon-scale forecasting — you don't train one model per product (that's millions of models).

```
Single-task:                    Multi-task:

  Model_A → forecast product A
  Model_B → forecast product B    Shared Model → forecast all products
  Model_C → forecast product C        │
  ...                                  ├── shared layers learn
  Model_N → forecast product N        │   "what demand looks like"
                                       │
  Millions of models.                  └── product-specific heads learn
  Can't scale.                             individual patterns

                                  One model. Millions of outputs.
```

The Goldilocks bandit decides how to allocate training compute across these tasks. Should the model spend more time learning grocery patterns or electronics patterns? The bandit figures it out.

### Results

Evaluated on the **M5 Walmart retail dataset** (3,049 products, 10 stores, 5.4 years of daily data — a public benchmark):

- Active sampling outperforms uniform random sampling
- The temperature sweet spot matters — too aggressive or too passive both hurt
- **Zero-shot generalization** to unseen retail data improves (the model generalizes better when trained on the right examples)

### Exploration vs. Exploitation — The RL Connection

This is the same fundamental trade-off from reinforcement learning:

```
Exploration vs Exploitation in RL:

  RL agent in a maze:
    Exploit: go to the room where you found food before
    Explore: check that corridor you've never visited

  Goldilocks bandit:
    Exploit: sample more from product groups that improved the model
    Explore: try product groups you haven't sampled much —
             they might be even more useful

  Same math. Different application.

  Common algorithms:
    - epsilon-greedy:  with probability ε, explore randomly
    - UCB (Upper Confidence Bound): pick the arm with highest
      (estimated reward + uncertainty bonus)
    - Thompson Sampling: maintain probability distributions over
      each arm's reward, sample from them
```

---

## How the Three Papers Fit Together

```
The Amazon forecasting pipeline, annotated:

  Millions of products
       │
       ├── New products (no history) ─────→ GEANN
       │   "Borrow patterns from similar      (2023)
       │    products via graph neural nets"
       │
       ├── All products during peak events ──→ SPADE
       │   "Decompose peak vs base demand      (2024)
       │    so the spike doesn't corrupt
       │    post-peak forecasts"
       │
       └── Training the model itself ────────→ Goldilocks
           "Pick the right training examples    (2025)
            — not too easy, not too hard —
            via multi-armed bandit"


Timeline of problems solved:

  GEANN (2023):     "What do I predict when I have NO data?"
  SPADE (2024):     "What do I predict when the data LIES to me?"
  Goldilocks (2025):"How do I TRAIN better across all products?"


  Data availability ──→ Data quality ──→ Training efficiency
  (cold start)          (peak corruption)  (sample selection)
```

### The Mathematical Thread

All three papers share Vincent's signature: **use mathematical structure to solve practical problems.**

| Paper | Structure Exploited | Math Tool |
|-------|-------------------|-----------|
| GEANN | Product catalog topology | Graph neural networks |
| SPADE | Temporal event structure | Attention-based decomposition |
| Goldilocks | Training difficulty distribution | Multi-armed bandits |

Each one identifies a *structural* property of the problem that a generic neural network wouldn't exploit, then builds that structure into the model architecture.

### A Shared Design Principle: Augment, Don't Replace

None of these papers throw away the base forecaster. All three are **modular augmentations**:

```
GEANN:      base MQ-CNN + graph augmentation module between encoder/decoder
SPADE:      base convolutions + peak attention pathway in parallel
Goldilocks: base training loop + bandit-based sampler wrapping the data loader

Each can be bolted onto existing production infrastructure
without rewriting the whole pipeline. This is why they
actually get deployed at Amazon scale.
```

---

## Connection to Your Study Topics

```
Your notes:                          Vincent's papers:

Attention (note 23)          ←→     SPADE uses attention for peak detection
  "softmax over scores"              "softmax over peak scores"
  "temperature controls                Goldilocks uses temperature to
   distribution sharpness"             control sampling concentration

Embeddings (note 27)         ←→     GEANN produces graph embeddings
  "similar things, nearby"           "similar products, shared forecast"
  "rare words get meaning             new products get demand patterns
   from context"                       from graph neighbors"

RL / exploration (note 25)   ←→     Goldilocks uses bandit algorithms
  "explore vs exploit"               "which training batch to sample?"

Transformers (note 19)       ←→     All three augment Transformer-based
  "self-attention over sequence"      forecasters with domain-specific priors

RAG (retrieval-augmented)    ←→     GEANN retrieves neighbor information
                                     to augment a forecaster that lacks
                                     internal knowledge (history)
```

---

## Takeaway

- **GEANN** solves cold-start by building a product similarity graph and using GNNs to transfer demand patterns from established products to new ones. Key lesson: at industrial scale, **structured domain knowledge (catalog hierarchy) beats data-driven similarity (learned embeddings)** — the k-NN graph is too noisy at 2M products
- **SPADE** solves peak corruption by decomposing demand into base + peak components via two parallel pathways. The **forward-fill** trick (replacing peak values with last non-peak before convolutions) is simple but remarkably effective. The **sparse peak attention** (masking non-peak positions inside softmax) constrains the model to only learn from historical peaks
- **Goldilocks** solves training inefficiency by framing example selection as a multi-armed bandit — exploring undersampled product groups while exploiting the most informative ones. Introduces a temperature parameter (same concept as attention temperature) to control how aggressively sampling concentrates
- Together they show a progression: data availability → data quality → training efficiency
- All three are **modular augmentations** that bolt onto existing production forecasters — designed to deploy, not just publish
- All three share the same philosophy: **inject domain structure into neural architectures** rather than hoping the network discovers it from raw data

---

*See also:* [Vincent Quenneville-Belair](researchers/vincent_quenneville_belair.md) | [Scaled Dot-Product Attention](23_scaled_dot_product_attention.md) | [Reinforcement Learning](25_reinforcement_learning_overview.md) | [Embedding Spaces](27_embedding_spaces_and_retrieval.md)
