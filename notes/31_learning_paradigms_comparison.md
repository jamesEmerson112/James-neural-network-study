# Learning Paradigms — Comparison and Data Assumptions

## What this note unpacks

**What's the difference between semi-supervised, few-shot, and self-supervised learning, and what data does each assume is available?** The tricky part is that these categories overlap and get muddled in practice. This note pins them down precisely, with a canonical table comparing all seven major paradigms (supervised, unsupervised, semi-supervised, self-supervised, few-shot, zero-shot, meta-learning) and the exact data each assumes.

---

## The master comparison table

| Paradigm | Labels? | Amount of data | Typical example | Loss structure |
|---|---|---|---|---|
| **Supervised** | All inputs labeled | Large labeled dataset | ImageNet classifier (1.2M labeled images) | Cross-entropy / MSE on $(x, y)$ pairs |
| **Unsupervised** | None | Large unlabeled dataset | K-means clustering, PCA, density estimation | Reconstruction, log-likelihood, clustering loss |
| **Semi-supervised** | Small labeled + large unlabeled | ~1K labeled + ~100K unlabeled | Pseudo-labeling for MNIST with 100 labels | Supervised loss on labeled + consistency/entropy/reconstruction on unlabeled |
| **Self-supervised** | None (labels generated from data) | Very large unlabeled | BERT (Wikipedia + BookCorpus); GPT (internet text); SimCLR (unlabeled ImageNet) | Task-specific proxy loss (MLM, contrastive, reconstruction) |
| **Few-shot** | Very small (1–5 per class) | Small new task + large pretrained model | GPT-3 in-context learning; prototypical networks on 5-way 5-shot | Distance-based or meta-learned; classical supervised on tiny $k$-shot set |
| **Zero-shot** | None for target classes | No examples at all, just class descriptions | CLIP text-to-image matching | Cross-modal similarity, no training on target |
| **Meta-learning** | Many small labeled tasks | "Train on tasks, test on a new task" | MAML (Model-Agnostic Meta-Learning) on Omniglot | Bi-level: inner task loss + outer meta-loss |
| **Transfer learning** | Labels on both source and target | Large source + small target | ImageNet pretrained → fine-tune on medical images | Supervised on source, then supervised on target |

**Legend:**
- "Labeled" = human-annotated ground truth
- "Unlabeled" = raw data with no human annotation
- Sizes are illustrative; actual amounts vary by problem

---

## The specific paradigms to know

### Semi-supervised learning

**Data assumption:** you have a **small amount of labeled data** and a **large amount of unlabeled data**.

```
  labeled data:     (x₁, y₁), (x₂, y₂), ..., (x_N, y_N)          N small (e.g., 100)
  unlabeled data:   x_{N+1}, x_{N+2}, ..., x_{N+M}               M large (e.g., 100,000)

  goal: train a classifier that uses BOTH sets
```

**Why it exists:** labels are expensive (medical images, legal documents) but raw data is cheap. Semi-supervised learning tries to extract maximum value from the unlabeled data by using it to shape the decision boundary without explicit labels.

**Core techniques:**

- **Pseudo-labeling** — train a model on the labeled set, use it to predict labels for unlabeled examples (the "pseudo-labels"), then retrain on the combined set. Iterate.
- **Consistency regularization** — augment unlabeled examples and require the model to produce the same prediction. Forces the model to be invariant to irrelevant perturbations. (Used in FixMatch, MixMatch, UDA.)
- **Entropy minimization** — penalize high-entropy predictions on unlabeled data, pushing the model toward confident predictions (i.e., decision boundaries in low-density regions).
- **Generative approaches** — train a VAE or GAN on all the data, then use the latent space as features for supervised learning on the labeled subset.

**When to use:** you have a clear set of classes but labeling is expensive, and unlabeled data is plentiful.

### Self-supervised learning (SSL)

**Data assumption:** you have **massive amounts of unlabeled data** and **no human labels at all**.

**Key insight:** you invent a task that lets you create supervision signal **from the data itself**. Common inventions:
- Mask some of the input, predict the missing part (BERT)
- Given a prefix, predict the next token (GPT)
- Given two augmented views of the same image, pull them together in embedding space (SimCLR)

The "labels" are generated automatically from the raw data — no human annotation needed. This is why SSL is sometimes described as "unsupervised learning that pretends to be supervised."

**Yann LeCun's framing:** SSL is the "dark matter of intelligence." Most of what humans learn comes from predicting what comes next in their experience (without any explicit reward or label), not from labeled examples. SSL is the closest ML analog.

**Why it matters for modern AI:** every large foundation model (GPT, BERT, CLIP, DALL-E, Stable Diffusion) is pretrained via SSL. The "pretrain on unlabeled data → fine-tune on small labeled task" paradigm is the foundation of modern NLP and is rapidly becoming the standard for vision and multimodal as well.

### Few-shot learning

**Data assumption:** you have a **small number of labeled examples per class** (typically 1 to 5) and you want to classify new examples. Critically, few-shot usually assumes you have **access to a large pretrained model** that was trained on lots of other (labeled or unlabeled) data.

The "few" refers to the number of examples **for the target task**, not total data availability.

**Terminology:**
- **$N$-way $k$-shot**: you have $N$ classes with $k$ labeled examples each. Example: "5-way 5-shot" means 5 classes with 5 examples each = 25 labeled examples total.
- **Support set**: the $N \cdot k$ labeled examples
- **Query set**: the new examples you want to classify

**Three main approaches:**

1. **Metric learning** (Siamese networks, prototypical networks, matching networks) — learn an embedding space where classes cluster, then classify new examples by nearest-neighbor distance to the support set.

2. **Meta-learning** (MAML) — train on many different few-shot tasks so the model learns to adapt quickly. The meta-trained model is then fine-tuned on the specific few-shot task in a handful of gradient steps.

3. **In-context learning** (GPT-3 and later LLMs) — no training at all for the few-shot task. Just paste the $k$ examples into the prompt and ask the model to predict on a new one. GPT-3 showed this works for text tasks, and it's now how most LLMs are used for few-shot.

**Canonical example — GPT-3 in-context learning:**

```
  Translate English to French:
  sea otter → loutre de mer
  plush giraffe → girafe en peluche
  cheese → [model predicts: fromage]
```

The model was never explicitly trained on English-French translation, but it had enough general language knowledge from pretraining that it can generalize from 2 examples in the context.

### Zero-shot learning

**Data assumption:** you have **no labeled examples of the target classes at all**. You have only a **description** of each class (a name, a text definition, or an image prototype).

**How it works:** you rely on a model that has learned a shared embedding space between the target domain and some auxiliary information (usually text descriptions). You embed both the target class descriptions and the input into that shared space, then classify by nearest-neighbor.

**Canonical example — CLIP:**

CLIP (OpenAI 2021) trains on 400 million (image, caption) pairs to learn a joint embedding space. To classify a new image, you:
1. Embed the image
2. Embed several text prompts like "a photo of a dog", "a photo of a cat", ..., "a photo of a zebra"
3. Pick the text whose embedding is closest to the image

You never see any labeled examples of the target classes. The model's general text-image understanding does the work.

### Meta-learning ("learning to learn")

**Data assumption:** you have **many small labeled tasks**, each one not big enough to solve on its own, but collectively enough to learn the *learning process*.

The meta-learner's goal isn't to solve any one task — it's to learn how to quickly adapt to new tasks. Test time: given a new task, the meta-learner should be able to solve it with very little data and a few gradient steps.

**Canonical example — MAML (Model-Agnostic Meta-Learning):**

Find a set of model parameters $\theta$ such that for *any* new task $\mathcal{T}$, a single gradient step on the task's training data produces a good model. Formally, minimize the meta-objective:

$$
\min_\theta \sum_{\mathcal{T}} \mathcal{L}_{\mathcal{T}}\!\left( \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}}(\theta) \right)
$$

Reading: "average loss across tasks, after one gradient step of fine-tuning." This favors initial parameters that are close to the optimum for every task — an easy-to-adapt starting point.

Meta-learning is the mechanism behind many few-shot algorithms. The two concepts often overlap.

---

## The "where does self-supervised fit?" question

Self-supervised learning sits in an awkward spot between supervised and unsupervised. It uses the *machinery* of supervised learning (inputs, targets, loss, backprop) but the targets are *generated from the data* rather than annotated by humans. From a practitioner's perspective, it's supervised. From a theoretical perspective, it's unsupervised.

**The framing that resolves this:** self-supervised learning is a *way of doing* unsupervised learning by casting it as supervised learning on a proxy task. The proxy task is designed so that solving it requires learning features that are useful for downstream tasks.

- **BERT's proxy task**: predict masked tokens → requires learning syntax, semantics, world knowledge
- **GPT's proxy task**: predict the next token → requires modeling everything about language and the world
- **SimCLR's proxy task**: recognize two augmentations of the same image → requires learning semantic object features
- **DINO's proxy task**: student matches teacher's output → requires emergent self-organized features

The "self" in self-supervised means "the data supervises itself." No human labels are involved.

---

## How the paradigms relate

```
               fully
             supervised
                 │
                 │  (lots of labeled data)
                 ▼
         supervised fine-tuning
              after
         pretraining (SSL)
                 │
                 │  (pretrained on unlabeled, fine-tune on small labeled)
                 ▼
        semi-supervised
          learning
                 │
                 │  (mix labeled and unlabeled in one training loop)
                 ▼
         self-supervised
          pretraining
                 │
                 │  (no labels at all; labels come from the data)
                 ▼
         unsupervised
            (pure)
```

**In modern deep learning, the dominant pattern is: self-supervised pretraining → supervised or few-shot fine-tuning.** Every major LLM and every modern vision foundation model follows this pattern. GPT, BERT, CLIP, DALL-E, LLaMA, Gemini — all of them are pretrained via SSL and then fine-tuned (or prompted) for downstream tasks.

---

## Data-to-paradigm decision tree

Given a data setup, here's how to identify the paradigm:

```
  Do you have labels?
    │
    ├── YES, for all data        → supervised
    │       but also want to use unlabeled → semi-supervised
    │
    ├── YES, but very few (1-5)  → few-shot (usually with pretrained model)
    │
    ├── YES, many small tasks    → meta-learning
    │
    ├── NO, but you have a
    │   pretrained model +
    │   class descriptions        → zero-shot
    │
    └── NO labels at all          → self-supervised (if you invent a proxy task)
                                  → unsupervised (if you cluster/density estimate)
```

---

## Quick definitions

Memorize these one-liners:

- **Supervised**: every example has a human label. Model learns $x \to y$.
- **Unsupervised**: no labels. Model learns structure ($x$ distribution, clusters, latent factors).
- **Semi-supervised**: small labeled set + large unlabeled set; model uses both.
- **Self-supervised**: no human labels; the model invents labels from the data (mask + predict, augment + match, etc.).
- **Few-shot**: 1–5 labeled examples per class; leverages a large pretrained model.
- **Zero-shot**: no labeled examples for the target classes; leverages class descriptions.
- **Meta-learning**: train on many tasks, learn to adapt quickly to new ones.

---

## History/lore

- **1960s — Vapnik & Chervonenkis** and the Soviet pattern-recognition school formalize supervised learning as statistical learning theory.
- **1990s — Semi-supervised learning** gains traction with papers like Blum & Mitchell's *Combining Labeled and Unlabeled Data with Co-Training* (1998) and Nigam et al.'s EM-based approaches.
- **2006 — Hinton, Osindero, Teh** publish *A Fast Learning Algorithm for Deep Belief Nets*. The key move is **unsupervised pretraining** — stack RBMs, train them unsupervised, then fine-tune with backprop. This is the proto-self-supervised approach and the match that lit the deep learning revolution.
- **2008 — Pascal Vincent, Yoshua Bengio, others** publish the **denoising autoencoder**. Corrupt input, learn to reconstruct clean. This is the direct ancestor of modern masked autoencoders (MAE) and DDPM.
- **2013 — Mikolov et al.** publish Word2Vec — self-supervised learning of word embeddings from raw text.
- **2017 — Vaswani et al.** publish the Transformer, which enables scaling self-supervised pretraining to huge models.
- **2018 — ELMo, GPT-1, BERT** all land within months of each other. The paradigm "pretrain on lots of text, fine-tune on small labeled task" becomes the default for NLP.
- **2020 — GPT-3** demonstrates massive-scale in-context learning. Few-shot learning via prompting (no gradient updates) becomes viable.
- **2020 — SimCLR, MoCo** show contrastive self-supervised learning can match supervised ImageNet pretraining. Self-supervised vision takes off.
- **2021 — CLIP** demonstrates zero-shot classification at scale by training on 400M image-text pairs.
- **2023 — Yann LeCun** gives his "dark matter of intelligence" talks, arguing SSL is the key to human-like AI because it's how humans learn most of what they know.

The arc: **every decade adds another paradigm, and the newer ones always incorporate the older ones as special cases.** GPT-3 does supervised learning (next-token prediction is labeled) on self-supervised data (text supervises itself) to enable few-shot in-context learning at inference time. All four paradigms in one model.

---

## Takeaway

- **Seven paradigms** to know: supervised, unsupervised, semi-supervised, self-supervised, few-shot, zero-shot, meta-learning. Transfer learning is the 8th but usually grouped with fine-tuning.
- **Semi-supervised** = small labeled + large unlabeled, used together in one training loop.
- **Self-supervised** = no human labels; the model invents labels from the data via a proxy task.
- **Few-shot** = 1–5 labeled examples per class, usually leveraging a large pretrained model.
- **The modern dominant pattern**: self-supervised pretraining → fine-tuning or few-shot prompting. Every major foundation model uses this.
