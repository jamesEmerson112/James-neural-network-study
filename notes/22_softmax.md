# Softmax — Turning Scores into Probabilities

> Part of the [Neural Network Study Timeline](00_timeline_neural_sequence_models.md). See also: [Backprop](09_backprop_the_wiggle_ratio.md), [Transformers](19_transformers.md), [BERT](20_bert.md).

---

## The Problem Softmax Solves

A neural network's final layer outputs raw numbers called **logits** — unbounded scores that can be negative, huge, or tiny. They mean nothing on their own.

```
Input: "The cat sat on the ___"

Network's raw output (logits) for 3 candidate words:
  "mat"    →  2.0
  "dog"    →  1.0
  "banana" →  0.1

These are NOT probabilities. They don't sum to 1. We can't say
"the model is 2.0% confident in mat" — that's meaningless.
```

We need a function that:
1. Makes everything **positive** (no negative probabilities)
2. Makes everything **sum to 1** (valid probability distribution)
3. **Preserves ordering** — the highest logit should get the highest probability
4. **Amplifies differences** — a score of 2.0 vs 1.0 should feel more decisive than "twice as likely"

Softmax does all four.

---

## The Formula

```
                    exp(x_i)
softmax(x_i) = ─────────────────
                Σ_j  exp(x_j)
```

Where `exp(x) = e^x` (Euler's number ≈ 2.718, raised to the power x).

### Worked Example

Logits: `[2.0, 1.0, 0.1]`

**Step 1: Exponentiate each logit**

```
exp(2.0) = e^2.0  = 7.389
exp(1.0) = e^1.0  = 2.718
exp(0.1) = e^0.1  = 1.105
```

**Step 2: Sum the exponentials**

```
7.389 + 2.718 + 1.105 = 11.212
```

**Step 3: Divide each by the sum**

```
softmax("mat")    = 7.389 / 11.212 = 0.659  (65.9%)
softmax("dog")    = 2.718 / 11.212 = 0.242  (24.2%)
softmax("banana") = 1.105 / 11.212 = 0.099  ( 9.9%)
                                      ─────
                                      1.000  ✓ sums to 1
```

The model is 65.9% confident the answer is "mat." The raw logit gap of 2.0 vs 1.0 became a probability gap of 65.9% vs 24.2% — the exponential amplified the difference.

### Why `exp()`?

| Property | Why it matters |
|---|---|
| `exp(x) > 0` for all x | Guarantees positive outputs — no negative probabilities |
| `exp(x)` is monotonically increasing | Preserves ranking — bigger logit → bigger probability |
| `exp(x)` grows exponentially | Amplifies differences — small gaps in logits become large gaps in probability |
| `exp(x)` is differentiable | We can backpropagate through it |

---

## Where Softmax Appears in This Course

Softmax shows up in three distinct roles. Same formula, different jobs.

### 1. Classification Output

The final layer of a classifier produces logits — one per class. Softmax converts them to probabilities so we can pick the most likely class.

```
Raw logits:     [2.1, -0.5, 0.3]     ← one per class (e.g., cat/dog/bird)
     ↓ softmax
Probabilities:  [0.73, 0.05, 0.12]   ← pick argmax → "cat"
```

This is the softmax mentioned in [BERT fine-tuning](20_bert.md): `[CLS] output → linear layer → softmax → class prediction`.

### 2. Attention Weights

In the Transformer attention mechanism, the dot product `Q · K^T` produces raw compatibility scores between every pair of tokens. Softmax turns these into attention weights (how much should token A attend to token B?).

```
attn = softmax(Q · K^T / √d_k) · V
       ^^^^^^^^^^^^^^^^^^^^^^^^
       This softmax ensures attention weights sum to 1
       across the key dimension — each query distributes
       100% of its attention across all keys
```

This is the softmax in [Assignment 3](12_assignment3_battle_plan.md)'s multi-head attention and in every Transformer layer in [BERT](20_bert.md).

### 3. Next-Token Prediction

Language models (GPT, etc.) output logits over the entire vocabulary at each position. Softmax converts these to a probability distribution over ~50,000 tokens, from which the next token is sampled.

```
Logits:        [1.2, -3.0, 0.5, ..., 4.1]    ← one per vocab token (50,000+)
     ↓ softmax
Probabilities: [0.003, 0.000, 0.002, ..., 0.051]
     ↓ sample
Next token:    "the"
```

---

## The Temperature Trick

Divide logits by a **temperature** `T` before applying softmax:

```
softmax(x_i / T)
```

Temperature controls how **sharp** or **flat** the distribution is:

| T | Effect | Distribution shape | When to use |
|---|---|---|---|
| T → 0 | Argmax — all probability on the highest logit | Spike | Greedy/deterministic generation |
| T = 1 | Standard softmax | Normal | Default — balanced |
| T → ∞ | Uniform — all classes equally likely | Flat | Maximum randomness/exploration |

### Worked Example

Logits: `[2.0, 1.0, 0.1]`

```
T = 0.5 (sharp):   softmax([4.0, 2.0, 0.2])  = [0.876, 0.118, 0.006]  ← very confident
T = 1.0 (default):  softmax([2.0, 1.0, 0.1])  = [0.659, 0.242, 0.099]  ← moderate
T = 2.0 (soft):    softmax([1.0, 0.5, 0.05]) = [0.422, 0.256, 0.163]  ← less decisive
```

**Where it shows up:**
- **LLM inference**: ChatGPT's "temperature" slider literally does this. Low T → factual/predictable, high T → creative/surprising
- **Knowledge distillation**: Hinton et al. (2015) — train a small model on a large model's soft probabilities at high temperature
- **Attention sharpness**: sometimes scaled attention scores use an implicit temperature via the `1/√d_k` factor

---

## Why CrossEntropyLoss Includes Softmax

In [Assignment 3](12_assignment3_battle_plan.md), you see: **"NO softmax in the final layer — CrossEntropyLoss includes it."** Here's why.

### The Naive Approach (numerically unstable)

```
Step 1: Compute softmax     → probabilities
Step 2: Compute log         → log-probabilities
Step 3: Compute NLL loss    → negative log-likelihood
```

The problem: if a logit is large (say 1000), then `exp(1000)` overflows to infinity. Then `inf / inf = NaN`. Your training crashes.

### The Log-Sum-Exp Trick (what PyTorch actually does)

Instead of computing softmax first and then taking the log, combine them:

```
log(softmax(x_i)) = log(exp(x_i) / Σ exp(x_j))
                   = x_i - log(Σ exp(x_j))
                   = x_i - log(Σ exp(x_j - max(x)))  - max(x)
                                      ↑
                           Subtract max first so exp() never overflows
```

This is called the **log-sum-exp trick**. It's mathematically identical but numerically stable.

### What This Means in PyTorch

```python
# DON'T do this (unstable + slower):
output = model(x)
probs = torch.softmax(output, dim=-1)
loss = -torch.log(probs[target])

# DO this (stable + faster):
output = model(x)                    # raw logits — NO softmax
loss = nn.CrossEntropyLoss()(output, target)  # includes softmax internally
```

`nn.CrossEntropyLoss` = `nn.LogSoftmax` + `nn.NLLLoss`, fused into one numerically stable operation. That's why your model's final layer should output **raw logits**, not probabilities.

---

## Softmax Gradient

When backpropagating through softmax (relevant to [Backprop: The Wiggle Ratio](09_backprop_the_wiggle_ratio.md)):

```
If p = softmax(x), then:

∂p_i / ∂x_j = p_i (δ_ij - p_j)

where δ_ij = 1 if i=j, 0 otherwise
```

In plain English:
- The gradient of a softmax output with respect to its own input is `p_i(1 - p_i)` — same form as logistic sigmoid derivative
- The gradient with respect to other inputs is `-p_i · p_j` — increasing one logit decreases all other probabilities

This is a **Jacobian matrix**, not a simple scalar derivative — because softmax outputs are interdependent (they must sum to 1). When you use `CrossEntropyLoss`, PyTorch computes this for you.

---

## Historical Note

Softmax didn't originate in machine learning. It comes from **statistical mechanics** — the physics of how particles distribute across energy states.

**1868 — Ludwig Boltzmann (Vienna)**
Boltzmann derived the probability that a particle occupies energy state `E_i` at temperature `T`:

```
P(E_i) = exp(-E_i / kT) / Z

where Z = Σ exp(-E_j / kT)    ← the "partition function"
```

This is softmax with a sign flip (lower energy = higher probability in physics, while higher logit = higher probability in ML). The temperature parameter `T` in LLM sampling is literally Boltzmann's temperature — same formula, same role.

**1982 — John Hopfield (Caltech → Princeton)**
Hopfield brought Boltzmann's energy-based framework into neural networks with the Hopfield network — a recurrent net whose dynamics minimize an energy function. This directly inspired the **Boltzmann machine** (Hinton & Sejnowski, 1983), which uses the Boltzmann distribution over network states.

Hopfield received the **2024 Nobel Prize in Physics** (shared with Geoffrey Hinton) for "foundational discoveries and inventions that enable machine learning with artificial neural networks."

**1990 — John Bridle**
Bridle formally proposed using the softmax function as the output activation for neural network classifiers in his paper "Training Stochastic Model Recognition Algorithms as Networks can Lead to Maximum Mutual Information Estimation of Parameters." He coined the name "softmax" — a smooth (differentiable) approximation of the argmax function.

```
Timeline:
Boltzmann (1868)  →  partition function (physics)
     ↓
Hopfield (1982)   →  energy-based neural nets (Boltzmann distribution in ML)
     ↓
Bridle (1990)     →  "softmax" named, standard output for classifiers
     ↓
Vaswani (2017)    →  softmax in attention: "Attention Is All You Need"
     ↓
Every LLM today   →  softmax over vocabulary at every token generation step
```

---

## The One-Sentence Summary

**Softmax is the function that turns a neural network's raw scores into a probability distribution — and it's been doing this since Boltzmann figured out how atoms distribute across energy states in 1868.**

---

## Cross-References

- [Backprop: The Wiggle Ratio](09_backprop_the_wiggle_ratio.md) — how gradients flow through softmax during training
- [Assignment 3 Battle Plan](12_assignment3_battle_plan.md) — where softmax appears in attention and the "no softmax in final layer" rule
- [Vanishing Gradient and tanh](13_vanishing_gradient_and_tanh.md) — related activation function issues
- [Transformers](19_transformers.md) — softmax in self-attention
- [BERT](20_bert.md) — softmax in fine-tuning classification heads
- [Master Timeline](00_timeline_neural_sequence_models.md) — where it all fits
