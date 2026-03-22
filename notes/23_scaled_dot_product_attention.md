# Scaled Dot-Product Attention — Why √d Is the Magic Number

> Part of the [Neural Network Study Timeline](00_timeline_neural_sequence_models.md). See also: [Softmax](22_softmax.md), [Transformers](19_transformers.md), [BERT](20_bert.md).

---

## The Attention Formula

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
```

Step by step:

```
1. Input vectors → project into Q (query), K (key), V (value)
2. Q · Kᵀ           → raw similarity scores (dot product)
3. ÷ √d_k           → scale down to stabilize variance
4. softmax           → convert to probabilities (weights sum to 1)
5. × V              → weighted sum of values
```

---

## Dot Product as Similarity

The dot product measures how **aligned** two vectors are — same family as cosine similarity, but with a key difference.

```
Dot product:       Q · K  =  |Q| × |K| × cos(θ)
Cosine similarity: Q · K / (|Q| × |K|)  =  cos(θ)
```

Cosine similarity **strips out magnitude** (vector length) and only measures the angle between vectors. Dot product keeps both angle and magnitude.

### What is magnitude?

Magnitude is the **length** of a vector — calculated using the Pythagorean theorem:

```
Vector [3, 4]    → magnitude = √(3² + 4²) = √25 = 5
Vector [6, 8]    → magnitude = √(6² + 8²) = √100 = 10
```

Same formula as computing the hypotenuse of a right triangle — because a 2D vector literally *is* a right triangle:

```
    ↑ 4
    |  /|
    | / |  ← hypotenuse = magnitude = 5
    |/  |
    +---→ 3
```

This generalizes to any dimension:

```
2D:   √(x² + y²)
3D:   √(x² + y² + z²)
768D: √(x₁² + x₂² + ... + x₇₆₈²)    ← BERT lives here
```

Pythagoras (~500 BC) all the way up to 768 dimensions. Same math.

### Why attention uses dot product, not cosine

| | Dot product | Cosine similarity |
|---|---|---|
| Measures | angle + magnitude | angle only |
| Range | (-∞, +∞) | [-1, 1] |
| Magnitude matters? | Yes | No |

**In attention**: magnitude carries useful signal — a vector with higher magnitude can indicate higher confidence or importance. We want to preserve that.

**In embedding search** (finding similar sentences): cosine similarity is preferred because you care about meaning (direction), not magnitude. "cat" and "CAT" should score the same.

---

## The √d Scaling — The Goldilocks Argument

### The problem

Say d_k = 512 (dimension of keys in the original Transformer). If Q and K are random vectors where each element has mean 0 and variance 1:

```
Q · K = q₁k₁ + q₂k₂ + ... + q₅₁₂k₅₁₂
```

Each product `qᵢkᵢ` has variance 1. When you sum 512 independent terms with variance 1:

```
Var(Q · K) = 512
StdDev(Q · K) = √512 ≈ 22.6
```

So raw dot products will typically range around **±45** (±2 standard deviations). These are huge numbers to feed into softmax.

### What softmax does with large inputs

```python
# Raw scores around ±45:
softmax([45, 44, 40, 30, 20])
# → [0.73, 0.27, 0.00, 0.00, 0.00]   ← near one-hot! Almost all weight on one token

# After dividing by √512 ≈ 22.6:
softmax([2.0, 1.95, 1.77, 1.33, 0.88])
# → [0.28, 0.27, 0.22, 0.14, 0.09]   ← spread out, model can learn
```

### Why not divide by d, or d², or something else?

The answer comes from variance math:

```
Var(Q · K) = d           → standard deviation = √d
Dividing by √d gives:    → Var(Q · K / √d) = d / d = 1
```

Dividing by √d is the **unique value that normalizes the variance back to 1** — giving softmax a stable input range regardless of dimension size.

| Scaling | Variance after | Softmax behavior |
|---|---|---|
| No scaling | d (grows with dimension) | Near one-hot → gradients vanish → training stalls |
| ÷ d | 1/d (shrinks with dimension) | Near uniform → model can't focus → learns nothing |
| ÷ √d | 1 (constant) | Peaked but flexible → healthy gradients → model learns |

Visually:

```
No scaling (variance = d):
  softmax → [██████████░░░░░░░░░░]  one token hogs all attention

÷ d (variance = 1/d):
  softmax → [████████████████████]  everything equal, can't distinguish

÷ √d (variance = 1):
  softmax → [██████░░████░░██░░█░]  peaked but flexible — just right
```

### Why this matters at scale

Without proper scaling, increasing d (for more model capacity) would break training — softmax saturates and gradients die. With √d scaling, transformers can go from d=512 (original) to d=12288+ (GPT-4 scale) and attention stays well-behaved.

---

## Historical Note

The √d scaling was introduced in the original "Attention Is All You Need" paper (Vaswani et al., 2017). The authors noted in a footnote that they suspected large dot products were pushing softmax into saturated regions, worked out the variance math, and √d fell out as the natural normalizer.

This connects to a deep thread in neural network history — keeping activations and gradients in well-behaved ranges is critical. See also: [Vanishing Gradient and tanh](13_vanishing_gradient_and_tanh.md) for the same principle in RNNs.

---

## The One-Sentence Summary

**Scaled dot-product attention measures token similarity via dot products, then divides by √d to keep softmax inputs at variance 1 — the unique scaling that works regardless of dimension size, letting transformers scale from hundreds to tens of thousands of dimensions.**

---

## Cross-References

- [Softmax](22_softmax.md) — how softmax converts scores to probabilities, temperature trick, numerical stability
- [Transformers](19_transformers.md) — high-level transformer architecture
- [BERT](20_bert.md) — encoder-only transformer using this attention mechanism
- [Vanishing Gradient and tanh](13_vanishing_gradient_and_tanh.md) — related problem of keeping gradients in stable ranges
- [Assignment 3 Battle Plan](12_assignment3_battle_plan.md) — implementing multi-head attention
- [Master Timeline](00_timeline_neural_sequence_models.md) — where it all fits
