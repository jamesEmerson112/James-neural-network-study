# The Vanishing Gradient Problem — Why RNNs Fail and LSTMs Exist

> *Last updated: 2026-03-17*
> *See also: [11_lstm_the_memory_machine.md](11_lstm_the_memory_machine.md) and [10_mlp_backprop_and_the_birth_of_rnns.md](10_mlp_backprop_and_the_birth_of_rnns.md)*

## The Problem Statement

> "In theory, RNNs are absolutely capable of handling such 'long-term dependencies.' A human could carefully pick parameters for them to solve toy problems of this form. Sadly, in practice, RNNs don't seem to be able to learn them."

Hochreiter (1991) and Bengio et al. (1994) found the fundamental reason: **vanishing gradients**.

## How an RNN Processes a Sequence

At each timestep:

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)
```

To learn that a token at step 100 affects output at step 200, backpropagation through time (BPTT) must send the gradient **back through 100 multiplications** of `W_hh`.

## The Math That Kills It

The gradient of the loss with respect to an early hidden state involves:

```
∂h_t / ∂h_k = ∏(i=k to t-1) W_hh · diag(tanh'(h_i))
```

A **chain of matrix multiplications**. What happens depends on the eigenvalues of `W_hh`:

| Eigenvalue magnitude | What happens | Effect |
|---|---|---|
| < 1 | Gradient **vanishes** exponentially | Network forgets distant past |
| > 1 | Gradient **explodes** exponentially | Training diverges, weights → NaN |
| = 1 | Gradient flows perfectly | Almost impossible to maintain in practice |

## Worked Example: Tracing Gradients Through 5 Steps

Setup: 1 hidden unit (scalar), `w = 1.5`, `h_0 = 0.5`

### Forward pass

```
h_0 = 0.5
h_1 = tanh(1.5 × 0.5)  = tanh(0.75)  = 0.635
h_2 = tanh(1.5 × 0.635) = tanh(0.953) = 0.741
h_3 = tanh(1.5 × 0.741) = tanh(1.112) = 0.804
h_4 = tanh(1.5 × 0.804) = tanh(1.206) = 0.836
h_5 = tanh(1.5 × 0.836) = tanh(1.254) = 0.849
```

Notice: `h` saturates toward ~0.85.

### Backward pass

We want `∂h_5 / ∂h_0`. By chain rule:

```
∂h_5/∂h_0 = ∂h_5/∂h_4 · ∂h_4/∂h_3 · ∂h_3/∂h_2 · ∂h_2/∂h_1 · ∂h_1/∂h_0
```

Each link: `∂h_t/∂h_{t-1} = tanh'(w · h_{t-1}) · w`

Using `tanh'(x) = 1 - tanh²(x)`:

```
Step 1: tanh'(0.75)  × 1.5 = (1 - 0.635²) × 1.5 = 0.597 × 1.5 = 0.895
Step 2: tanh'(0.953) × 1.5 = (1 - 0.741²) × 1.5 = 0.451 × 1.5 = 0.677
Step 3: tanh'(1.112) × 1.5 = (1 - 0.804²) × 1.5 = 0.354 × 1.5 = 0.531
Step 4: tanh'(1.206) × 1.5 = (1 - 0.836²) × 1.5 = 0.301 × 1.5 = 0.452
Step 5: tanh'(1.254) × 1.5 = (1 - 0.849²) × 1.5 = 0.279 × 1.5 = 0.419
```

### Total gradient across 5 steps

```
∂h_5/∂h_0 = 0.895 × 0.677 × 0.531 × 0.452 × 0.419 = 0.061
```

After just **5 steps**, the gradient is **6% of the original signal**.

### Scale it up (average factor ≈ 0.6 per step)

| Steps back | Gradient magnitude | What the network "hears" |
|---|---|---|
| 5 | 0.6⁵ = 0.078 | 7.8% — weak but detectable |
| 10 | 0.6¹⁰ = 0.006 | 0.6% — nearly invisible |
| 20 | 0.6²⁰ = 0.0000366 | 0.004% — functionally zero |
| 50 | 0.6⁵⁰ ≈ 1.3 × 10⁻¹¹ | dead |
| 100 | 0.6¹⁰⁰ ≈ 6.5 × 10⁻²³ | **smaller than an atom** |

## Why tanh Is the Culprit

### tanh'(x) = 1 - tanh²(x) — always ≤ 1

```
x = 0.0  →  tanh' = 1.00   (perfect, but only at zero)
x = 0.5  →  tanh' = 0.79
x = 1.0  →  tanh' = 0.42
x = 1.5  →  tanh' = 0.18
x = 2.0  →  tanh' = 0.07
x = 3.0  →  tanh' = 0.01   (basically dead)
```

Each backward step multiplies by **two things**:

```
tanh'(x)  ×  w
  ↑            ↑
 ≤ 1        the weight
(usually much less)
```

The cruel part: as the RNN trains and hidden states get pushed toward larger values (tanh saturates at ±1), the derivatives get **even smaller**. The better it fits forward, the worse gradients flow backward. **Self-defeating.**

### The exploding case

If `w = 3.0` and hidden states stay near zero (where tanh'≈1):

```
factor per step ≈ 1.0 × 3.0 = 3.0
After 50 steps: 3^50 ≈ 7 × 10²³ → NaN, training crashes
```

## Wait — What IS tanh?

### Not the geometric tangent

`tan(θ) = opposite / adjacent` — that's trigonometric tangent (triangles).

`tanh(x)` is **hyperbolic tangent** — completely unrelated to geometry:

```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
```

Pronunciation: **"tan-H"** or **"tanch"**. The H = hyperbolic.

It's just a **squashing function**: takes any real number, outputs between -1 and +1.

```
        1 |          _______________
          |        /
          |      /
        0 |----/----
          |  /
          |/
       -1 |_______________

      input: -∞ ←————————→ +∞
      output: always between -1 and +1
```

### The hyperbolic family

```
sinh(x) = (eˣ - e⁻ˣ) / 2        "sinch" or "shin"
cosh(x) = (eˣ + e⁻ˣ) / 2        "cosh" (rhymes with gosh)
tanh(x) = sinh(x) / cosh(x)      "tanch" or "tan-H"
```

The "hyperbolic" name comes from 1700s mathematicians who noticed these functions had similar algebraic identities to trig functions — so they reused the names with an "h". That's the entire reason. No geometric intuition needed for deep learning.

### Activation functions used in deep learning

| Function | Range | Used for |
|---|---|---|
| **sigmoid** (σ) | (0, 1) | Gates — "how much to let through" |
| **tanh** | (-1, +1) | Hidden states — "what value to store" |
| **ReLU** | [0, ∞) | Modern default — killed tanh in most networks post-2012 |

Sigmoid and tanh are related: `σ(x) = (1 + tanh(x/2)) / 2`

### Why tanh lost to ReLU (in most networks)

```
tanh:  gradient ≤ 1, saturates → vanishing gradient
ReLU:  gradient = 1 (positive), 0 (negative) → simple, trains faster
```

ReLU replaced tanh after AlexNet (2012). But LSTMs still use tanh and sigmoid internally for gates — which is what you implement in Assignment 3.

## How LSTM Solves the Vanishing Gradient

The key insight — **additive** cell state update:

```
RNN:   h_t = tanh(W · h_{t-1} + ...)        ← multiplicative → gradient decays
LSTM:  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t     ← ADDITIVE → gradient flows freely
```

When the forget gate `f_t ≈ 1`, the gradient passes through **unchanged** — like a highway. This is the "constant error carousel."

```
RNN gradient:  ∏ (tanh'(x_i) · w)     ← product of numbers < 1 → vanishes
LSTM gradient: ∏ (f_i)                 ← forget gate ≈ 1 when needed → preserved
```

No tanh derivative in the cell state path. No weight matrix multiplication. Just the forget gate.

## The Progression (Assignment 3 Arc)

```
RNN:         Vanishing gradient — can't learn beyond ~10-20 steps
    ↓ fix: additive cell state + gates
LSTM:        Learns long-range dependencies — but still sequential bottleneck
    ↓ fix: let decoder look at all encoder states
Seq2Seq+Attn: Attention bypasses the sequential bottleneck
    ↓ fix: remove recurrence entirely
Transformer: All-to-all attention in parallel — no vanishing gradient across positions
```

Each model in the assignment directly addresses the limitations of the previous one.
