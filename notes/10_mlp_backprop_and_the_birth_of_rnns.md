# MLP, Backprop, and the Birth of RNNs

> **Three researchers at two universities turned a 24-year-old idea into the algorithm that resurrected neural networks — then immediately asked: "What if the layers were time?"**

---

## Part 1: The Three Resurrectioners

### David Rumelhart (1942–2011) — The Cognitive Scientist

B.A. in psychology and mathematics (University of South Dakota), Ph.D. in mathematical psychology (Stanford). Professor of Psychology at UCSD from 1967. Co-founded UCSD's Institute for Cognitive Science. Won a MacArthur Fellowship in 1987 — the year after the paper.

Rumelhart wasn't an engineer. He wanted to understand how **humans** learn internal representations. Neural networks were his model of the mind.

### Geoffrey Hinton — The Psychologist Turned AI Pioneer

B.A. in Experimental Psychology from King's College, Cambridge (1970) — **not** physics, despite the common myth. Ph.D. in Artificial Intelligence from the University of Edinburgh (1978). In 1986 he was at **Carnegie Mellon** (1982–1987), not UCSD — though he'd been a visiting scholar at UCSD earlier (1978–1980).

Left CMU in 1987 for the University of Toronto, partly because he opposed Reagan-era military funding of AI research. Nobel Prize in Physics, 2024 — for the neural network work he started in the 1980s.

### Ronald J. Williams (1945–2024) — The Mathematician

B.S. in Mathematics from Caltech, M.A. and Ph.D. in Mathematics from UCSD. Member of the PDP Research Group at UCSD (1983–1986), then professor of Computer Science at Northeastern University for 22+ years.

Williams brought the mathematical rigor — he's the one who later formalized the REINFORCE algorithm for reinforcement learning.

### The PDP Group

The **Parallel Distributed Processing Research Group** at UCSD, led by Rumelhart and James McClelland. In 1986 they published the two-volume *Parallel Distributed Processing: Explorations in the Microstructure of Cognition* (MIT Press) — over 30,000 citations. Chapter 8 of Volume 1, by Rumelhart, Hinton, and Williams, was the full treatment of backpropagation. The 4-page Nature paper was the condensed version.

---

## Part 2: What Is an MLP?

A **Multi-Layer Perceptron** is what you get when you stack Rosenblatt's perceptrons into layers and connect every neuron to every neuron in the next layer:

```
         INPUT LAYER          HIDDEN LAYER 1        HIDDEN LAYER 2        OUTPUT LAYER
         (raw data)           (learned features)    (higher features)     (answer)

          x₁ ●─────────────────● h₁ ──────────────────● h₃ ──────────────────● y₁
              │╲              ╱│╲                    ╱│╲                    ╱│
              │ ╲            ╱ │ ╲                  ╱ │ ╲                  ╱ │
              │  ╲          ╱  │  ╲                ╱  │  ╲                ╱  │
              │   ╲        ╱   │   ╲              ╱   │   ╲              ╱   │
          x₂ ●─────╲──────╱───● h₂ ╲────────────╱────● h₄ ╲────────────╱────● y₂
              │     ╲    ╱     │     ╲          ╱     │     ╲          ╱
              │      ╲  ╱      │      ╲        ╱      │      ╲        ╱
          x₃ ●────────╲╱──────●       ╲      ╱       │       ╲      ╱
                                        ╲    ╱        │        ╲    ╱
                                         ╲  ╱         │         ╲  ╱
                                          ╲╱          │          ╲╱
                                                      │

         Each arrow = one weight.
         Every neuron connects to EVERY neuron in the next layer.
         That's "fully connected" or "dense."
```

### What each neuron does

```
neuron output = activation( Σ(inputs × weights) + bias )
              = σ( w₁x₁ + w₂x₂ + w₃x₃ + b )

  1. Multiply each input by its weight       (linear combination)
  2. Add bias                                 (shift the decision boundary)
  3. Pass through activation function         (introduce nonlinearity)
```

### Why the activation function matters

Without activation functions, stacking layers does nothing — matrix × matrix = still a matrix. A 100-layer linear network collapses into a single layer. The activation function (sigmoid, tanh, ReLU) is what makes depth useful.

### How it differs from a single Perceptron

| | Single Perceptron (1957) | MLP (1986) |
|---|---|---|
| Layers | 1 | 2+ |
| Activation | Hard step (0 or 1) | Smooth (sigmoid, tanh, ReLU) |
| Can solve XOR? | No | Yes |
| How to train | Perceptron rule | Backpropagation |
| Decision boundary | One straight line | Arbitrary curves |

---

## Part 3: Backprop Through an MLP — The Sigmoid Problem

Note 09 covers the intuition of backprop — wiggle ratios chained backward. Here we focus on what happens when those ratios pass through a **sigmoid activation**, because this is where the story leads to RNNs and eventually to LSTMs.

### Sigmoid: the activation Rumelhart used

```
σ(z) = 1 / (1 + e⁻ᶻ)

Output always between 0 and 1.

              1.0 ─────────────────────────────────╭──────────
                                                 ╱
              0.5 ──────────────────────────────╱─────────────
                                              ╱
              0.0 ──────────────╭────────────╱────────────────
                          -6   -4   -2    0    2    4    6
                                          z

Derivative: σ'(z) = σ(z) × (1 - σ(z))

Peak derivative: σ'(0) = 0.5 × 0.5 = 0.25

    THE MAXIMUM gradient through a sigmoid is 0.25.
    Every sigmoid layer SHRINKS the gradient by at least 75%.
```

### Gradient flow through a 4-layer MLP

```
        Layer 1         Layer 2         Layer 3         Layer 4        Loss
         σ               σ               σ               σ
          │               │               │               │             │
          │   ×0.25       │   ×0.25       │   ×0.25       │   ×0.25    │
          │◄──────────────│◄──────────────│◄──────────────│◄────────────│
          │               │               │               │             │
    gradient = ?     gradient = ?    gradient = ?    gradient = ?    gradient = 1.0
              │               │               │               │
              ▼               ▼               ▼               ▼
         0.25⁴           0.25³           0.25²           0.25¹
        = 0.004          = 0.016         = 0.0625         = 0.25

    Layer 4 feels the error clearly (0.25).
    Layer 1 barely feels anything  (0.004).
```

This is why Rumelhart's 1986 networks were typically **shallow** — 1 or 2 hidden layers. Go deeper and the early layers stop learning. (See note 09, Part 5 for the general vanishing gradient explanation with wiggle ratios.)

---

## Part 4: The XOR Victory

The XOR problem killed the Perceptron (note 06). Rumelhart's team solved it in the PDP book (Chapter 8) with **2 hidden neurons** — the simplest possible MLP:

```
THE ARCHITECTURE RUMELHART ACTUALLY USED:

    Input Layer          Hidden Layer           Output Layer
    (2 neurons)          (2 neurons)            (1 neuron)

     x₁ ●──────────────── h₁ ●
         │╲              ╱│    ╲
         │  ╲          ╱  │      ╲
         │    ╲      ╱    │        ● output
         │      ╲  ╱      │      ╱
         │        ╲╱      │    ╱
     x₂ ●──────────────── h₂ ●


    h₁ learns: "Is at least one input on?"         (≈ OR)
    h₂ learns: "Are both inputs on?"               (≈ AND)
    output learns: "h₁ on BUT h₂ off?"             (= XOR)
```

The Nature paper didn't actually feature XOR — it was a 4-page paper focused on a more impressive demo: the **family tree experiment**. Two isomorphic family trees (English and Italian), 24 people, 12 relationship types. The network learned distributed representations where English and Italian counterparts produced **similar internal activations** — it discovered the abstract structure of kinship without being told.

But XOR was the symbolic victory. Minsky said multi-layer networks couldn't be trained. Rumelhart trained one. Game over.

(See note 09, Part 6 for the detailed XOR walkthrough with weights and step-by-step computation.)

---

## Part 5: From MLP to RNN — "What If the Layers Were Time?"

An MLP takes a **fixed-size** input and produces a **fixed-size** output:

```
MLP:    [x₁, x₂, x₃]  →  network  →  [y₁, y₂]

Input must always be exactly 3 numbers.
Output must always be exactly 2 numbers.
```

But language doesn't come in fixed sizes. "Cat" is 3 letters. "The cat sat on the mat" is 6 words. You can't hardcode the input width.

### The conceptual leap

Rumelhart, Hinton, and Williams described it in the final section of PDP Chapter 8 ("Training Recurrent Neural Nets"): **share the same weights across every time step**, and **feed the hidden state back into the next step**.

```
MLP (fixed input):

    x₁ ──→ [Layer 1] ──→ [Layer 2] ──→ [Layer 3] ──→ output
            (weights A)   (weights B)   (weights C)
            all different weights


RNN (sequence, any length):

    x₁ ──→ [Layer] ──→ x₂ ──→ [Layer] ──→ x₃ ──→ [Layer] ──→ output
           (weights W)        (weights W)        (weights W)
               │                   │                   │
               └──── h₁ ──────────└──── h₂ ────────────┘
                  hidden state      hidden state
                  carries forward   carries forward

    SAME weights at every step. Hidden state = memory.
```

This is the key insight: **an RNN is an MLP where the "layers" are time steps, the weights are shared, and each layer passes a hidden state to the next**.

---

## Part 6: RNN Architecture

### The folded view (what you usually see)

```
              ┌──────────┐
              │          │
    x_t ────→ │  RNN     │────→ y_t
              │  Cell    │
    h_{t-1} ─→│          │──→ h_t
              └──────────┘
                   ▲  │
                   │  │
                   └──┘
                 recurrent
                 connection
```

The loop arrow means: "take h_t and feed it back as h_{t-1} at the next step."

### The unrolled view (what actually happens during training)

```
    x₁           x₂           x₃           x₄
     │            │            │            │
     ▼            ▼            ▼            ▼
  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
  │ RNN  │──→│ RNN  │──→│ RNN  │──→│ RNN  │──→ h₄
  │ Cell │    │ Cell │    │ Cell │    │ Cell │    (final
  └──────┘    └──────┘    └──────┘    └──────┘    hidden
h₀──┘  │         │  │         │  │         │  │   state)
       ▼         ▼         ▼         ▼
      y₁        y₂        y₃        y₄

   ALL FOUR BOXES SHARE THE SAME WEIGHTS.
   This is one network unrolled across 4 time steps.
```

### The equation

```
h_t = tanh( W_xh · x_t  +  W_hh · h_{t-1}  +  b_h )
      ────  ─────────────   ──────────────────  ────
      activation  new input     previous memory    bias

y_t = W_hy · h_t + b_y

Where:
  W_xh = input-to-hidden weights     (same at every step)
  W_hh = hidden-to-hidden weights    (same at every step — this IS the recurrence)
  W_hy = hidden-to-output weights
  h_t  = hidden state at time t      (the "memory")
  x_t  = input at time t
```

### Why tanh instead of sigmoid?

```
Sigmoid:  output range [0, 1],   max derivative = 0.25
Tanh:     output range [-1, 1],  max derivative = 1.0

Tanh lets gradients flow better (max derivative 4× larger).
Also, tanh is zero-centered — negative values are meaningful.

But tanh still squashes. 1.0 is the MAX derivative.
In practice, tanh'(z) < 1 for most z values.
The gradient still shrinks at every step. Just slower.
```

### Two early RNN variants

| | Jordan Network (1986) | Elman Network (1990) |
|---|---|---|
| Creator | Michael Jordan (UCSD) | Jeffrey Elman (UCSD) |
| Feeds back | **Output** to context units | **Hidden state** to context units |
| Memory | Remembers what it **said** | Remembers what it **thought** |
| Paper | "Serial Order" (tech report) | "Finding Structure in Time" |

Elman's design — feeding back the hidden state — became the standard RNN architecture. When people say "vanilla RNN" today, they mean the Elman network.

---

## Part 7: BPTT — Backpropagation Through Time

### The core idea

Unroll the RNN → it's just a very deep MLP with **shared weights** → run standard backprop on it.

Paul Werbos formalized this in his 1990 paper "Backpropagation Through Time: What It Does and How to Do It." He'd described backprop for neural nets in his 1974 PhD thesis; BPTT was the extension to recurrent networks.

### Visual: gradient flowing backward through time

```
    x₁           x₂           x₃           x₄
     │            │            │            │
     ▼            ▼            ▼            ▼
  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
  │ RNN  │──→│ RNN  │──→│ RNN  │──→│ RNN  │
  │ Cell │    │ Cell │    │ Cell │    │ Cell │
  └──────┘    └──────┘    └──────┘    └──────┘
     │            │            │            │
     ▼            ▼            ▼            ▼
    y₁           y₂           y₃           y₄

                 FORWARD PASS: ──────────────→  (left to right)

  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
  │ ∂L   │◄───│ ∂L   │◄───│ ∂L   │◄───│ ∂L   │◄─── Loss
  │ ∂h₁  │    │ ∂h₂  │    │ ∂h₃  │    │ ∂h₄  │
  └──────┘    └──────┘    └──────┘    └──────┘

                 BACKWARD PASS: ◄──────────────  (right to left)

  Gradient for W_hh at step t depends on ALL future steps.
  To update the shared weights, SUM the gradients from every time step.
```

### The connection to Kelley's rockets (note 09)

```
KELLEY'S ROCKETS (1960)              BPTT (1990)
─────────────────────                ────────────
Time stages: t₁, t₂, ..., tₙ       Time steps: x₁, x₂, ..., xₙ
State: position, velocity            State: hidden vector h_t
Controls: thrust angle               Controls: weights W
Cost: fuel burned at landing          Cost: loss at output
Propagate ∂cost/∂control backward    Propagate ∂loss/∂weight backward

SAME MATH. Kelley optimized flight paths through sequential stages.
BPTT optimizes predictions through sequential time steps.
The unrolled RNN IS Kelley's stage-by-stage system — with shared weights.
```

---

## Part 8: The Cliff — Why RNNs Break

Here's where the sigmoid derivative (Part 3) and BPTT (Part 7) collide.

### The math of forgetting

At each time step, the gradient passes through:
1. The activation function derivative (tanh': max 1.0, usually < 1)
2. The hidden-to-hidden weight matrix W_hh

```
Gradient at step t from a loss at step T:

∂L/∂h_t = ∂L/∂h_T × ∏(k=t to T-1) [ tanh'(h_k) × W_hh ]

If |tanh'(h_k) × W_hh| < 1 at each step:

Step back 5:    0.8⁵   = 0.328        ← still learning
Step back 10:   0.8¹⁰  = 0.107        ← struggling
Step back 20:   0.8²⁰  = 0.012        ← almost nothing
Step back 50:   0.8⁵⁰  = 0.00001      ← dead
Step back 100:  0.8¹⁰⁰ = 0.00000000002 ← the gradient is gone
```

### What this means for language

```
"The cat that I saw yesterday at the park near the old oak tree was ___"
  ↑                                                                  ↑
  word 1                                                          word 15

The verb "was" needs to agree with "cat" (singular).
That's 14 time steps of gradient multiplication.

0.8¹⁴ = 0.044

The gradient from "was" reaches "cat" at 4.4% strength.
The network can barely learn this dependency.

Now imagine 100 words between subject and verb:
0.8¹⁰⁰ ≈ 0  →  No learning happens. The RNN forgets "cat."
```

### Hochreiter's discovery

Sepp Hochreiter formally proved this in his 1991 diploma thesis at the Technical University of Munich: *"Untersuchungen zu dynamischen neuronalen Netzen"* ("Investigations into Dynamic Neural Networks"), supervised by Jürgen Schmidhuber. Written in German, it wasn't widely read — but the math was devastating.

Bengio, Simard, and Frasconi confirmed it in English in 1994: *"Learning Long-Term Dependencies with Gradient Descent is Difficult"* (IEEE Transactions on Neural Networks).

**This is why Hochreiter and Schmidhuber invented LSTM in 1997** — the cell state highway that keeps the gradient near 1.0 across time steps. (See notes 07 and 08 for how LSTM gates work and how to implement them.)

### The fix preview

```
VANILLA RNN:                              LSTM:

    h₁ ──tanh──→ h₂ ──tanh──→ h₃         c₁ ───────────→ c₂ ───────────→ c₃
         ×0.8         ×0.8                      ×f (≈1.0)      ×f (≈1.0)
                                                + new info      + new info
    Gradient: 0.8 × 0.8 = 0.64
    After 50 steps: 0.8⁵⁰ ≈ 0             Gradient: 1.0 × 1.0 = 1.0
                                           After 50 steps: 1.0⁵⁰ = 1.0

    RNN forgets.                           LSTM remembers.

    The cell state c_t is the highway.
    The forget gate f_t controls the on-ramp.
    When f ≈ 1, information flows unchanged. (Note 08: f_t * c_{t-1})
```

---

## Part 9: The Visual Timeline — MLP to Transformers

```
1986        1986        1990        1990        1991        1994        1997
 │           │           │           │           │           │           │
 ▼           ▼           ▼           ▼           ▼           ▼           ▼
MLP +       Jordan &    Elman       Werbos      Hochreiter  Bengio     LSTM
Backprop    early RNN   Network     formalizes  proves      confirms   (Hochreiter &
(Rumelhart  (PDP Ch.8)  "Finding    BPTT        vanishing   it in      Schmidhuber)
 Hinton      recurrent  Structure              gradient    English
 Williams)   connections in Time"               (German
             described                           thesis)
 │                                                                      │
 │  ◄───────── The Resurrection ──────────►  ◄── The Problem ──►       │
 │                                                                      │
 └──────────────────────── 11 years ────────────────────────────────────┘
         from backprop working to the fix for its biggest flaw


2014        2014        2017         2018+
 │           │           │            │
 ▼           ▼           ▼            ▼
GRU         Seq2Seq +   Transformer  BERT, GPT
(Cho et al) Attention   "Attention   Scale wins
2 gates     (Bahdanau)  Is All You
instead                 Need"
of 3                    (Vaswani)
 │                       │
 │  ◄── RNN era ──►     │  ◄── Attention era ──►
 │                       │
 Drop recurrence entirely. Self-attention looks at ALL
 positions in parallel — no more sequential bottleneck,
 no more vanishing gradient through time.
```

---

## The One-Sentence Story of This Note

**Rumelhart, Hinton, and Williams showed that stacking perceptrons into layers and chaining wiggle ratios backward could learn anything (MLP + backprop) — then immediately extended it to sequences by sharing weights across time (RNN) — but the same chain of wiggle ratios that made learning possible also made it decay exponentially over long sequences, which is why LSTM was invented 11 years later.**

---

## Cross-References

| Topic | See |
|-------|-----|
| Wiggle ratios & chain rule intuition | Note 09 — *Backpropagation: The Wiggle Ratio* |
| Minsky's XOR critique & the AI winter | Note 06 — *Minsky's Perceptrons — How One Book Killed a Field* |
| LSTM vs Seq2Seq vs Transformer comparison | Note 07 |
| LSTM gate equations & implementation | Note 08 — *Phase 1: Naive LSTM* |
| Full historical timeline | Note 00 — *Timeline: Neural Networks → Sequence Models → Transformers* |
