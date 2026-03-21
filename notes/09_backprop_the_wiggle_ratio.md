# Backpropagation: The Wiggle Ratio

> **The partial derivative is just the wiggle ratio — nothing more.**
> If I wiggle this input, how much does the output wiggle? Chain those ratios together and you know how the very first knob affects the very last number. That's backprop.

---

## Why This Matters

Every time you train a neural network — LSTM, Seq2Seq, Transformer — backprop is running. Understanding it intuitively means understanding WHY training works, WHY gradients vanish, WHY LSTMs needed gates, and WHY Kelley's 1960 rocket math is the same algorithm Rumelhart used in 1986.

---

## Part 1: What "Sensitivity" Actually Means

Forget calculus notation. Sensitivity just means:

**"If I wiggle this input, how much does the output wiggle?"**

```
DEAD SIMPLE EXAMPLE:

  You set your thermostat to 70°F.
  Room reaches 70°F.

  You bump it to 71°F.
  Room reaches 71°F.

  Sensitivity = 1°F out / 1°F in = 1.0
  "The room responds exactly as much as you asked."

  Now imagine your thermostat is broken:
  You bump it to 71°F.
  Room reaches 75°F.

  Sensitivity = 5°F out / 1°F in = 5.0
  "The room OVERREACTS to your input."
```

That ratio — output wiggle / input wiggle — **IS** the partial derivative. That's all ∂output/∂input means.

---

## Part 2: A Full Backward Pass with Actual Numbers

Tiny network: 2 layers, 1 neuron each, trying to predict that the answer should be **1.0**.

### Forward Pass

```
INPUT        LAYER 1           LAYER 2          LOSS
 x=2.0       w₁=0.3            w₂=0.5
              │                  │
              ▼                  ▼
         x × w₁ = 0.6      0.6 × w₂ = 0.3     (0.3 - 1.0)² = 0.49
              │                  │                    │
           hidden=0.6        output=0.3          loss=0.49
                                                 (we wanted 1.0, got 0.3)
                                                 that's BAD
```

We got 0.3, wanted 1.0. Loss = 0.49. Now the question: **which weight should we change, and by how much?**

### Backward Pass — Step by Step, No Symbols, Just Numbers

**Step 1: How sensitive is the loss to the output?**

```
output = 0.3  →  loss = (0.3 - 1.0)² = 0.49
output = 0.4  →  loss = (0.4 - 1.0)² = 0.36     ← loss dropped by 0.13!
output = 0.5  →  loss = (0.5 - 1.0)² = 0.25      ← dropped more!

Sensitivity: when output goes up by 0.1, loss goes down by ~0.13

Exact: ∂loss/∂output = 2 × (0.3 - 1.0) = -1.4

Translation: "Every +1 unit of output REDUCES loss by 1.4"
             "Output is VERY influential on loss right now"
             "We desperately need more output"
```

**Step 2: How sensitive is the output to w₂?**

```
output = hidden × w₂ = 0.6 × w₂

w₂ = 0.5  →  output = 0.6 × 0.5 = 0.30
w₂ = 0.6  →  output = 0.6 × 0.6 = 0.36
w₂ = 0.7  →  output = 0.6 × 0.7 = 0.42

Sensitivity: when w₂ goes up by 0.1, output goes up by 0.06

Exact: ∂output/∂w₂ = 0.6 (the hidden value)

Translation: "w₂ has moderate influence — limited by how big
              the hidden layer value is"
```

**Step 3: CHAIN THEM — how sensitive is loss to w₂?**

```
This is the chain rule, but watch — it's just multiplication:

loss sensitivity to w₂ = (loss sensitivity to output)
                          × (output sensitivity to w₂)

                        = -1.4 × 0.6
                        = -0.84

Translation: "If I increase w₂ by a tiny amount,
              loss DECREASES by 0.84 × that amount"

SO: increase w₂! That's gradient descent.

w₂_new = 0.5 - (learning_rate × -0.84)
       = 0.5 + 0.084    (with lr=0.1)
       = 0.584           ← w₂ got bigger, which makes output bigger,
                            which gets us closer to 1.0
```

**Step 4: Keep going backward — how sensitive is loss to w₁?**

This is where multi-layer gets interesting:

```
loss is sensitive to output by:        -1.4
output is sensitive to hidden by:       w₂ = 0.5
hidden is sensitive to w₁ by:           x = 2.0

Chain them ALL:

loss sensitivity to w₁ = -1.4 × 0.5 × 2.0 = -1.4

Translation: "w₁ actually has MORE influence than w₂ here
              because the input (2.0) amplifies it"

w₁_new = 0.3 - (0.1 × -1.4)
       = 0.3 + 0.14
       = 0.44
```

### Verify — Run Forward Again with Updated Weights

```
BEFORE:  x=2.0 → ×w₁(0.3) → 0.6 → ×w₂(0.5) → 0.30 → loss=0.49
AFTER:   x=2.0 → ×w₁(0.44)→ 0.88→ ×w₂(0.584)→ 0.51 → loss=0.24

Loss dropped from 0.49 to 0.24 in ONE step!
Output went from 0.30 to 0.51 (closer to our target of 1.0)
```

---

## Part 3: Why It's the Same as Kelley's Rockets

```
SAME LOGIC, DIFFERENT WORDS:

NEURAL NET                              ROCKET (Kelley, 1960)
──────────                              ──────────────────────
"If I change w₁ by 0.01,               "If I change thrust angle
 how much does the loss                  at t₁ by 0.01°,
 change?"                                how much extra fuel
                                         do I burn by landing?"

Wiggle w₁ →                             Wiggle angle₁ →
  hidden changes →                        position at t₂ changes →
    output changes →                        drag at t₃ changes →
      loss changes                            fuel at tₙ changes

Sensitivity flows backward:             Sensitivity flows backward:
  loss → output → hidden → w₁             fuel → tₙ → t₃ → t₂ → angle₁

Update w₁ to reduce loss                Update angle₁ to reduce fuel
```

Kelley published "Gradient Theory of Optimal Flight Paths" in the ARS Journal (American Rocket Society, 1960). He was optimizing ICBM trajectories, reentry vehicles, and aircraft fuel efficiency. The math is identical to backprop — propagate the cost gradient backward through sequential stages using the chain rule.

He just never applied it to neural networks. Filed under "rocket science" for 26 years.

---

## Part 4: The Chain Rule Is Just Chained Wiggle Ratios

The chain rule in calculus looks scary:

```
∂L/∂w₁ = ∂L/∂a₃ × ∂a₃/∂a₂ × ∂a₂/∂w₁
```

But it's literally:

```
"How much does the loss wiggle     "How much does layer 3 wiggle
 when layer 3 wiggles?"        ×    when layer 2 wiggles?"
                               ×   "How much does layer 2 wiggle
                                    when w₁ wiggles?"

= "How much does the loss wiggle when w₁ wiggles?"
```

Each link in the chain is one layer's wiggle ratio. Multiply them all together and you've chained through the whole network. That's why it's called the chain rule.

---

## Part 5: Why Gradients Vanish (and Why LSTMs Fix It)

Now you can see the problem. Each wiggle ratio is typically **less than 1** (because of sigmoid/tanh activations):

```
DEEP NETWORK (10 layers):

∂L/∂w₁ = 0.8 × 0.7 × 0.6 × 0.9 × 0.5 × 0.8 × 0.7 × 0.6 × 0.9 × 0.5

        = 0.8 × 0.7 × 0.6 × ...  ← each multiplication SHRINKS it

        = 0.003

Translation: "By the time the error signal reaches layer 1,
              it's been multiplied by 0.003.
              Layer 1 barely feels anything.
              It can't learn."

THIS IS THE VANISHING GRADIENT PROBLEM (Hochreiter, 1990)
```

For RNNs it's even worse — the "layers" are TIME STEPS, so a 100-word sentence means 100 multiplications:

```
RNN processing "The cat that I saw yesterday at the park was ..."

Word 100 → word 99 → word 98 → ... → word 1

Each step multiplies by ~0.8:  0.8¹⁰⁰ = 0.00000000000000000002

The gradient from "was" never reaches "cat."
The RNN forgets.
```

**LSTM solution (Hochreiter & Schmidhuber, 1997):** Add a **cell state** highway that flows through time with a wiggle ratio of **~1.0**:

```
LSTM cell state:

c₁ ──→ c₂ ──→ c₃ ──→ ... ──→ c₁₀₀

Each step: c_new = forget_gate × c_old + input_gate × new_info

If forget_gate ≈ 1:  the gradient flows through UNCHANGED.
Wiggle ratio = 1.0 at every step.
1.0¹⁰⁰ = 1.0  ← gradient survives!

"The cat" information reaches "was" 100 steps later.
```

That's the whole point of Assignment Phase 1: you're building the gates that keep the wiggle ratio near 1.0 so gradients survive across long sequences.

---

## Part 6: How Layers Build Abstraction (Multi-Layer Intuition)

A single layer can only draw a straight line through the data. Multiple layers build features on features:

```
WHAT EACH LAYER "LEARNS":

Layer 1 weights:    edges, colors, brightness
                         ↓
Layer 2 weights:    textures, corners, simple shapes
                         ↓
Layer 3 weights:    eyes, ears, wheels, windows
                         ↓
Layer 4 weights:    faces, cars, cats, dogs
                         ↓
Output weights:     "this is a cat" (0.94)

EACH LAYER'S WEIGHTS = a different LEVEL OF ABSTRACTION
    pixels → edges → parts → objects → answer
```

Layer 1 doesn't answer the question. It builds features. Layer 2 combines those features into higher features. Layer N uses everything below it to answer.

### XOR: The Simplest Multi-Layer Example

XOR is the problem that killed the Perceptron (Minsky & Papert, 1969) and required multi-layer networks to solve:

```
INPUT       DESIRED OUTPUT       WHY IT'S HARD
(0, 0)  →   0                    No single straight line
(0, 1)  →   1                    can separate 0s from 1s
(1, 0)  →   1
(1, 1)  →   0

WITH 2 LAYERS:

Layer 1, Neuron A (weights: [1, 1], bias: -0.5):
  "Are EITHER of the inputs on?"        → OR gate

Layer 1, Neuron B (weights: [1, 1], bias: -1.5):
  "Are BOTH inputs on?"                 → AND gate

Layer 2, Neuron C (weights: [+1, -2], bias: -0.5):
  "Is A on BUT B off?"                  → OR minus AND = XOR!

  Input   →  A (OR)  →  B (AND)  →  C: A - 2B  →  Output
  (0,0)       0          0           0 - 0 = 0      0 ✓
  (0,1)       1          0           1 - 0 = 1      1 ✓
  (1,0)       1          0           1 - 0 = 1      1 ✓
  (1,1)       1          1           1 - 2 = -1     0 ✓

Layer 1 asked two simple yes/no questions.
Layer 2 combined the answers.
Neither layer alone could solve XOR. Together they can.
```

---

## Part 7: The Timeline of "Almost Backprop"

```
1960  Kelley         gradient backward through flight paths    (aerospace)
1962  Dreyfus        same thing, cleaner, dynamic programming  (RAND/defense)
1962  Rosenblatt     coins "back-propagating errors"           (neural nets)
                     but can't implement it
1970  Linnainmaa     automatic differentiation (general math)  (Finland)
1974  Werbos         applies it to neural nets in PhD thesis   (Harvard)
                     but nobody reads it
1986  Rumelhart      publishes in Nature, everyone notices     (FINALLY)
      Hinton
      Williams

      26 YEARS from Kelley to adoption.
      The math existed. The connection didn't.
```

The tragedy: if someone in 1962 had walked from the aerospace department to the neural network lab and said "hey, your back-propagating errors thing? we've been doing that for rockets" — the AI winter might never have happened.

---

## The One-Sentence Version

**Backprop = measure how much the final answer wiggles when you wiggle each weight, by chaining wiggle ratios backward through every layer, then nudge every weight in the direction that reduces the error.**

That's it. Kelley did it for rockets. Rumelhart did it for neurons. Your LSTM assignment does it for sequences. Same wiggle ratios, same chain, same idea.
