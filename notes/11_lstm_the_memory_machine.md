# LSTM: The Memory Machine

> **The RNN's problem was simple: gradients die over time. The fix was equally simple: give them a highway that doesn't squash. LSTM is that highway.**

---

## Part 1: The Problem (Brief)

RNNs forget. The gradient flowing backward through time multiplies by tanh' × W_hh at every step. After 50 steps: 0.8⁵⁰ ≈ 0. The error signal from word 50 never reaches word 1. The network can't learn long-range dependencies.

(Full derivation: note 10, Part 8. Intuition with wiggle ratios: note 09, Part 5.)

LSTM fixes this by adding a **second path** — the cell state — where information flows with **addition** instead of multiplication. Addition doesn't shrink gradients. That's the entire insight.

---

## Part 2: Hochreiter & Schmidhuber

### Sepp Hochreiter

In 1991, as a student at the Technical University of Munich, he wrote a diploma thesis (*"Untersuchungen zu dynamischen neuronalen Netzen"* — "Investigations into Dynamic Neural Networks") that **proved** vanishing gradients were a fundamental problem, not just a practical inconvenience. His advisor: Jürgen Schmidhuber.

### Jürgen Schmidhuber

Scientific Director of IDSIA (Swiss AI Lab) in Lugano, Switzerland from 1995. He supervised Hochreiter's thesis and co-authored the LSTM paper.

### The Paper

**"Long Short-Term Memory"** — *Neural Computation*, Volume 9, Issue 8, Pages 1735–1780, 1997. Communicated by Ronald Williams (the same Williams from the 1986 backprop paper).

The core concept: the **Constant Error Carousel (CEC)** — a self-connected unit with a fixed weight of 1.0. Error flows through it unchanged. Gates control what enters and exits, but the carousel itself preserves the gradient perfectly.

### What most people don't know

The **original 1997 LSTM had no forget gate**. Only two gates:
- Input gate: what goes into memory
- Output gate: what comes out of memory

The cell state could only **accumulate** — never erase. This worked for pre-segmented sequences but broke on continuous streams. The forget gate was added in 2000 by Gers, Schmidhuber, and Cummins (*"Learning to Forget: Continual Prediction with LSTM"*, Neural Computation 12(10)).

The LSTM you implement in the assignment is the **post-2000 version** with the forget gate included.

---

## Part 2b: The Gender Example — What Gates Actually Do

The classic way to understand LSTM gates: a language model predicting pronouns.

```
"The queen sat on her throne. The king entered and he ..."
```

While processing "The queen sat on her throne", the cell state encodes (among many things) something like **"current subject = female"** so it predicts "her".

Then "The king entered" arrives:

```
Step 1-6:  "The queen sat on her throne"
           Cell state: [..., female, ...]
           → predicts "her" ✓

Step 7:    "The king"
           Forget gate (f_t): "new subject — erase old gender info"
           Input gate (i_t):  "yes, write new info"
           Cell gate (g_t):   "the new value is: male"
           Output gate (o_t): "expose this for pronoun prediction"
           Cell state: [..., male, ...]
           → predicts "he" ✓
```

There's no literal "gender neuron." The hidden dimensions are just numbers. Through training, some combination of dimensions *ends up* encoding subject gender because that's useful for next-word prediction. The network figured that out on its own.

**The four gates in plain English:**

```
Forget (f_t):  "Is the old info still relevant? If not, erase it"
Input (i_t):   "Is there new info worth storing?" — the VALVE (sigmoid: 0-1)
Cell (g_t):    "What are the new values?" — the WATER (tanh: -1 to +1)
Output (o_t):  "Which parts of memory to expose right now?"

Sigmoid = the valve (how much)
tanh = the water (what content)
You need both.
```

### Why sigmoid on the input gate?

Without sigmoid, tanh alone would write to **every** dimension every timestep. Sigmoid controls **which** dimensions get written:

```
C̃_t (tanh):  [ 0.8,  -0.3,   0.5,  -0.9]    ← "proposed values"
i_t (σ):     [ 0.01,  0.99,  0.02,   0.98]   ← "which ones matter"

i_t ⊙ C̃_t:  [0.008, -0.297, 0.01,  -0.882]  ← only dims 2 and 4 get written
```

### Why tanh on the output?

The cell state `C_t` is unbounded — additions accumulate over time. tanh normalizes it back to (-1, +1) before exposing. The cell state is the **vault** (raw, unbounded); the hidden state is the **window** (normalized, filtered view).

```
C_t (vault):   the real memory — raw, unbounded, carries across time
h_t (window):  tanh(C_t) × o_t — normalized view, exposed to the world
```

---

## Part 3: The Big Picture — Two Parallel Tracks

This is the key diagram. Everything else is details:

```
THE TWO HIGHWAYS:

                    ×f_t            ×f_t            ×f_t
    Cell state:  c₀ ──→(forget)──→ c₁ ──→(forget)──→ c₂ ──→(forget)──→ c₃
                         + ↑               + ↑               + ↑
                       i_t×g_t           i_t×g_t           i_t×g_t
                     (new info)        (new info)        (new info)

                        │                  │                  │
                     tanh + o_t         tanh + o_t         tanh + o_t
                        │                  │                  │
                        ▼                  ▼                  ▼
    Hidden state: h₀ ────────────→ h₁ ────────────→ h₂ ────────────→ h₃


    Cell state  = LONG-TERM memory.  Gradient-friendly (additive path).
    Hidden state = WORKING memory.   Squashed through tanh for output.

    The cell state is the HIGHWAY.
    The hidden state is the OFF-RAMP.
```

### Compare to vanilla RNN

```
RNN (one path, squashed at every step):

    h₀ ──tanh──→ h₁ ──tanh──→ h₂ ──tanh──→ h₃
         ×W_hh        ×W_hh        ×W_hh
         ×tanh'       ×tanh'       ×tanh'      ← gradient shrinks EVERY step


LSTM (two paths — cell state skips the squashing):

    c₀ ────────→ c₁ ────────→ c₂ ────────→ c₃  ← gradient flows freely
                                                    (just multiplied by f_t ≈ 1)
    h₀ ────────→ h₁ ────────→ h₂ ────────→ h₃  ← output is still squashed,
                                                    but that's fine — the
                                                    gradient takes the top path
```

---

## Part 4: Gate by Gate — Why 4, Not 2, Not 3

Every gate uses **sigmoid** (output 0 to 1) because a gate is a **dial**:
- 0 = "block everything"
- 1 = "pass everything"
- 0.7 = "pass 70%"

Sigmoid is the only standard activation that naturally maps to [0, 1]. That's why gates use sigmoid, not tanh or ReLU.

---

### Gate 1: Forget Gate (f_t) — "What should I erase from memory?"

```
f_t = σ( x_t @ W_if + b_if  +  h_{t-1} @ W_hf + b_hf )
      ──
      sigmoid
      (0 to 1)

Then:  c_t = f_t × c_{t-1} + ...
                ↑
                this multiplication ERASES old memory

f_t = 1.0  →  keep ALL of c_{t-1}     (remember everything)
f_t = 0.0  →  erase ALL of c_{t-1}    (forget everything)
f_t = 0.7  →  keep 70% of c_{t-1}     (fade slightly)
```

**Analogy:** You're reading a book. You finish a chapter about cats. New chapter is about dogs. The forget gate says: "erase the cat stuff (f ≈ 0), but keep the setting description (f ≈ 1) — we're still in the same park."

**Practical tip:** Initialize the forget gate bias to **1.0** (not 0). This makes σ(1.0) ≈ 0.73, so the network starts by **remembering most things** and learns what to forget. With bias = 0, σ(0) = 0.5, and the network immediately discards half its memory before learning anything. This one trick (Jozefowicz et al., 2015) can make the difference between LSTM working and failing.

---

### Gate 2: Input Gate (i_t) — "What new info is worth storing?"

```
i_t = σ( x_t @ W_ii + b_ii  +  h_{t-1} @ W_hi + b_hi )

i_t = 1.0  →  fully write new info to cell state
i_t = 0.0  →  ignore this input completely
i_t = 0.3  →  write 30% of new info (not very important)
```

**Analogy:** A bouncer at the door of memory. Not everything you read deserves to be remembered. "The" and "a" are low-importance — input gate stays low. "explosion" after a sequence of calm words — input gate spikes.

---

### Gate 3: Cell Gate / Candidate (g_t) — "What are the proposed new values?"

```
g_t = tanh( x_t @ W_ig + b_ig  +  h_{t-1} @ W_hg + b_hg )
      ────
      tanh (NOT sigmoid!)
      output: [-1, +1]
```

**Why tanh, not sigmoid?** This isn't a gate (a dial). This is the actual **content** to be written. Content needs to be both positive ("this feature is present") and negative ("this feature is absent"). Sigmoid can only output positive values. Tanh gives the full range [-1, +1].

**Analogy:** The input gate is the bouncer. The cell gate is the **guest** trying to get in. The bouncer (i_t) decides how much of the guest (g_t) actually enters.

```
What actually gets written to memory:

    i_t × g_t

    = (how much to write) × (what to write)
    = 0.3 × [-0.8, +0.5]
    = [-0.24, +0.15]        ← only 30% of the proposed values get in
```

---

### Gate 4: Output Gate (o_t) — "What should I expose as my answer?"

```
o_t = σ( x_t @ W_io + b_io  +  h_{t-1} @ W_ho + b_ho )

Then:  h_t = o_t × tanh(c_t)
```

**Why do we need this?** The cell state stores EVERYTHING the network has ever decided to remember. But at any given time step, only some of that is relevant to the current output.

**Analogy:** Your brain knows thousands of facts. But when someone asks "What's 2+2?", you don't dump everything — your output gate filters for just the math-relevant neurons.

```
c_t = [0.9, -0.3, 0.7, 0.1]     (cell state: all accumulated knowledge)
o_t = [1.0,  0.0, 0.8, 0.2]     (output gate: what's relevant NOW)

h_t = o_t × tanh(c_t)
    = [1.0, 0.0, 0.8, 0.2] × tanh([0.9, -0.3, 0.7, 0.1])
    = [1.0, 0.0, 0.8, 0.2] × [0.72, -0.29, 0.60, 0.10]
    = [0.72, 0.00, 0.48, 0.02]

    Dimension 1: fully exposed (o=1.0)
    Dimension 2: completely hidden (o=0.0) — it's in memory but not in output
    Dimension 3: mostly exposed (o=0.8)
    Dimension 4: barely exposed (o=0.2)
```

---

### Why 4 gates and not fewer?

| If you removed... | What breaks |
|---|---|
| Forget gate | Memory grows forever, can't reset context (original 1997 problem) |
| Input gate | Every input overwrites memory — no selectivity |
| Cell gate (candidate) | No new content to write — memory is frozen |
| Output gate | Everything in memory leaks into output — no filtering |

You need all 4. GRU (Part 8) shows what happens when you reduce to 2 — it works, but with trade-offs.

---

## Part 5: The Cell State Update — Why Addition Saves Gradients

This is the single most important equation in the LSTM:

```
c_t = f_t × c_{t-1}  +  i_t × g_t
      ──────────────     ──────────
      what to KEEP       what to ADD
      from old memory    as new memory
```

### Why this saves gradients

Take the derivative:

```
∂c_t / ∂c_{t-1} = f_t

That's it. The gradient through the cell state is just the forget gate value.

If f_t ≈ 1 (remember):  gradient ≈ 1.0   → passes through unchanged
If f_t ≈ 0 (forget):    gradient ≈ 0.0   → intentionally blocked


Contrast with vanilla RNN:

∂h_t / ∂h_{t-1} = tanh'(z) × W_hh

    tanh'(z) is almost always < 1
    Then multiplied by W_hh
    Product is typically 0.5–0.9
    After 50 steps: 0.8⁵⁰ ≈ 0.00001  ← DEAD
```

### The key difference: ADDITIVE vs MULTIPLICATIVE

```
RNN:   h_t = tanh( W × h_{t-1} + ... )     ← h_{t-1} is INSIDE the tanh
                                                every step squashes
                                                gradient: tanh' × W (< 1)

LSTM:  c_t = f_t × c_{t-1} + i_t × g_t     ← c_{t-1} is OUTSIDE any activation
                                                it's just multiplied by f_t
                                                gradient: f_t (≈ 1 when remembering)
```

The cell state **bypasses the activation function**. That's the whole trick. The gradient takes the highway on top, not the squashed road on the bottom.

---

## Part 6: Full Numerical Walkthrough

Let's process "the cat sat" through an LSTM cell with hidden_size=2. Small enough to follow every number.

### Setup

```
hidden_size = 2,  input_size = 3 (one-hot-ish)

Initial states:
  h₀ = [0, 0]
  c₀ = [0, 0]

Inputs (simplified embeddings):
  x₁ = "the" = [1, 0, 0]
  x₂ = "cat" = [0, 1, 0]
  x₃ = "sat" = [0, 0, 1]

I'll use small made-up weights. In reality these are learned.
For clarity, I'll show the RESULT of each gate, not every matrix multiply.
```

### Time step 1: "the"

```
Input: x₁ = [1, 0, 0],  h₀ = [0, 0],  c₀ = [0, 0]

Gate computations (after W×x + W×h + b, then activation):

  f₁ = σ(...)  = [0.6, 0.7]      forget gate: moderate retention
  i₁ = σ(...)  = [0.8, 0.3]      input gate: dim 1 important, dim 2 not
  g₁ = tanh(.) = [0.5, -0.4]     candidate: proposed new values
  o₁ = σ(...)  = [0.9, 0.5]      output gate: dim 1 exposed, dim 2 half

Cell state update:
  c₁ = f₁ × c₀      +  i₁ × g₁
     = [0.6, 0.7]×[0,0] + [0.8, 0.3]×[0.5, -0.4]
     = [0, 0]            + [0.4, -0.12]
     = [0.4, -0.12]

  c₀ was zeros, so forget gate doesn't matter yet.
  Memory now holds: [0.4, -0.12]  ("the" encoded)

Hidden state update:
  h₁ = o₁ × tanh(c₁)
     = [0.9, 0.5] × tanh([0.4, -0.12])
     = [0.9, 0.5] × [0.38, -0.12]
     = [0.34, -0.06]
```

### Time step 2: "cat"

```
Input: x₂ = [0, 1, 0],  h₁ = [0.34, -0.06],  c₁ = [0.4, -0.12]

Gate results:
  f₂ = σ(...)  = [0.9, 0.3]      forget: keep dim 1 (high), erase dim 2 (low)
  i₂ = σ(...)  = [0.7, 0.9]      input: dim 2 now important!
  g₂ = tanh(.) = [0.2, 0.8]      candidate: "cat" features
  o₂ = σ(...)  = [0.6, 0.8]      output: both dims somewhat exposed

Cell state update:
  c₂ = f₂ × c₁       +  i₂ × g₂
     = [0.9, 0.3]×[0.4, -0.12] + [0.7, 0.9]×[0.2, 0.8]
     = [0.36, -0.036]           + [0.14, 0.72]
     = [0.50, 0.68]
                  ↑
                  dim 2 went from -0.12 to 0.68
                  forget gate ERASED most of old dim 2 (×0.3)
                  input gate WROTE strong new value (+0.72)
                  "cat" overwrote "the" in dimension 2

Hidden state:
  h₂ = o₂ × tanh(c₂)
     = [0.6, 0.8] × tanh([0.50, 0.68])
     = [0.6, 0.8] × [0.46, 0.59]
     = [0.28, 0.47]
```

### Time step 3: "sat"

```
Input: x₃ = [0, 0, 1],  h₂ = [0.28, 0.47],  c₂ = [0.50, 0.68]

Gate results:
  f₃ = σ(...)  = [0.95, 0.9]     forget: keep almost everything!
  i₃ = σ(...)  = [0.4, 0.2]      input: not much new to write
  g₃ = tanh(.) = [0.6, -0.3]     candidate: "sat" features
  o₃ = σ(...)  = [0.8, 0.9]      output: expose both dims

Cell state update:
  c₃ = f₃ × c₂       +  i₃ × g₃
     = [0.95, 0.9]×[0.50, 0.68] + [0.4, 0.2]×[0.6, -0.3]
     = [0.475, 0.612]            + [0.24, -0.06]
     = [0.715, 0.552]
                  ↑
                  Both dims PRESERVED from step 2 (f ≈ 0.9)
                  Small addition from "sat"
                  The network REMEMBERS "cat" through step 3!

Hidden state:
  h₃ = o₃ × tanh(c₃)
     = [0.8, 0.9] × tanh([0.715, 0.552])
     = [0.8, 0.9] × [0.61, 0.50]
     = [0.49, 0.45]
```

### Gradient comparison: backward from step 3 to step 1

```
THROUGH THE CELL STATE (LSTM):

  ∂c₃/∂c₂ = f₃ = [0.95, 0.9]
  ∂c₂/∂c₁ = f₂ = [0.9, 0.3]

  ∂c₃/∂c₁ = f₃ × f₂ = [0.95×0.9, 0.9×0.3] = [0.855, 0.27]

  Dim 1: 85.5% of gradient survives!  The network remembered.
  Dim 2: 27% — the network CHOSE to forget (f₂ was 0.3).
         That's not a bug — that's the forget gate working.


IF THIS WERE A VANILLA RNN (same 3 steps):

  ∂h₃/∂h₁ = tanh'(z₂) × W × tanh'(z₁) × W
           ≈ 0.8 × W × 0.8 × W
           ≈ 0.64 × W²

  With typical W, this is ~0.3–0.5.
  After 50 steps: basically zero.
```

---

## Part 7: The Hidden State Update

```
h_t = o_t × tanh(c_t)
```

Why tanh on c_t? The cell state can grow large (it's additive — values accumulate over time). Tanh squashes it back to [-1, 1] for a well-behaved output. But this tanh is only on the **output path** — the cell state itself is NOT squashed when passing to the next step. That's the key.

```
Cell state path (gradient highway):

  c_{t-1}  ──────────────────→  c_t  ──────────────────→  c_{t+1}
           NO tanh here              NO tanh here
           just ×f + new            just ×f + new


Output path (squashed for use):

  c_t ──→ tanh ──→ × o_t ──→ h_t ──→ (used for predictions, next step's gates)
           ↑                    ↑
           squashed for         filtered for
           bounded output       relevance
```

The gradient can take EITHER path backward. Through the cell state (top), it survives. Through the hidden state (bottom), it squashes — but it doesn't have to. The cell state highway is always available as an alternative route.

---

## Part 8: GRU — The Simplified Version

Cho et al. (2014), *"Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"* (EMNLP). The key idea: **merge cell state and hidden state into one**, reduce 4 gates to 2.

### GRU equations

```
z_t = σ( x_t @ W_xz + h_{t-1} @ W_hz + b_z )      ← UPDATE gate
r_t = σ( x_t @ W_xr + h_{t-1} @ W_hr + b_r )      ← RESET gate

h̃_t = tanh( x_t @ W_xh + (r_t × h_{t-1}) @ W_hh + b_h )   ← candidate

h_t  = z_t × h_{t-1}  +  (1 - z_t) × h̃_t           ← final state
       ──────────────     ─────────────────
       what to KEEP        what to REPLACE with
```

### The clever trick: coupled gates

```
LSTM has SEPARATE forget and input gates:
  c_t = f_t × c_{t-1} + i_t × g_t
        ↑                ↑
        independent      independent     (f and i can both be 1.0 or both 0.0)

GRU COUPLES them:
  h_t = z_t × h_{t-1} + (1 - z_t) × h̃_t
        ↑                ↑
        z_t              (1 - z_t)       (if you keep 70%, you write 30%)

  z_t = 1.0:  keep ALL old state,  add NOTHING new     (pure memory)
  z_t = 0.0:  erase ALL old state, use ALL new          (pure reset)
  z_t = 0.7:  keep 70% old,        add 30% new          (blended)

  You can't keep 100% AND write 100%. That's the trade-off.
```

### LSTM vs GRU comparison

| | LSTM | GRU |
|---|---|---|
| States | 2 (cell + hidden) | 1 (hidden only) |
| Gates | 4 (forget, input, cell, output) | 2 (update, reset) |
| Parameters | ~4 × (n² + n×m) | ~3 × (n² + n×m) |
| Forget/input | Independent | Coupled (z and 1-z) |
| Very long sequences | Better (separate cell state highway) | Slightly worse |
| Training speed | Slower (more params) | Faster (~25% fewer params) |
| When to use | Long documents, complex dependencies | Shorter sequences, faster training |

In practice, the difference is often small. Jozefowicz et al. (2015) found that with proper forget gate bias initialization (bias=1), LSTM and GRU perform comparably on most tasks.

---

## Part 9: Connecting to Your Assignment Code

The file `models/naive/LSTM.py` implements the post-2000 LSTM (with forget gate) from scratch.

### Weight mapping

```
EQUATION                                           CODE PARAMETERS
────────                                           ───────────────
i_t = σ( x_t @ W_ii + b_ii + h_{t-1} @ W_hi + b_hi )   W_ii, b_ii, W_hi, b_hi
f_t = σ( x_t @ W_if + b_if + h_{t-1} @ W_hf + b_hf )   W_if, b_if, W_hf, b_hf
g_t = tanh(x_t @ W_ig + b_ig + h_{t-1} @ W_hg + b_hg)  W_ig, b_ig, W_hg, b_hg
o_t = σ( x_t @ W_io + b_io + h_{t-1} @ W_ho + b_ho )   W_io, b_io, W_ho, b_ho

Naming convention:
  W_  i  f  →  Weight for Input-to-Forget gate
      ↑  ↑
      │  └─ which gate (i=input, f=forget, g=cell, o=output)
      └──── source (i=input x_t, h=hidden h_{t-1})

Total: 8 weight matrices + 8 bias vectors = 16 nn.Parameters
```

### Weight shapes

```
W_i* : (input_size, hidden_size)    ← multiplied as: x_t @ W_i*
W_h* : (hidden_size, hidden_size)   ← multiplied as: h_{t-1} @ W_h*
b_*  : (hidden_size,)               ← added after matmul

Why x @ W and NOT W @ x?
  x shape:     (batch, input_size)
  W shape:     (input_size, hidden_size)
  x @ W shape: (batch, hidden_size)  ✓

  If W were (hidden_size, input_size), you'd need W @ x.T — messier.
  The assignment shapes weights so x @ W works WITHOUT transposing.
```

### Init order (must match autograder)

```python
# 1. Input gate
self.W_ii = nn.Parameter(torch.Tensor(input_size, hidden_size))
self.b_ii = nn.Parameter(torch.Tensor(hidden_size))
self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
self.b_hi = nn.Parameter(torch.Tensor(hidden_size))

# 2. Forget gate
self.W_if = nn.Parameter(torch.Tensor(input_size, hidden_size))
self.b_if = nn.Parameter(torch.Tensor(hidden_size))
self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
self.b_hf = nn.Parameter(torch.Tensor(hidden_size))

# 3. Cell gate (candidate)
self.W_ig = nn.Parameter(torch.Tensor(input_size, hidden_size))
self.b_ig = nn.Parameter(torch.Tensor(hidden_size))
self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
self.b_hg = nn.Parameter(torch.Tensor(hidden_size))

# 4. Output gate
self.W_io = nn.Parameter(torch.Tensor(input_size, hidden_size))
self.b_io = nn.Parameter(torch.Tensor(hidden_size))
self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
self.b_ho = nn.Parameter(torch.Tensor(hidden_size))

# Activations
self.sigmoid = nn.Sigmoid()
self.tanh = nn.Tanh()
```

### Forward pass structure

```python
def forward(self, x):
    # x shape: (batch, sequence_length, input_size)
    batch_size, seq_len, _ = x.shape

    h_t = torch.zeros(batch_size, self.hidden_size)
    c_t = torch.zeros(batch_size, self.hidden_size)

    for t in range(seq_len):
        x_t = x[:, t, :]  # (batch, input_size)

        i_t = self.sigmoid(x_t @ self.W_ii + self.b_ii + h_t @ self.W_hi + self.b_hi)
        f_t = self.sigmoid(x_t @ self.W_if + self.b_if + h_t @ self.W_hf + self.b_hf)
        g_t = self.tanh(x_t @ self.W_ig + self.b_ig + h_t @ self.W_hg + self.b_hg)
        o_t = self.sigmoid(x_t @ self.W_io + self.b_io + h_t @ self.W_ho + self.b_ho)

        c_t = f_t * c_t + i_t * g_t    # cell state update
        h_t = o_t * self.tanh(c_t)      # hidden state update

    return (h_t, c_t)
```

### Common gotcha from note 08

Note 08 lists the output gate as `o_t = tanh(...)`. In the standard LSTM formulation, the output gate uses **sigmoid** (like all gates), not tanh. The tanh in the hidden state equation `h_t = o_t * tanh(c_t)` is applied to the **cell state**, not to the gate itself. The code above uses sigmoid for o_t, which matches the standard formulation and PyTorch's `nn.LSTM`.

---

## Part 10: Visual — LSTM Cell Internals

```
                         CELL STATE (the highway)
        c_{t-1} ───────────────(×)─────────────(+)───────────────── c_t
                                ↑               ↑                    │
                                │               │                    │
                             ┌──┴──┐        ┌───┴────┐              │
                             │     │        │        │              │
                             │ f_t │        │ i_t×g_t│              │
                             │     │        │        │              │
                             └──┬──┘        └───┬────┘              │
                                │               │                    │
                                │          ┌────┴────┐              │
                                │          │    ×    │              │
                                │       ┌──┴─┐  ┌─┴──┐            │
                                │       │ i_t│  │ g_t│            │
                                │       │  σ │  │tanh│            │
                                │       └──┬─┘  └─┬──┘            │
                                │          │      │                │
                         ┌──────┴──────────┴──────┴──┐      ┌─────┴─────┐
                         │                           │      │           │
     x_t ──────────────→ │     ALL GATES RECEIVE:    │      │   tanh    │
                         │     x_t and h_{t-1}       │      │           │
     h_{t-1} ──────────→ │                           │      └─────┬─────┘
                         │                           │            │
                         └──────────┬────────────────┘         ┌──┴──┐
                                    │                          │     │
                                 ┌──┴──┐                       │ o_t │
                                 │ f_t │                       │  σ  │
                                 │  σ  │                       └──┬──┘
                                 └─────┘                          │
                                                               ┌──┴──┐
                                                               │  ×  │
                                                               └──┬──┘
                                                                  │
                                                                  ▼
                                                         h_t ───────── → (output & next step)


SUMMARY OF DATA FLOW:
  1. x_t and h_{t-1} enter from the left
  2. Four gates compute simultaneously (f, i, g, o)
  3. Forget gate (f_t) selectively erases old cell state
  4. Input gate (i_t) × candidate (g_t) adds new information
  5. Cell state updates: c_t = f×c + i×g  (top highway)
  6. Output gate (o_t) filters tanh(c_t) to produce h_t (bottom path)
  7. h_t goes to output AND feeds into the next time step's gates
```

---

## The One-Sentence Story

**LSTM adds a cell state highway above the RNN's hidden state — information flows through it with addition (not squashing), so the gradient survives across hundreds of time steps, and four learned gates control what to forget, what to store, what to write, and what to expose.**

---

## Cross-References

| Topic | See |
|-------|-----|
| Vanishing gradient math deep dive (tanh, worked example) | Note 13 |
| Vanishing gradient intuition (wiggle ratios) | Note 09, Part 5 |
| RNN cliff: math of gradient decay through time | Note 10, Part 8 |
| LSTM implementation checklist (weight shapes, init order) | Note 08 |
| LSTM vs Seq2Seq vs Transformer comparison | Note 07 |
| Hochreiter 1991 thesis & Bengio 1994 confirmation | Note 10, Part 8 |
| Historical timeline (1997 LSTM in context) | Note 00 |
| From MLP to RNN: the conceptual leap | Note 10, Part 5 |
