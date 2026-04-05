# 07. Markov Chain vs RNN vs LSTM vs Transformer

## Why this note exists

While reading [03_forward_process.md](03_forward_process.md), a natural instinct hits: *"a Markov chain sounds like the hidden state of an RNN or LSTM — each step only depends on the previous state."* That instinct is partly right and partly wrong, and the difference matters. This note pins it down, then answers the follow-up: *"but it's not a transformer, right?"*

---

## The shared instinct

All four mechanisms — Markov chain, RNN, LSTM, Transformer — are answers to the same question:

> **"When you're at step $t$ in a sequence, how do you know about the past?"**

But they answer it in four very different ways. Let's walk them.

---

## 1. Markov chain — "the state is the data"

Formal definition (from [03_forward_process.md](03_forward_process.md) Part 1):

$$
P(x_{t+1} \mid x_t, x_{t-1}, \ldots, x_0) = P(x_{t+1} \mid x_t)
$$

Key traits:

- **State = the raw data itself.** In DDPM's forward process, $x_t$ *is* a full image tensor. There's no separate "summary variable."
- **No learned parameters.** The transition rule (add Gaussian noise with variance $\beta_t$) is fixed by the designer.
- **First-order.** You only need yesterday's state to predict today's; day-before-yesterday is forgotten.
- **"Memoryless given the present."** This does not mean the chain has amnesia. It means the present state is assumed to contain everything useful from the past.

Mental picture: a pachinko ball bouncing from peg to peg. Its next position depends on where it is now and a random deflection. Its history of bounces doesn't matter.

---

## 2. RNN hidden state — "a learned summary of the past"

Cross-reference: [../10_mlp_backprop_and_the_birth_of_rnns.md](../10_mlp_backprop_and_the_birth_of_rnns.md).

The recurrent neural network (Elman, 1990; Jordan, 1986) introduced the idea of a **hidden state** $h_t$ that is updated at each step:

$$
\begin{aligned}
h_t &= f_W(h_{t-1}, x_t) \\
y_t &= g_V(h_t)
\end{aligned}
$$

Where:

- $x_t$ is the input at step $t$ (a word, a pixel, whatever).
- $h_t$ is the hidden state — a fixed-size vector that is supposed to summarize everything useful from the input sequence so far.
- $f_W$ and $g_V$ are neural networks with *learned* weights $W$ and $V$.

Key traits:

- **The state is a learned compression**, not the raw data. The RNN decides, through training, what to cram into $h_t$.
- **Markov in the hidden state.** Given $h_t$, the next hidden state $h_{t+1}$ depends only on $h_t$ and the next input. So $h$ is technically a Markov chain — but a Markov chain whose state representation was *learned* to make prediction easier.
- **Parameters $W, V$ are the same at every timestep.** That's the "recurrent" part — one set of weights, applied repeatedly as the sequence scrolls by.
- **Short effective memory.** In practice, vanilla RNNs forget things from 10–20 steps ago because gradients vanish during backprop-through-time (see [../13_vanishing_gradient_and_tanh.md](../13_vanishing_gradient_and_tanh.md)).

Mental picture: a runner carrying a backpack. The backpack has fixed size, so they have to decide at each step what to pack in and what to throw out. The runner's training taught them what's worth keeping.

---

## 3. LSTM cell state — "a runner with a better backpack"

Cross-reference: [../11_lstm_the_memory_machine.md](../11_lstm_the_memory_machine.md).

The Long Short-Term Memory network (Hochreiter & Schmidhuber, 1997) is an RNN with a better memory architecture. It carries *two* state tensors:

- $h_t$ — the "hidden state" (the same idea as RNN)
- $c_t$ — the "cell state" (a long-term memory lane)

Plus three learned gates that decide, at every timestep:

- **Forget gate**: what to erase from $c_t$
- **Input gate**: what new info to write into $c_t$
- **Output gate**: what part of $c_t$ to expose as $h_t$

Key traits:

- **Still Markov**, but now over the joint state $(h_t, c_t)$. Given those two, the next step is determined.
- **Much longer effective memory** — because the cell state $c_t$ can carry information forward nearly untouched through the forget gate, gradients flow back many more timesteps without vanishing.
- **Same parameter-sharing across time** as RNN.

Mental picture: the runner now has a backpack *plus* a notepad. The notepad can hold stable information indefinitely, and the runner has learned rules for when to write to it, erase from it, and read from it.

---

## 4. Transformer — "forget running summaries, read the whole past"

Cross-reference: [../19_transformers.md](../19_transformers.md).

The Transformer (Vaswani et al., "Attention is All You Need," 2017) threw out the entire sequential-state idea. Instead of carrying a running summary forward, it keeps *every* past position in memory and, at every step, does a lookup over all of them via **attention**.

How the query at position $t$ "sees the past":

$$
\text{output}_t = \sum_{s \leq t} \text{attention\_weight}(t, s) \cdot \text{value}_s
$$

There is no $h_t$ being passed from step to step. There is no running state at all. The mechanism is:

- **Store all past positions**. In a causal language model, at step $t$ you have access to positions $0, 1, \ldots, t$.
- **Compute attention scores** between the current query (from position $t$) and every past key (from positions $0..t$).
- **Take a weighted sum** of the past values using those scores.

Key traits:

- **Not Markov in the sequential sense.** The output at step $t$ depends directly on *every* past step, not just the previous one. No compression step in between.
- **No running state variable.** There's nothing like $h_t$ to carry forward. Each position's representation is re-derived from scratch by attending over the past.
- **Memory scales with sequence length.** Storing all past positions costs $O(T)$ memory; computing attention over them costs $O(T^2)$ time. This is why long contexts are expensive for transformers but cheap for RNNs.
- **Parameters are shared across positions** via the same attention heads and MLPs, just like RNN weights were shared across time.

Mental picture: instead of a runner with a backpack, imagine a librarian with a card catalog. At every step, the librarian re-reads every card in the catalog to find the relevant ones. There's no summary slip — the raw cards are always available.

---

## Comparison table

| Property | Markov chain (DDPM) | RNN | LSTM | Transformer |
|---|---|---|---|---|
| State type | Raw data (pixels) | Learned hidden vector $h_t$ | Learned $(h_t, c_t)$ | None — all past kept explicitly |
| Learned params? | None | $W, V$ | $W, V$, gate weights | Attention + MLP weights |
| Markov order | 1 | 1 (in $h_t$) | 1 (in $h_t, c_t$) | Not Markov — full history |
| Effective memory | As long as signal survives | ~10–20 steps | ~100s of steps | Entire context window |
| Memory cost per step | $O(\text{state size})$ | $O(\text{hidden size})$ | $O(\text{hidden + cell size})$ | $O(\text{sequence length})$ |
| Compute cost per step | $O(1)$ | $O(1)$ | $O(1)$ | $O(\text{sequence length})$ |

---

## Where DDPM's forward process lands

DDPM's forward process is a Markov chain, but an *unusual* one for machine learning purposes:

1. **The state is the raw image tensor $x_t$.** No learned compression, no hidden variable. The full pixel grid is the state.
2. **The transition rule is fixed math**, not a neural network. $x_t = \sqrt{1 - \beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon_t$ is hardcoded.
3. **The "memory" is the current noise level.** Because $\bar{\alpha}_t$ is deterministic and known, you can read off how much signal vs noise is left just from the timestep index.

So DDPM is using Markov chains the way physicists and statisticians have used them for a century — as a mathematical tool for modeling a diffusion process — not the way modern deep learning uses them to justify sequential architectures.

**Crucially, the neural network in DDPM (the UNet) is not recurrent.** It is called $T$ times during sampling, but each call is *independent* — the UNet doesn't carry hidden state from step to step. The "state" that flows from step to step is just the noisy image itself, which is passed back in on the next call. There is no $h_t$, no $c_t$, no attention over past $x$ values. Each reverse step is a fresh forward pass through a stateless network.

---

## So, is your instinct right?

**Partially.** Here's the accurate version:

> *"The DDPM forward process is a Markov chain — like an RNN's hidden state is also first-order Markov. But RNNs and LSTMs learn their state representation ($h_t$ is a compressed summary the network invents), whereas DDPM's state is literally just the raw pixel tensor with some noise mixed in."*

And:

> *"It's definitely not a transformer. Transformers abandoned the Markov/sequential-state idea entirely — they store and re-read all past positions at every step. DDPM's forward process passes a single state forward one step at a time, which is the opposite philosophy."*

The confusion is understandable because the word "state" gets reused across these three communities (physics, RNNs, transformers). But once you see the four side by side, they stop blurring together.

---

## History/lore

- **1906 — Andrey Markov** invents Markov chains in St. Petersburg. His first application is analyzing the sequence of vowels and consonants in Pushkin's novel *Eugene Onegin*, to prove that the law of large numbers holds for dependent sequences. The motivation was partly a theological argument with Pavel Nekrasov about free will — Nekrasov had claimed independence was a precondition for statistical law, and Markov wanted to refute that using dependent sequences.
- **1986 — Michael Jordan** (the cognitive scientist, not the basketball player) publishes Jordan networks, one of the first recurrent architectures.
- **1990 — Jeffrey Elman** publishes Elman networks ("Finding Structure in Time"), formalizing the idea of a hidden state that persists across timesteps in a simple recurrent network. This is what most people today mean by "RNN."
- **1997 — Sepp Hochreiter & Jürgen Schmidhuber** publish LSTM, solving the vanishing-gradient problem that plagued Elman-style RNNs. Hochreiter's 1991 diploma thesis had already diagnosed the problem; LSTM was the fix.
- **2014 — Dzmitry Bahdanau, Cho, Bengio** introduce attention as an addition to seq2seq RNNs — still recurrent, but now with a lookup mechanism over encoder states.
- **2017 — Vaswani et al.** publish "Attention is All You Need," showing you can drop the recurrence entirely and use attention alone. This is when the Markov-chain mental model finally stops being useful for sequence modeling in NLP.
- **2020 — Ho, Jain, Abbeel** publish the DDPM paper, using a century-old Markov chain concept (from Sohl-Dickstein's 2015 physics-inspired framing) for image generation. The irony: at the same moment transformers were killing off recurrence in language, Markov chains were making a comeback in image generation.

The lesson: the same abstraction (a Markov chain) can serve completely different roles depending on what you put in the state variable. Physicists use raw coordinates. RNNs use learned hidden vectors. DDPM uses raw pixel tensors. Same math, wildly different uses.

---

## Takeaway

- **Markov chain**: state is the data, transitions are fixed, memoryless given the present.
- **RNN**: state is a learned compression, transitions are a learned network, effectively Markov in the hidden state.
- **LSTM**: same as RNN but with gated memory for longer-range dependencies.
- **Transformer**: not Markov; stores all past positions and re-reads them via attention at every step.
- **DDPM forward process**: a Markov chain with raw pixels as the state and no learned parameters. It is *not* an RNN, LSTM, or transformer. The UNet it trains is a stateless function called $T$ times, not a recurrent network.

Your instinct connecting Markov chains to RNN memory was the right kind of pattern-matching — both are sequential mechanisms where "the present summarizes the past." The difference is whether the summary is *raw data* (DDPM) or a *learned vector* (RNN/LSTM). Transformers sit outside this whole family by keeping the entire past explicit.
