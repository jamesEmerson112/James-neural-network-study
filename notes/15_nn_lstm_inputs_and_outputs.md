# What nn.LSTM Actually Returns

> **You built LSTM by hand in Phase 1. Now nn.LSTM does it for you — but the outputs look different. This note explains what comes out and why.**

---

## Phase 1 (your naive LSTM) vs nn.LSTM

```python
# YOUR naive LSTM (Phase 1):
h_t, c_t = lstm(x)
# Returns: final hidden state, final cell state
# That's it. Two tensors.

# PyTorch's nn.LSTM:
output, (h_n, c_n) = lstm(x)
# Returns: ALL hidden states, PLUS final (hidden, cell) as a tuple
# Three tensors, packed in a specific way.
```

The difference: nn.LSTM gives you **everything**, not just the final state.

---

## What each return value is

### `output` — hidden state at EVERY timestep

```
Input sentence: "The cat sat on the mat"
                  t=0 t=1 t=2 t=3 t=4 t=5

output = [h_0, h_1, h_2, h_3, h_4, h_5]

Shape: (batch, seq_len, hidden_size)
       e.g. (32, 6, 256)
```

This is the full history of what the LSTM "thought" at each word. Most of the time you don't need this — UNLESS you're using **attention**, where the decoder looks back at every one of these.

```
Without attention: output is returned but mostly ignored
With attention:    output is the memory bank the decoder searches through
```

### `h_n` — final hidden state only

```
h_n = h_5 (the last one from output)

Shape: (num_layers, batch, hidden_size)
       e.g. (1, 32, 256)

This is the "summary" of the whole sentence.
This is the baton pass to the decoder.
```

### `c_n` — final cell state only

```
c_n = the cell state at the last timestep

Shape: (num_layers, batch, hidden_size)
       e.g. (1, 32, 256)

The long-term memory. Decoder's LSTM needs this to continue
"thinking" where the encoder left off.
```

---

## Why the tuple packing?

```python
output, hidden = self.rnn(embedded)

# If nn.RNN:   hidden = h_n                    (one tensor)
# If nn.LSTM:  hidden = (h_n, c_n)             (tuple of two!)
```

PyTorch packs h_n and c_n into a tuple so that `nn.RNN` and `nn.LSTM` have the same calling convention. You always write:

```python
output, hidden = self.rnn(embedded)
```

But `hidden` is secretly different depending on the model type:

```
nn.RNN:   hidden = tensor
nn.LSTM:  hidden = (tensor, tensor)
```

That's why the encoder has to unpack it:

```python
if self.model_type == "LSTM":
    hidden, cell_state = hidden    # unpack the tuple
    # transform only hidden...
    hidden = (hidden, cell_state)  # repack for the decoder
```

---

## Visual: what flows where in Seq2Seq

```
ENCODER:
                    output (all timesteps)
                    ┌──────────────────────────────┐
                    │  h_0   h_1   h_2   h_3   h_4 │ → saved for attention
                    └──────────────────────────────┘
                                                 │
"The cat sat on mat" → [Embed] → [LSTM] ────────┤
                                                 │
                    ┌────────────────────────┐   │
                    │  h_n (final hidden)    │───┤→ projected → tanh → baton pass
                    │  c_n (final cell)      │───┘→ passed through untouched
                    └────────────────────────┘


DECODER receives:
  - output      (all encoder hidden states)  → for attention
  - h_n         (projected, tanh'd)          → decoder's initial hidden state
  - c_n         (untouched)                  → decoder's initial cell state
```

---

## Connecting back to Phase 1

What `nn.LSTM` does internally is exactly what you coded:

```python
# This is what nn.LSTM runs behind the scenes:
for t in range(seq_len):
    i_t = sigmoid(x_t @ W_ii + b_ii + h_t @ W_hi + b_hi)
    f_t = sigmoid(x_t @ W_if + b_if + h_t @ W_hf + b_hf)
    g_t = tanh(x_t @ W_ig + b_ig + h_t @ W_hg + b_hg)
    o_t = sigmoid(x_t @ W_io + b_io + h_t @ W_ho + b_ho)
    c_t = f_t * c_t + i_t * g_t
    h_t = o_t * tanh(c_t)

    all_hidden_states.append(h_t)    # ← nn.LSTM saves every one of these

output = stack(all_hidden_states)    # ← that's the "output" return value
h_n = h_t                           # ← final hidden state
c_n = c_t                           # ← final cell state
```

Your Phase 1 code only returned the final (h_t, c_t). nn.LSTM also saves every h_t along the way — that's the only difference.

---

## Cross-References

| Topic | See |
|-------|-----|
| LSTM gate mechanics (what happens inside each timestep) | Note 11, Parts 4–6 |
| Why cell state exists (gradient highway) | Note 11, Part 5 |
| Encoder full structure (Embedding → Dropout → LSTM → projection) | Note 14, Part 3 |
| Why attention needs all hidden states | Note 14, Part 4 |
