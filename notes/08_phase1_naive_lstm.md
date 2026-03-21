# Phase 1: Naive LSTM

## What it is
A single LSTM cell built from scratch using `nn.Parameter` (no `nn.LSTM`). It's a learning exercise — not used by Seq2Seq or Transformer.

## File
`models/naive/LSTM.py`

## The 4 Gates

```
i_t = sigmoid(x_t @ W_ii + b_ii + h_{t-1} @ W_hi + b_hi)   # input gate: what new info to store
f_t = sigmoid(x_t @ W_if + b_if + h_{t-1} @ W_hf + b_hf)   # forget gate: what to throw away
g_t = tanh(x_t @ W_ig + b_ig + h_{t-1} @ W_hg + b_hg)      # cell gate: candidate new values
o_t = tanh(x_t @ W_io + b_io + h_{t-1} @ W_ho + b_ho)       # output gate: what to output
```

## Cell & hidden state update

```
c_t = f_t * c_{t-1} + i_t * g_t    # new cell state
h_t = o_t * tanh(c_t)              # new hidden state
```

## What to implement

### `__init__`
- 8 weight matrices (`nn.Parameter`): W_ii, W_hi, W_if, W_hf, W_ig, W_hg, W_io, W_ho
- 8 bias vectors (`nn.Parameter`): b_ii, b_hi, b_if, b_hf, b_ig, b_hg, b_io, b_ho
- 2 activations: `nn.Sigmoid()`, `nn.Tanh()`
- Weight shapes: W_i* is (input_size, hidden_size), W_h* is (hidden_size, hidden_size)
- **Do NOT transpose weights** — they're shaped so `x @ W` works directly

### `forward(x)` where x is (batch, sequence, feature)
- Init h_t and c_t to zeros: shape (batch, hidden_size)
- Loop over sequence dimension (timesteps)
- At each step, compute all 4 gates, update c_t and h_t
- Return final (h_t, c_t)

## Init order (must match for autograder)
1. input gate (W_ii, b_ii, W_hi, b_hi)
2. forget gate (W_if, b_if, W_hf, b_hf)
3. cell gate (W_ig, b_ig, W_hg, b_hg)
4. output gate (W_io, b_io, W_ho, b_ho)

## Gotchas
- Weights are NOT transposed in forward — shape them as (input_size, hidden_size) so `x @ W` works
- Do NOT apply tanh/linear to cell state — only hidden state gets `o_t * tanh(c_t)`
- `init_hidden()` is already provided — it does xavier init on weights, zeros on biases
