# Assignment 3 Battle Plan: NLP Machine Translation

> **This is not a theory note. This is a battle plan.**
> When you sit down to code, follow this. No re-reading templates or notebooks.

---

## Overview Map

4 phases. 13 TODO blocks. The dependency chain:

```
Phase 1 (Naive LSTM)  ──→  standalone, do first
Phase 2 (Seq2Seq)     ──→  builds on Phase 1 concepts (uses nn.LSTM, not your naive one)
Phase 3 (Transformer) ──→  independent of Phase 2
Phase 4 (Full Trans.) ──→  independent of Phase 2, builds on Phase 3 concepts
```

| Phase | Files | What You're Building | TODOs |
|-------|-------|---------------------|-------|
| 1 | `models/naive/LSTM.py` | LSTM cell from scratch — 16 params, loop over time | 2 |
| 2 | `models/seq2seq/Encoder.py`, `Decoder.py`, `Seq2Seq.py` | Encoder-decoder with optional attention | 6 |
| 3 | `models/Transformer.py` (class `TransformerTranslator`) | Single-layer transformer from scratch | 3 |
| 4 | `models/Transformer.py` (class `FullTransformerTranslator`) | `nn.Transformer` wrapper with masks | 2 |

---

## Phase 1: Naive LSTM

**File:** `models/naive/LSTM.py`

### TODO 1 — `__init__(self, input_size, hidden_size)`

Declare 16 parameters + 2 activations. **Order is autograder-critical.**

```
For each gate, the pattern is:
  W_i*  (input_size, hidden_size)     ← input weights
  b_i*  (hidden_size,)               ← input bias
  W_h*  (hidden_size, hidden_size)   ← hidden weights
  b_h*  (hidden_size,)               ← hidden bias
```

Init order — gate by gate:

```python
# 1. Input gate (i_t)
self.W_ii = nn.Parameter(torch.Tensor(input_size, hidden_size))
self.b_ii = nn.Parameter(torch.Tensor(hidden_size))
self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
self.b_hi = nn.Parameter(torch.Tensor(hidden_size))

# 2. Forget gate (f_t)
self.W_if = nn.Parameter(torch.Tensor(input_size, hidden_size))
self.b_if = nn.Parameter(torch.Tensor(hidden_size))
self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
self.b_hf = nn.Parameter(torch.Tensor(hidden_size))

# 3. Cell gate (g_t)
self.W_ig = nn.Parameter(torch.Tensor(input_size, hidden_size))
self.b_ig = nn.Parameter(torch.Tensor(hidden_size))
self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
self.b_hg = nn.Parameter(torch.Tensor(hidden_size))

# 4. Output gate (o_t)
self.W_io = nn.Parameter(torch.Tensor(input_size, hidden_size))
self.b_io = nn.Parameter(torch.Tensor(hidden_size))
self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
self.b_ho = nn.Parameter(torch.Tensor(hidden_size))

# Activations
self.sigmoid = nn.Sigmoid()
self.tanh = nn.Tanh()
```

### TODO 2 — `forward(self, x)`

```
Input:   x        (batch, seq_len, input_size)
Output:  (h_t, c_t)   both (batch, hidden_size)
```

The loop:

```python
batch_size, seq_len, _ = x.shape
h_t = torch.zeros(batch_size, self.hidden_size)   # on same device as x
c_t = torch.zeros(batch_size, self.hidden_size)

for t in range(seq_len):
    x_t = x[:, t, :]                                # (batch, input_size)

    i_t = sigmoid( x_t @ W_ii + b_ii + h_t @ W_hi + b_hi )
    f_t = sigmoid( x_t @ W_if + b_if + h_t @ W_hf + b_hf )
    g_t = tanh(    x_t @ W_ig + b_ig + h_t @ W_hg + b_hg )
    o_t = sigmoid( x_t @ W_io + b_io + h_t @ W_ho + b_ho )

    c_t = f_t * c_t + i_t * g_t       # cell state update
    h_t = o_t * tanh(c_t)             # hidden state update

return (h_t, c_t)
```

### Gotchas

- **Do NOT transpose weights.** Shapes are set so `x @ W` works directly.
- **Output gate uses sigmoid** (like all gates), not tanh. The tanh is on `c_t`.
- Xavier init is handled by the provided `init_hidden()` — don't add your own.
- Return ONLY the **final** h_t and c_t, not the full sequence.

### Unit Test

`unit_test_values('lstm')` — expected shapes: h_t `(4, 4)`, c_t `(4, 4)`, atol=1e-3.

---

## Phase 2: Seq2Seq

**Uses `nn.RNN` / `nn.LSTM` (built-in), NOT your naive LSTM.**

### 2a. Encoder (`models/seq2seq/Encoder.py`)

**Constructor args:** `input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"`

#### TODO 1 — `__init__` layers (in this order)

```python
# 1. Embedding
nn.Embedding(input_size, emb_size)

# 2. Recurrent layer (choose by model_type)
nn.RNN(emb_size, encoder_hidden_size)        # if model_type == "RNN"
nn.LSTM(emb_size, encoder_hidden_size)       # if model_type == "LSTM"

# 3. Projection: Linear → ReLU → Linear
nn.Linear(encoder_hidden_size, encoder_hidden_size)
nn.ReLU()
nn.Linear(encoder_hidden_size, decoder_hidden_size)   # ← must be decoder_hidden_size!

# 4. Dropout
nn.Dropout(dropout)
```

#### TODO 2 — `forward(self, input)`

```
Input:   input    (batch, seq_len)
Output:  output   (from RNN)
         hidden   (projected, tanh'd)
```

```
1. Embed input → (batch, seq_len, emb_size)
2. Apply dropout to embeddings
3. Feed through RNN/LSTM → get output and hidden
4. Project hidden through Linear → ReLU → Linear
5. Apply tanh to projected hidden state
6. If LSTM: return output, (tanh(projected_h), projected_c)
             Do NOT apply tanh/linear to cell state — only to hidden
   If RNN:  return output, tanh(projected_h)
```

**Gotcha:** The linear projection output is `decoder_hidden_size` — this is how encoder and decoder hidden sizes can differ.

**Unit test:** `unit_test_values('encoder')` — output `(5, 2, 2)`, hidden `(1, 5, 2)`.

---

### 2b. Decoder (`models/seq2seq/Decoder.py`)

**Constructor args:** `emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN", attention=False`

#### TODO 1 — `__init__` layers (in this order)

```python
# 1. Embedding
nn.Embedding(output_size, emb_size)

# 2. Recurrent layer
nn.RNN(emb_size, decoder_hidden_size)                    # if no attention
nn.LSTM(emb_size, decoder_hidden_size)                   # if no attention, LSTM
# BUT if attention=True, RNN input size changes:
nn.RNN(decoder_hidden_size, decoder_hidden_size)         # attention ON
nn.LSTM(decoder_hidden_size, decoder_hidden_size)        # attention ON, LSTM

# 3. Output projection
nn.Linear(decoder_hidden_size, output_size)
nn.LogSoftmax(dim=1)

# 4. Dropout
nn.Dropout(dropout)

# 5. If attention=True: downsize layer
nn.Linear(emb_size + encoder_hidden_size, decoder_hidden_size)
```

#### TODO 2 — `compute_attention(self, hidden, encoder_outputs)`

```
Input:   hidden           (1, N, hidden_dim)
         encoder_outputs  (N, T, hidden_dim)
Output:  attention_prob   (N, 1, T)
```

Uses **cosine similarity** (NOT dot product):

```
cos_sim(q, K) = q @ K^T / (|q| × |K|)
```

Can use the formula directly or `torch.nn.functional.cosine_similarity`.

Then softmax over T dimension to get attention probabilities.

#### TODO 3 — `forward(self, input, hidden, encoder_outputs=None)`

```
Input:   input            (N, 1)   ← single token
         hidden           (1, N, decoder_hidden_size)
         encoder_outputs  (N, T, encoder_hidden_size)   ← for attention
Output:  output           (N, output_size)
         hidden           (1, N, decoder_hidden_size)
```

```
1. Embed input → dropout
2. If attention:
   a. compute_attention(hidden, encoder_outputs) → attn_prob (N, 1, T)
      NOTE: for LSTM, pass only hidden state h, not cell state c
   b. context = attn_prob @ encoder_outputs → (N, 1, hidden_dim)
   c. concat(context, embedded) along feature dim  ← context FIRST
   d. Feed through attention linear layer
   e. This becomes the RNN input
3. If no attention:
   embedded goes directly to RNN
4. RNN → output
5. Linear → LogSoftmax → final output (N, output_size)
6. Return output and hidden
```

**Gotchas:**
- Attention uses **cosine similarity**, not dot product
- Attention only uses LSTM **hidden state** (not cell state)
- Concatenation order: `(context, dropout_output)` — context first
- RNN `input_size` changes when attention is on (`decoder_hidden_size` instead of `emb_size`)

**Unit tests:** `unit_test_values('decoder')` — output `(5, 10)`, hidden `(1, 5, 2)`.
`unit_test_values('attention')` — expected attention `(5, 1, 2)`.

---

### 2c. Seq2Seq (`models/seq2seq/Seq2Seq.py`)

**Constructor args:** `encoder, decoder, device`

#### TODO 1 — `__init__`

Two lines. Store encoder and decoder, move to device:

```python
self.encoder = encoder.to(device)
self.decoder = decoder.to(device)
```

#### TODO 2 — `forward(self, source)`

```
Input:   source   (batch, seq_len)
Output:  outputs  (batch, seq_len, decoder.output_size)
```

```python
batch_size, seq_len = source.shape

# 1. Encode
encoder_outputs, hidden = self.encoder(source)

# 2. First decoder input = <sos> token (first token of source)
input = source[:, 0].unsqueeze(1)          # (N, 1)

# 3. Init outputs
outputs = torch.zeros(batch_size, seq_len, self.decoder.output_size).to(device)

# 4. Decode loop
for t in range(1, seq_len):               # start at 1, not 0
    output, hidden = self.decoder(input, hidden, encoder_outputs)
    outputs[:, t] = output
    input = output.argmax(dim=1).unsqueeze(1)    # greedy: (N, 1)

return outputs
```

**Gotcha:** Loop starts at `t=1` because `t=0` is the `<sos>` token (already known).

**Unit tests:** `unit_test_values('seq2seq')` — `(2, 2, 8)`.
`unit_test_values('seq2seq_attention')` — `(2, 2, 8)`.

---

## Phase 3: TransformerTranslator

**File:** `models/Transformer.py`, class `TransformerTranslator`

**Constructor args:** `input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43`

### Already initialized for you (Deliverable 2 support)

```python
# Head 1
self.k1 = nn.Linear(hidden_dim, dim_k)
self.v1 = nn.Linear(hidden_dim, dim_v)
self.q1 = nn.Linear(hidden_dim, dim_q)

# Head 2
self.k2 = nn.Linear(hidden_dim, dim_k)
self.v2 = nn.Linear(hidden_dim, dim_v)
self.q2 = nn.Linear(hidden_dim, dim_q)

self.softmax = nn.Softmax(dim=2)
self.attention_head_projection = nn.Linear(dim_v * num_heads, hidden_dim)
self.norm_mh = nn.LayerNorm(hidden_dim)
```

### Deliverable 1 — Embeddings (`__init__` + `embed()`)

Init:
```python
self.embeddingL = nn.Embedding(input_size, hidden_dim)
self.posembeddingL = nn.Embedding(max_length, hidden_dim)
```

Forward (`embed` method):
```
Input:   inputs   (N, T)
Output:  embeddings   (N, T, hidden_dim)
```
```python
positions = torch.arange(T).to(device)       # [0, 1, ..., T-1]
return embeddingL(inputs) + posembeddingL(positions)
```

Uses **learned positional embeddings** (not sine/cosine).

### Deliverable 2 — Multi-Head Attention (`multi_head_attention()`)

```
Input:   inputs   (N, T, hidden_dim)
Output:  outputs  (N, T, hidden_dim)
```

```
For each head i:
  Q_i = q_i(inputs)                         (N, T, dim_q)
  K_i = k_i(inputs)                         (N, T, dim_k)
  V_i = v_i(inputs)                         (N, T, dim_v)
  attn_i = softmax(Q_i @ K_i^T / sqrt(dim_k)) @ V_i    (N, T, dim_v)

Concat all heads along last dim → (N, T, dim_v * num_heads)
Project: attention_head_projection → (N, T, hidden_dim)
Add residual: output + inputs
LayerNorm: norm_mh
```

### Deliverable 3 — Feed-Forward Layer (`__init__` + `feedforward_layer()`)

Init:
```python
self.ff_linear1 = nn.Linear(hidden_dim, dim_feedforward)
self.ff_linear2 = nn.Linear(dim_feedforward, hidden_dim)
self.norm_ff = nn.LayerNorm(hidden_dim)
```

Forward:
```
Input:   inputs   (N, T, hidden_dim)
Output:  outputs  (N, T, hidden_dim)

Linear1 → ReLU → Linear2 → add residual (inputs) → LayerNorm
```

### Deliverable 4 — Final Layer (`__init__` + `final_layer()`)

Init:
```python
self.final_linear = nn.Linear(hidden_dim, output_size)
```

Forward:
```
Input:   inputs   (N, T, hidden_dim)
Output:  outputs  (N, T, output_size)

Just the linear. NO softmax — CrossEntropyLoss includes it.
```

### Deliverable 5 — Full Forward (`forward()`)

```python
def forward(self, inputs):
    # inputs: (N, T)
    x = self.embed(inputs)                # (N, T, hidden_dim)
    x = self.multi_head_attention(x)      # (N, T, hidden_dim)
    x = self.feedforward_layer(x)         # (N, T, hidden_dim)
    x = self.final_layer(x)              # (N, T, output_size)
    return x
```

**Unit tests:** `unit_test_values('d1')` through `unit_test_values('d4')`.

---

## Phase 4: FullTransformerTranslator

**File:** `models/Transformer.py`, class `FullTransformerTranslator`

**Constructor args:** `input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, num_layers_enc=2, num_layers_dec=2, dropout=0.2, max_length=43, ignore_index=1`

`self.pad_idx = ignore_index` is already set.

### Deliverable 1 — nn.Transformer

```python
self.transformer = nn.Transformer(
    d_model=hidden_dim,
    nhead=num_heads,
    num_encoder_layers=num_layers_enc,
    num_decoder_layers=num_layers_dec,
    dim_feedforward=dim_feedforward,
    dropout=dropout
)
```

### Deliverable 2 — Embeddings (ORDER IS CRITICAL)

```python
self.srcembeddingL = nn.Embedding(input_size, hidden_dim)      # 1
self.tgtembeddingL = nn.Embedding(output_size, hidden_dim)     # 2
self.srcposembeddingL = nn.Embedding(max_length, hidden_dim)   # 3
self.tgtposembeddingL = nn.Embedding(max_length, hidden_dim)   # 4
```

Do NOT change this order.

### Deliverable 3 — Final Layer

```python
self.final_linear = nn.Linear(hidden_dim, output_size)
```

### Deliverable 4 — Training Forward (`forward(self, src, tgt)`)

```
Input:   src   (N, T)
         tgt   (N, T)
Output:  outputs   (N, T, output_size)
```

```python
# 1. Shift target right (prepend SOS)
tgt = self.add_start_token(tgt)           # already provided

# 2. Embed src and tgt (token + positional)
src_positions = torch.arange(src.shape[1]).to(device)
tgt_positions = torch.arange(tgt.shape[1]).to(device)
src_emb = srcembeddingL(src) + srcposembeddingL(src_positions)
tgt_emb = tgtembeddingL(tgt) + tgtposembeddingL(tgt_positions)

# 3. Create masks
tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])   # causal mask
src_key_padding_mask = (src == self.pad_idx)     # (N, T) boolean
tgt_key_padding_mask = (tgt == self.pad_idx)     # (N, T) boolean

# 4. nn.Transformer expects (T, N, H) — transpose!
#    Check whether batch_first is set. If not, transpose.
# 5. Call transformer
output = self.transformer(
    src_emb, tgt_emb,
    tgt_mask=tgt_mask,
    src_key_padding_mask=src_key_padding_mask,
    tgt_key_padding_mask=tgt_key_padding_mask
)

# 6. Final linear layer
return self.final_linear(output)
```

**Gotcha:** Check if `nn.Transformer` defaults to `batch_first=False`. If so, you need to permute `(N, T, H)` → `(T, N, H)` before calling, and permute back after.

### Deliverable 5 — `generate_translation(self, src)`

```
Input:   src      (N, T)
Output:  outputs  (N, T, output_size)
```

```python
batch_size, seq_len = src.shape

# 1. Init
outputs = torch.zeros(batch_size, seq_len, self.output_size).to(device)
tgt = torch.full((batch_size, seq_len), self.pad_idx).long().to(device)

# 2. Autoregressive loop
for t in range(seq_len):
    logits = self.forward(src, tgt)          # (N, T, output_size)
    outputs[:, t] = logits[:, t]
    tgt[:, t] = logits[:, t].argmax(dim=-1)  # greedy decode

return outputs
```

**Unit tests:** `unit_test_values('full_trans_fwd')` — shape `(3, 43, 5)`.
`unit_test_values('full_trans_translate')`.

---

## Training Cheat Sheet

| Parameter | Seq2Seq | Transformer |
|-----------|---------|-------------|
| Embedding size | 128 | 128 (hidden_dim) |
| Hidden size | 128 | 128 |
| Learning rate | 1e-3 | 1e-3 |
| Optimizer | Adam | Adam |
| Gradient clip | 0.5 | 0.5 |
| Dropout | 0.2 | 0.2 |
| dim_feedforward | — | 2048 |
| num_heads | — | 2 |
| dim_k/v/q | — | 96 |
| max_length | — | 43 |

- **Loss:** `CrossEntropyLoss(ignore_index=PAD_IDX)` — PAD_IDX = 1
- **Scheduler:** `ReduceLROnPlateau` (optional)
- **Batch processing:** Data may be transposed `(seq_len, batch)` in utils — check `utils.py` train/evaluate functions

---

## Gotcha Checklist

```
PHASE 1 — Naive LSTM
  [ ] Do NOT transpose weights — x @ W works directly
  [ ] Output gate uses sigmoid (not tanh)
  [ ] Return only FINAL (h_t, c_t), not full sequence
  [ ] init_hidden() handles Xavier init — don't add your own

PHASE 2 — Seq2Seq
  [ ] Use nn.RNN / nn.LSTM (built-in), NOT your naive LSTM
  [ ] Encoder: Linear projection output must be decoder_hidden_size
  [ ] Encoder: tanh on hidden state ONLY — not on cell state
  [ ] Encoder: Dropout goes AFTER embedding, BEFORE RNN
  [ ] Decoder: Attention uses cosine similarity, not dot product
  [ ] Decoder: Attention only on hidden state, not cell state
  [ ] Decoder: concat order is (context, embedded) — context first
  [ ] Decoder: RNN input_size changes when attention=True
  [ ] Seq2Seq: First input is source[:, 0] (<sos> token)
  [ ] Seq2Seq: Decode loop starts at t=1, not t=0

PHASE 3 — TransformerTranslator
  [ ] Learned positional embeddings (not sine/cosine)
  [ ] Scale attention by sqrt(dim_k)
  [ ] Add + Norm after both multi-head attention AND feed-forward
  [ ] Final layer: NO softmax (CrossEntropyLoss includes it)

PHASE 4 — FullTransformerTranslator
  [ ] Embedding init order: src, tgt, src_pos, tgt_pos — do NOT change
  [ ] Call add_start_token(tgt) at the start of forward
  [ ] Create causal mask for target
  [ ] Create padding masks for both src and tgt
  [ ] Check batch_first — nn.Transformer defaults to (T, N, H)
  [ ] generate_translation: tgt starts as all pad tokens
```

---

## Cross-References

| Topic | See |
|-------|-----|
| LSTM gate mechanics (why 4 gates, numerical walkthrough) | Note 11, Parts 4–6 |
| Weight naming convention and shapes | Note 11, Part 9 |
| RNN architecture, vanishing gradient math | Note 10, Part 8 |
| Backprop intuition (wiggle ratios) | Note 09, Part 5 |
| LSTM implementation checklist | Note 08 |
