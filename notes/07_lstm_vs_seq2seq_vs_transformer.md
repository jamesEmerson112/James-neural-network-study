# LSTM vs Seq2Seq with Attention vs Transformer

| | **LSTM** | **Seq2Seq + Attention** | **Transformer** |
|---|---|---|---|
| **Architecture** | Single recurrent cell | Encoder-decoder with RNN/LSTM | Self-attention encoder-decoder |
| **Processes input** | Sequentially, step by step | Sequentially (both sides) | All positions in parallel |
| **Context handling** | Fixed hidden state vector | Attention over all encoder outputs | Full self-attention over all tokens |
| **Long-range dependencies** | Struggles (vanishing gradients) | Better (attention helps) | Best (direct access to all positions) |
| **Key mechanism** | Gates (input, forget, output) | Gates + cosine similarity attention | Scaled dot-product multi-head attention |
| **Positional info** | Implicit from sequential processing | Implicit from sequential processing | Explicit positional embeddings required |
| **Training speed** | Slow (sequential) | Slow (sequential) | Fast (parallelizable) |
| **Translation quality** | Worst (no decoder) | Good | Best |
| **In this assignment** | `models/naive/LSTM.py` | `models/seq2seq/` | `models/Transformer.py` |

## TL;DR

- **LSTM** → basic sequence memory
- **Seq2Seq** → lets the decoder "look back" at encoder outputs via attention
- **Transformer** → drops recurrence entirely, uses self-attention for full parallelism and better long-range modeling
