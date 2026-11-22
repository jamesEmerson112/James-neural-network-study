# Memory for LLMs

> See also: [Long-term memory benchmarks](../21b_long_term_memory_benchmarks.md), [LSTM, the memory machine](../11_lstm_the_memory_machine.md).

This folder covers how a language model can use more material than fits in its context window. One way keeps memory inside the model, as stored keys and values that attention can fetch again. The other keeps memory outside the model, as text that an agent searches and pastes into its prompt.

- `01_emllm_vs_recmem_video.mp4` is a Manim explainer, 5 minutes 22 seconds long at 1080p30. It compares EM-LLM (ICLR 2025) with RecMem (Findings of ACL 2026). EM-LLM cuts one long input into events where the model is surprised, refines the cuts with graph modularity on attention keys, and fetches events in every layer by similarity and by neighbourhood in time. RecMem stores every conversation turn raw and calls an LLM to write episodes and facts only when a topic recurs. The video then runs one six-turn chat through both rules and shows each paper's results against its own baselines. Its companion note, with every formula, number and source, is [EM-LLM vs RecMem](01_emllm_vs_recmem.md).
- `01_emllm_vs_recmem_video.py` renders it. No LaTeX is needed, because every formula is drawn with Pango markup.

Both papers are in the repo under `human_cognitive_ai/`.
