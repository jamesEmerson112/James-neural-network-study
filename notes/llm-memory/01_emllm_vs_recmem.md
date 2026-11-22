# EM-LLM vs RecMem

> Companion note to `01_emllm_vs_recmem_video.mp4`. See [the overview](00_overview.md).

The video is a Manim explainer at 1080p30 that compares two ways of giving a language model memory beyond its context window. EM-LLM, by Fountas and colleagues at Huawei Noah's Ark Lab London and University College London (ICLR 2025), keeps the keys and values of one very long input inside the model, cut into events where the model is surprised. RecMem, by Dai and colleagues at CUHK, BUPT and Wuhan University with Huawei (Findings of ACL 2026), keeps an agent's conversation as text outside the model and spends LLM calls on a topic only when it recurs.

The toy numbers on screen are computed by the script, which asserts the claims the captions make about them. One example is that exactly the four designed spikes beat the surprise threshold. Another is that modularity peaks at the true boundary. The paper numbers are copied from the two papers' tables, and the video never puts the two papers' scores on one axis, because they use different benchmarks and metrics.

Render it from this folder (Manim Community; no LaTeX needed):

```
manim --resolution 1920,1080 --frame_rate 30 01_emllm_vs_recmem_video.py MemoryTwoWays
```

Setting `MEM_ONLY=emllm_cut,same_chat` renders only the named sections. Every section starts and ends on an empty frame, so sections rendered separately can be joined with ffmpeg's concat demuxer.

## Timeline

The video runs 5 minutes 48 seconds.

| Time | Section |
|---|---|
| 0:00 | Title |
| 0:09 | Where the memory lives: per-layer key-value events inside the model, text stores outside it |
| 0:40 | EM-LLM step 1: surprise, the moving threshold, and the events it cuts |
| 1:09 | EM-LLM step 2: sliding a late boundary to the position with the highest modularity |
| 1:50 | EM-LLM step 3: the similarity buffer, the contiguity buffer and the attention context |
| 2:17 | RecMem writing: raw turns, the similarity cone, consolidation on recurrence and merge-first |
| 3:21 | RecMem reading: one search over three stores, then an unchanged LLM |
| 3:39 | One six-turn chat under both rules |
| 4:17 | Results, each against the baselines in its own paper |
| 4:42 | Side-by-side table |
| 5:01 | Where the ideas come from: the psychology each paper borrows, as two timelines |
| 5:27 | Summary and one labelled synthesis |

## EM-LLM in three steps

EM-LLM works on a single long input, such as a book or a long document, with a pretrained model and no training. Its direct predecessor is InfLLM (Xiao et al., 2024), which also stores past keys and values outside the attention window and fetches blocks of them back. InfLLM cuts the input into fixed-size blocks, while EM-LLM cuts it where the content changes.

**Step 1: cut where surprise spikes.** The surprise of token $x_t$ is how badly the model predicted it, $-\log P(x_t \mid x_1, \ldots, x_{t-1})$. A token starts a new event when its surprise beats a moving threshold,

$$T_t = \mu_{t-\tau:t} + \gamma\,\sigma_{t-\tau:t},$$

which is the mean of the last $\tau$ surprises plus $\gamma$ standard deviations. The threshold therefore adapts to how surprising the text usually is. The video's toy stream has 64 tokens with $\gamma = 3$ and $\tau = 10$, and four designed spikes at tokens 13, 27, 40 and 52. The script asserts that exactly those four tokens beat the threshold. The paper sweeps $\gamma$ per model, and its PG-19 segmentation study uses $\gamma = 10^{-3}$.

**Step 2: refine the cuts with graph modularity.** Each attention head defines a similarity graph over the tokens, $A^h_{ij} = (K^h_i)^\top K^h_j$, from the tokens' keys. For each pair of consecutive surprise boundaries $(\alpha, \beta)$, Algorithm 1 searches $\hat\beta \in (\alpha, \beta]$ and moves $\beta$ to the position that maximizes the metric. A boundary can therefore move back toward the previous one, but never past its surprise position. With modularity as the metric, the score is the similarity inside events minus what a random graph with the same degrees would put there:

$$Q = \frac{1}{2m}\sum_{i,j}\left(A_{ij} - \frac{d_i d_j}{2m}\right)\delta(c_i, c_j), \qquad d_i = \sum_j A_{ij}, \quad 2m = \sum_{i,j} A_{ij}.$$

The paper prints the normalizer as $1/(4m)$, and a constant factor does not change where the maximum is. The paper also tests conductance, and modularity gave the best results. The video's toy has 18 tokens whose true events have 7, 5 and 6 tokens. Surprise put the second boundary at token 14, two tokens late, where $Q = 0.303$. Searching positions 8 to 14, the best is token 12, where $Q = 0.390$. For simplicity the toy scores the modularity of the whole three-event split, where the paper's $f(A, \{\alpha, \hat\beta\})$ scores the split around the boundary being moved. The refinement costs time but no LLM calls. With Mistral-7B at 4K+2K, one 512-token chunk takes 1.40 s in InfLLM, 1.57 s with surprise alone and 2.27 s with surprise plus refinement.

**Step 3: fetch events in every layer.** Events are kept in CPU memory, each with a few representative tokens, as in InfLLM. For each new chunk, every layer separately fills two buffers. The similarity buffer $k_s$ holds the events whose representative keys best match the chunk's queries, found by k-nearest-neighbour search. The contiguity buffer $k_c$ holds the events next to those hits in time. Attention then runs over 128 initial tokens kept as attention sinks, the $k = k_s + k_c$ fetched events, and the local context. The local plus retrieved sizes are 4K+2K tokens for Mistral-7B, 4K+4K for LLaMA-3 and LLaMA-3.1, and 1K+3K for Phi-3 and Phi-3.5. The contiguity buffer copies a habit from human free recall: people who recall one item tend to recall the items studied next to it, more often forward than backward (Howard and Kahana's temporal context model, 2002).

The main cost is storage. With 32 layers and 32 key-value heads in half precision, the key-value cache takes 9.8 GB at 20K tokens and 488.3 GB at 1M tokens.

## RecMem: write and read

RecMem stores each conversation turn $u_i$, meaning one user message, the assistant's reply and a timestamp, as a subconscious unit with an embedding:

$$s_i = (v_i, u_i), \qquad v_i = \Phi(u_i).$$

Storing a unit needs no LLM call. When a new unit arrives, RecMem first looks for the nearest existing episode $E^\star$ and merges the turn into it if they are similar enough:

$$E^\star \leftarrow \text{LLM}_{\text{merge}}(E^\star, u_i) \quad \text{if} \quad \cos(v_i, v_{E^\star}) \ge \theta_{\text{sim}}.$$

This merge-first step keeps one topic from forking into parallel episodes. Otherwise, RecMem retrieves the top-$k$ most similar units $N_i$ from the subconscious store and keeps the close ones,

$$R_i = \{\, s_j \in N_i \mid \cos(v_i, v_j) \ge \theta_{\text{sim}} \,\}.$$

If $|R_i| \ge \theta_{\text{count}}$, the cluster $C_i = R_i \cup \{s_i\}$ is consolidated. One LLM call writes one to three episodes, each a time-anchored narrative of three to six sentences. A second call, the semantic refinement, extracts fewer than ten atomic facts and recovers details the episode summarized away. At question time, one embedding search fetches 10 raw turns, 5 episodes and 10 facts by default, and the answering model only ever sees that text. That is why RecMem runs on API-only models; the paper uses GPT-4o-mini and GPT-4.1-mini with text-embedding-3-small.

The paper's thresholds are $\theta_{\text{sim}} = 0.7$ with $\theta_{\text{count}} = 5$ on LoCoMo, and $\theta_{\text{sim}} = 0.6$ with $\theta_{\text{count}} = 4$ on LongMemEval-S.

**Counting convention.** The paper can be read two ways here, and the readings differ by one. Read strictly, $R_i$ holds the matching units already in the store, and $C_i = R_i \cup \{s_i\}$ adds the new unit separately, so $|R_i|$ counts earlier turns only. But the paper also says each unit is indexed into the store immediately, and its appendix walk-through sets $\theta_{\text{count}} = 2$ and fires on a topic's second mention, described as a topic "observed in at least two interaction units". That reading counts the new turn too. The video follows the earlier-turns reading and says so on screen. Its write demo counts earlier matches, and its six-turn chat uses a toy rule that fires on one earlier match. Under the appendix reading, the same chat would use $\theta_{\text{count}} = 2$.

The write demo in the video uses 7 turns, $\theta_{\text{sim}} = 0.7$ and a toy $\theta_{\text{count}} = 3$. It draws embeddings as directions on a circle, so the similarity test becomes a cone with half-angle $\arccos 0.7 \approx 45.6^\circ$. Turns 1 to 6 find $|R| = 0, 0, 1, 0, 2, 3$ earlier matches. Turn 6 consolidates the four cake turns with two calls, and turn 7 merges into the cake episode with one call. That makes 3 LLM calls in seven turns, where an eager memory system, which runs extraction on every turn, makes at least 7.

## The six-turn chat

The chat is illustrative: order a birthday cake, ask how to wash dark jeans, ask what to write on the cake, report a cancelled flight, move the cake pickup, and say thanks. EM-LLM's illustrative surprises are 1.0, 2.9, 2.2, 3.7, 2.5 and 0.6 against a threshold of 1.8, so it cuts before turns 2, 3, 4 and 5. That gives five events, and the cancelled flight is an event of its own. RecMem, with a toy rule that fires on one earlier match, consolidates the cake topic at turn 3, merges turn 5 into that episode, and leaves the jeans, the flight and the thanks raw.

The example shows the two signals pulling in opposite directions. EM-LLM marks what is new, and RecMem keeps what comes back. EM-LLM's events stay in time order, while RecMem's episode gathers turns 1, 3 and 5 by meaning.

## Results shown

EM-LLM against RAG and the full context, with LLaMA-3.1-8B (paper Table 9):

| Benchmark | RAG (NV-Embed-v2) | Full context | EM-LLM (S, 4K+4K) |
|---|---|---|---|
| LongBench average | 36.44 | 39.30 | 51.58 |
| ∞-Bench average | 58.97 | 66.33 | 66.66 |

Against InfLLM (paper Table 1), EM-LLM's LongBench average is higher with all five base models. The pairs are 41.9 to 43.7 for Mistral-7B-v0.2, 47.0 to 47.2 for LLaMA-3-8B, 51.1 to 51.3 for LLaMA-3.1-8B, 34.5 to 35.4 for Phi-3-mini, and 34.2 to 34.9 for Phi-3.5-mini.

RecMem against agent memory systems on LoCoMo, with GPT-4.1-mini (paper Tables 1 and 2):

| Method | LLM-judge overall | Memory-construction tokens | Query tokens |
|---|---|---|---|
| Full context | 84.18 | 0 | 31.52K |
| Naive RAG | 54.42 | 0 | 6.28K |
| MemoryOS | 67.60 | 400.7K | 5.04K |
| Mem0 | 62.92 | 1,520.8K | 2.11K |
| A-Mem | 68.83 | 1,459.9K | 5.56K |
| RecMem | 81.10 | 193.2K | 2.75K |

On LongMemEval-S, RecMem scores 76.80 with 365.5K construction tokens. Full context wins on LoCoMo, whose conversations average about 16K tokens, and falls to last place on LongMemEval-S, whose histories average about 115K. Construction tokens fall 87.3% against Mem0 and 86.8% against A-Mem on LoCoMo.

## Side by side

| | EM-LLM | RecMem |
|---|---|---|
| Memory unit | Events of keys and values | Turns, episodes and facts, as text |
| Written when | Surprise beats a moving threshold | Similar turns recur $\theta_{\text{count}}$ times |
| LLM calls to write | None; refinement is graph math on keys | Only for topics that recur |
| Recall | k-NN on keys, in every layer and chunk | Embedding search, once per question |
| Model access | Open weights, with attention patched | Any chat model plus an embedder |
| Main cost | Key-value storage, 488 GB per 1M tokens | LLM calls for each recurring topic |
| Blind spot | No summaries and no merging | One-off events stay raw |

The two methods work at different layers of the stack. EM-LLM changes attention inside one forward pass, and RecMem manages text between the turns of an agent. They could be stacked, with RecMem handling memory across sessions for a model that uses EM-LLM to read one very long input.

The closing idea of the video is a synthesis, not a claim in either paper. RecMem's limitations section asks for a lightweight salience signal beyond pure recurrence, so that rare but important events get consolidated. EM-LLM's surprise score is one candidate. Computing it needs token log-probabilities over the conversation, which API models do not expose for input text, so a small local model would have to supply them.

## Where the ideas come from

Each paper borrows a different part of the psychology of memory, and the video's second-to-last section draws the two lineages as timelines.

EM-LLM's lineage starts with Howard and Kahana's temporal context model (2002). It explains why, in free recall, people tend to recall next the items studied near the last one, more often forward than backward, and the contiguity buffer copies that habit. Event segmentation theory (Zacks and colleagues, 2007) holds that people split continuous experience into events at moments of high prediction error, which is the idea behind cutting at surprise. Kumar and colleagues (2023) showed that Bayesian surprise from a language model predicts where human listeners mark event boundaries in stories. EM-LLM also cites the finding that event boundaries are where memories are retrieved from (Michelmann, Hasson and Norman, 2023).

RecMem's lineage starts with Atkinson and Shiffrin's multi-store model (1968), in which information passes from fast, transient stores into a stable long-term store. Its subconscious store plays the transient role. Complementary Learning Systems (McClelland, McNaughton and O'Reilly, 1995) adds that a fast hippocampal system records experiences and the neocortex slowly extracts what repeats, which is what the recurrence trigger stands in for. Kumaran, Hassabis and McClelland revisited the theory in 2016 with an eye on artificial agents.

## Sources

- Z. Fountas, M. A. Benfeghoul, A. Oomerjee, F. Christopoulou, G. Lampouras, H. Bou-Ammar and J. Wang, "Human-like Episodic Memory for Infinite Context LLMs," ICLR 2025, arXiv:2407.09450. The numbers here come from v2 (October 2024), the copy in `human_cognitive_ai/Human-Like Episodic Memory for Infinite Context LLMs - UCL - 2025.pdf`.
- Z. Dai, S. Deng, S. Guan, Y. Tian, X. Yao, X. Yan and J. Cheng, "RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents," Findings of ACL 2026, pages 32353–32376, arXiv:2605.16045. The repo copy is `human_cognitive_ai/RecMem - Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents - CUHK, BUPT, Wuhan University - 2026.pdf`.
- The benchmarks RecMem reports on are described in [Long-term memory benchmarks](../21b_long_term_memory_benchmarks.md).
