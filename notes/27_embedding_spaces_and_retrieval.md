# Embedding Spaces and Retrieval

Beyond Word2Vec's flat Euclidean vectors — hyperbolic geometry, dimensionality, RAG pipelines, and the lossy/lossless tradeoff.

---

## The evolution of embeddings

| Method | Year | Space | Context-aware? | Key advance |
|--------|------|-------|---------------|-------------|
| Word2Vec (Mikolov) | 2013 | Euclidean $\mathbb{R}^{300}$ | No — one vector per word | Learned embeddings from raw text |
| Poincaré (Nickel & Kiela, FAIR) | 2017 | Hyperbolic $\mathbb{H}^2$ | No — one vector per entity | Geometry matches hierarchical structure |
| ELMo (Peters, Zettlemoyer) | 2018 | Euclidean $\mathbb{R}^{1024}$ | Yes — vector changes per sentence | Same word, different meaning → different vector |
| BERT (Devlin) | 2018 | Euclidean $\mathbb{R}^{768}$ | Yes — bidirectional attention | Deeper context, self-supervised pretraining at scale |

**Luke Zettlemoyer** co-authored ELMo at the Allen Institute for AI / University of Washington. ELMo broke the "one word = one vector" assumption — the vector for "bank" differs depending on whether the sentence is about rivers or finance. This is the direct bridge from Word2Vec to modern LLMs; BERT and GPT both build on this insight.

---

## Poincaré embeddings — hyperbolic geometry for hierarchies

Word2Vec embeds words in flat Euclidean space ($\mathbb{R}^n$) — every direction and region is equivalent. Poincaré embeddings use **hyperbolic space** instead, which naturally represents hierarchical (tree-like) relationships.

**Why hyperbolic works for trees:** In the Poincaré disk, the space near the boundary is exponentially "larger" than it looks visually. Points that appear close near the edge are actually far apart in hyperbolic distance. A tree's node count grows exponentially with depth — hyperbolic space's volume grows exponentially with radius. The geometry matches the data structure.

```
EUCLIDEAN DISK (flat)                  POINCARÉ DISK (hyperbolic)

  ┌─────────────────┐                   ┌─────────────────┐
  │                 │                   │     ·  ·  ·     │
  │    ·     ·      │                   │   · · · · · ·   │
  │  ·    ·    ·    │                   │  · · · · · · ·  │
  │    ·     ·      │                   │ · · · ROOT · · ·│
  │  ·    ·    ·    │                   │  · · · · · · ·  │
  │    ·     ·      │                   │   · · · · · ·   │
  │                 │                   │     ·  ·  ·     │
  └─────────────────┘                   └─────────────────┘

  Uniform density everywhere.           Root near center, leaves
  Runs out of room fast for             spread toward boundary where
  hierarchical data.                    there's exponentially more room.
```

**Concrete comparison:**

| Dimensions | Euclidean (Word2Vec-style) | Hyperbolic (Poincaré) |
|-----------|---------------------------|----------------------|
| 2D | ~50 words before relationships collapse | Entire WordNet taxonomy faithfully |
| 5D | Rough clusters, ~10K entities | 100K+ entities with high fidelity |
| 200D | Full vocabulary, fine-grained analogies | Overkill — 5D already sufficient |

Poincaré embeddings are a **real working embedding space**, not a visualization trick. The model trains and runs inference in hyperbolic space. Downstream tasks (link prediction, taxonomy completion) use hyperbolic distances directly.

**Paper:** Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical Representations," NeurIPS 2017.

---

## Dimensionality — why 300D and not 3D?

Word2Vec uses 300 dimensions. BERT uses 768. GPT-style models go up to 12,288. Nobody can visualize any of these — but the math works identically to 2D/3D. A vector in $\mathbb{R}^{300}$ is just a list of 300 numbers. Dot products, cosine similarity, vector addition — same formulas, longer sums.

**The capacity argument:** Each dimension is a degree of freedom for encoding one sliver of meaning. In 2D, if "king" must be close to "queen" AND close to "royal" AND far from "banana" AND maintain the analogy direction... you run out of room. More dimensions = more nuance before relationships start interfering.

```
Dimensions vs capability:

  2-3D     ██                          ~50 words, relationships collapse
  50D      ██████████                  ~10K words, rough clusters
  300D     ██████████████████████████  ~1M words, fine-grained analogies
  768D     ████████████████████████████████  BERT-scale, contextual
  1000D+   ████████████████████████████████  diminishing returns, overfitting risk
```

No single dimension maps to one clean concept (like "gender" or "formality"). The representation is **distributed** — meaning is spread across all dimensions simultaneously.

---

## Visualization tools vs real embedding spaces

```
PURPOSE COMPARISON:

t-SNE / UMAP                           Poincaré embeddings
─────────────                           ────────────────────
Take high-dimensional vectors           The model ACTUALLY LEARNS
  → crush to 2D                         and OPERATES in this space
  → just for human eyeballs
                                        Distances are real and meaningful.
Distances in the plot are               Downstream tasks use hyperbolic
  approximate and often misleading.     distances directly.

No model trains or runs                 Training + inference happen here.
  inference in this space.

PURPOSE: human understanding            PURPOSE: machine learning
```

| | Purpose | Used at training time? | Distances meaningful? |
|---|---------|----------------------|----------------------|
| t-SNE | Visualization only | No — post-hoc projection | Approximate, often misleading |
| UMAP | Visualization only | No — post-hoc projection | Better than t-SNE, still lossy |
| Word2Vec space | Training + inference | Yes | Yes — cosine similarity is the core metric |
| Poincaré disk | Training + inference | Yes | Yes — hyperbolic distance encodes hierarchy |

---

## RAG and agent memory

Retrieval-Augmented Generation (RAG) is the practical application of embeddings at scale. The pipeline:

```
YOUR DOCUMENTS (markdown, PDFs, code)
        │
        ▼
┌──────────────────┐
│  CHUNK           │   Split into paragraphs or ~500 token windows
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  EMBED           │   Each chunk → BERT/ada-002 embedding → 768-1536D vector
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  STORE           │   Vectors → vector database (Pinecone, ChromaDB, Weaviate)
└──────────────────┘


AT QUERY TIME:

┌──────────────────┐
│  User question   │ → same embedding model → query vector
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  SIMILARITY      │   cosine similarity search → top-K most relevant chunks
│  SEARCH          │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  INJECT          │   Top-K chunks → inserted into LLM's context window
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  LLM ANSWERS     │   With your documents as grounding
└──────────────────┘
```

**Obsidian-as-agent-memory:** Some developers connect Obsidian (markdown note-taking app) to a vector database, embedding all their notes. An AI agent can then semantically search personal notes to answer questions grounded in the user's own knowledge base. This is RAG applied to personal knowledge management.

**Claude Code's approach is simpler:** The `MEMORY.md` index + individual memory files use keyword/metadata matching, not vector search. Less sophisticated but works for the scale of a single project.

**The stack connecting quiz topics to RAG:**
1. **Word2Vec** (quiz_5_10) → proved you can learn useful vectors from raw text
2. **Self-supervised pretraining** (quiz_5_08) → how BERT/ada-002 learned embeddings
3. **Dimensionality** → why those vectors are 768D, not 3D
4. **Poincaré** → alternative geometry for hierarchical retrieval (active research)

---

## Karpathy's views — embeddings vs raw context

From Andrej Karpathy's public talks:

**"Intro to Large Language Models" (November 2023):** Described the LLM as an operating system. The context window is working memory (fast, limited). Embedding-based search over documents is long-term memory (slower, unlimited capacity).

**"State of GPT" (Microsoft Build 2023):** Walked through the RAG pipeline explicitly — documents → chunk → embed → vector DB → query → cosine similarity → inject top-K. Called it the most practically important pattern for customizing LLMs to your own data.

**His nuance:** As context windows grow (4K → 128K → 1M+), brute-force "just read everything" starts competing with embedding-based retrieval. The trend line suggests the crossover keeps moving.

---

## Lossy vs lossless — the fundamental tradeoff

```
ORIGINAL TEXT (lossless, full fidelity)
═══════════════════════════════════════════════════════
"Bellman published his dynamic programming paper in 1957
 while at RAND Corporation. He chose the name 'dynamic
 programming' partly to hide the mathematical nature of
 his work from his boss, who was hostile to research."
═══════════════════════════════════════════════════════
         │                                    │
    EMBED (compress)                     STUFF RAW INTO
    into 768 floats                      CONTEXT WINDOW
         │                                    │
         ▼                                    ▼
┌─────────────────────┐          ┌──────────────────────────┐
│ [0.023, -0.41, ...]  │          │ Every. Single. Word.     │
│                     │          │ Exactly as written.      │
│ PRESERVED:          │          │ Nothing lost.            │
│ ✓ "about Bellman"   │          │                          │
│ ✓ "about RL/DP"     │          │ Cost: ~50 tokens of      │
│ ✓ "historical tone" │          │ context budget           │
│                     │          │                          │
│ LOST:               │          │ LLM reads it directly,   │
│ ✗ "hostile" (exact  │          │ answers any question     │
│    word + connotation)│         │ perfectly.               │
│ ✗ year = 1957 vs    │          │                          │
│    ~"mid 1900s"     │          │                          │
│ ✗ boss relationship │          │                          │
│ ✗ causal chain      │          │                          │
└─────────────────────┘          └──────────────────────────┘
```

**At scale — why embeddings win anyway:**

| | 1,000 notes | 10,000 notes | 100,000 notes |
|---|---|---|---|
| **Raw context** | ~2M tokens — barely fits 1M window | ~20M tokens — doesn't fit anywhere | Impossible |
| **Embeddings + top-K** | Works, slight accuracy loss | Works, same cost | Works, same cost |

**The practical spectrum today:**

| Scale | Best strategy |
|-------|--------------|
| A few files | Read raw — no embedding needed |
| Hundreds of notes | Could go either way |
| Thousands of docs (startup codebase) | Hybrid — embeddings for search, raw once you find the right file |
| Millions of records (enterprise) | Embeddings mandatory — raw is physically impossible |

Analogies: JPEG (lossy, compact, searchable) vs RAW photo (lossless, huge, perfect). MP3 vs WAV. The compression is always lossy, but at scale you must compress.

---

## Takeaway

- Poincaré embeddings use curved (hyperbolic) space to encode hierarchies in far fewer dimensions than Euclidean space — 5D hyperbolic ≈ 200D Euclidean for tree data
- Embedding dimensionality (300D → 12288D) exists because each dimension encodes one degree of freedom for meaning; 2D/3D simply doesn't have enough capacity
- t-SNE/UMAP are visualization-only projections; Poincaré is a real working space models train in
- RAG = chunk → embed → vector DB → query → top-K → inject into context — the production application of everything in the Word2Vec lineage
- Embeddings trade accuracy for scale; raw context trades scale for accuracy; as context windows grow, the crossover keeps shifting

---

*See also:* [quiz5/quiz_5_10_word2vec_deep_dive.md](quiz5/quiz_5_10_word2vec_deep_dive.md) · [quiz5/quiz_5_08_self_supervised_tasks_catalog.md](quiz5/quiz_5_08_self_supervised_tasks_catalog.md) · [28_how_llms_process_text.md](28_how_llms_process_text.md)
