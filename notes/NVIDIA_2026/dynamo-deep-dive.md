# NVIDIA Dynamo — Deep Dive
**Context:** GTC 2026 (March 2026), San Jose
**Status:** Open-source, production-available (announced GTC 2025, expanded GTC 2026)

---

## 1. What Is Dynamo?

NVIDIA Dynamo is a **datacenter-scale distributed inference serving framework** — software, not hardware.

It is the **orchestration layer** that sits above individual inference engines (SGLang, vLLM, TensorRT-LLM) and coordinates how requests flow across a cluster of GPUs. Jensen Huang has called it the **"inference operating system"** for AI factories.

Think of it this way:
- Inference engines (SGLang, vLLM) = the **cars** (execute forward passes on individual GPUs)
- Dynamo = the **traffic control system** (decides which car handles each request, moves data between cars)

**Built in:** Rust (performance-critical core) + Python (extensibility/APIs)
**Open source:** Yes — [github.com/ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)
**Successor to:** NVIDIA Triton Inference Server

---

## 2. The Problem Dynamo Solves

### Background: Prefill vs Decode (the two phases of LLM inference)

Every LLM inference request has two distinct computational phases:

```
Input tokens → [PREFILL] → KV Cache → [DECODE] → Output tokens
                   ↑                      ↑
           Compute-bound            Memory-bandwidth-bound
           (processes all           (generates one token
           input tokens at once)    at a time, reads KV cache)
```

**Prefill:** Processes all the input tokens simultaneously. Heavy matrix multiplications → GPU compute-bound.
**Decode:** Generates output tokens one by one. Each step must read the entire KV cache → memory-bandwidth-bound.

### The Problem on a Single GPU

If you run both phases on the same GPU:
- Prefill hogs compute → decode stalls (bad latency for users)
- Decode hogs bandwidth → prefill queues up (bad throughput for operators)
- KV caches from different users compete for the same VRAM
- You can't scale one phase without the other

### What Dynamo Does

Dynamo solves this by:

1. **Disaggregated prefill/decode** — routes prefill to separate GPU pools optimized for compute, routes decode to separate GPU pools optimized for memory bandwidth
2. **KV-cache-aware routing** — routes new requests to GPUs that already hold the relevant cached context (avoids recomputing what's already been computed)
3. **Distributed KV cache management** — tracks KV cache locations across an entire cluster (not just one GPU)
4. **NIXL (NVIDIA Inference Transfer Library)** — high-speed, non-blocking, direct GPU-to-GPU KV cache transfer (VRAM → VRAM)

---

## 3. Dynamo's Four Core Components

```
┌──────────────────────────────────────────────────────────┐
│                    NVIDIA Dynamo                          │
│                                                          │
│  ┌──────────┐  ┌────────────────┐  ┌──────────────────┐ │
│  │ Planner  │  │  Smart Router  │  │ Distributed KV   │ │
│  │          │  │                │  │ Cache Manager    │ │
│  │ Plans    │  │ Decides which  │  │                  │ │
│  │ request  │  │ GPU/node gets  │  │ Tracks prefix    │ │
│  │ execution│  │ each request   │  │ blocks in radix  │ │
│  └──────────┘  └────────────────┘  │ tree across all  │ │
│                                    │ workers          │ │
│  ┌────────────────────────────────┐└──────────────────┘ │
│  │ NIXL (Inference Transfer Lib) │                      │
│  │ Non-blocking GPU VRAM→VRAM KV  │                      │
│  │ transfer during decode         │                      │
│  └────────────────────────────────┘                      │
└──────────────────────────────────────────────────────────┘
```

### KV-Cache-Aware Routing (how it works)

The router maintains a **radix tree** tracking which prefix blocks (chunks of KV cache) are cached on each worker.

When a new request arrives, the cost function is:

```
cost = overlap_score_weight * prefill_blocks + decode_blocks
```

- High weight → favor workers that already have the relevant cache (minimize Time to First Token / TTFT)
- Low weight → distribute evenly (minimize Inter-Token Latency / ITL)

The router picks the lowest-cost worker. This state is persisted via NATS JetStream so new router replicas can sync on startup.

---

## 4. How Dynamo Fits Into the Full Inference Stack

```
┌────────────────────────────────────────────────┐
│           NVIDIA Dynamo (orchestration)         │  ← Routes requests, manages cluster state
│     "Traffic control" across all GPU nodes      │
├────────────────────────────────────────────────┤
│   SGLang  │    vLLM   │   TensorRT-LLM         │  ← Inference engines (one per node/GPU group)
│     (the "cars" that execute forward passes)    │
├────────────────────────────────────────────────┤
│               FlashInfer (kernels)              │  ← GPU kernel library (raw compute)
├────────────────────────────────────────────────┤
│              NVIDIA GPU Hardware                │
│    Rubin / Rubin CPX / LPX (Groq LPUs)         │
└────────────────────────────────────────────────┘
```

Dynamo does NOT execute inference itself — it orchestrates engines that do. On a single GPU, you don't need Dynamo at all.

---

## 5. Dynamo and the Pareto Frontier

### The Efficiency Tradeoff Graph

Jensen's GTC keynote opened with a scatterplot with two axes:
- **Y-axis:** Tokens/second per Megawatt (energy efficiency — bulk throughput)
- **X-axis:** Tokens/second per User (interactivity — per-user latency)

These are in tension: optimizing for one usually hurts the other. The **Pareto frontier** is the curve connecting the optimal tradeoffs — the set of configurations where you cannot improve one axis without sacrificing the other.

### How Dynamo Pushes the Frontier

Without Dynamo (monolithic serving):
- Prefill and decode compete on the same GPU
- Can't independently scale compute-heavy prefill vs memory-heavy decode
- Stuck at a suboptimal point on the Pareto curve

With Dynamo (disaggregated):
- Prefill workers run on compute-optimized hardware (e.g., Rubin CPX with massive DRAM for long contexts)
- Decode workers run on bandwidth-optimized hardware (e.g., LPX with 150 TB/s SRAM bandwidth from Groq LPUs)
- KV-cache reuse via intelligent routing cuts redundant computation
- Result: **better throughput AND better latency** simultaneously — moves the entire curve up-and-right

### Measured Results

| Workload | Hardware | Dynamo Improvement |
|---|---|---|
| DeepSeek-R1 671B (reasoning) | GB200 NVL72 | 30x throughput increase (tokens/sec/GPU) |
| General LLM serving | H200 (FP8) | Shifts Pareto frontier — blue/red curves in InferenceMAX benchmarks |
| GCore production | Multi-node | 6x higher throughput + 2x lower latency vs naive serving |

The NVIDIA InferenceMAX benchmark suite specifically uses the Pareto frontier to visualize Dynamo + TensorRT-LLM vs baseline, showing the full curve shift.

---

## 6. Why Dynamo Is Discussed Alongside Rubin GPUs and Groq LPUs

At GTC 2026, NVIDIA presented a unified inference hardware + software story:

```
Hardware tier:                 Software tier (Dynamo):
─────────────────────          ─────────────────────────────────
Rubin CPX GPU                  Handles PREFILL for long contexts
(GDDR7, massive DRAM,          (compute-bound, 1M+ token windows)
 million-token contexts)  ←──  Routes prefill requests here

Rubin GPU (NVL72)              Handles mixed workloads
(288 GB HBM4, 22 TB/s)    ←──  Balanced throughput/latency zone
 Bulk token generation

LPX rack (Groq LPUs)           Handles DECODE for real-time use
(500 MB SRAM per LPU,    ←──  Routes decode/interactive requests
 150 TB/s SRAM bandwidth)      here (memory-bandwidth-bound)
 256 LPUs per rack
```

**Dynamo is the glue** that routes each phase of each request to the right hardware tier. Without Dynamo, you can't efficiently use all three hardware types in one system.

This is why Jensen says NVIDIA is building an "AI factory" — it's not just a GPU, it's a software-orchestrated pipeline across specialized hardware, similar to how a factory has specialized machines for each production stage.

---

## 7. The Groq Acquisition Context ("Mellanox Moment")

Jensen explicitly framed the Groq deal as analogous to NVIDIA's 2020 Mellanox acquisition:

| Deal | What it added | Why it mattered |
|---|---|---|
| Mellanox (2020, ~$7B) | InfiniBand networking | Networking is the bottleneck in multi-GPU training — NVIDIA bought the solution |
| Groq (2025, ~$20B licensing) | LPU technology (SRAM-heavy, low-latency decode) | Low-latency interactive inference is the bottleneck in agentic AI — NVIDIA bought the solution |

**The Groq LPU problem it solves:**
NVIDIA GPUs (HBM-based) have massive compute but limited bandwidth for decode-phase inference. Each token generated requires reading the entire KV cache. For real-time agentic systems that need <100ms latency, HBM bandwidth is the wall.

Groq's LPU architecture is entirely SRAM-based (no HBM):
- 500 MB SRAM per LPU chip
- 150 TB/s bandwidth (vs 22 TB/s for Rubin HBM4) — ~7x higher
- Optimized specifically for decode: low batch size, ultra-low latency
- Weakness: limited total memory (SRAM is expensive and small), not good for prefill

Dynamo routes decode-phase traffic to LPX racks. Dynamo is what makes the Groq LPU useful at datacenter scale — without the orchestration layer, you can't dynamically split requests between GPU and LPU tiers.

---

## 8. Key Technical Details

| Property | Detail |
|---|---|
| Language | Rust (core) + Python (APIs/extensibility) |
| License | Open source (Apache 2.0) |
| GitHub | [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) |
| Backends supported | SGLang, vLLM, TensorRT-LLM |
| KV transfer mechanism | NIXL (direct VRAM-to-VRAM, non-blocking) |
| Cache state storage | NATS JetStream (persistent, distributed) |
| Deployment | Kubernetes-native (EKS, AKS, GKE all documented) |
| Production adoption | AWS EKS, Azure AKS, GCP AI Hypercomputer, Gcore |
| GTC 2026 sessions | ~20 dedicated Dynamo sessions at the conference |

---

## 9. Summary: Why Dynamo Matters for This Study

If you're studying LLM inference, Dynamo is the answer to: *"How do you efficiently serve LLMs at scale across many GPUs?"*

Key insights to remember:
1. **Prefill is compute-bound; decode is memory-bandwidth-bound** — they need different hardware
2. **Disaggregated serving** (Dynamo's core idea) separates them, enabling independent scaling
3. **KV cache reuse** via smart routing is a huge efficiency multiplier (avoids recomputing the same prompt prefixes)
4. **The Pareto frontier** is the conceptual frame: Dynamo + specialized hardware lets you be better on BOTH throughput and latency, not just one
5. **Dynamo is the "inference OS"** — it's what makes a heterogeneous cluster of Rubin GPUs, Rubin CPX, and LPX/Groq LPUs work as one coherent system

---

## Cross-References

- See `gtc2026-keynote.md` for the Pareto frontier graph details and hardware specs
- See `llm-inference-stack.md` for where Dynamo fits in the full training-to-serving pipeline
- See `llm-inference-stack.md` → FlashInfer section for what runs *inside* the inference engines Dynamo orchestrates
