# Wearable AI Startup — Inference Routing Opportunity

*Date: March 30, 2026*

---

## 1. The Message

> "hey James, noticed you in the SensAI Hackademy #introduce-yourself community. we're building a wearable AI startup, currently focused on routing each inference request to the right model size in real time on constrained hardware. we're hiring a founding engineer who lives in this space. open to a quick chat?"

Source: SensAI Hackademy #introduce-yourself channel

---

## 2. What They're Actually Building (Technical Breakdown)

**"Routing each inference request to the right model size"** means:

A wearable device with multiple model tiers available (e.g., tiny/small/medium/large), and a **router/dispatcher** that decides per-request which model to use based on:

- **Query complexity** — "what time is it?" → tiny model. "Summarize the last hour of conversation" → large model.
- **Latency budget** — urgent requests → smallest model that can handle it. Background tasks → larger model.
- **Battery/power state** — low battery → route to smallest model or offload to phone/cloud.
- **Hardware load** — if the on-device accelerator is busy, route to a different tier or queue.
- **Context length** — short input → small model. Long input → model with larger context window.

This is essentially **NVIDIA Dynamo's routing problem scaled down to a wearable**:

```
DYNAMO (datacenter)                    WEARABLE ROUTER (edge)
─────────────────                      ──────────────────────
GPU cluster with multiple model        Wearable with multiple quantized
replicas of different sizes            model tiers (4-bit, 8-bit, fp16)
     ↓                                      ↓
Smart router picks optimal GPU         Smart router picks optimal model
based on KV-cache, load, latency       based on complexity, battery, latency
     ↓                                      ↓
Prefill/decode disaggregation          On-device vs. phone vs. cloud split
```

**"On constrained hardware"** means:
- Limited RAM (maybe 1-4GB on a wearable SoC)
- Limited compute (mobile NPU or small GPU, not datacenter)
- Power budget (battery life matters)
- Thermal constraints (can't run hot inference on your face)
- Storage limits (can't store 20 models — need quantized variants)

---

## 3. Skills Match — How My Study Work Maps

### A. Model Size/Cost Benchmarking → DIRECT MATCH

| What I've Studied | Where | Why It Matters |
|-------------------|-------|----------------|
| Parameter Golf competition — fitting models into 16MB | `notes/James_notes_deepLearning_team_project/08_parameter-golf.md` | They need models that fit on constrained hardware. I've studied how to compress models to extreme sizes while maintaining quality (BPB metrics). |
| Comparing nanoGPT (124M-1.5B params) vs llm.c vs nanochat (560M params) | `notes/James_notes_deepLearning_team_project/02_comparison-table.md` | Understanding the performance characteristics of different model sizes is exactly what their router needs to know. |
| Benchmark evaluation (DCLM CORE, ARC, MMLU, GSM8K) | `notes/James_notes_deepLearning_team_project/06_datasets-benchmarks-comparison.md` | How to measure if a smaller model is "good enough" for a given request — the core routing decision. |
| GPT-4o-mini at 87.5% accuracy for 40x less cost (MongoDB RAG project) | External project: MongoDB-Agentic-Context-Window | Proved that smaller models can handle most requests. The routing problem is about knowing WHEN the small model is sufficient. |

### B. Inference Orchestration/Routing → EXCELLENT MATCH

| What I've Studied | Where | Why It Matters |
|-------------------|-------|----------------|
| NVIDIA Dynamo — smart routing across GPU clusters | `notes/llm-inference-stack/llm-inference-stack.md` | Dynamo IS a model router. KV-cache-aware routing, prefill/decode disaggregation, Pareto frontier optimization. Their wearable router is this concept at smaller scale. |
| SGLang + FlashInfer (29% faster than vLLM) | Same file | Understanding inference engine internals — what makes inference fast or slow, and where bottlenecks appear. |
| Prefill = compute-bound, Decode = memory-bandwidth-bound | Same file | The router needs to understand WHY different model sizes have different latency profiles. This is the answer. |
| vLLM PagedAttention | Same file | Memory-efficient inference — critical on constrained hardware where you can't waste RAM. |

### C. Constrained Hardware / RTOS → PERFECT FIT (3 Years Production Experience)

| Experience / Knowledge | Where | Why It Matters |
|------------------------|-------|----------------|
| **3 years production ThreadX at JCM Global** — daily RTOS development on embedded hardware, UART/BLE data buffers | Professional experience (2023–present) | This isn't studied knowledge — I ship production code on ThreadX daily. Wearable AI devices run an RTOS, and I already work in that world professionally. |
| RTOS landscape (FreeRTOS, Zephyr, ThreadX, NuttX) | `notes/NVIDIA_2026/rtos-ros-nvidia-research.md` | Broad understanding of the RTOS ecosystem beyond just ThreadX — can evaluate which RTOS fits their wearable's constraints. |
| Micro-ROS + ThreadX gap analysis | `notes/NVIDIA_2026/micro-ros-threadx-opportunity.md` | Deep analysis of safety-certified RTOS integration — shows I think about system-level architecture on embedded devices. |
| Real-time constraints (hard vs. soft, PREEMPT_RT, Xenomai) | Same files | "Real-time routing" means the routing decision itself must meet latency constraints. I understand what that requires at the OS level. |
| Two-layer architecture (Linux for AI + RTOS for hardware control) | RTOS research notes | Their wearable likely has a similar split — ML inference on one core/processor, hardware control on another. I work with this kind of architecture at JCM. |

### D. CUDA / Low-Level Optimization → GOOD FOUNDATION

| What I've Studied | Where | Why It Matters |
|-------------------|-------|----------------|
| llm.c — pure C/CUDA implementation, 7% faster than PyTorch | `notes/James_notes_deepLearning_team_project/04_llmc-notes.md` | Proves you can get significant speedups by dropping to C/CUDA. On constrained hardware, every % matters. |
| FlashInfer kernels (MLSys 2025 Best Paper) | `notes/llm-inference-stack/llm-inference-stack.md` | Attention kernels, GEMM, MoE, sampling — the actual compute primitives that run during inference. |
| CUDA programming guide + parallel programming patterns | `CUDA class/` directory | Foundational GPU programming knowledge. |

### E. Active Team Project → DRY RUN FOR THIS JOB

| What We're Building | Where | Why It Matters |
|---------------------|-------|----------------|
| Forking nanochat, fitting into 16MB, 10-min H100 training budget | `notes/James_notes_deepLearning_team_project/10_approach-draft.md` | This IS the constrained inference problem. Smaller model, strict constraints, measuring quality vs. size tradeoffs. |
| Gated attention modifications (NeurIPS 2025 inspired) | Same file | Architectural changes to improve quality within constraints — exactly what you'd do to improve the small model tier in a routing system. |
| SFT + GRPO post-training under constraints | Same file | Fine-tuning compressed models for specific tasks — relevant to making each model tier good at its routing-assigned workload. |

---

## 4. Contrast with Mira (Lessons Learned)

| | Mira | This Startup |
|---|---|---|
| **Core technical problem** | API calls to Gemini on ODM hardware | Model routing + on-device inference on constrained hardware |
| **ML depth** | None — "we don't use RAG," just long context window | Deep — custom routing, model selection, quantization, edge deployment |
| **Hardware ownership** | White-labeled from MYVU/DreamSmart | Building their own inference stack (implied by "constrained hardware" focus) |
| **Technical moat** | None identified | The router itself + optimized model tiers |
| **My skills match** | Overqualified / mismatched | Right-sized — this is exactly what I've been studying |
| **Founding engineer role** | Unclear what "highly technical" meant to CTO | Clear problem statement: routing + constrained inference |

**Lesson from Mira:** Ask direct technical questions early. Validate the actual stack before investing prep time.

---

## 5. Questions to Ask on the Chat

### Technical depth (show you understand the problem)

1. **"What's your current routing strategy — is it rule-based (if complexity > threshold → large model) or learned (a small classifier that predicts which model tier to use)?"**
   *Shows you know there are different approaches to the routing decision.*

2. **"Are the model tiers different architectures, or the same architecture at different quantization levels (fp16/int8/int4)?"**
   *Shows you understand quantization as a model-size lever.*

3. **"Where does inference happen — fully on the wearable SoC, split between wearable and phone, or with a cloud fallback?"**
   *Learn whether this is true edge inference or hybrid. Directly relevant to your RTOS + inference stack knowledge.*

4. **"What's your latency target per request, and how do you measure it end-to-end?"**
   *Shows real-time systems thinking. Learn if their constraints are similar to Mira's 700ms or tighter.*

5. **"What SoC/NPU are you targeting? Qualcomm QCS series, MediaTek, or custom silicon?"**
   *Shows hardware awareness. The answer tells you a lot about their stage and ambition.*

### Company evaluation (learn from Mira red flags)

6. **"How many people are on the team currently, and what's the split between hardware and software?"**
   *7-person Mira was a red flag. Understand team structure.*

7. **"What's your current stage — do you have a working prototype routing requests, or is this still in research?"**
   *Prototype = real engineering. Research-only = higher risk.*

8. **"What does the founding engineer role look like day-to-day? Is it more systems/infra, more ML research, or both?"**
   *Mira's role turned out to be API integration despite the job posting. Clarify early.*

---

## 6. Talking Points for the Conversation

### Lead with the intersection

> "I've been studying this exact intersection — I spent the last few months going deep on both the LLM inference stack (NVIDIA Dynamo, SGLang, vLLM) and constrained embedded systems (RTOS, real-time scheduling, safety-certified platforms). What drew me in is that the routing problem on a wearable is essentially Dynamo's request orchestration scaled down to an edge device with power and memory constraints."

### Concrete proof points

> "I'm currently working on a team project where we're fitting a language model into a 16MB artifact with a 10-minute training budget on H100s — it's a compressed model competition. The core challenge is measuring quality degradation as you shrink the model, which is exactly the data a routing system needs to make good decisions."

> "My RAG benchmarking project showed that GPT-4o-mini achieves 87.5% of GPT-4's accuracy at 40x lower cost. That's the kind of size-quality tradeoff analysis your router needs to make per-request in real time."

> "I've been working with ThreadX professionally for 3 years at JCM Global, handling UART/BLE data buffers on constrained embedded systems. So the RTOS side isn't something I've just read about — it's my day job. I also have broader context on FreeRTOS, Zephyr, and real-time scheduling constraints from my own research."

### Be honest about gaps

> "I have 3 years of production embedded/RTOS experience, so the constrained hardware side is real professional experience, not just study. The gap is specifically on production model routing — my Dynamo knowledge is from studying the architecture, and my constrained inference work is the ongoing team project. But I'm closer to this than most candidates because I already live in both worlds: embedded systems professionally and ML inference optimization in my research."

---

## 7. Why This Matters

This opportunity combines ALL the threads of my study:
- Neural network fundamentals (how models work at the architecture level)
- Inference optimization (how to make models run fast and cheap)
- RTOS/embedded systems (how constrained hardware behaves)
- Model benchmarking (how to measure quality vs. size tradeoffs)
- The team project (hands-on compressed model work)

Unlike Mira, where my skills were overqualified for API calls on ODM hardware, this role needs someone who understands **both** the ML inference stack **and** the embedded systems constraints. That intersection is rare — and it's exactly where I've been positioning myself.
