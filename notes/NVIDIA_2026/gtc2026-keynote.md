# GTC 2026 Keynote Notes — Jensen Huang
**Event:** NVIDIA GTC 2026, San Jose
**Date:** March 17, 2026

---

## Table of Contents

**[Part I: The Problem & The Vision](#part-i-the-problem--the-vision)**
1. [The Inference Efficiency Graph (Pareto Frontier)](#1-the-inference-efficiency-graph-pareto-frontier)
2. [Jensen's ROI / Profit Pitch](#2-jensens-roi--profit-pitch)

**[Part II: Hardware Platform](#part-ii-hardware-platform)**
3. [GPU Architecture Generations](#3-gpu-architecture-generations)
4. [Groq 3 LPU + LPX Rack](#4-groq-3-lpu--lpx-rack)
5. [Rubin CPX](#5-rubin-cpx)
6. [Roadmap: Rubin Ultra → Feynman](#6-roadmap-rubin-ultra--feynman)

**[Part III: Software Stack](#part-iii-software-stack)**
7. [NVIDIA Dynamo — The Inference OS](#7-nvidia-dynamo--the-inference-os)
8. [OpenClaw + NVIDIA NemoClaw](#8-openclaw--nvidia-nemoclaw)
9. [Nemotron Coalition](#9-nemotron-coalition)

**[Part IV: Infrastructure & Ecosystem](#part-iv-infrastructure--ecosystem)**
10. [DSX AI (Omniverse DSX Blueprint)](#10-dsx-ai-omniverse-dsx-blueprint)
11. [BlueField ASTRA](#11-bluefield-astra)
12. [Switch (Las Vegas)](#12-switch-las-vegas--the-physical-ai-factory-layer)
13. [NVIDIA Space-1 (Vera Rubin Module)](#13-nvidia-space-1-vera-rubin-module)

**[Part V: Physical AI](#part-v-physical-ai)**
14. [Robotics at GTC 2026](#14-robotics-at-gtc-2026)
15. [Automotive Partners (NVIDIA DRIVE)](#15-automotive-partners-nvidia-drive)

**[Part VI: Personal Notes & Career](#part-vi-personal-notes--career)**
16. [RTOS + ROS 2 Connection](#16-rtos--ros-2-connection)

**[Appendix](#appendix)**
- [Quick Reference Table](#quick-reference-key-numbers-to-remember)
- [Glossary](#glossary)
- [Related Files](#related-files)

---

# Part I: The Problem & The Vision

## 1. The Inference Efficiency Graph (Pareto Frontier)

Jensen opened with a scatter plot that frames the core problem in AI infrastructure.

**Axes:**
- **Y-axis — TPS/MW:** Tokens Per Second per Megawatt = *energy efficiency* (output per unit of power)
- **X-axis — TPS/user:** Tokens Per Second per User = *interactivity* (how responsive the system feels to each individual)

**The tradeoff:**
You can maximize bulk throughput (left side, high TPS/MW) OR minimize per-user latency (right side, high TPS/user) — but not both simultaneously at current tech levels.

**Three zones on the graph:**

| Zone | TPS/MW | TPS/User | Use case |
|---|---|---|---|
| Bulk tokens (left) | Very high (3.5M+ t/s/MW) | Low | Batch processing, offline jobs |
| Goldilocks (center) | Balanced | Balanced | Production inference APIs |
| Low-latency (right) | Lower | High | Real-time chat, agentic loops |

**Why GPT MoE 128k vs 400k appear on the graph:**
- These are context window sizes (tokens) for GPT Mixture-of-Experts models
- **128k context** (gpt-oss-120b class) — more energy efficient, higher TPS/MW; placed center-left
- **400k context** (GPT-5.2, released Dec 2025) — larger KV cache (~3x memory), trades efficiency for reasoning depth; placed further right-and-down
- Longer context = more memory bandwidth demand → shifts a model *right and down* (more capable/interactive, but heavier on power per token)
- Note: gpt-oss-120b achieves 1.5M tokens/sec on Blackwell GB200 NVL72

**NVIDIA's goal:**
"Raise the entire Pareto curve" — Rubin + LPX aim to be better on *both* axes simultaneously vs. Blackwell/Hopper.

---

## 2. Jensen's ROI / Profit Pitch

Jensen framed Rubin as a **business investment story**, not just a tech upgrade.

**The core argument:**
Inference is where data center revenue is generated. Lower cost-per-token = faster payback on hardware.

**Key claims:**
- **$1 trillion in combined Blackwell + Vera Rubin purchase orders** projected through 2027
- Rubin delivers **10x lower cost-per-token** vs. Blackwell
- Rubin achieves **50x better performance-per-watt** vs. Grace Blackwell (NVL72 comparison; Jensen initially said 30x, corrected to 50x)
- NVIDIA claims "lowest cost-per-token in the world" on Rubin

**The "Mellanox moment" framing:**
Groq acquisition = for inference what Mellanox was for networking — a missing piece that completes the stack.

**Agentic era framing:**
Autonomous agents running inference loops 24/7 → massively more inference demand than interactive chat → Rubin's efficiency directly monetizable.

---

# Part II: Hardware Platform

## 3. GPU Architecture Generations

### Hopper (2022–2023)
- **Flagship:** H100 / H200
- **Process:** TSMC N4, 80B transistors
- **H100 memory:** 80 GB HBM3, 3 TB/s bandwidth, 700W TDP
- **H200 memory:** 141 GB HBM3e (nearly 2x H100), 4.8 TB/s bandwidth (~1.4x H100)
  - H200 achieves 31,000+ tokens/sec on Llama 2 70B — 45% faster than H100
- **Key innovation:** First NVIDIA architecture with a Transformer Engine (dynamic FP16→FP8 precision switching)
- **Use cases:** LLM training and inference at scale

---

### Blackwell (2024–2025)
- **Chips:** B100, B200, B300 (Blackwell Ultra)
- **Process:** TSMC 4NP, 208B transistors (dual-die)
- **B200 peak:** 9 PFLOPS FP4 dense / 18 PFLOPS FP4 sparse
- **B300 (Ultra):** 15 PFLOPS FP4 dense, 288 GB HBM3e, 8 TB/s bandwidth, 1,400W TDP
- **NVLink 5.0:** 1.8 TB/s bidirectional per GPU (14x faster than PCIe Gen5)
- **NVL72 config:** 72 Blackwell GPUs + 36 Grace CPUs in a liquid-cooled rack
  - GB300 NVL72 delivers 1.1 ExaFLOPS FP4
- **vs Hopper:** 10x throughput/MW for MoE models

---

### Vera Rubin (2026 — announced GTC 2026)
Named after astronomer **Vera Rubin** (who provided key evidence for dark matter).
Architecture theme: **Agentic AI** — autonomous reasoning, planning, physical interaction.

**Chip specs (Rubin GPU):**

| Spec | Blackwell (B200) | Rubin |
|---|---|---|
| Process | TSMC 4NP | TSMC N3 (3nm) |
| Transistors | 208B | 336B (+1.6x) |
| Memory | 192 GB HBM3e | 288 GB HBM4 (~1.5x) |
| Memory BW | 8 TB/s | 22 TB/s (+2.75x) |
| NVLink | 5.0 — 1.8 TB/s | 6.0 — 3.6 TB/s (+2x) |
| FP4 inference | ~9 PFLOPS | 50 PFLOPS (+5.5x) |
| Performance/watt | baseline | 10x better |

**NVL72 rack (Rubin):**
- 72 Rubin GPUs + 36 Vera CPUs, all-to-all NVLink switching
- 260 TB/s total aggregate NVLink bandwidth
- **3.6 ExaFLOPS** inference / **2.5 ExaFLOPS** training
- vs Blackwell NVL72: 5x inference throughput, 10x lower cost per token, 4x fewer GPUs for MoE training

**Shipping timeline:**
- Samples to AWS / Azure / GCP / Oracle: Q2–Q3 2026
- Full production ramp: early 2027

**Launch customers (announced GTC 2026):**
- AI labs: Anthropic, Meta, Mistral AI, OpenAI
- Hyperscalers: AWS, Google Cloud, Microsoft Azure, Oracle Cloud
- Cloud partners: CoreWeave, Crusoe, Lambda, Nebius, Nscale, Together AI
- OEMs: Cisco, Dell, HPE, Lenovo, Supermicro
- Notable: Thinking Machines Lab — multiyear gigawatt-scale Vera Rubin deployment

**No public GitHub repos yet** for Vera Rubin NVL72 specifically (searched `github.com/NVIDIA` — none as of GTC day).

---

### Vera CPU (paired with Rubin in NVL72)

"World's first processor purpose-built for agentic AI and reinforcement learning."

| Spec | Detail |
|---|---|
| Cores | 88 custom NVIDIA "Olympus" Armv9.2 cores |
| Memory | LPDDR5X |
| Bandwidth | 1.2 TB/s (2x Grace, at half the power) |
| Interconnect | 2nd-gen NVIDIA Scalable Coherency Fabric |
| Density | 256 per rack (or 36 per NVL72 rack paired with 72 Rubin GPUs) |

**vs Grace CPU (Blackwell era):** Grace used Arm Neoverse V2 cores + 1st-gen fabric. Vera uses NVIDIA's own Olympus cores + 2nd-gen fabric. Purpose shift: Grace was a host CPU for GPUs; Vera is designed to *drive* agentic reasoning loops and RL training directly.

---

### The Full Vera Rubin Platform — 6 Chips

| Chip | Role |
|---|---|
| **Rubin GPU** | AI compute — 336B transistors, 288 GB HBM4, 50 PFLOPS FP4 |
| **Vera CPU** | Orchestration — 88 Olympus Arm cores, 1.2 TB/s |
| **BlueField-4 DPU** | Infrastructure control + security (ASTRA) — 64 Arm cores, 800 Gb/s |
| **ConnectX-9 SuperNIC** | GPU-to-network — 800 Gb/s, controlled via BlueField ASTRA |
| **NVLink 6 Switch** | Scale-up GPU interconnect — 3.6 TB/s/GPU, 260 TB/s total |
| **Spectrum-6 Ethernet** | Scale-out fabric — TSMC 3nm, 200G SerDes |

---

## 4. Groq 3 LPU + LPX Rack

### LPU — Language Processing Unit
An inference-only ASIC (not a GPU). Originally called "Tensor Streaming Processor" at Groq, renamed LPU post-ChatGPT.

**The core difference from a GPU — SRAM vs HBM:**

| Property | GPU (Rubin) | LPU (Groq 3) |
|---|---|---|
| Memory type | Off-chip HBM4 | On-chip SRAM only |
| Bandwidth | 22 TB/s | 150 TB/s per chip |
| Latency | Hundreds of nanoseconds | Nanoseconds |
| Energy/bit | ~6 picojoules | ~0.3 picojoules (20x cheaper) |
| Execution model | Parallel, dynamic scheduling | Deterministic, compiler-scheduled |
| Best for | Training + prefill (compute-dense) | Decode (memory-bandwidth-bound) |

**Why SRAM matters for decode:** Each output token requires loading every model weight from memory once. GPUs sit idle waiting for HBM. SRAM eliminates that bottleneck — 150 TB/s vs 22 TB/s, at 20x lower energy.

**Trade-off:** SRAM is physically large. One Groq 3 LPU = 500 MB. A 70B FP8 model = ~70 GB → needs 140+ LPUs just to hold the weights. The LPX rack (256 LPUs = 128 GB pooled SRAM) solves this.

### LPX Rack Specs (GTC 2026 announcement)

LPX is a **complementary inference rack** built on Groq LPU technology — from NVIDIA's acquisition of Groq (framed by Jensen as the "Mellanox moment" of inference).

- **256 Groq 3 LPUs per rack** (4x increase from first-gen Groq racks)
- Each Groq 3 LPU: 500 MB SRAM, **150 TB/s bandwidth** (vs Rubin GPU's 22 TB/s HBM4)
- Total rack: 128 GB SRAM, **40 PB/s combined bandwidth**
- PCBs: 52-layer M9 Q-glass boards with larger on-chip memory

**Why LPX exists alongside Rubin GPUs:**
The Pareto frontier shows Rubin GPUs excel at *bulk throughput* (left zone).
LPX racks — with their massive SRAM bandwidth — handle *low-latency, per-user interactive inference* (right zone).
Together they allow NVIDIA to cover the full inference spectrum.

### C2C Interconnect (RealScale)
- **Switchless, direct connection** — each LPU connects to neighbors directly, no switch in between
- Topology: similar to a dragonfly-plus pattern
- **Why no switch?** The LPU's deterministic execution model means the compiler pre-assigns every memory transfer to a specific time slot. A runtime switch would add unpredictable latency, breaking the deterministic model.
- **Plesiosynchronous protocol** — chips run at nearly the same clock with bounded drift; compiler accounts for this, making the entire 256-chip rack behave as one logical chip
- Link speed: 30 Gbps/lane, 4 lanes per direction per link
- LPX rack total scale-up bandwidth: **640 TB/s**

### Quick Reference: LPU Ecosystem Terms

| Term | Definition |
|---|---|
| **FPGA** | Xilinx Alveo U55C — used to prototype LPU architecture before ASIC tapeout. Not in production hardware. |
| **Host CPU** | Accepts inference requests, dispatches to LPU array, runs framework interface (PyTorch/JAX → LPU compiler), coordinates KV cache handoff between GPU prefill and LPU decode via NVLink Fusion. In Rubin NVL72, this role is played by the Vera CPU. |
| **BF4** | Most likely BlueField-4 DPU (see [section 11](#11-bluefield-astra)). Possibly also a 4-bit Brain Float numeric format — Groq uses TruePoint (INT8/FP8 weights, 100-bit accumulation) instead; whether Groq 3 adds BF4 is unconfirmed. <!-- TODO: confirm from session --> |

---

## 5. Rubin CPX

A variant Rubin GPU designed specifically for **massive-context inference workloads** (very long sequences like GPT-5.2's 400k context). Where standard Rubin GPUs use HBM4, CPX is optimized with high-capacity GDDR7 DRAM to hold enormous KV caches for million-token-plus contexts.

In the Dynamo disaggregated serving model, CPX handles the **prefill phase** — the compute-heavy processing of long input contexts — while standard Rubin NVL72 handles balanced workloads and LPX handles low-latency decode.

---

## 6. Roadmap: Rubin Ultra → Feynman

<!-- TODO: confirm Oberon — seen on stage, likely a switch/networking family name -->

Jensen compared generations with a use-case framing (not just specs):

| Generation | Year | Theme | Named After |
|---|---|---|---|
| Hopper | 2022 | LLM training at scale | Grace Hopper (computer scientist) |
| Blackwell | 2024–25 | Scaling inference; MoE models | David Blackwell (statistician) |
| Vera Rubin | 2026 | Agentic AI; reasoning; physical AI | Vera Rubin (astronomer) |
| Rubin Ultra | 2027 | — | — |
| Feynman | ~2028 | TBD | Richard Feynman (physicist) |

**Oberon** — appeared on stage in roadmap context. <!-- TODO: confirm what family this belongs to — possibly NVLink switch generations or networking silicon, not GPU compute -->

---

# Part III: Software Stack

## 7. NVIDIA Dynamo — The Inference OS

Dynamo is **pure software** — the "inference operating system" that ties all three hardware tiers (Rubin GPU, Rubin CPX, LPX) together. Open source (Apache 2.0, `github.com/ai-dynamo/dynamo`), built in Rust (core) + Python (APIs), successor to NVIDIA Triton Inference Server.

**The core problem it solves:** LLM inference has two phases with fundamentally different resource needs — **prefill** (compute-bound, processes all input tokens at once) and **decode** (memory-bandwidth-bound, generates tokens one-by-one). Running both on the same GPU wastes resources. Dynamo **disaggregates** them:

| Hardware tier | Optimized for | Dynamo routes... |
|---|---|---|
| **Rubin CPX** (high DRAM) | Prefill — long context (1M+ tokens) | Compute-heavy prefill |
| **Rubin NVL72** (288 GB HBM4) | Balanced bulk throughput | Mixed workloads |
| **LPX rack / Groq 3** (SRAM, 150 TB/s) | Decode — low-latency token generation | Interactive real-time decode |

**Key mechanisms:** A smart router using a radix tree tracks which KV cache blocks live on which workers across the cluster, routing new requests to workers that already hold relevant cached context. KV cache transfers between tiers use NIXL (direct VRAM-to-VRAM, non-blocking). State synced via NATS JetStream.

**Measured results:** 30x throughput on DeepSeek-R1 671B (GB200 NVL72); 6x higher throughput + 2x lower latency in Gcore production.

> **See [`dynamo-deep-dive.md`](dynamo-deep-dive.md) for full technical details** — architecture diagrams, component breakdown, Pareto frontier analysis, and Groq acquisition context.

---

## 8. OpenClaw + NVIDIA NemoClaw

### OpenClaw — The Open-Source Personal Agent Framework

**Origin:** Created by **Peter Steinberger** (Austrian developer, founded PSPDFKit — a PDF SDK company acquired for ~$100M).
- Nov 2025: released as **Clawdbot**
- Jan 27, 2026: renamed **Moltbot** (Anthropic trademark objection), then renamed **OpenClaw** 3 days later
- Growth: 310,000+ GitHub stars, 58,000+ forks, 1,200+ contributors in ~60 days — surpassed React's 10-year record
- Jensen at GTC 2026: *"probably the single most important release of software, you know, probably ever"*

**GitHub:** `github.com/openclaw/openclaw` — MIT licensed (310k+ stars, 58k+ forks in ~60 days — fastest-growing OSS project in GitHub history)

**What it does:** An always-on, local-first AI agent that navigates file systems, uses software tools, and executes multi-step workflows autonomously — no cloud dependency, no human needed at each step. Data stored as Markdown files on disk (local-first, user owns their data).

**The token demand math:**
- Single interactive prompt: ~1 response
- Agentic task loop: ~1,000x more tokens
- Always-on background OpenClaw agent: up to **1,000,000x more tokens** than a single prompt → this is the entire GTC 2026 inference demand narrative

**How Peter makes money:** He doesn't — from OpenClaw. It was MIT-licensed and he ran it *at a loss* (OpenAI subsidized token costs). On Feb 14, 2026, Steinberger **joined OpenAI**. OpenClaw moved to an independent **501(c)(3) open-source foundation**. OpenAI sponsors the project but does not own the code. Monetization was reputational — it led directly to his OpenAI hire.

**Why NVIDIA built NemoClaw:** When OpenAI acquired OpenClaw, enterprise customers who'd built on it faced vendor lock-in risk with OpenAI controlling the platform. NVIDIA stepped in with an independent, GPU-native enterprise alternative.

**Practical OpenClaw use cases (the "put your workflows in an agent" category):**
- Email triage — reads inbox on schedule, delivers prioritized summaries to Slack/Telegram
- Dependency + security monitoring — scans project files weekly, flags vulnerabilities
- Social media repurposing — monitors RSS/blog, auto-generates platform-specific posts (10+ hrs/week saved)
- Newsletter drafting — researches, drafts, schedules, avoids repeating past issues
- Meeting transcription + action-item extraction — emails summaries to participants
- Competitor intelligence — scrapes competitor sites for product/pricing changes
- SEC/earnings monitoring — alerts on specific company filings
- Code review pipelines — automated PR review agents
- Video production pipelines — end-to-end content creation automation

---

### NVIDIA NemoClaw — The Enterprise Layer on Top of OpenClaw

Think: **OpenClaw = engine. NemoClaw = full enterprise vehicle.**

NemoClaw adds on top of OpenClaw:
- Enterprise security, privacy controls, audit logs
- Authentication + authorization layers
- Tool-use framework for enterprise SaaS integrations
- Hardware-optimized deployment on NVIDIA DGX Spark / NVL72 racks
- Integration with NeMo framework, Nemotron models, NVIDIA NIM microservices
- **Hardware-agnostic** — runs on NVIDIA, AMD, Intel, CPU-only

**What Jensen's "put your company policies into NemoClaw" means:**
You load your company's **policies, workflows, permissions, and data access rules** into NemoClaw — the same rules you'd give a new employee at onboarding. The agents then execute multi-step tasks (CRM updates, data processing, security monitoring, content generation, business decisions) **autonomously within those guardrails**, without a human at each step.

**Practical use cases (by partner):**

| Partner | Agent use case |
|---|---|
| **Salesforce** | CRM automation, sales pipeline updates, customer service workflows |
| **Adobe** | Creative workflow automation, content generation agents |
| **Cisco** | Security monitoring, network configuration agents |
| **CrowdStrike** | Threat detection and response agents |
| **Google** | Cloud infrastructure agents |

**NemoClaw vs Dynamo — not competing:**

| | NemoClaw | Dynamo |
|---|---|---|
| Layer | Application — *what* agents do | Infrastructure — *how* inference is served |
| Problem | Enterprise task automation | Efficient GPU cluster inference routing |
| Relationship | Generates inference requests | Serves those requests efficiently |

**The open-source business model:** Partners get free usage in exchange for contributing — same playbook as CUDA built its ecosystem. NVIDIA makes money on the hardware that NemoClaw agents drive inference demand on.

---

## 9. Nemotron Coalition

**What it is:** A partner ecosystem initiative (not a formal consortium) to co-develop **Nemotron 4** — the next generation of NVIDIA's open-weight AI models.

Jensen claimed: *"Nemotron 3 Ultra will be the best base model in the world"* — the coalition is the vehicle for building what comes next.

**Named coalition members:**
- **Black Forest Labs** (open-source image/video generation)
- **Perplexity** (AI-powered search + reasoning)
- **Mistral** (open-weight LLMs)
- **Cursor** (AI developer tooling)

**Nemotron 3 model family (current gen, Dec 2025):**

| Variant | Size | Notes |
|---|---|---|
| Nano | ~30B MoE | Smallest |
| Super | 120B total / 12B active MoE | Already launched; 5x throughput vs prior gen |
| Ultra | TBD | Pending H1 2026; "best base model in the world" claim |

Architecture: **Hybrid Mamba-Transformer MoE** — open weights, training recipes on Hugging Face + GitHub.

**How it connects to the rest of the stack:**
- **Nemotron** (models) → powers **NemoClaw** (enterprise agents) → inference served by **Dynamo** → runs on **Rubin/LPX** hardware
- The coalition partners contribute fine-tuning data and vertical specialization (search, code, creative, security)

**Companies already building on Nemotron:** Accenture, Cadence, CrowdStrike, Deloitte, EY, Oracle Cloud, Palantir, ServiceNow (co-developed Apriel Nemotron 15B), Siemens, Synopsys, Zoom.

---

# Part IV: Infrastructure & Ecosystem

## 10. DSX AI (Omniverse DSX Blueprint)

NVIDIA's **full-stack AI factory infrastructure platform** — treats the entire data center (building, power, cooling, compute, networking, storage) as one co-designed system optimized for AI at gigawatt scale. First announced GTC Washington DC (Oct 2025); elevated to a Vera Rubin launch centerpiece at GTC 2026 with 200+ partners.

Note: "DSX" is not officially expanded — NVIDIA uses it as a brand name (like DGX). Speculation: "Data center System eXperience" or similar, but unconfirmed.

**The four problems DSX solves:**
1. **Power walls** — fixed grid contracts cap how many GPU racks you can add
2. **Stranded grid capacity** — ~100 GW of global grid power sits underutilized because data centers draw fixed max load instead of participating in demand response
3. **Fragmented IT/OT** — building systems (cooling, power) and compute systems run as separate silos, blocking holistic optimization
4. **Long build cycles** — gigawatt-scale factory design currently involves physical trial-and-error

**Three components:**

| Component | What it does | Key number |
|---|---|---|
| **DSX Boost** | NVIDIA Max-Q efficiency tuning at cluster level, not just GPU level — runs workloads at optimal performance-per-watt operating point | +30% GPU throughput within same power contract |
| **DSX Flex** | Translates electric grid signals into cluster-level power events; data center becomes a grid-flexible asset, absorbing/shedding load dynamically | Unlocks ~100 GW stranded grid capacity globally |
| **DSX Exchange** | Secure IT/OT integration fabric — links building ops (cooling, power, safety) to NVIDIA software stack via real-time APIs; dissolves the IT/OT boundary | Enables Omniverse digital twin of full facility |

**Vera Rubin DSX AI Factory Reference Design (GTC 2026):**
Co-designed validated blueprint covering compute (Rubin NVL72) + networking (Spectrum-XGS) + storage + power + cooling as one system. Goal: maximize tokens/watt, improve resiliency, accelerate time-to-production for new AI factory builds.

**Real deployments already running on DSX:**
- Switch's 2 GW site in Georgia
- Stargate facility (1.2 GW, Abilene, Texas)
- Validation site: Digital Realty, Manassas, Virginia

---

## 11. BlueField ASTRA

Note: What was initially heard as "Blueshield" during the keynote is actually **BlueField ASTRA** — no product called Blueshield exists. The name likely comes from "BlueField" + its security/shield function.

**Full name:** BlueField **A**dvanced **S**ecure **T**rusted **R**esource **A**rchitecture
**Chip:** BlueField-4 DPU (one of the 6 Rubin platform chips)

**Problem it solves:**
In cloud bare-metal AI infrastructure, if the host OS is compromised, an attacker can tamper with network provisioning, routing, and fabric config. Traditional setup: host OS configures NICs and network fabric → tenant workloads can interfere.

**ASTRA's fix:**
- BlueField-4 DPU becomes the **sole trusted control point** for all I/O in/out of the compute node
- Sits between host OS and both ConnectX-9 SuperNICs + NVLink fabric via dedicated sideband
- SuperNIC control plane is **completely isolated from host OS** — tenants cannot see or tamper with it
- Enables secure multi-tenant bare-metal AI (no hypervisor overhead)
- Covers CPU + GPU + NVLink domains simultaneously (**3rd-generation Confidential Computing**)
- Extends trust from front-end (north-south) all the way into east-west AI fabric (GPU-to-GPU NVLink)

**Generational jump:** BlueField-3 (Blackwell) did not have this. ASTRA is a BlueField-4 first.

---

## 12. Switch (Las Vegas) — The Physical AI Factory Layer

Switch kept coming up in the keynote because it's the **physical infrastructure story** behind Jensen's "AI factory" framing.

**What Switch is:**
- Data center colocation operator, Las Vegas HQ, founded 2000 by Rob Roy
- Taken private 2022 ($11B, DigitalBridge + IFM Investors)
- Not a cloud provider or GPU maker — they build and operate the actual buildings

**Why they're relevant at GTC 2026:**
- Hosted **CoreWeave's world-first NVIDIA GB300 NVL72 cloud deployment** inside their Las Vegas EVO AI Factory
- This is the full stack Jensen described: NVIDIA hardware → CoreWeave cloud operator → Switch physical facility — already running in Las Vegas before Vera Rubin ships
- Official **Omniverse DSX Blueprint** ecosystem partner — using NVIDIA's digital twin tools to design and manage their AI Factory lifecycle
- EVO AI Factory product: **2+ MW per cabinet**, liquid-cooled, explicitly designed to scale with NVIDIA's roadmap "from Blackwell, Rubin, and beyond"
- Had a booth (#91) in the DSX AI Infrastructure Pavilion at GTC 2026

**The narrative fit:**
Jensen is arguing AI compute = industrial infrastructure. Switch is building the industrial plants. The Las Vegas campus (2M+ sq ft) is one of the largest data center footprints in the world and is already hosting the hardware generations Jensen is showing on stage.

---

## 13. NVIDIA Space-1 (Vera Rubin Module)

**What it is:** A **space-qualified AI compute module** derived from the Vera Rubin GPU architecture — built for deployment aboard satellites and orbital data centers (ODCs). Not a rack, not a cooling system — actual satellite hardware.

Jensen's framing: *"Going out to space and starting data centers there."*

**Why it's hard:** Jensen specifically said *"we're working on how to deal with radiation now"* — radiation hardening is the active unsolved engineering challenge for Space-1. (Note: the liquid cooling / "45°C warm water" discussion at GTC was about the **terrestrial** NVL72, not Space-1.)

Space constraints vs. ground:
- Must run on solar power → extreme energy efficiency
- Must survive radiation
- Must fit within satellite SWaP limits (Size, Weight, and Power)

**Space partner ecosystem announced:**
Aetherflux, Axiom Space, Kepler Communications, Planet Labs, Sophia Space, Starcloud

**Availability:** "Later date" — not shipping at announcement. Jetson Orin and IGX Thor are the current space-ready products.

---

# Part V: Physical AI

## 14. Robotics at GTC 2026

<!-- TODO: confirm "Redblock robot" — not found in any official GTC 2026 coverage; may be misheard name -->

**On stage during Jensen's keynote:**
- **1X Technologies — NEO** (humanoid, Norwegian): autonomous domestic tasks using GR00T N1
- **Disney Research — BDX Droids** (Star Wars-inspired): joined Huang on stage; co-developed Newton physics engine with NVIDIA + Google DeepMind

**GR00T / Isaac platform partners (broad ecosystem):**

| Company | Robot | Notes |
|---|---|---|
| Agility Robotics | Digit | Amazon-backed humanoid |
| Figure AI | Figure | Uses NVIDIA Helix VLA model |
| Boston Dynamics | Electric Atlas | Electric humanoid |
| NEURA Robotics | — | German humanoid |
| Unitree Robotics | G1 / H1 | Chinese humanoid |
| Apptronik | Apollo | — |
| XPENG Robotics | Iron | — |
| ABB Robotics | Industrial | Physical AI at scale partnership |
| WORKR | — | $25/hr robotic worker demo at NVIDIA booth |

**NVIDIA technologies powering these robots:**
- **Isaac Lab** — RL simulation training
- **Isaac GR00T N1.6** — humanoid foundation model
- **Cosmos** — world foundation model / synthetic data for sim-to-real
- **Omniverse** — digital twins
- **Jetson Thor / T4000** — on-robot compute

---

## 15. Automotive Partners (NVIDIA DRIVE)

Jensen announced new NVIDIA DRIVE partnerships with:
- **BYD**, **Hyundai**, **Nexon** (spelling TBC), **Mercedes**, **Toyota**
<!-- TODO: confirm full partner list and any new DRIVE Thor / DRIVE Orin announcements -->

---

# Part VI: Personal Notes & Career

## 16. RTOS + ROS 2 Connection

**Every robot and autonomous vehicle Jensen showed today needs an RTOS underneath.**

The AI/GPU layer gets the stage time, but safety-critical control (braking, steering, motor control, sensor fusion) runs on a separate real-time operating system. Eclipse ThreadX fits directly into this stack at three intersection points:

1. **NVIDIA DRIVE + Automotive ECUs** — DRIVE handles AI inference; safety-critical vehicle control runs on a separate ASIL-certified RTOS. The BYD/Toyota/Mercedes partnerships all need this split architecture.
2. **Robotics: micro-ROS on RTOS** — NVIDIA Isaac / ROS 2 runs at the application layer; micro-ROS handles real-time actuator control on RTOS backends. Every humanoid on stage has this layered architecture.
3. **Eclipse SDV Ecosystem** — Eclipse Foundation hosts both ThreadX and the Software Defined Vehicle initiative. Several of Jensen's automotive partners are SDV members.

**RTOS vs ROS 2 — they are NOT the same thing:**

| | RTOS (e.g. ThreadX) | ROS 2 |
|---|---|---|
| Actually an OS? | Yes — runs on microcontrollers | No — it's a middleware framework |
| Runs on | MCUs (STM32, ESP32) | Linux (Ubuntu on Jetson) |
| Timing | Hard real-time (guaranteed) | Best-effort (not guaranteed) |
| Purpose | Control hardware directly (motors, sensors) | Coordinate robot modules (planning, perception) |
| Memory | Kilobytes | Gigabytes |
| Example task | "spin motor at exactly 1kHz" | "plan a path around obstacle" |

A robot needs **both**: ROS 2 is the brain's language, RTOS is the muscles' guarantee. **micro-ROS** is the translator — it lets the RTOS side speak ROS 2's publish/subscribe protocol so the brain (Jetson + Linux) can send commands to the muscles (MCU + ThreadX) without custom serial protocols.

```
 Jetson (Linux)          Network           STM32 (ThreadX)
 +------------+          (UDP)            +----------------+
 |  ROS 2     | ──── messages ──────────> | micro-ROS      |
 |  "move to  |                           | receives msg   |
 |   45 deg"  |                           | calls ThreadX  |
 +------------+                           | tx_thread to   |
                                          | run motor PID  |
                                          +----------------+
```

> **For detailed research** on the RTOS + ROS landscape, career paths, and the micro-ROS ThreadX opportunity, see:
> - [`rtos-ros-nvidia-research.md`](rtos-ros-nvidia-research.md) — full landscape analysis, NVIDIA specifics, skills, and hiring
> - [`micro-ros-threadx-opportunity.md`](micro-ros-threadx-opportunity.md) — the gap analysis, execution roadmap, and career value chain

---

# Appendix

## Quick Reference: Key Numbers to Remember

| Metric | Hopper H100 | Blackwell B200 | Rubin |
|---|---|---|---|
| Memory | 80 GB HBM3 | 192 GB HBM3e | 288 GB HBM4 |
| Memory BW | 3 TB/s | 8 TB/s | 22 TB/s |
| FP4 inference | — | 9 PFLOPS | 50 PFLOPS |
| NVLink | 4.0 | 5.0 | 6.0 |
| NVL72 ExaFLOPS (inf) | — | 1.1 | 3.6 |
| Process node | N4 (4nm) | N4P (4nm) | N3 (3nm) |

---

## Glossary

| Term | Definition |
|---|---|
| **BF4** | BlueField-4 DPU — NVIDIA's Data Processing Unit for infrastructure control + security |
| **C2C** | Chip-to-Chip — direct inter-chip interconnect (used in LPX rack's RealScale topology) |
| **CPX** | Rubin GPU variant optimized for massive-context inference with high-capacity DRAM |
| **DPU** | Data Processing Unit — offloads networking, storage, and security from CPUs/GPUs |
| **DSX** | NVIDIA's AI factory infrastructure platform (brand name, not officially expanded) |
| **FP4 / FP8** | 4-bit / 8-bit floating-point formats used for efficient AI inference |
| **GR00T** | NVIDIA's humanoid robot foundation model |
| **HBM** | High Bandwidth Memory — stacked DRAM used in GPUs (HBM3, HBM3e, HBM4) |
| **LPU** | Language Processing Unit — Groq's SRAM-based inference ASIC |
| **LPX** | Rack form factor for Groq LPUs (256 LPUs, 128 GB pooled SRAM) |
| **MoE** | Mixture of Experts — model architecture where only a subset of parameters activate per token |
| **NIXL** | NVIDIA Inference Transfer Library — non-blocking VRAM-to-VRAM KV cache transfer |
| **NVL72** | 72-GPU rack configuration with all-to-all NVLink switching |
| **PFLOPS** | PetaFLOPS — 10^15 floating-point operations per second |
| **SRAM** | Static Random-Access Memory — fast, on-chip memory used in LPUs |
| **SWaP** | Size, Weight, and Power — design constraints for space/embedded hardware |
| **TDP** | Thermal Design Power — maximum heat a chip is designed to dissipate |
| **TPS** | Tokens Per Second — throughput metric for LLM inference |

---

## Related Files

| File | Description |
|---|---|
| [`dynamo-deep-dive.md`](dynamo-deep-dive.md) | Full technical deep dive on NVIDIA Dynamo inference OS — architecture, components, Pareto analysis |
| [`gtc2026-visual-map.md`](gtc2026-visual-map.md) | ASCII visual map — 6 diagrams connecting all GTC 2026 announcements into one coherent picture |
| [`rtos-ros-nvidia-research.md`](rtos-ros-nvidia-research.md) | RTOS + ROS 2 + NVIDIA robotics landscape research — ecosystem status, career paths, hiring |
| [`micro-ros-threadx-opportunity.md`](micro-ros-threadx-opportunity.md) | Gap analysis for micro-ROS ThreadX port — opportunity, execution roadmap, career value |
