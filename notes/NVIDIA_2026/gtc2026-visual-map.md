# GTC 2026 Visual Map — ASCII Diagrams

*Big-picture connections between all major GTC 2026 announcements.*
*For detailed component diagrams, see companion files linked at bottom.*

**Diagrams:**
1. [Master Stack — "The NVIDIA AI Factory"](#1-master-stack--the-nvidia-ai-factory)
2. [The Pareto Frontier — Inference Tradeoff](#2-the-pareto-frontier--inference-tradeoff)
3. [Hardware Evolution Timeline](#3-hardware-evolution-timeline)
4. [Vera Rubin NVL72 Rack — 6-Chip Architecture](#4-vera-rubin-nvl72-rack--6-chip-architecture)
5. [Inference Request Flow Through Dynamo](#5-inference-request-flow-through-dynamo)
6. [Ecosystem Partner Map](#6-ecosystem-partner-map)
7. [Physical AI — From AI Brain to Motor Control](#7-physical-ai--from-ai-brain-to-motor-control)

---

## 1. Master Stack — "The NVIDIA AI Factory"

How every major announcement layers together into one vertical story:

```
 ╔══════════════════════════════════════════════════════════════╗
 ║                   THE NVIDIA AI FACTORY                      ║
 ╠══════════════════════════════════════════════════════════════╣
 ║                                                              ║
 ║  LAYER 6 — APPLICATIONS                                     ║
 ║  ┌─────────────┐ ┌─────────────┐ ┌────────────────────────┐ ║
 ║  │  NemoClaw    │ │  OpenClaw   │ │  Enterprise Agents     │ ║
 ║  │  (enterprise │ │  (personal  │ │  (Salesforce, Adobe,   │ ║
 ║  │   agents)    │ │   agents)   │ │   Cisco, CrowdStrike)  │ ║
 ║  └──────┬──────┘ └──────┬──────┘ └───────────┬────────────┘ ║
 ║         └───────────────┼────────────────────┘              ║
 ║                         v                                    ║
 ║  LAYER 5 — MODELS                                           ║
 ║  ┌─────────────┐ ┌─────────────┐ ┌────────────────────────┐ ║
 ║  │ Nemotron 3/4 │ │  Partner    │ │  Nemotron Coalition    │ ║
 ║  │ (Mamba-MoE)  │ │  Models     │ │  (Mistral, Perplexity, │ ║
 ║  │ Nano/Super/  │ │  (GPT,Llama,│ │   Cursor,Black Forest) │ ║
 ║  │ Ultra        │ │   Claude)   │ │                        │ ║
 ║  └──────┬──────┘ └──────┬──────┘ └───────────┬────────────┘ ║
 ║         └───────────────┼────────────────────┘              ║
 ║                         v                                    ║
 ║  LAYER 4 — INFERENCE OS                                     ║
 ║  ┌──────────────────────────────────────────────────────┐   ║
 ║  │                   NVIDIA Dynamo                       │   ║
 ║  │  Smart Router + KV Cache Manager + NIXL transfers     │   ║
 ║  │  Routes prefill/decode to optimal hardware tier       │   ║
 ║  │  (Apache 2.0, Rust + Python, open source)             │   ║
 ║  └─────────────────────┬────────────────────────────────┘   ║
 ║                        v                                     ║
 ║  LAYER 3 — HARDWARE (three tiers)                           ║
 ║  ┌───────────────┐ ┌──────────────┐ ┌───────────────────┐  ║
 ║  │ Rubin CPX     │ │ Rubin NVL72  │ │ LPX Rack          │  ║
 ║  │ (prefill,     │ │ (balanced,   │ │ (decode,           │  ║
 ║  │  long context)│ │  bulk tokens)│ │  low-latency)      │  ║
 ║  │ GDDR7 DRAM    │ │ 288GB HBM4  │ │ 256 Groq 3 LPUs   │  ║
 ║  │               │ │ 50 PFLOPS   │ │ 150 TB/s SRAM/chip │  ║
 ║  └───────────────┘ └──────────────┘ └───────────────────┘  ║
 ║                        v                                     ║
 ║  LAYER 2 — INFRASTRUCTURE                                   ║
 ║  ┌─────────────┐ ┌─────────────┐ ┌────────────────────────┐ ║
 ║  │ DSX AI      │ │ BlueField   │ │ Spectrum-6 Ethernet    │ ║
 ║  │ (factory    │ │ ASTRA       │ │ (scale-out fabric)     │ ║
 ║  │  platform)  │ │ (security)  │ │                        │ ║
 ║  │ Boost/Flex/ │ │ BF-4 DPU   │ │ ConnectX-9 SuperNIC    │ ║
 ║  │ Exchange    │ │ 3rd-gen CC  │ │                        │ ║
 ║  └─────────────┘ └─────────────┘ └────────────────────────┘ ║
 ║                        v                                     ║
 ║  LAYER 1 — PHYSICAL AI                                      ║
 ║  ┌─────────────┐ ┌─────────────┐ ┌────────────────────────┐ ║
 ║  │ Robotics    │ │ Automotive  │ │ Space                  │ ║
 ║  │ Isaac/GR00T │ │ NVIDIA DRIVE│ │ Space-1 Module         │ ║
 ║  │ 1X, Figure, │ │ BYD,Toyota, │ │ (Vera Rubin derivative)│ ║
 ║  │ ABB, Disney │ │ Mercedes    │ │ Orbital data centers   │ ║
 ║  └─────────────┘ └─────────────┘ └────────────────────────┘ ║
 ║                                                              ║
 ╚══════════════════════════════════════════════════════════════╝
```

**The token demand thesis driving this stack:**

```
  Single chat prompt:    ~1 response            (       1x tokens)
  Agentic task loop:     ~1,000 responses        (   1,000x tokens)
  Always-on agent:       ~1,000,000 responses    (1,000,000x tokens)
  (OpenClaw / NemoClaw)                                  |
                                                         v
                           THIS explosive demand is why you need
                           Rubin + LPX + Dynamo at factory scale
```

---

## 2. The Pareto Frontier — Inference Tradeoff

```
  TPS/MW
  (energy
  efficiency)
     ^
     |
 3.5M|  *  *                          BULK ZONE
     |    * *    Rubin CPX              (batch processing,
     |      *  * handles                 offline jobs)
 2.5M|       o*  prefill
     |         *  *
     |           *  *                  GOLDILOCKS ZONE
 1.5M|             *  *                 (production APIs)
     |               * *
     |     Rubin NVL72  *
 1.0M|     balanced zone  o  *
     |                      *  *       LOW-LATENCY ZONE
     |                         *  *     (real-time chat,
 0.5M|                  LPX handles *    agentic loops)
     |                   decode      *
     |                                 *
     +--------+--------+--------+--------+---> TPS/user
              50      100      200      400    (interactivity)

  o = GPT MoE models Jensen showed on stage:
      Upper-left o : gpt-oss-120b (128k context)
      Lower-right o: GPT-5.2 (400k context, ~3x KV cache)
      Longer context = more memory BW → shifts RIGHT + DOWN

  ════════════════════════════════════════════════
  NVIDIA's goal: raise the ENTIRE Pareto curve
  Rubin + LPX = better on BOTH axes vs Blackwell
  ════════════════════════════════════════════════
```

---

## 3. Hardware Evolution Timeline

```
  2022-23        2024-25         2026          2027        ~2028
    |               |              |              |            |
    v               v              v              v            v
 ┌─────────┐  ┌───────────┐  ┌───────────┐  ┌─────────┐  ┌────────┐
 │ HOPPER  │  │ BLACKWELL │  │VERA RUBIN │  │ RUBIN   │  │FEYNMAN │
 │         │  │           │  │           │  │ ULTRA   │  │        │
 │H100 80GB│  │B200 192GB │  │336B xstrs │  │ (TBD)   │  │ (TBD)  │
 │  HBM3   │  │  HBM3e    │  │TSMC N3    │  │         │  │        │
 │H200 141G│  │B300U 288GB│  │288GB HBM4 │  │         │  │        │
 │  HBM3e  │  │  15 PFLOPS│  │22 TB/s BW │  │         │  │        │
 │80B, N4  │  │208B, 4NP  │  │50 PFLOPS  │  │         │  │        │
 │NVLink 4 │  │9 PFLOPS(B2│  │NVLink 6   │  │         │  │        │
 │3-4.8TB/s│  │NVLink 5   │  │3.6 TB/s   │  │         │  │        │
 │         │  │1.8 TB/s   │  │           │  │         │  │        │
 └────┬────┘  └─────┬─────┘  └─────┬─────┘  └────┬────┘  └───┬────┘
  Grace        David          Vera          (TBD)       Richard
  Hopper       Blackwell      Rubin                     Feynman
      │              │              │              │           │
      └──────────────┴──────────────┴──────────────┴───────────┘
       Theme:        Theme:         Theme:
       LLM           Scaling        Agentic AI
       training      inference      + Physical AI

  KEY JUMPS (Blackwell B200 → Vera Rubin):
  Memory BW:  8 → 22 TB/s    (+2.75x)
  FP4:        9 → 50 PFLOPS  (+5.5x)
  NVLink:     1.8 → 3.6 TB/s (+2x)
  Perf/watt:  10x improvement
  Cost/token: 10x reduction
```

---

## 4. Vera Rubin NVL72 Rack — 6-Chip Architecture

```
 ┌─────────────────────────────────────────────────────────────┐
 │              VERA RUBIN NVL72 RACK (one rack)               │
 │                                                             │
 │  ┌─────────────────────────────────────────────────────┐   │
 │  │              NVLink 6 Switch Fabric                  │   │
 │  │           260 TB/s total bandwidth                   │   │
 │  │           3.6 TB/s per GPU (all-to-all)              │   │
 │  └────┬──────────┬──────────┬──────────┬───────────────┘   │
 │       │          │          │          │                    │
 │       v          v          v          v                    │
 │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐              │
 │  │ Rubin  │ │ Rubin  │ │ Rubin  │ │ Rubin  │  ... x72     │
 │  │  GPU   │ │  GPU   │ │  GPU   │ │  GPU   │  GPUs total  │
 │  │ 336B   │ │ 336B   │ │ 336B   │ │ 336B   │              │
 │  │ 288GB  │ │ 288GB  │ │ 288GB  │ │ 288GB  │              │
 │  │ HBM4   │ │ HBM4   │ │ HBM4   │ │ HBM4   │              │
 │  └────────┘ └────────┘ └────────┘ └────────┘              │
 │       ^          ^          ^          ^                    │
 │       │          │          │          │                    │
 │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐              │
 │  │ Vera   │ │ Vera   │ │ Vera   │ │ Vera   │  ... x36     │
 │  │  CPU   │ │  CPU   │ │  CPU   │ │  CPU   │  CPUs total  │
 │  │ 88     │ │ 88     │ │ 88     │ │ 88     │  (Olympus    │
 │  │Olympus │ │Olympus │ │Olympus │ │Olympus │   Armv9.2)   │
 │  │ cores  │ │ cores  │ │ cores  │ │ cores  │              │
 │  └────────┘ └────────┘ └────────┘ └────────┘              │
 │                                                             │
 │  ┌──────────────────────┐  ┌──────────────────────┐        │
 │  │   BlueField-4 DPU    │  │   ConnectX-9 SuperNIC│        │
 │  │   (ASTRA security)   │  │   (800 Gb/s)         │        │
 │  │   64 Arm cores       │──│   Controlled by BF-4 │        │
 │  │   Sideband control   │  │   Host OS isolated   │        │
 │  └──────────┬───────────┘  └──────────┬───────────┘        │
 │             │                         │                     │
 │             v                         v                     │
 │  ┌─────────────────────────────────────────────────────┐   │
 │  │            Spectrum-6 Ethernet Switch                │   │
 │  │         TSMC 3nm   |   200G SerDes                   │   │
 │  │         Scale-out to other racks                     │   │
 │  └─────────────────────────────────────────────────────┘   │
 │                                                             │
 │  RACK TOTALS: 3.6 ExaFLOPS inference | 2.5 ExaFLOPS train  │
 └─────────────────────────────────────────────────────────────┘
```

---

## 5. Inference Request Flow Through Dynamo

```
                          ┌─────────────┐
                          │  User sends  │
                          │  request     │
                          └──────┬──────┘
                                 │
                                 v
                   ┌─────────────────────────┐
                   │      Dynamo Router       │
                   │  (KV-cache-aware radix   │
                   │   tree lookup)           │
                   └──────┬──────────────────┘
                          │
              ┌───────────┴───────────┐
              │ Does a worker already  │
              │ have cached KV blocks? │
              └───┬───────────────┬───┘
                  │               │
              YES │           NO  │
              (reuse)       (compute)
                  │               │
                  v               v
    ┌─────────────────────────────────────┐
    │         PREFILL PHASE               │
    │  Routed to: Rubin CPX              │
    │  (compute-bound, long context)      │
    │  Processes all input tokens at once  │
    └──────────────┬──────────────────────┘
                   │
                   │  KV cache transfer via NIXL
                   │  (direct VRAM → VRAM, non-blocking)
                   │
                   v
    ┌─────────────────────────────────────┐
    │         DECODE PHASE                │
    │  Routed to: LPX Rack (Groq LPUs)  │
    │  (bandwidth-bound, low-latency)     │
    │  Generates tokens one-by-one        │
    │  150 TB/s SRAM >> 22 TB/s HBM      │
    └──────────────┬──────────────────────┘
                   │
                   v
           ┌──────────────┐
           │   Response    │
           │   streamed    │
           │   to user     │
           └──────────────┘


    ALTERNATIVE ROUTES (Dynamo decides per-request):
    ┌────────────────────────────────────────────────┐
    │  Rubin CPX   → prefill for 1M+ token contexts  │
    │  Rubin NVL72 → balanced mixed workloads         │
    │  LPX Rack    → ultra-low-latency decode         │
    └────────────────────────────────────────────────┘

    State synced across cluster via NATS JetStream
```

---

## 6. Ecosystem Partner Map

```
 ┌──────────────────────────────────────────────────────────┐
 │                GTC 2026 ECOSYSTEM PARTNERS                │
 │            (who plugs in at which layer)                   │
 ├──────────────────────────────────────────────────────────┤
 │                                                           │
 │  AI LABS ────────────────────> MODELS + HARDWARE          │
 │  ┌──────────────────────────┐                             │
 │  │ Anthropic  OpenAI  Meta  │  Launch customers for       │
 │  │ Mistral    Perplexity    │  Vera Rubin NVL72           │
 │  │ Black Forest Labs Cursor │  + Nemotron Coalition       │
 │  └──────────────────────────┘                             │
 │                                                           │
 │  CLOUD ──────────────────────> INFRASTRUCTURE             │
 │  ┌──────────────────────────┐                             │
 │  │ AWS   Azure  GCP  Oracle │  Rubin samples Q2-Q3 2026  │
 │  │ CoreWeave  Crusoe Lambda │  Dynamo on EKS/AKS/GKE     │
 │  │ Nebius Nscale Together AI│                              │
 │  └──────────────────────────┘                             │
 │                                                           │
 │  ENTERPRISE ─────────────────> NEMOCLAW AGENTS            │
 │  ┌──────────────────────────┐                             │
 │  │ Salesforce (CRM)         │  Company policies →         │
 │  │ Adobe (creative)         │  autonomous agent            │
 │  │ Cisco (security/network) │  execution within            │
 │  │ CrowdStrike (threats)    │  guardrails                  │
 │  │ ServiceNow Palantir Zoom │                              │
 │  └──────────────────────────┘                             │
 │                                                           │
 │  ROBOTICS ───────────────────> PHYSICAL AI                │
 │  ┌──────────────────────────┐                             │
 │  │ 1X Technologies (NEO)    │  Isaac Lab / GR00T N1.6     │
 │  │ Figure AI (Helix VLA)    │  Cosmos world model          │
 │  │ Boston Dynamics (Atlas)  │  Jetson Thor compute         │
 │  │ ABB Robotics (industrial)│  Omniverse digital twins     │
 │  │ Disney Research (BDX)    │                              │
 │  │ Agility Unitree Apptronk │                              │
 │  └──────────────────────────┘                             │
 │                                                           │
 │  AUTOMOTIVE ─────────────────> NVIDIA DRIVE               │
 │  ┌──────────────────────────┐                             │
 │  │ BYD  Toyota  Mercedes    │  DRIVE AGX platform          │
 │  │ Hyundai  Nexon           │  Autonomous driving stack    │
 │  └──────────────────────────┘                             │
 │                                                           │
 │  INFRASTRUCTURE ─────────────> DSX AI FACTORIES           │
 │  ┌──────────────────────────┐                             │
 │  │ Switch (2 GW, Georgia)   │  Physical facilities         │
 │  │ Digital Realty (valid.)  │  Omniverse DSX Blueprint     │
 │  │ Stargate (1.2GW,Abilene)│  Power/cooling/compute       │
 │  │ Dell HPE Lenovo Suprmicro│  co-designed as one system   │
 │  │ Thinking Machines Lab    │                              │
 │  └──────────────────────────┘                             │
 │                                                           │
 │  SPACE ──────────────────────> SPACE-1 MODULE             │
 │  ┌──────────────────────────┐                             │
 │  │ Aetherflux  Axiom Space  │  Vera Rubin derivative       │
 │  │ Kepler Comms Planet Labs │  Orbital data centers        │
 │  │ Sophia Space  Starcloud  │  Radiation hardening WIP     │
 │  └──────────────────────────┘                             │
 │                                                           │
 └──────────────────────────────────────────────────────────┘
```

---

## 7. Physical AI — From AI Brain to Motor Control

Every robot and autonomous vehicle Jensen showed needs this split:
the AI brain runs on Linux, but the muscles need a real-time OS.

```
  ┌──────────────────────────────────────────────────┐
  │  NVIDIA Jetson / DRIVE AGX  (Linux)               │
  │  AI inference: Isaac ROS, perception, planning    │
  │  Runs: ROS 2 middleware on Ubuntu                 │
  └────────────────────────┬─────────────────────────┘
                           │ ROS 2 messages (DDS / Zenoh)
                           v
  ┌──────────────────────────────────────────────────┐
  │  micro-ROS bridge  (DDS-XRCE protocol)            │
  │  Translates ROS 2 pub/sub to MCU-size packets     │
  └────────────────────────┬─────────────────────────┘
                           │
                           v
  ┌──────────────────────────────────────────────────┐
  │  RTOS on microcontroller                          │
  │  ┌──────────┐ ┌──────────┐ ┌────────────────┐   │
  │  │ FreeRTOS │ │ Zephyr   │ │ ThreadX  (*)   │   │
  │  └──────────┘ └──────────┘ └────────────────┘   │
  │  Hard real-time: guaranteed sub-ms timing         │
  │                                                   │
  │  (*) ThreadX has IEC 61508 + ISO 26262 safety     │
  │      certs, but NO micro-ROS port exists yet      │
  └────────────────────────┬─────────────────────────┘
                           │ Direct hardware control
                           v
  ┌──────────────────────────────────────────────────┐
  │  MCU + Actuators                                  │
  │  STM32, ESP32, Renesas RA                         │
  │  Motors, sensors, brakes, steering                │
  └──────────────────────────────────────────────────┘

  SAME SPLIT IN AUTOMOTIVE:
  NVIDIA DRIVE (AI on Thor) <-> QNX/RTOS (safety control)
  BYD, Toyota, Mercedes all use this architecture

  See: micro-ros-threadx-opportunity.md for gap analysis
  See: rtos-ros-nvidia-research.md for full landscape
```

---

## Cross-References

- [gtc2026-keynote.md](gtc2026-keynote.md) — Full keynote notes
- [dynamo-deep-dive.md](dynamo-deep-dive.md) — Dynamo architecture deep dive
- [micro-ros-threadx-opportunity.md](micro-ros-threadx-opportunity.md) — RTOS gap analysis
- [rtos-ros-nvidia-research.md](rtos-ros-nvidia-research.md) — RTOS + ROS 2 research
