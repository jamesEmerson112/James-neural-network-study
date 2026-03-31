# Mira — Competitive Landscape & White-Label Hardware Intel

*Updated: March 30, 2026 — Post-interview debrief*

---

## 1. The White-Label Revelation

A Mira employee said:

> "I would go to Captify or Even Realities. Captify literally uses the same white-labeled glasses from the same manufacturer."

### What this means

- Mira does **not manufacture its own glasses hardware**. They source from an ODM (Original Design Manufacturer).
- **Captify** confirmed uses **MYVU (DreamSmart Group)** — a subsidiary of Geely Automotive (Meizu's parent) based in Hangzhou/Shenzhen. MYVU makes 43g AR glasses with waveguide displays, BLE 5.2/5.3, running their "Flyme AR" OS.
- The implication: **Mira likely also uses MYVU/DreamSmart hardware** (or the same Shenzhen supply chain) and layers their own software on top.
- The glasses (dual waveguide, ~39-43g, BLE 5.2, microphones, no camera) are essentially **commodity hardware** available to multiple brands.

### What the ODM is

**MYVU / DreamSmart Group:**
- Subsidiary of Geely Automotive (which acquired Meizu)
- Based in Hangzhou, China
- Makes the world's lightest monochromatic AR glasses (43g)
- Optical engine supplier: **Meta-Bounds** (provides waveguide technology)
- Runs Flyme AR OS + Flyme AI large model (for their own brand)
- Expanding to Southeast Asia (11 countries via Digital Way Group)
- Strategic vision: "Phone + XR + Smart Car" ecosystem

Other major ODM players in this space: Beautaste (10M+ glasses/year, factories in Vietnam/Thailand/China), Vuzix (enterprise), and various Shenzhen manufacturers.

---

## 2. Competitor Breakdown

### Captify — Same Hardware, Accessibility Focus

| | Details |
|---|---|
| **Product** | Captioning glasses for deaf/hard of hearing |
| **Founded by** | Tom Pritsky (lives with hearing loss, built TranscribeGlass v1) & Jason Gui (hardware veteran, shipped 10K+ smart glasses to 100+ countries) |
| **Hardware** | MYVU (DreamSmart Group) — confirmed white-label partnership |
| **Software** | MYVU AR companion app handles transcription; cloud ASR from Microsoft, Google, Amazon |
| **Killer feature** | Real-time captions projected on lens in green text |
| **Price** | **$99 USD** (vs Mira's $479-$799) |
| **Battery** | 4 hours continuous captioning, 48 hours standby |
| **Languages** | 13 (English, Spanish, French, Italian, German, Chinese, Japanese, Russian, Vietnamese, Malaysian, Indonesian, Turkish, Thai) |
| **Funding** | Kickstarter-funded |
| **Target market** | Deaf/hard of hearing community — pure accessibility play |

**Key insight:** Captify is NOT a direct competitor to Mira in the AI/memory space. They use the same hardware but target a completely different market (accessibility vs. "second brain"). Their $99 price point vs Mira's $479+ shows the hardware itself is cheap — Mira's margin is in the AI subscription ($20/mo).

### Even Realities — Display-First, Chinese-Backed

| | Details |
|---|---|
| **Product** | G1 (2024) and G2 (CES 2026) smart glasses |
| **Founded by** | Wang Xiaoyi (Will Wang) — ex-Apple (iPhone, Apple Watch), ex-Anker, ex-JMGO |
| **Team** | ~70 employees, hires from Apple, Samsung, Philips, Mykita, Lindberg |
| **Hardware** | **Appears to be in-house design** in Shenzhen (Shenzhen Yiwen Technology Co., Ltd.), 41 patents, proprietary "HAOS" MicroLED optical engine |
| **Display** | MicroLED waveguide, 640x200, green monochrome, 1000 nits, 20Hz |
| **Software** | Even AI (frontend) → Perplexity / ChatGPT (backends); G2 adds "Conversate" (always-listening ambient AI) |
| **Price** | **$599** (G1 and G2 base) |
| **Battery** | 1.5-2 days (160mAh + 2000mAh charging case) |
| **Weight** | 44g (G1) |
| **Languages** | 13+ |
| **Funding** | >$10M across 3 rounds (Series A July 2024), 100% Chinese VC |
| **Target** | Consumer/prosumer — professionals wanting discreet AI in meetings |

**Key features:** Teleprompt (scripted text on lens), Translate, Navigate, Transcribe, QuickNotes, smart notifications. G2 adds R1 companion ring.

**Security concerns (flagged by Suzu Labs):**
- 12 compromised employee credentials on dark web (unrotated for 1+ year)
- Extensive device fingerprinting (IMEI, installed apps, clipboard, calendar, location)
- Voiceprints shared with unnamed third-party providers
- Parent company subject to China's National Intelligence Law (Article 7)
- Privacy policy URL returned 404 at time of investigation

**Key insight:** Even Realities is bigger (70 people vs 7), better funded, and designs their own hardware. Their "Conversate" feature on G2 is a direct competitor to Mira's "remember everything." However, their privacy story is a **major vulnerability** — Mira's "audio deleted after processing, transcripts on phone only" is a significant advantage in the US/EU market.

### Mira vs. Competitors Summary

| Dimension | Mira | Captify | Even Realities |
|-----------|------|---------|----------------|
| **Team size** | 7 | Small (Kickstarter) | ~70 |
| **Funding** | $6.6M (GC, Naval) | Kickstarter | >$10M (Chinese VC) |
| **Hardware** | ODM (likely MYVU) | ODM (MYVU confirmed) | In-house (Shenzhen) |
| **Price** | $479-$799 | $99 | $599 |
| **AI subscription** | $20/mo | None | None (free tier only) |
| **Core bet** | AI/memory ("second brain") | Accessibility (captions) | Display/HUD (teleprompt) |
| **AI backend** | Gemini + proprietary RAG | Microsoft/Google/Amazon ASR | Perplexity/ChatGPT |
| **Privacy** | Strong (local storage, audio deleted) | Standard | Weak (Chinese law, fingerprinting) |
| **Moat** | Software/AI pipeline | Price + accessibility market | Hardware design + patents |

---

## 3. What This Means for Mira's Moat

### Hardware is NOT the moat

If Captify can sell the same physical glasses for $99, the hardware is a commodity. Mira charges $479-$799 for the same (or similar) chassis. The ~$400-$700 premium is **entirely the software/AI layer**:

1. **Transcription pipeline** (Soniox) — always-on, streaming
2. **RAG over conversations** — the "remember everything" retrieval system
3. **Gemini integration** — generation layer for answering questions
4. **iOS companion app** — session management, privacy controls
5. **Sub-700ms latency** — end-to-end pipeline optimization

### The REAL competitive advantages

1. **Privacy positioning** — "No camera, audio deleted, transcripts on YOUR phone." This is Mira's strongest narrative vs Even Realities (Chinese data law) and vs Meta Ray-Ban (camera = social stigma).

2. **AI pipeline quality** — The RAG system that retrieves the right memory from hours of conversation. This is where James's MongoDB-Agentic-Context-Window experience maps directly.

3. **Latency engineering** — Sub-700ms across the full pipeline. This is a systems engineering problem, not just an AI problem. RTOS thinking (time-budget allocation across pipeline stages) applies.

4. **Subscription revenue** — $20/mo recurring vs competitors' one-time hardware sales. Sustainable business model IF the AI is good enough to retain subscribers.

### The vulnerability

The employee recommended competitors. This could signal:
- Internal frustration (7-person team, possibly overworked)
- Skepticism about execution ability vs better-funded competitors
- OR just honest assessment that the market is commoditizing fast

---

## 4. How My Skills Map to the REAL Differentiator

Since hardware is commoditized, the value should be in software/AI. But post-interview, the picture is different:

| My Skill | What I Assumed | Reality at Mira |
|----------|---------------|-----------------|
| **RAG pipeline** | Core to their product | They don't use RAG at all |
| **LLM cost optimization** | Critical for margins | Relevant — they're sending full transcripts to Gemini, which is expensive |
| **RTOS/C experience** | Firmware optimization | Hardware is ODM; Mira doesn't write firmware |
| **Neural network understanding** | Debug/optimize AI layer | The "AI layer" is Gemini API calls — limited scope for this |
| **LLM inference stack** | Deploy/optimize inference | They use hosted Gemini, not self-hosted inference |
| **Firmware awareness** | BLE/power optimization | Not their problem — MYVU handles it |

### Honest takeaway

My skills are **overqualified for what Mira actually needs** (API integration on ODM hardware) and **mismatched on the specific stack** (they don't use RAG). The skills themselves are strong — they just fit better at a company doing deeper AI/ML work.

### These skills WOULD matter at

- A company actually building RAG pipelines (search, enterprise AI, knowledge management)
- A company doing on-device ML (real embedded AI, not cloud API calls)
- A company optimizing LLM inference at scale (not just calling Gemini)
- A team that needs someone who can bridge systems engineering and ML

---

## 5. Post-Interview Debrief (March 30, 2026)

### "We don't use RAG"

The CTO (Caine Ardayfio) stated directly: **Mira does not use RAG.** This contradicts the job posting and the "remember everything" marketing. The actual tech stack appears to be:

```
Mics → BLE → Phone → Soniox transcription → Full transcript into Gemini's context window → Answer
```

No embedding. No vector DB. No retrieval. No reranking. Just dump conversation history into Gemini's long context window (1M+ tokens) and let the model handle it.

**What this means:**
- The "AI layer" is essentially **Gemini API calls** — no proprietary retrieval pipeline
- Cost scales linearly with usage (more conversation history = more tokens = more money per query)
- Latency scales with context length (700ms target gets harder as transcripts grow)
- The MYVU supplier could replicate this trivially (they already have Flyme AI)
- **There is no technical moat** — hardware is ODM, software is API calls

### Software layer not differentiated from supplier

Upon closer inspection, Mira's software features don't clearly differentiate from what MYVU/DreamSmart already ships with their Flyme AR OS:
- MYVU already has: transcription, AI assistant, translation, display interface
- Mira adds: Gemini instead of Flyme AI, "memory" branding, privacy emphasis (local storage)
- The delta is thin — essentially a different LLM plugged into a similar pipeline

### CTO interview impressions

- **Asked repeatedly for "very highly technical" projects** but couldn't specify what domain or problem he wanted to see. This is vague for someone leading a 7-person technical team.
- **Appeared distracted and disengaged** during the presentation of both public and closed-source work.
- **Body language suggested he had no intention of working with me** before the interview started — seemed like going through the motions.
- Context: 22-year-old Harvard dropout who went viral with prototype demos (80M+ views). Strong at marketing/fundraising, unclear on production engineering depth.

### Pattern recognition: Three red flags

| Signal | What it suggests |
|--------|-----------------|
| ML employee recommending competitors ("go to Captify or Even Realities") | Internal team may not believe in the product |
| Software not differentiated from ODM supplier | Value proposition is weaker than marketed |
| CTO disengaged + "we don't use RAG" | Technical depth may be shallow; product is simpler than the job posting implies |

### Lesson learned

My RAG project (MongoDB-Agentic-Context-Window) is actually **more sophisticated than what Mira is building**. The prep was based on their job posting mentioning RAG, which turned out to be inaccurate. In future interviews: ask directly about the tech stack BEFORE preparing a presentation around assumptions.

### Is the long-context-only approach valid?

To be fair, "just use Gemini's long context window" is a legitimate architectural bet:
- Gemini's context window is growing (1M+ tokens)
- Simpler architecture = fewer things to break
- Google is making long context cheaper over time

But the risks are real:
- **Cost:** Sending 500K tokens per query at scale burns money fast
- **Latency:** Long context = slower inference, threatening the 700ms target
- **Vendor lock-in:** Entirely dependent on Google's pricing and availability
- **No fallback:** If Gemini gets expensive or slow, there's no retrieval layer to reduce token count

RAG exists precisely because long context alone doesn't scale economically. Mira is betting it will. Time will tell.

---

## 6. New Interview Questions Based on This Intel

1. "How do you think about hardware differentiation when the waveguide/display supply chain is becoming commoditized?" *(Shows you understand the market)*
2. "What's the split in your 700ms budget across BLE, transcription, retrieval, and generation?" *(Shows systems thinking)*
3. "With Captify at $99 on similar hardware, how do you justify the premium to customers long-term?" *(Shows business thinking — answer should be: the AI subscription is the value)*
4. "Are you considering running parts of the RAG pipeline on-device (phone) to reduce cloud dependency and improve privacy further?" *(Shows technical depth)*

---

## Sources

- [Captify Official](https://captify.glass/)
- [Captify Kickstarter](https://www.kickstarter.com/projects/captify/captify-captions-for-real-life)
- [Captify + MYVU Partnership](https://captify.glass/pages/captify-myvu)
- [Even Realities Official - G1](https://www.evenrealities.com/g1)
- [Even Realities Official - G2](https://www.evenrealities.com/smart-glasses)
- [Even Realities Company](https://www.evenrealities.com/company)
- [Tom's Guide - G1 Review](https://www.tomsguide.com/computing/smart-glasses/even-realities-g1-smart-glasses-review)
- [Tom's Guide - G2 Review](https://www.tomsguide.com/computing/smart-glasses/even-realities-g2-smart-glasses-review)
- [Suzu Labs Security Investigation](https://suzulabs.com/suzu-labs-blog/internal-analysis-even-realities-g2-smart-glasses-security-privacy-investigation)
- [DreamSmart Group / MYVU](https://www.dreamsmart.com/en)
- [DreamSmart Southeast Asia Expansion](https://autonews.gasgoo.com/icv/70030994.html)
- [Mira Official](https://www.trymira.com/)
- [General Catalyst - Seeding Mira](https://www.generalcatalyst.com/stories/seeding-the-future-with-mira)
- [Mira $6.6M Funding](https://pulse2.com/mira-6-6-million-seed-funding/)
- [Gizmodo - Mira Review](https://gizmodo.com/oh-great-smart-glasses-that-record-everything-you-say-2000699011)
- [Hearing Tracker - AR Captioning Glasses Review](https://www.hearingtracker.com/hearing-glasses/hear-with-your-eyes-five-ar-live-captioning-glasses)
- [Top 8 AI Glasses Manufacturers 2026](https://www.beautasteyewear.com/blog/sunglasses-blog/top-8-ai-glasses-manufacturers/)
