# Vincent Quenneville-Belair

From simulating black holes to teaching machines to hear.

---

## Career arc

```
McGill (2008)                    Joint Honours Math + Physics
  │                              iGEM 2008 (biology competition — did the math modeling)
  ▼
University of Minnesota          MSc Applied Math (2011)
(2008-2015)                      MCS Computer Science (2014)
  │                              PhD Applied Math (2015)
  │                              Advisor: Douglas N. Arnold (FEEC creator)
  │                              Thesis: finite element simulations of general relativity
  │                              NSERC Alexander Graham Bell Scholarship
  ▼
Columbia University              Chu Assistant Professor of Applied Mathematics
(2015-2017)                      Gravitational wave propagation from black hole collisions
  │                              LIGO detected gravitational waves Sep 2015 —
  │                              3 months after his thesis defense
  ▼
Amazon #1 (2017-2019)            ML Scientist, Supply Chain & Marketing
  │                              Demand forecasting, causal impact models
  │                              Taught internal courses on matrix factorization
  ▼
Meta / Facebook AI               ML Engineer, PyTorch team
(2019-2021)                      Created torchaudio (200+ contributors)
  │                              OneCycleLR scheduler, chainable schedulers
  │                              PyTorch emeritus maintainer
  ▼
Amazon #2 (2021-present)         ML Scientist, Supply Chain Forecasting
                                 SPADE, GEANN, Goldilocks papers
```

---

## Key contributions

### Torchaudio

PyTorch's official audio/speech processing library. Not an "Audio Language Model" — it's infrastructure, like torchvision is for images.

```
What torchaudio provides:

Raw audio (WAV, MP3, FLAC)
    ↓  I/O: loading, saving, format conversion
Waveform
    ↓  Transforms: spectrograms, MFCCs, mel-frequency, resampling
Features
    ↓  Pre-trained models: Wav2Vec2, HuBERT (speech recognition),
    │                      Tacotron2 (text-to-speech)
    ↓  CTC decoders, forced alignment, speech enhancement
Outputs


Where torchaudio sits in the modern audio LLM stack:

Raw audio → [torchaudio transforms] → [speech encoder (Wav2Vec2/HuBERT/Whisper)]
    → audio tokens/embeddings → [LLM backbone (GPT-4o, Gemini)]
    → text or audio output

Torchaudio lives in the first few links of this chain.
```

### PyTorch optimizers

- **OneCycleLR** (PR #25324) — implements Leslie Smith's Super-Convergence (2018). Anneals learning rate from initial → maximum → minimum much lower than start. One of the most widely used LR schedulers in PyTorch.
- **Chainable schedulers** (PRs #24352, #26423) — architectural change allowing composition of multiple LR schedulers together.
- **Scheduler ordering warnings** (PR #31125) — warnings when chainable schedulers are used in incorrect order.

---

## Publications

| Year | Title | Venue | Summary |
|------|-------|-------|---------|
| 2009 | On Taylor Dispersion Effects for Transient Solutions in Geothermal Heating Systems | Int. J. Heat Mass Transfer | Minimum pipe length for geothermal systems accounting for flow-diffusion interaction |
| 2014 | Uniform-in-Time Superconvergence of the HDG Methods for the Acoustic Wave Equation | Mathematics of Computation | With Bernardo Cockburn. Proved HDG methods converge faster than expected for acoustic waves, and the accuracy doesn't degrade over time |
| 2015 | A New Approach to Finite Element Simulations of General Relativity | PhD Thesis, UMN | Approximated Weyl curvature directly via Einstein-Bianchi system using FEEC; borrowed stability from plate bending theory |
| 2021 | TorchAudio: Building Blocks for Audio and Speech Processing | ICASSP 2022 | System paper for torchaudio v0.10, 24 co-authors |
| 2023 | Distribution-Free Multi-Horizon Forecasting and Vending System | KDD (MileTS) | "Vending machine" API — request any forecast at any lead time, span, or quantile. 13.4% improvement on extreme quantiles |
| 2023 | GEANN: Scalable Graph Augmentations for Multi-Horizon Time Series Forecasting | KDD (DLG) | Cold-start forecasting via graph neural networks — predict demand for products with zero sales history |
| 2024 | Effectively Leveraging Exogenous Information Across Neural Forecasters | NeurIPS Workshop | SSM layers to capture temporal dynamics of promotions, holidays, weather |
| 2024 | SPADE: Split Peak Attention DEcomposition | NeurIPS Workshop | Separates peak events (Prime Day) from normal demand. 30% improvement on worst post-peak forecasts |
| 2025 | Goldilocks: Active Sampling Bandit for Multi-Task Forecasting | KDD (AI4SupplyChain) | Multi-armed bandit selects training examples — "not too hard, not too easy." Connects directly to exploration vs exploitation (quiz_5_03) |

### The acoustic wave paper in detail

**"Uniform-in-Time Superconvergence of the HDG Methods for the Acoustic Wave Equation"** (2014, with Bernardo Cockburn)

HDG = Hybridizable Discontinuous Galerkin. You break the computational domain into cells, allow the solution to jump between cells, then introduce unknowns on cell boundaries that let you solve cell-by-cell (static condensation).

**Superconvergence:** Normally, degree-$k$ polynomials give convergence at rate $k+1$. Cockburn and Vincent proved you get better-than-expected accuracy. The critical result: this accuracy **doesn't degrade over time**. In wave simulations, tiny errors normally accumulate and compound. They proved that doesn't happen with HDG.

This paper foreshadowed his audio work — acoustic wave simulation → audio processing library, five years later. The connection is more thematic than technical, but the domain knowledge carried.

---

## Audio ML debugging — why it's harder than text

```
TEXT LLMs — tight feedback loop:

  Input: "What is 2+2?"
  Output: "5"
  You: I can SEE that's wrong. Immediately. No tools needed.


AUDIO MODELS — broken feedback loop:

  Input: [waveform]     What does "wrong" even look like?
  Output: [waveform]
                ┌───────────────┬──────────────┬──────────────┐
                ▼               ▼              ▼              │
         Spectrogram?     Play it back?   Transcribe first?   │
         Can you spot     (takes real     (then you're just   │
         the bug?         time — can't    debugging text      │
         Didn't think so. skim audio)     again, not audio)   │
                                                              │
```

| | Text | Audio |
|---|---|---|
| Training data | Scrape the internet — trillions of tokens | Must record, transcribe, align |
| Annotation | Read and label — fast | Listen and label — real-time minimum |
| Error taxonomy | Wrong word, wrong fact — discrete | Wrong pitch? emphasis? timing? — continuous, subjective |
| Debug speed | Read output → 2 seconds | Play clip → full duration minimum |

Whisper (OpenAI, 2022) was a breakthrough partly because it collapses audio back into text as fast as possible — debug in your native high-bandwidth modality (reading) instead of low-bandwidth (listening).

---

## The three career phases

```
Phase 1: Applied Math (2009-2015)     Phase 2: Open Source (2019-2021)     Phase 3: Forecasting (2021-2025)

Geothermal heat transfer              TorchAudio                          Distribution-free forecasting
  → Acoustic waves                      (200+ contributors)                 → Graph augmentation (GEANN)
    → General relativity               PyTorch optimizers                     → Peak decomposition (SPADE)
                                        (OneCycleLR)                            → Active sampling (Goldilocks)

"Prove it's correct"                  "Build it for everyone"              "Make it work at scale"
```

**Common thread across all phases:** Using mathematical structure to solve practical problems — the topological structure of differential forms for general relativity, the algebraic structure of PyTorch's autograd for audio processing, the graph structure of product catalogs for cold-start forecasting.

---

## Takeaway

- PhD simulated black holes using finite element methods; LIGO detected gravitational waves 3 months after his defense
- Created torchaudio (PyTorch's audio library), coordinated 200+ contributors at Meta
- Torchaudio is infrastructure, not an Audio Language Model — it provides the building blocks (spectrograms, pre-trained speech models) that audio LLMs are built on
- His 2025 Goldilocks paper uses multi-armed bandits for training sample selection — direct connection to RL exploration vs exploitation
- Career arc: pure math → open source framework → applied ML at Amazon

---

*See also:* [../29_pytorch_production_deployment.md](../29_pytorch_production_deployment.md) · [alexandr_wang.md](alexandr_wang.md)
