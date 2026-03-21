# LLM Benchmarks — What They Actually Measure

## Emotional Intelligence

| Benchmark | What it measures | Link |
|---|---|---|
| **EQ-Bench 3** | Emotional intelligence in multi-turn roleplay — empathy, psychological insight, social dexterity | [eqbench.com](https://eqbench.com/) |
| **EmpathyBench** | AI emotional intelligence testing | [empathybench.com](https://www.empathybench.com/eq) |
| **HEART** | Emotional support across 5 dimensions — empathy, attunement, resonance | Academic (2026) |
| **EmotionQueen** | Key event recognition, implicit emotional recognition, intention recognition | Academic |

Key finding: bigger/newer models don't always score higher on EQ. GPT-4o beats GPT-5 on EQ-Bench. Kimi-K2 leads. Optimizing for reasoning/coding can degrade emotional nuance.

GPT-4.5 was marketed as "high EQ" by OpenAI, but no independent benchmark backs this claim. EQ-Bench 3 scored it lower than GPT-4o.

---

## Working Memory / Multi-Task / Conversational Quality

These capture the "feels good to talk to" factor that EQ benchmarks miss.

| Benchmark | What it measures |
|---|---|
| **ChatBot Arena (LMSYS)** | Human preference — users pick which model they prefer in blind A/B tests. Closest to "which model do I enjoy using?" [lmarena.ai](https://lmarena.ai) |
| **MT-Bench** | Multi-turn conversation quality — coherence, helpfulness, flow across turns |
| **WildBench** | Real-world conversation ability rated by humans |
| **IFEval** | Instruction following — can the model handle multiple simultaneous constraints in one prompt? |
| **LOFT** | Long-context faithfulness — can the model track multiple things at once? |

---

## The Gap Between Benchmarks and Experience

What feels like "high EQ" in daily use is often a mix of:
1. **Working memory** — holding multiple threads without dropping context
2. **Instruction following under complexity** — juggling parallel asks cleanly
3. **Conversational naturalness** — tone, pacing, knowing when to be brief
4. **Task switching** — moving between topics without losing coherence

No single benchmark captures all of this. ChatBot Arena (human preference) is the closest proxy.

---

## Standard Capability Benchmarks (for reference)

| Benchmark | Domain |
|---|---|
| **MMLU** | General knowledge across 57 subjects |
| **HumanEval / SWE-bench** | Code generation and software engineering |
| **MATH / GSM8K** | Mathematical reasoning |
| **Open LLM Leaderboard** | Aggregate ranking of open-source models [huggingface.co](https://huggingface.co/spaces/open-llm-leaderboard) |

---

## Closed-Source Model Degradation

OpenAI frequently updates model weights behind the same API name. The GPT-4.5 you use today may be a different model than last month — same name, different behavior. Open-source models don't have this problem: weights are frozen, what you download is what you get.
