# GELU, SiLU and Recent LLM Activations

> Companion note to `02_gelu_silu_and_recent_activations_video.mp4`, the second video in this series. See [the overview](00_overview.md) for video 1.

The video is about five minutes long at 1080p30 and has two parts. Part 1 looks again at GELU and SiLU, the two smooth gates that video 1 introduced, and shows how the Swish-$\beta$ family connects them to each other and to ReLU. Part 2 covers activation functions that released LLMs adopted between 2024 and 2026. Every number in the video is computed by the script itself, and every adoption claim comes from a paper, a model card or a model's `config.json`.

Render it from this folder (Manim Community; no LaTeX needed):

```
manim --resolution 1920,1080 --frame_rate 30 02_gelu_silu_and_recent_activations_video.py RecentActivations
```

## Part 1: GELU and SiLU

### GELU is the average of a random gate

Hendrycks and Gimpel (2016) started from a stochastic regularizer, a source of training noise like dropout. Multiply the input $x$ by a mask $m \sim \mathrm{Bernoulli}(\Phi(x))$, where $\Phi$ is the standard Gaussian CDF. The mask keeps $x$ with probability $\Phi(x)$ and zeroes it otherwise. The expected output is

$$\mathbb{E}[m\,x] = \Phi(x)\cdot x + \big(1-\Phi(x)\big)\cdot 0 = x\,\Phi(x),$$

and that deterministic curve is GELU. At $x = 0.5$ the mask keeps the input about $69\%$ of the time, so $\mathrm{GELU}(0.5) \approx 0.35$.

### Two approximations

The same paper gives two cheaper forms:

$$\mathrm{GELU}(x) \approx 0.5\,x\left(1+\tanh\!\left(\sqrt{2/\pi}\,\big(x+0.044715\,x^3\big)\right)\right) \qquad\text{and}\qquad \mathrm{GELU}(x) \approx x\,\sigma(1.702\,x).$$

On $[-6, 6]$ the script finds that the tanh form is never more than $0.00047$ from the exact GELU, while the sigmoid form is off by up to $0.0203$. GPT-2's released code writes the tanh form out in full, and its Hugging Face config names it `gelu_new`.

### The Swish-$\beta$ family

Ramachandran, Zoph and Le (2017) wrote Swish as $x\,\sigma(\beta x)$. One knob, $\beta$, connects several curves:

| $\beta$ | Curve |
|---|---|
| $0$ | the straight line $x/2$ |
| $1$ | SiLU, $x\,\sigma(x)$ |
| $1.702$ | GELU's sigmoid approximation |
| $\beta \to \infty$ | ReLU, because $\sigma(\beta x)$ becomes a step |

Every member is a rescaled SiLU, $x\,\sigma(\beta x) = \mathrm{SiLU}(\beta x)/\beta$. So the dip of any member sits at $x = -1.2785/\beta$ with depth $-0.2785/\beta$, and it shrinks toward ReLU's corner as $\beta$ grows.

### Neither curve is monotonic

Both curves dip below zero and then come back toward it as $x \to -\infty$. The script finds the minima numerically by golden-section search:

| Function | Minimum value | At $x$ |
|---|---|---|
| GELU | $-0.1700$ | $-0.7518$ |
| SiLU | $-0.2785$ | $-1.2785$ |

Left of each minimum the slope is negative, so a more negative input gives a larger output. For example $\mathrm{SiLU}(-3) \approx -0.142$, which is larger than $\mathrm{SiLU}(-1.28) \approx -0.278$. ReLU, sigmoid and tanh are all monotonic, so they have no such dip.

## Part 2: activations in recent LLMs, 2024 to 2026

| Function | Formula | Where it sits | Introduced | Adopted in released LLMs |
|---|---|---|---|---|
| Squared ReLU | $\max(0,x)^2$ | feed-forward block | So et al., Primer (2021), found by searching over transformer designs | Nemotron-4 15B (2024); Nemotron-H and Arcee AFM-4.5B (2025, `relu2` in `config.json`); Nemotron 3 Nano (2025, in its MoE experts) |
| Clamped SwiGLU | $\min(g,7)\,\sigma\big(1.702\min(g,7)\big)\cdot\big(\mathrm{clamp}(u,-7,7)+1\big)$ | feed-forward block (MoE experts) | OpenAI, gpt-oss (August 2025) | gpt-oss-20b and gpt-oss-120b (2025); DeepSeek-V4.1-Flash also sets `swiglu_limit: 10.0` (2026) |
| xIELU | see the formula below | feed-forward block | Huang and Schlag (November 2024) | Swiss AI's Apertus (September 2025, `hidden_act: "xielu"`) |
| Sigmoid gate on attention | $Y' = Y \odot \sigma(XW_\theta)$ | attention output, not the feed-forward block | Qiu et al., Qwen team (May 2025) | Qwen3-Next (September 2025, model card lists Gated Attention layers); Qwen3.5 (February 2026) and Qwen3.8 (August 2026), both with `attn_output_gate: true` in `config.json` |

### Squared ReLU

Squared ReLU keeps ReLU's zero for negative inputs and squares the positive side, so its slope is $2x$ and grows with the input. So et al. found it while searching for more efficient transformer language models. The Primer abstract credits two changes for most of Primer's gains: squaring the ReLU and adding a depthwise convolution after the query, key and value projections. NVIDIA picked it up three years later for Nemotron-4 and kept it in the hybrid Mamba-transformer Nemotron-H and Nemotron 3 Nano models.

### gpt-oss's clamped SwiGLU

OpenAI's reference implementation splits the expert's projection into a gate input $g$ and a linear input $u$, then computes

$$g \leftarrow \min(g, 7), \qquad u \leftarrow \mathrm{clamp}(u, -7, 7), \qquad \mathrm{out} = g\,\sigma(1.702\,g)\cdot(u+1).$$

The gate $g\,\sigma(1.702\,g)$ uses the same constant as GELU's sigmoid approximation from 2016, so it traces a GELU-shaped curve rather than SiLU's. The model card does not say why that constant was chosen, and the video does not guess. The card calls the design "unconventional, including clamping and a residual connection". The reference code adds $1$ to the linear branch, with a comment calling it an extra bias of $1$. With both clamps, the script finds that every output of the gated product lies between $-42.0$ and $56.0$. DeepSeek-V4.1-Flash's `config.json` (2026) also sets a `swiglu_limit`, of $10.0$, but the exact form of its clamp is not checked here.

The sigmoid form of GELU did ship before gpt-oss. OpenAI's CLIP code (2021) defines `QuickGELU` as $x\,\sigma(1.702\,x)$, used as a plain activation rather than as the gate of a GLU. The video's summary caption gives all three dates: the sigmoid form dates from 2016, CLIP used it as QuickGELU in 2021, and gpt-oss put it inside a SwiGLU gate in 2025.

### xIELU

Huang and Schlag derived xIELU by choosing a trainable gradient and integrating it. The gradient grows linearly for positive inputs and can be negative for negative inputs. The Transformers implementation computes

$$\mathrm{xIELU}(x) = \begin{cases} \alpha_p\,x^2 + \beta\,x & x > 0 \\ \alpha_n\,(e^{x}-1) - \alpha_n\,x + \beta\,x & x \le 0 \end{cases}$$

with $\alpha_p = \mathrm{softplus}(\cdot) > 0$ and $\alpha_n = \beta + \mathrm{softplus}(\cdot) > \beta$ learned in each layer and $\beta = 0.5$ fixed. Both $\alpha$ parameters start at $0.8$. With those initial values the curve has a minimum of $-0.206$ at $x \approx -0.98$, and as $x \to -\infty$ its slope tends to $-(\alpha_n - \beta)$, so it climbs again. A trained model's curve will differ, because the $\alpha$ values move during training. The paper reports lower perplexity than $\mathrm{ReLU}^2$ and SwiGLU at matched compute and parameter count, in 1.1B and 3B Llama models trained on 125B tokens of FineWeb Edu.

### A sigmoid gate on the attention output

Qiu et al. report that a head-specific sigmoid gate applied to the output of scaled dot-product attention, $Y' = Y \odot \sigma(XW_\theta)$, consistently improves performance. Here $X$ is the hidden state after pre-normalization, the same input the attention reads. This is an element-wise sigmoid, but it sits in attention, not in the feed-forward block. Qwen3-Next's model card lists Gated Attention layers next to Gated DeltaNet layers, and the Transformers implementation of Qwen3-Next multiplies the attention output by `torch.sigmoid(gate)`. The Qwen3.5 and Qwen3.8 configs switch on `attn_output_gate`.

## Leads that did not make the video

These were checked and left out, because the sourcing rule needs a primary source for both the formula and an LLM adoption.

- Dynamic Tanh, $\mathrm{DyT}(x) = \tanh(\alpha x)$ (Zhu, Chen, He, LeCun and Liu, 2025), replaces normalization layers in transformers. No released LLM using it was found.
- PolyCom (Zhuo et al., 2025) and Smooth-SwiGLU (2024) were not verified within the research time, and no released LLM using them was found.
- The modded-nanogpt speedrun reportedly uses squared ReLU, but it is a training benchmark repository rather than a released model with a model card.
- Gemma's configs reportedly use `gelu_pytorch_tanh`, but the Gemma config files are gated behind a login and could not be read.

## Limits of the adoption claims

The adoption years in the video are the earliest released models found during a short search, not proven first uses. Nemotron-4 (2024) is the earliest released LLM found using squared ReLU, but an earlier one may exist. "Clamped SwiGLU, introduced 2025" means no released model with a SwiGLU clamp before gpt-oss was found.

## Sources

- GELU definition, the Bernoulli-mask motivation and both approximations: Hendrycks and Gimpel, "Gaussian Error Linear Units (GELUs)", https://arxiv.org/abs/1606.08415
- GPT-2's tanh-form `gelu` function: https://github.com/openai/gpt-2/blob/master/src/model.py
- GPT-2's `activation_function: "gelu_new"`: https://huggingface.co/openai-community/gpt2/blob/main/config.json
- Swish as $x\,\sigma(\beta x)$: Ramachandran, Zoph and Le, "Searching for Activation Functions", https://arxiv.org/abs/1710.05941
- Squared ReLU in Primer: So et al., "Primer: Searching for Efficient Transformers for Language Modeling", https://arxiv.org/abs/2109.08668
- Squared ReLU in Nemotron-4 15B's MLP layers: https://arxiv.org/abs/2402.16819
- `mlp_hidden_act: "relu2"` in Nemotron-H: https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/main/config.json
- Squared ReLU in Nemotron 3 Nano's MoE layers: https://arxiv.org/abs/2512.20848
- `hidden_act: "relu2"` in Arcee AFM-4.5B: https://huggingface.co/arcee-ai/AFM-4.5B/blob/main/config.json
- gpt-oss clamped SwiGLU code (alpha 1.702, limit 7, the $+1$): https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py
- gpt-oss model card, "unconventional, including clamping and a residual connection": https://arxiv.org/abs/2508.10925
- gpt-oss `swiglu_limit: 7.0`: https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json
- DeepSeek-V4.1-Flash `swiglu_limit: 10.0`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json
- xIELU: Huang and Schlag, "Deriving Activation Functions Using Integration", https://arxiv.org/abs/2411.13010
- xIELU implementation and initial values: https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
- Apertus `hidden_act: "xielu"`: https://huggingface.co/swiss-ai/Apertus-8B-2509/blob/main/config.json
- Apertus technical report (September 2025): https://arxiv.org/abs/2509.14233
- Gated attention: Qiu et al., "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free", https://arxiv.org/abs/2505.06708
- Qwen3-Next's Gated Attention layers: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Qwen3-Next's `attn_output * torch.sigmoid(gate)` in Transformers: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py
- Qwen3.5 `attn_output_gate: true`: https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json
- Qwen3.8 `attn_output_gate: true`: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
- `QuickGELU`, $x\,\sigma(1.702x)$, in OpenAI's CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
- Dynamic Tanh (left out of the video): Zhu et al., "Transformers without Normalization", https://arxiv.org/abs/2503.10622

Release months for models without a dated paper (Nemotron-H, AFM-4.5B, Qwen3-Next, Qwen3.5, Qwen3.8, DeepSeek-V4.1-Flash) come from the `createdAt` date of each Hugging Face repository.
