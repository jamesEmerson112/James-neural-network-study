# Generative Models — The Taxonomy of Learning to Create

> Part of the [Neural Network Study Timeline](00_timeline_neural_sequence_models.md). See also: [Softmax](22_softmax.md) (Boltzmann connection), [Transformers](19_transformers.md), [BERT](20_bert.md).

---

## The Big Question

**Discriminative models** answer: "Is this a cat or a dog?"
**Generative models** answer: "What does a cat look like?" — then generate one.

More formally: a generative model learns the **probability distribution** of the training data, p(x). Once it knows "what real data looks like" statistically, it can sample from that distribution to create new data.

The question that splits the entire field: **how do you model p(x)?**

---

## The Taxonomy (Goodfellow, NIPS 2016)

This tree — from Ian Goodfellow's famous 2016 tutorial — classifies every generative model by how it handles the probability distribution.

```
                            Generative Models
                           /                  \
                  Maximum Likelihood          Direct
                  (learn p(x))               (skip p(x) entirely)
                 /              \                    \
          Explicit              Implicit              GAN
          Density               Density
         (write down p(x))     (never write p(x))
        /           \                 \
   Tractable     Approximate      Markov Chain
   (compute       (can't compute      → GSN
    exactly)       exactly)
   |              /         \
   |         Variational   Markov Chain
   |            → VAE      → Boltzmann machine
   |
   ├── Fully visible belief nets
   ├── NADE
   ├── MADE
   ├── PixelRNN / PixelCNN
   └── Change of variables (normalizing flows)
```

Let's walk through each branch.

---

## Maximum Likelihood — "Find the Best Explanation"

Most generative models use **maximum likelihood**: find model parameters θ that maximize the probability of the training data.

```
θ* = argmax  Σ  log p(x_i ; θ)
       θ    i=1

In English: "Find the parameters that make the training data
             most probable under our model."
```

This is the same log-likelihood that shows up everywhere in statistics. The question is: what form does p(x) take?

---

## Explicit Density → Tractable

These models **write down an exact formula** for p(x) and can **compute it directly**. No approximations needed.

### Autoregressive Factorization (Fully Visible Belief Nets)

The key trick: use the chain rule of probability to decompose p(x) into a product of conditionals.

```
p(x) = p(x₁) · p(x₂|x₁) · p(x₃|x₁,x₂) · ... · p(x_n|x₁,...,x_{n-1})
```

Each term is a simple conditional distribution — predict one variable given all previous ones. This is **exactly** how autoregressive language models (GPT) work: predict the next token given all previous tokens.

For images, this means predicting one pixel at a time given all previous pixels.

### NADE (2011) — Neural Autoregressive Density Estimator

**Who**: Hugo Larochelle & Iain Murray

The first model to use a neural network to parameterize each conditional p(x_i | x₁,...,x_{i-1}). Each conditional is a small neural net that takes all previous variables as input.

**Problem**: you need a separate forward pass for each variable — slow for high-dimensional data.

### MADE (2015) — Masked Autoregressive Density Estimator

**Who**: Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle

Clever trick: use a **single** neural network but with **masked connections** so that each output only depends on previous inputs. The mask enforces the autoregressive property without needing separate networks.

```
Standard neural net:        MADE (masked):
  x₁ x₂ x₃ → [full] → y     x₁ x₂ x₃ → [masked] → y
  every output sees              y₂ can only see x₁
  every input                    y₃ can only see x₁, x₂
```

Much faster than NADE — one forward pass for all outputs.

### PixelRNN / PixelCNN (2016) — Autoregressive Image Generation

**Who**: Aäron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu (DeepMind)

Apply the autoregressive idea to images: generate one pixel at a time, left-to-right, top-to-bottom. Each pixel is conditioned on all previously generated pixels.

- **PixelRNN**: uses LSTMs to model dependencies — high quality but slow (sequential)
- **PixelCNN**: uses masked convolutions — faster (parallelizable) but slightly lower quality

```
Generating a 32×32 image:
  Pixel 1 → Pixel 2 → ... → Pixel 1024
  Each pixel conditioned on ALL previous pixels

That's 1024 sequential steps for a tiny image.
For a 256×256 image: 65,536 steps. Painfully slow.
```

This speed problem is what motivated alternative approaches like VAEs and GANs.

### Change of Variables / Normalizing Flows (2015)

**Who**: Danilo Rezende & Shakir Mohamed (DeepMind)

Start with a simple distribution (like a Gaussian), then apply a series of **invertible transformations** to warp it into the complex data distribution. Because the transformations are invertible, you can compute the exact probability using the change-of-variables formula from calculus.

```
z ~ N(0, 1)        ← simple starting distribution
x = f(z)           ← learned invertible transformation
p(x) = p(z) · |det(df/dx)|⁻¹   ← exact density via Jacobian
```

The constraint: every transformation must be invertible, which limits what the network can do.

---

## Explicit Density → Approximate

These models write down a formula for p(x) but **can't compute it exactly** — they need approximations.

### Variational → VAE (2013)

**Who**: Diederik Kingma & Max Welling (University of Amsterdam)

**Full name**: Variational Autoencoder

The idea: learn a **latent space** (compressed representation) and a decoder that maps from latent space back to data space.

```
Encoder: x → z  (compress data into latent code)
Decoder: z → x  (reconstruct data from latent code)

The "variational" part: the encoder doesn't output a single z,
it outputs a DISTRIBUTION over z (mean + variance).
You sample from that distribution during training.
```

**Why "variational"?** The true posterior p(z|x) is intractable (can't compute it). So we **approximate** it with a simpler distribution q(z|x) and optimize a lower bound called the **ELBO** (Evidence Lower Bound).

```
log p(x) ≥ ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
                   ──────────────    ──────────────────────
                   reconstruction     regularization
                   quality            (keep latent space smooth)
```

**Strengths**: fast generation (just sample z and decode), smooth latent space for interpolation
**Weakness**: outputs tend to be blurry (the model averages over uncertainty)

### Markov Chain → Boltzmann Machine (1985)

**Who**: Geoffrey Hinton & Terrence Sejnowski

An **energy-based model** — assigns an energy E(x) to every possible configuration, then uses the Boltzmann distribution to define probabilities:

```
p(x) = exp(-E(x)) / Z

where Z = Σ exp(-E(x'))   ← partition function (sum over ALL possible x)
          x'
```

This is **the same formula as softmax** (see [Softmax note](22_softmax.md)) — because softmax IS the Boltzmann distribution. The connection runs deep:

```
Boltzmann (1868) → statistical mechanics → Boltzmann distribution
     ↓
Hopfield (1982) → energy-based neural nets
     ↓
Hinton & Sejnowski (1985) → Boltzmann machine (uses Boltzmann distribution in a neural net)
     ↓
Bridle (1990) → calls it "softmax" for classifiers
     ↓
Vaswani (2017) → softmax in attention
```

**The problem**: computing Z requires summing over ALL possible configurations — exponentially many. So training uses **Markov Chain Monte Carlo (MCMC)** sampling to approximate the gradient, which is slow and finicky.

**Restricted Boltzmann Machines (RBMs)** simplified this by restricting connections (no hidden-to-hidden or visible-to-visible links), making sampling tractable. Hinton's **deep belief nets** (2006) stacked RBMs and helped trigger the deep learning revolution — the same revolution that eventually made Boltzmann machines obsolete.

---

## Implicit Density

These models learn to generate data **without ever writing down p(x)**.

### Markov Chain → GSN (2013)

**Who**: Yoshua Bengio et al. (University of Montreal)

**Full name**: Generative Stochastic Networks

The idea: learn a Markov chain transition operator that, when run for many steps, produces samples from the data distribution — without ever defining p(x) explicitly. Think of it as "learn how to walk around in data space so that where you end up looks like real data."

Historically important as a bridge between Boltzmann machines and GANs, but largely superseded.

---

## Direct — GANs (2014)

**Who**: Ian Goodfellow et al. (University of Montreal)

GANs take a radically different approach: **skip the density entirely**. Don't model p(x) at all — just learn to generate samples that are indistinguishable from real data.

### How It Works

Two networks compete:

```
Generator (G):     Takes random noise z → generates fake data G(z)
Discriminator (D): Takes data → outputs "real" or "fake"

Training loop:
  1. G generates fake samples
  2. D tries to tell real from fake
  3. G improves to fool D
  4. D improves to catch G
  5. Repeat until D can't tell the difference
```

This is a **minimax game**:

```
min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]

Discriminator wants to maximize: correctly classify real vs fake
Generator wants to minimize: make discriminator wrong
```

### The Bar Story

Goodfellow came up with GANs during a going-away party at a bar in Montreal in 2014. Friends were discussing generative models and their problems. Goodfellow suggested the adversarial idea, went home, coded it up that night, and **it worked on the first try**. He submitted the paper to NIPS 2014, and it became one of the most cited papers in deep learning history.

Yann LeCun called GANs "the most interesting idea in the last 10 years in ML."

### Why GANs Matter

| Strength | Why |
|---|---|
| Sharp, realistic outputs | No blurriness (unlike VAEs) — the discriminator penalizes any fuzziness |
| Fast generation | One forward pass through G (unlike autoregressive models that generate pixel by pixel) |
| No density estimation needed | Avoids the intractable partition function problem entirely |

| Weakness | Why |
|---|---|
| Mode collapse | G learns to produce only a few types of outputs instead of the full diversity |
| Training instability | The minimax game is notoriously hard to balance — G and D must improve at similar rates |
| No density evaluation | Can't compute p(x) — can't tell you how likely a given image is |

---

## Beyond the Slide — What Came After

Goodfellow's 2016 taxonomy predates some major developments:

### Diffusion Models (2020)

**Who**: Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)

**Paper**: "Denoising Diffusion Probabilistic Models" (DDPM)

The idea: **gradually add noise** to data until it becomes pure Gaussian noise, then learn to **reverse the process** step by step.

```
Forward process (fixed):
  Real image → slightly noisy → more noisy → ... → pure noise

Reverse process (learned):
  Pure noise → slightly less noisy → ... → generated image
```

Each denoising step is a small neural network prediction. Unlike GANs, training is stable (no adversarial game), and unlike VAEs, outputs are sharp.

**Where it fits in the taxonomy**: explicit density, tractable — the model learns the score function ∇log p(x), which can be used to compute densities.

### Stable Diffusion (2022)

**Who**: Robin Rombach et al. (CompVis group, Ludwig Maximilian University of Munich)

Key innovation: run diffusion in a **compressed latent space** (not pixel space), making it practical for high-resolution images. Open-sourced, kicked off the AI art explosion.

```
Previous diffusion:  noise ↔ pixel space (slow, 256×256 max)
Latent diffusion:    noise ↔ latent space → decode to pixels (fast, 512×512+)
```

### The Current Landscape (2024–2025)

```
Image generation:    Diffusion models dominate (Stable Diffusion, DALL-E 3, Midjourney)
Text generation:     Autoregressive transformers dominate (GPT, Claude, LLaMA)
Video generation:    Diffusion + transformers (Sora, Runway)
Audio generation:    Autoregressive + diffusion hybrids
```

GANs, once the king of image generation, have been largely displaced by diffusion models for quality — though GANs remain faster at inference time.

---

## Summary Table

| Model | Year | Density Type | Key Idea | Strength | Weakness |
|---|---|---|---|---|---|
| Boltzmann machine | 1985 | Explicit, approximate | Energy-based, Boltzmann distribution | Theoretically elegant | Slow MCMC training |
| NADE | 2011 | Explicit, tractable | Neural autoregressive conditionals | Exact likelihood | Slow (sequential) |
| VAE | 2013 | Explicit, approximate | Encoder-decoder with latent space | Fast, smooth latent space | Blurry outputs |
| GSN | 2013 | Implicit | Learned Markov chain | No density needed | Largely superseded |
| GAN | 2014 | Direct (none) | Generator vs discriminator game | Sharp outputs, fast | Unstable training, mode collapse |
| MADE | 2015 | Explicit, tractable | Masked autoencoder | Fast (single pass) | Limited expressiveness |
| Normalizing Flows | 2015 | Explicit, tractable | Invertible transformations | Exact likelihood + fast | Invertibility constraint |
| PixelRNN/CNN | 2016 | Explicit, tractable | Autoregressive pixel generation | High quality | Extremely slow |
| Diffusion (DDPM) | 2020 | Explicit, tractable | Gradual denoising | Sharp + stable training | Slow sampling (many steps) |

---

## The One-Sentence Summary

**Every generative model answers the same question — "what does real data look like?" — but they differ in whether they write down the probability formula (explicit), approximate it (variational/MCMC), or skip it entirely (GANs).**

---

## Historical Note

This taxonomy is from Ian Goodfellow's NIPS 2016 tutorial "Generative Adversarial Networks." Goodfellow — who invented GANs in 2014 at the University of Montreal under Yoshua Bengio — created this tree to situate GANs within the broader landscape of generative modeling. The tutorial remains one of the clearest high-level guides to the field.

The three Montreal professors — Bengio, Goodfellow (student), and their collaborators — are responsible for an extraordinary portion of this tree: GANs, GSN, and foundational work on VAEs and autoregressive models. Montreal in the 2010s was to generative models what Vienna in the 1920s was to physics.

---

## Cross-References

- [Softmax](22_softmax.md) — the Boltzmann distribution connection (softmax IS the Boltzmann distribution)
- [Scaled Dot-Product Attention](23_scaled_dot_product_attention.md) — attention mechanism in transformers
- [Transformers](19_transformers.md) — decoder-only transformers are autoregressive generative models
- [BERT](20_bert.md) — encoder-only, NOT generative (discriminative/understanding)
- [Backprop: The Wiggle Ratio](09_backprop_the_wiggle_ratio.md) — how all these models train
- [Master Timeline](00_timeline_neural_sequence_models.md) — where each model fits historically
