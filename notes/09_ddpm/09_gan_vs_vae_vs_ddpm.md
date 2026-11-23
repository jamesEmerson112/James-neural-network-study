# 09. GAN vs VAE vs DDPM

## Why this note exists

DDPM is one of three major families of deep generative models. The other two are **GANs** (Generative Adversarial Networks, Goodfellow 2014) and **VAEs** (Variational Autoencoders, Kingma & Welling 2014). All three solve the same problem — *given a dataset, learn to sample new data that looks like it* — but they solve it in three wildly different ways, with three different training procedures, three different failure modes, and three different strengths.

This note places DDPM among its peers. It mirrors [07_markov_vs_rnn_lstm_transformer.md](07_markov_vs_rnn_lstm_transformer.md): three architectures, three defining ideas, three mental pictures, one comparison table.

---

## The shared question

All three families try to answer:

> **"Given samples from an unknown distribution $p_{\text{data}}(x)$, build a model you can sample from that matches it."**

For images, $p_{\text{data}}$ is "the distribution of real photos" or "the distribution of MNIST digits." You never see the full distribution — just a training set of draws from it. The goal is to train a model $p_\theta$ whose samples are indistinguishable from real draws.

The three families disagree on:
- **What the model is** (implicit generator, encoder+decoder, denoising chain)
- **How to train it** (adversarial game, variational bound, regression on noise)
- **How to sample from it** (one forward pass, one forward pass, $T$ forward passes)

---

## 1. GAN — the adversarial game

**Core idea**: train two networks against each other. A **generator** $G$ maps random noise $z$ to a fake image $G(z)$. A **discriminator** $D$ tries to tell real images from generator output. The generator's goal is to fool the discriminator; the discriminator's goal is to catch the generator. At equilibrium, the generator has learned to produce samples the discriminator can't distinguish from real.

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] \;+\; \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

```
  GAN — two-player game

        z (noise)               x (real image)
         │                       │
         ▼                       │
       ┌───┐                     │
       │ G │──► G(z) ──┐         │
       └───┘           │         │
                       ▼         ▼
                      ┌─────────────┐
                      │      D      │──► "real or fake?"
                      └─────────────┘

  G learns: fool D
  D learns: catch G
  training = minimax equilibrium
```

**Sampling**: one forward pass through $G$ — fast. Feed random noise in, image comes out.

**Strengths**: sharp, high-fidelity samples. Fast inference. No likelihood computation needed.

**Weaknesses**:
- **Training instability** — the minimax game can oscillate, collapse, or fail to converge. Infamous in practice.
- **Mode collapse** — generator learns to produce only a few kinds of samples that reliably fool the discriminator, ignoring the rest of the data distribution.
- **No explicit likelihood** — you can't ask "how probable is this image under the model?" The density $p_\theta(x)$ is implicit.
- **No latent structure** — the mapping from noise $z$ to image is uninterpretable; you can't easily edit an image by manipulating its latent code.

**Mental picture**: a counterfeiter and a detective. The counterfeiter prints fake bills; the detective examines them. Each round, the counterfeiter gets better at mimicking real bills, and the detective gets better at spotting fakes. Equilibrium is reached when the counterfeiter's bills are indistinguishable from real ones.

---

## 2. VAE — encoder + decoder, trained with a variational bound

**Core idea**: learn a latent space. An **encoder** $q_\phi(z \mid x)$ maps an input image to a distribution over latent codes. A **decoder** $p_\theta(x \mid z)$ maps a latent code back to an image. The training objective is the **Evidence Lower Bound (ELBO)** — a tractable lower bound on the log-likelihood of the data:

$$
\mathcal{L}_{\text{ELBO}}(x) = \mathbb{E}_{q_\phi(z \mid x)}\!\left[\log p_\theta(x \mid z)\right] \;-\; D_{\text{KL}}\!\left(q_\phi(z \mid x) \,\|\, p(z)\right)
$$

The first term is a reconstruction loss (decoder should reproduce $x$ from the encoded $z$). The second term is a KL divergence pulling the encoder's output toward a standard Gaussian prior $p(z) = \mathcal{N}(0, \mathbf{I})$. Together they force the latent space to be both informative and well-structured.

```
  VAE — encoder compresses, decoder reconstructs

         x                          x̂ (reconstruction)
         │                          ▲
         ▼                          │
       ┌─────┐                    ┌─────┐
       │ enc │──► q(z│x) ──► z ──►│ dec │
       └─────┘     │               └─────┘
                   │
                   └──── KL ──── N(0, I)  prior

  reconstruction loss pulls x̂ toward x
  KL loss pulls q(z│x) toward N(0, I)
```

**Sampling**: draw $z \sim \mathcal{N}(0, \mathbf{I})$, pass through decoder — one forward pass. Fast.

**Strengths**:
- **Explicit likelihood bound** — you can compute and optimize the ELBO.
- **Structured latent space** — the Gaussian prior forces latents into a well-behaved space. You can interpolate, arithmetic on latents, etc.
- **Stable training** — just minimize ELBO, no adversarial game.

**Weaknesses**:
- **Blurry samples** — the reconstruction loss (typically Gaussian/MSE or Bernoulli) averages over modes, producing characteristically soft outputs compared to GANs.
- **The ELBO is a lower bound, not the actual likelihood** — there's a gap, and the model is not directly maximizing what you care about.
- **Posterior collapse** — in some settings the encoder ignores $x$ and outputs the prior, making $z$ uninformative.

**Mental picture**: a compression algorithm. The encoder is the zip, the decoder is the unzip. You train them together so that every image round-trips cleanly, *and* so that the compressed codes cluster in a well-structured space (a big Gaussian ball). Once trained, you generate by picking a random point in the ball and running the unzip.

---

## 3. DDPM — learn to reverse a fixed noising process

**Core idea**: define a fixed forward process that slowly corrupts images into pure Gaussian noise over $T$ steps (see [03_forward_process.md](03_forward_process.md)). Train a network to reverse one step of this process. At inference, start from pure noise and iteratively apply the learned reverse step $T$ times.

Training loss is plain MSE between true and predicted noise:

$$
\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{x_0,\, t,\, \varepsilon}\!\left[\, \| \varepsilon - \varepsilon_\theta(x_t, t) \|^2 \,\right]
$$

```
  DDPM — denoise through T small steps

  x_T    x_{T−1}  x_{T−2}   ...    x_1    x_0
  ██░░   ██░░     █▓░       ...    █▓     █
  ██░░   █▓░░     █▓░              █▓     █
  ░▓░░   ░▓░░     ░▓░              ░░     ·

   ←── network called T times, each a tiny denoise step ──
   
  no encoder, no discriminator — just a noise predictor
  trained with plain MSE on "what noise did I add?"
```

**Sampling**: $T$ forward passes through the network (hundreds to thousands). Slow.

**Strengths**:
- **Training is trivial** — plain regression on noise. No adversarial game, no ELBO weighting, no posterior collapse.
- **Mode coverage** — because the training objective touches every noise level, the model learns the full distribution. Essentially no mode collapse.
- **State-of-the-art sample quality** — sharper than VAE, more diverse than GAN. This is what powers Stable Diffusion, DALL-E 2, Imagen, Midjourney.
- **Principled**: the reverse process has a clean probabilistic interpretation via the posterior $q(x_{t-1} \mid x_t, x_0)$ (see [06_reverse_process.md](06_reverse_process.md)).

**Weaknesses**:
- **Slow sampling** — $T$ network calls per image. DDIM, consistency models, latent diffusion, and distillation methods have been invented specifically to fix this.
- **No latent space by default** — unlike VAE, DDPM doesn't give you a structured $z$ to manipulate. (Latent Diffusion Models fix this by running DDPM in VAE latent space.)
- **High compute** — training is expensive because the network has to handle every noise level from clean to pure noise.

**Mental picture**: a sculptor chipping away at a block of marble. The block is pure noise; the sculpture is the final image. Each chisel stroke (each reverse step) removes a tiny bit of noise, guided by the network's learned sense of "what this should look like." After enough strokes, a clean image emerges from what was originally random.

---

## Comparison table

| Property | GAN | VAE | DDPM |
|---|---|---|---|
| Year | 2014 | 2014 | 2020 |
| Core idea | Two-player game | Encoder + decoder with variational bound | Reverse a fixed noising process |
| Training loss | Minimax (adversarial) | ELBO (reconstruction + KL) | MSE on predicted noise |
| Number of networks | 2 ($G$ + $D$) | 2 ($q_\phi$ + $p_\theta$) | 1 ($\varepsilon_\theta$) |
| Sampling speed | 1 forward pass (fast) | 1 forward pass (fast) | $T$ forward passes (slow) |
| Training stability | Notoriously unstable | Stable | Very stable |
| Sample sharpness | Very sharp | Often blurry | Very sharp |
| Mode coverage | Prone to collapse | Good | Excellent |
| Explicit likelihood | No | Lower bound (ELBO) | Tractable (via the chain) |
| Latent space structure | None | Gaussian-structured | None (unless latent diffusion) |
| Key failure mode | Mode collapse, instability | Blurry output, posterior collapse | Slow sampling |
| Signature applications | StyleGAN, BigGAN, NVIDIA face synthesis | Disentangled representations, molecular design | Stable Diffusion, DALL-E 2, Imagen, Midjourney |

---

## Where each lives in the landscape

Think of the three families as three points on a **trilemma triangle**: training stability, sample quality, sampling speed. Historically you could pick two out of three.

```
                      sample quality
                           ▲
                           │
                         DDPM
                        /     \
                       /       \
                      /         \
                     /           \
                    /             \
                  VAE ───────────── GAN
                training          sampling
                stability         speed
```

- **GAN**: sharp samples, fast sampling — but unstable training.
- **VAE**: stable training, fast sampling — but blurry samples.
- **DDPM**: stable training, sharp samples — but slow sampling.

Post-2022 research (DDIM, consistency models, latent diffusion, flow matching) has been closing the speed gap on DDPM so effectively that the trilemma is essentially solved in DDPM's favor. This is why the dominant 2023–2026 generative model architectures — Stable Diffusion, DALL-E 3, Midjourney, Imagen, Sora — are all diffusion-based rather than GAN- or VAE-based.

---

## Which inspired which

The three families are not independent — they share machinery and ideas:

- **VAE → DDPM**: DDPM inherits the **reparameterization trick** from VAE (Kingma & Welling 2014). Every time DDPM writes $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon$, it is doing exactly what VAE does: sample a standard Gaussian, then deterministically transform it. See [04_visualizing_one_noising_step.md](04_visualizing_one_noising_step.md). Also, DDPM can be derived as a hierarchical VAE with $T$ latent layers and a fixed (not learned) encoder — the forward process is the encoder, the reverse process is the decoder.

- **VAE → DDPM (author lineage)**: **Diederik Kingma**, co-author of the VAE paper, later co-authored *Variational Diffusion Models* (2021) with Tim Salimans, Ben Poole, and Jonathan Ho — explicitly framing diffusion as a hierarchical VAE and showing that DDPM, VAE, and score-matching are all views of the same underlying object. The conceptual loop closed.

- **Physics → DDPM**: the "diffusion" framing itself comes from **Sohl-Dickstein et al., 2015** (*Deep Unsupervised Learning using Nonequilibrium Thermodynamics*), which borrowed the idea from statistical physics — specifically the thermodynamics of diffusion processes. **Surya Ganguli's group at Stanford** was bringing physics intuition into ML at the time, which is why the paper was born there and not in a traditional ML lab. Ho, Jain, Abbeel's 2020 paper simplified and scaled this up to become practical for image generation.

- **GANs ↔ DDPMs**: mostly independent lineages. GANs never borrowed heavily from VAEs or diffusion, and diffusion models barely borrowed from GANs. The two are different philosophies — implicit adversarial learning vs explicit probabilistic modeling.

---

## History/lore

- **2014 — Ian Goodfellow** publishes *Generative Adversarial Networks* at NeurIPS. Goodfellow was a PhD student in **Yoshua Bengio's lab at Université de Montréal**. The story: he conceived the idea during a beer-fueled argument with labmates at **Les Trois Brasseurs** (a brewpub in downtown Montreal) about how to train a generative model, prototyped it that night, and had it working by morning. GANs dominated generative modeling from 2014–2020.
- **2014 — Diederik Kingma & Max Welling** publish *Auto-Encoding Variational Bayes* at ICLR, introducing VAEs and the reparameterization trick in the same paper. Kingma's PhD thesis at the University of Amsterdam was built around this work.
- **2015 — Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, Surya Ganguli** publish *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* at ICML, introducing the diffusion-based generative model framework. The paper was undercited for five years — the ML community didn't understand how big a deal it was until 2020.
- **2020 — Jonathan Ho, Ajay Jain, Pieter Abbeel** publish *Denoising Diffusion Probabilistic Models* at NeurIPS. This is the paper that made diffusion practical: simplified Sohl-Dickstein's framework, showed how to train with the simple MSE loss, and produced image quality competitive with GANs for the first time. The paper is what turned diffusion models from an academic curiosity into the dominant generative architecture.
- **2021 — Prafulla Dhariwal & Alex Nichol** (*Diffusion Models Beat GANs on Image Synthesis*, OpenAI, NeurIPS 2021) show diffusion models beating BigGAN on ImageNet at comparable compute budgets. This was the moment the ML community collectively stopped betting on GANs and started betting on diffusion. **Dhariwal later co-founded Anthropic**; **Nichol became a key contributor on OpenAI's DALL-E series**. Every image-generation startup founded after 2021 has been diffusion-based, not GAN-based.
- **2022 — Latent Diffusion / Stable Diffusion** (Rombach et al., CompVis / Runway / Stability AI) makes diffusion cheap enough to run on consumer hardware by doing it in VAE latent space rather than pixel space. This is the breakthrough that put image generation in every developer's hands.
- **2023–2026** — Diffusion models become the dominant generative architecture for images (Stable Diffusion, DALL-E 3, Midjourney, Imagen), video (Sora, Veo, Runway Gen-3), audio (AudioLDM, Stable Audio), and increasingly language (diffusion LMs — still experimental). GANs are essentially retired for high-stakes generative applications except in niches like real-time face synthesis.

The irony of timing: diffusion models were invented (2015) before transformers (2017), but took five years longer to hit the mainstream. When they finally did, they took over completely.

---

## Takeaway

- **GAN**: adversarial two-player game, sharp + fast, unstable training, prone to mode collapse.
- **VAE**: encoder + decoder with ELBO, stable + structured latent, characteristically blurry.
- **DDPM**: denoise a fixed Markov chain, stable + sharp + mode-covering, slow sampling.
- All three share the same goal ("sample from $p_{\text{data}}$") and all three use Gaussians somewhere in the pipeline, but they assemble those pieces in three incompatible ways.
- DDPM inherits the **reparameterization trick** from VAE and can be framed as a hierarchical VAE with fixed encoder. GANs are the outsider family, mostly independent.
- As of 2026, diffusion has won the sample-quality trilemma. Speed has been patched by DDIM, consistency models, and latent diffusion. The dominant image, video, and audio generative models are all diffusion-based.

When you see a new generative-model paper, the first question to ask is: *which of these three families is it in, or is it a hybrid?* Almost every paper since 2020 either extends one of the three or combines them (e.g., VAE latent space + DDPM reverse process = Latent Diffusion).
