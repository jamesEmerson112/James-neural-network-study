# 09. GANs and VAEs — Training, Objectives, and the Reparameterization Trick

## What this note unpacks

Quiz topic 9 asks: "**How do GANs and VAEs work? What are their objectives, training procedures, and losses?**" Plus: "what is the reparameterization trick?"

This note is self-contained (no need to flip to other files during the quiz) and covers:
- **GAN**: minimax objective, generator vs discriminator, training loop, mode collapse, why the minimax game converges in theory
- **VAE**: the ELBO, encoder/decoder, reconstruction + KL losses, training loop
- **The reparameterization trick**: the trick VAEs need so gradients can flow through sampling — with a full derivation
- Comparison table and history

For a deeper comparison that also includes DDPM (the third family), see [../ddpm/09_gan_vs_vae_vs_ddpm.md](../ddpm/09_gan_vs_vae_vs_ddpm.md). That file is out of quiz 5 scope (DDPM isn't tested) but it's the canonical source.

---

## The shared goal

Both GANs and VAEs try to solve the same problem:

> Given a dataset of samples (e.g., images of faces), **train a model that can generate new samples that look like they came from the same distribution.**

The "same distribution" is $p_{\text{data}}(x)$ — the true distribution of real images. You don't know it explicitly; you only see samples from it. Both GANs and VAEs try to learn a model $p_\theta(x)$ and then draw from it.

They differ in how:
- **GAN**: learn implicitly via adversarial game
- **VAE**: learn explicitly via maximum likelihood (bounded by ELBO)

---

## GANs

### Core idea

Two networks train against each other:

- **Generator $G$**: takes random noise $z$ as input, outputs a fake image $G(z)$. Goal: fool the discriminator into thinking fakes are real.
- **Discriminator $D$**: takes an image (real or fake) as input, outputs a scalar probability that the image is real. Goal: correctly classify real vs fake.

Training is a **minimax game**: each network tries to win against the other. At equilibrium, the generator has learned to produce samples that are statistically indistinguishable from real data, and the discriminator is reduced to guessing.

```
  z (noise) ──► G ──► G(z) (fake)
                       │
                       └──┐
                          │
  x (real data) ────────► D ──► "real or fake?"
```

### The minimax objective

$$
\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\!\left[\log D(x)\right] + \mathbb{E}_{z \sim p_z}\!\left[\log(1 - D(G(z)))\right]
$$

**Reading the objective:**

- **$\mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)]$**: for real images, the discriminator wants to output $D(x) = 1$, so $\log D(x) = 0$ (maximum). For fake images that got labeled as real, this term approaches $-\infty$.
- **$\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$**: for generator output $G(z)$, the discriminator wants to output $D(G(z)) = 0$, so $\log(1 - D(G(z))) = 0$ (maximum). The generator wants the opposite: it wants $D(G(z)) = 1$, which makes $\log(1 - D(G(z)))$ go to $-\infty$ — which is *good for $G$* because $G$ is minimizing.
- $D$ **maximizes**: wants to assign 1 to reals and 0 to fakes.
- $G$ **minimizes**: wants $D$ to assign 1 to its fakes (i.e., wants the second term to go to $-\infty$).

### The training loop

```
algorithm: GAN training
  initialize G and D with random parameters
  loop forever:
    # --- Discriminator step ---
    sample mini-batch of real images {x_1, ..., x_m} from p_data
    sample mini-batch of noise {z_1, ..., z_m} from p_z
    compute fake images: g_i = G(z_i)
    
    # gradient ASCENT on D
    update D to maximize:  (1/m) Σ_i [log D(x_i) + log(1 - D(g_i))]
    
    # --- Generator step ---
    sample mini-batch of noise {z_1, ..., z_m} from p_z
    compute fake images: g_i = G(z_i)
    
    # gradient DESCENT on G
    update G to minimize:  (1/m) Σ_i log(1 - D(G(z_i)))
    
    # (in practice, often replace with maximizing log D(G(z_i)) — same fixed point, better gradients early in training)
```

**Practical detail**: the generator's loss $\log(1 - D(G(z)))$ has a vanishing gradient early in training when $D(G(z)) \approx 0$. Goodfellow's 2014 paper recommends the **non-saturating** alternative: maximize $\log D(G(z))$ instead. Same fixed point, much better gradients. This is what everyone uses in practice.

### Why it works (in theory)

**Claim** (from Goodfellow 2014): if $G$ and $D$ have enough capacity, and training converges to the global minimax equilibrium, then $p_G = p_{\text{data}}$ exactly.

**Proof sketch:**

1. For a fixed $G$, the optimal discriminator is $D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}$. (Take derivative of $V(D, G)$ with respect to $D(x)$ and set to 0.)

2. Substituting $D^*$ back into $V(D, G)$ and simplifying, you get:

$$
V(D^*, G) = -\log 4 + 2 \cdot D_{\text{JS}}(p_{\text{data}} \,\|\, p_G)
$$

where $D_{\text{JS}}$ is the **Jensen-Shannon divergence**.

3. $D_{\text{JS}}$ is zero iff $p_{\text{data}} = p_G$, so minimizing $V$ over $G$ minimizes the JS divergence, which is zero exactly when the two distributions match. At that point, $D^*(x) = 1/2$ everywhere — the discriminator can only guess randomly, because fakes look identical to real.

In practice, deep networks don't have infinite capacity, training doesn't reach the exact global minimum, and the game oscillates instead of converging smoothly. That's why GANs are notoriously unstable. But the theory says: **if it worked perfectly, the generator would produce samples from the true data distribution.**

### Failure modes

- **Mode collapse**: $G$ learns to produce a few outputs that always fool $D$, ignoring the rest of the data distribution. Generated samples look realistic individually but have no diversity.
- **Training instability**: the minimax game can oscillate, diverge, or collapse. Both networks need to improve at compatible rates; if $D$ gets too good too fast, $G$ has no useful gradient; if $G$ gets too good too fast, $D$ can't catch up.
- **No explicit likelihood**: you can't ask "how probable is this sample under the model?" GANs model the density implicitly via their samples, not explicitly.
- **Vanishing gradients**: when $D$ is too good, $\log(1 - D(G(z)))$ saturates and the generator stops learning. The non-saturating loss partially fixes this.

**Fixes:**
- **DCGAN** (2015): specific architectural rules (convolutions, batch norm) that made GANs stable for images.
- **WGAN** (2017): replace the discriminator's sigmoid output with a linear "critic" scoring Wasserstein distance. Much more stable, no more mode collapse.
- **Spectral normalization**, **self-attention**, **progressive growing** (StyleGAN): various architectural and regularization fixes.

---

## VAEs

### Core idea

Two networks: an encoder that maps data to a latent distribution, and a decoder that maps latents back to data. Train by maximizing the marginal log-likelihood of the data — or rather, a tractable lower bound on it, called the ELBO.

```
  x (input) ──► encoder q_φ(z|x) ──► z (sample from latent) ──► decoder p_θ(x|z) ──► x̂ (reconstruction)
                        │
                        │
                        ▼
                  KL divergence to prior p(z) = N(0, I)
```

### The objective: ELBO

The marginal log-likelihood of a data point $x$ is:

$$
\log p_\theta(x) = \log \int p_\theta(x, z) \, dz = \log \int p_\theta(x \mid z) p(z) \, dz
$$

This integral is intractable for a deep decoder (you can't integrate over all possible $z$). The VAE trick: introduce a **variational posterior** $q_\phi(z \mid x)$ (the encoder) that approximates the true posterior, and derive a lower bound:

$$
\log p_\theta(x) \geq \underbrace{\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]}_{\text{reconstruction term}} - \underbrace{D_{\text{KL}}\!\left(q_\phi(z \mid x) \,\|\, p(z)\right)}_{\text{KL term}} = \mathcal{L}_{\text{ELBO}}(x)
$$

**Derivation (one line via Jensen's inequality):**

$$
\log p_\theta(x) = \log \int p_\theta(x, z) \, dz = \log \int q_\phi(z \mid x) \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \, dz \;\geq\; \int q_\phi(z \mid x) \log \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \, dz
$$

The last step uses **Jensen's inequality** ($\log$ is concave, so $\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$). Expanding the right side and rearranging gives the ELBO formula above.

The ELBO is an inequality — it's a **lower bound** on $\log p_\theta(x)$. The gap between the ELBO and the true log-likelihood is exactly $D_{\text{KL}}(q_\phi(z \mid x) \| p_\theta(z \mid x))$ — how far the variational posterior is from the true posterior. As $q_\phi$ gets closer to the true posterior, the gap closes.

### Reading the ELBO

$$
\mathcal{L}_{\text{ELBO}}(x) = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{\text{KL}}(q_\phi(z \mid x) \| p(z))
$$

Two terms, each doing a specific job:

1. **Reconstruction term** $\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]$:
   - Encoder produces a distribution over $z$ given $x$
   - Sample a $z$
   - Decoder tries to reconstruct $x$ from $z$
   - The log-probability of the original $x$ under the decoder's output
   - **Maximizing** this term means the decoder should reproduce $x$ from the encoded $z$
   - In practice, for Gaussian decoder output, this becomes MSE; for Bernoulli, binary cross-entropy

2. **KL term** $-D_{\text{KL}}(q_\phi(z \mid x) \| p(z))$:
   - Penalizes the encoder's output distribution from straying too far from the prior $p(z) = \mathcal{N}(0, \mathbf{I})$
   - Without this term, the encoder could store $x$ directly in $z$ (no compression, no generalization)
   - With this term, the encoder is forced to use a **structured Gaussian-shaped latent space**, which enables sampling

**The training objective is to maximize ELBO**, or equivalently, **minimize negative ELBO**.

### The reparameterization trick — full derivation

**The problem.** The ELBO contains a sampling step:

$$
\mathbb{E}_{q_\phi(z \mid x)}\!\left[ \log p_\theta(x \mid z) \right]
$$

During training, you need to compute this expectation. The standard Monte Carlo estimate is: sample $z \sim q_\phi(z \mid x)$, compute $\log p_\theta(x \mid z)$, treat that as an unbiased estimate of the expectation.

But now you want the gradient of the expectation with respect to $\phi$ (the encoder parameters). You **can't** just take the gradient of the sample, because the sampling step itself depends on $\phi$. Sampling is not differentiable — you can't backprop through it.

**The trick.** Rewrite the sampling step so the randomness is **external** to the model. Instead of:

$$
z \sim q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x),\ \sigma^2_\phi(x))
$$

(which has $\phi$ inside the sampling operation), do:

$$
\varepsilon \sim \mathcal{N}(0, \mathbf{I}) \qquad \text{and} \qquad z = \mu_\phi(x) + \sigma_\phi(x) \cdot \varepsilon
$$

**The sample $\varepsilon$ doesn't depend on $\phi$** — it's just a fresh standard normal. The dependence on $\phi$ is now in the **deterministic computation** $z = \mu_\phi(x) + \sigma_\phi(x) \cdot \varepsilon$, which is a simple differentiable function of the encoder outputs.

**Why this works:** if $\varepsilon \sim \mathcal{N}(0, 1)$, then $\mu + \sigma \cdot \varepsilon \sim \mathcal{N}(\mu, \sigma^2)$. So $z = \mu_\phi(x) + \sigma_\phi(x) \cdot \varepsilon$ has the correct distribution. You've exchanged "sample $z$ from a parameterized distribution" for "sample $\varepsilon$ from a fixed distribution, then deterministically transform."

**Why it matters:**

```
  Before reparameterization:                    After reparameterization:

  x ──► enc_φ ──► μ, σ                          x ──► enc_φ ──► μ, σ
                    │                                            │
                    ▼                                            ▼
            z ~ N(μ, σ²)  ✗ non-differentiable         z = μ + σ·ε   (ε sampled outside)
                    │                                            │
                    ▼                                            ▼
                  dec_θ                                         dec_θ
                    │                                            │
                    ▼                                            ▼
                  loss                                          loss

  Gradient w.r.t. φ: BROKEN at sampling         Gradient w.r.t. φ: flows through μ, σ
```

**The gradient flow:** with the reparameterization, the gradient of the reconstruction loss with respect to $\phi$ flows through $z = \mu_\phi(x) + \sigma_\phi(x) \cdot \varepsilon$ using standard chain rule — $\varepsilon$ is a constant (just a sampled number). Before the reparameterization, the gradient couldn't flow through the sampling operation because "sample from $\mathcal{N}(\mu, \sigma^2)$" isn't differentiable with respect to $\mu$ and $\sigma$.

**The identity that makes it all work:** for any differentiable function $f$,

$$
\nabla_\phi \mathbb{E}_{z \sim \mathcal{N}(\mu_\phi, \sigma^2_\phi)}[f(z)] = \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, 1)}\!\left[ \nabla_\phi f(\mu_\phi + \sigma_\phi \cdot \varepsilon) \right]
$$

The left side is intractable; the right side is estimable by sampling $\varepsilon$ and applying standard backprop.

This is Kingma and Welling's main contribution in the 2013 VAE paper, and it's the single idea that makes VAE training work. DDPM ([../ddpm/09_gan_vs_vae_vs_ddpm.md](../ddpm/09_gan_vs_vae_vs_ddpm.md)) inherits this trick wholesale: every time DDPM writes $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \varepsilon$, that's the reparameterization trick, with the "encoder" being a fixed noise schedule instead of a learned network.

### The KL term in closed form

When $q_\phi(z \mid x) = \mathcal{N}(\mu, \sigma^2)$ and $p(z) = \mathcal{N}(0, \mathbf{I})$, the KL divergence has a **closed-form expression** (no sampling needed):

$$
D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = \frac{1}{2} \sum_{j=1}^{d}\!\left( \sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2 \right)
$$

where $d$ is the latent dimension. This is the formula you see in every VAE implementation. The derivation involves integrating Gaussians, which is tedious but standard.

### The training loop

```
algorithm: VAE training
  initialize encoder q_φ and decoder p_θ with random parameters
  loop forever:
    sample mini-batch of real data {x_1, ..., x_m}
    for each x_i:
      # encoder produces (μ, σ)
      μ_i, σ_i ← q_φ(x_i)
      
      # reparameterize
      ε_i ← sample from N(0, I)
      z_i ← μ_i + σ_i · ε_i
      
      # decoder reconstructs
      x̂_i ← p_θ(z_i)
      
      # compute ELBO
      recon_loss_i ← −log p_θ(x_i | z_i)      # usually MSE or BCE
      kl_loss_i    ← KL(N(μ_i, σ_i²) ‖ N(0, I))   # closed form
      loss_i       ← recon_loss_i + kl_loss_i
    
    total_loss ← (1/m) Σ_i loss_i
    update (θ, φ) via gradient descent on total_loss
```

That's the entire VAE training loop. Just backprop on a sum of reconstruction and KL losses. Much simpler than GANs — no adversarial game, no alternating updates, no minimax. This is why VAE training is stable and GAN training is not.

### Failure modes

- **Blurry reconstructions**: with a Gaussian output decoder, the reconstruction loss is MSE, which averages over modes. Result: VAEs produce characteristically soft, blurry images — the mean of many possible reconstructions rather than a sharp sample.
- **Posterior collapse**: the encoder ignores $x$ and outputs the prior $\mathcal{N}(0, \mathbf{I})$. Happens when the decoder is powerful enough to reconstruct $x$ without useful latents. The KL term becomes zero (good for minimization!) but $z$ carries no information about $x$.
- **ELBO is a lower bound, not exact likelihood**: there's a gap between ELBO and $\log p_\theta(x)$, and you're not directly maximizing what you care about.

---

## GAN vs VAE comparison

| | GAN | VAE |
|---|---|---|
| Core idea | Two-player adversarial game | Encoder + decoder + variational bound |
| Loss | Minimax $V(D, G)$ | ELBO (reconstruction + KL) |
| Networks | Generator + Discriminator | Encoder + Decoder |
| Training | Alternating gradient updates | Joint gradient descent on ELBO |
| Sampling | $G(z)$ for $z \sim p_z$ | $p_\theta(x \mid z)$ for $z \sim \mathcal{N}(0, I)$ |
| Training stability | Notoriously unstable | Very stable |
| Sample quality | Sharp | Blurry (with Gaussian output) |
| Explicit likelihood | No | Lower bound (ELBO) |
| Latent space structure | None (implicit) | Gaussian-structured |
| Typical failure | Mode collapse | Posterior collapse, blur |
| Invented | 2014 (Goodfellow) | 2013/2014 (Kingma & Welling) |

**Pedagogical summary:**
- **VAE**: "compress and uncompress, plus force the compressed codes to be Gaussian" — easier training, blurrier output
- **GAN**: "two networks play a game where one makes fakes and the other catches them" — harder training, sharper output

---

## History/lore

- **2013 — Diederik Kingma & Max Welling** (University of Amsterdam) post *Auto-Encoding Variational Bayes* on arXiv in December; it appears at ICLR 2014. The paper introduces the VAE and the **reparameterization trick**. Kingma's PhD at Amsterdam was built around this work.
- **2014 — Ian Goodfellow** (Bengio's lab at Université de Montréal) publishes *Generative Adversarial Networks* at NeurIPS 2014. The origin story: during a beer-fueled argument with labmates at **Les Trois Brasseurs** (a Montreal brewpub), Goodfellow conceived the adversarial training idea, went home, prototyped it that night, and had it working by morning. GANs dominated generative modeling from 2014 to 2020.
- **2015 — DCGAN** (Radford, Metz, Chintala) is the first GAN architecture that reliably generates high-quality images. Alec Radford (then 24) later co-authored GPT-1.
- **2016 — Pixel-RNN, Pixel-CNN** (van den Oord et al.) introduce autoregressive image generation as a third alternative.
- **2017 — WGAN** (Arjovsky, Chintala, Bottou) replaces the Jensen-Shannon divergence with Wasserstein distance, fixing many of the stability issues of vanilla GANs. Arjovsky was a math PhD student at NYU Courant, and the mathematical rigor shows.
- **2018 — StyleGAN** (Karras, NVIDIA) produces photorealistic human faces that spawn thispersondoesnotexist.com. Public realization that AI can fabricate people who never existed; the deepfake era begins.
- **2020 — DDPM** (Ho, Jain, Abbeel) inherits the reparameterization trick from VAE and applies it to a sequence of noising steps, producing a third generative family that eventually overtakes GANs for image generation.
- **2021 — Kingma himself** returns to the diffusion lineage and co-authors *Variational Diffusion Models*, explicitly framing diffusion as a hierarchical VAE. The loop closes: the inventor of the reparameterization trick helps formalize how DDPM uses it.

**Direct lineage of the reparameterization trick**: Kingma & Welling 2013 (VAE) → Kingma et al. 2021 (VDM) → inherited by every modern diffusion model including Stable Diffusion and DALL-E 2.

---

## Takeaway

- **GAN objective**: $\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$
- **GAN training**: alternate discriminator and generator updates. Unstable, but produces sharp samples.
- **VAE objective (ELBO)**: $\mathcal{L}_{\text{ELBO}}(x) = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{\text{KL}}(q_\phi(z \mid x) \| p(z))$
- **VAE training**: gradient descent on the sum of reconstruction and KL losses. Stable, but produces blurry samples.
- **Reparameterization trick**: $z = \mu_\phi(x) + \sigma_\phi(x) \cdot \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, \mathbf{I})$. Pushes randomness outside the differentiable path so gradients can flow through $\mu_\phi$ and $\sigma_\phi$.
- **GAN failure modes**: mode collapse, training instability, no likelihood
- **VAE failure modes**: blurry samples, posterior collapse

Next note: [10_word2vec_deep_dive.md](10_word2vec_deep_dive.md) — the last quiz topic: Skip-gram, CBOW, and negative sampling.
