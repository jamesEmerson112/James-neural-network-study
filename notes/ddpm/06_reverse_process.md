# 06. The Reverse Process — Rebuilding the Image

The forward process destroys; the reverse process creates. This is where the neural network lives and where actual "generation" happens.

---

## The goal

We want a function that takes pure Gaussian noise $x_T \sim \mathcal{N}(0, \mathbf{I})$ and iteratively transforms it into a clean image $x_0$. If we can model the reverse transitions

$$
p(x_{t-1} \mid x_t)
$$

then we can start from noise and walk backwards: $x_T \to x_{T-1} \to \ldots \to x_1 \to x_0$.

---

## Why we can't just invert the forward process

The forward process is easy because we're always adding noise — Gaussian in, Gaussian out. But the reverse is genuinely hard.

Imagine you see a blurry, noisy image $x_t$. The question "what was $x_{t-1}$?" has infinitely many possible answers — *any* clean-ish image that could have been corrupted into $x_t$ is a valid candidate. The reverse transition $p(x_{t-1} \mid x_t)$ is a complicated, multi-modal distribution that depends on the full data distribution of images.

**The trick**: if we condition on the original image $x_0$ as well, the reverse transition becomes tractable:

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\ \tilde{\mu}_t(x_t, x_0),\ \tilde{\beta}_t \mathbf{I}\right)
$$

This is a Gaussian with a closed-form mean and variance (both derivable from the forward process). The problem is we don't know $x_0$ at sampling time — that's exactly what we're trying to generate. So we train a neural network to *estimate the thing we need* and use it as a stand-in.

---

## What the neural network actually predicts

Here's the beautiful simplification from the 2020 paper: **the network predicts the noise $\varepsilon$ that was added, not the image.**

Why noise instead of the image? Because during training, we know exactly what noise we added (we sampled it ourselves). So we have a perfect ground-truth label for every training step:

$$
\text{training\_loss} = \text{MSE}\!\left(\varepsilon_\theta(x_t, t),\ \varepsilon\right)
$$

Where:

- $\varepsilon$ is the actual noise we sampled and added to create $x_t$
- $\varepsilon_\theta(x_t, t)$ is the network's prediction ($\theta$ denotes the network's parameters)
- MSE means mean-squared error

This is just a regression problem. No GAN-style instability, no VAE-style ELBO hand-wringing. The network simply learns: *given a noisy image and a timestep, output the noise pattern*.

### Why this works mathematically

By rearranging the forward process equation $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\varepsilon$, we can express $x_0$ in terms of $x_t$ and $\varepsilon$:

$$
x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\,\varepsilon}{\sqrt{\bar{\alpha}_t}}
$$

So knowing $\varepsilon$ and $x_t$ is equivalent to knowing $x_0$. The neural network learning to predict $\varepsilon$ is, in a mathematical sense, learning to reconstruct $x_0$ — just parameterized in a way that turns out to be numerically well-behaved during training.

---

## The training algorithm (Algorithm 1 from the DDPM paper)

Simplified pseudocode:

```
repeat:
    x_0 = sample an image from the dataset
    t   = sample a timestep uniformly from {0, 1, ..., T-1}
    eps = sample noise ~ N(0, I)
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    loss = || eps - eps_theta(x_t, t) ||^2
    take a gradient step on loss
```

Three critical things to notice:

1. **Random $t$**: each training sample uses a different random timestep. This is why we need the closed-form $q(x_t \mid x_0)$ — we can jump to any $t$ in one operation.
2. **One network, all timesteps**: the same network handles every noise level. The timestep $t$ is passed in as an input (typically embedded via a sinusoidal position encoding, similar to transformers).
3. **Trivial loss function**: just MSE. No adversarial tricks, no KL weights to tune.

---

## The sampling algorithm (Algorithm 2 from the DDPM paper)

Simplified pseudocode:

```
x_T = sample from N(0, I)
for t = T-1 down to 0:
    z = sample from N(0, I)   if t > 0, else z = 0
    predicted_noise = eps_theta(x_t, t)
    mu    = (1 / sqrt(alpha_t)) * ( x_t - (beta_t / sqrt(1 - alpha_bar_t)) * predicted_noise )
    sigma = sqrt(beta_t)
    x_{t-1} = mu + sigma * z
return x_0
```

The mean formula, unpacked:

$$
\mu = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\varepsilon_\theta(x_t, t) \right)
$$

This is the posterior mean $q(x_{t-1} \mid x_t, x_0)$ where $x_0$ is replaced by the implicit estimate from the predicted noise. Step by step:

1. $\varepsilon_\theta(x_t, t)$ is the network's guess at what noise is in $x_t$.
2. $\dfrac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\varepsilon_\theta(x_t, t)$ is a scaled version of the noise to subtract.
3. $x_t - (\text{that scaled noise})$ is a denoised version of $x_t$.
4. Dividing by $\sqrt{\alpha_t}$ rescales it properly for the previous timestep.
5. The final $+\,\sigma \cdot z$ adds a little randomness back in, because the true reverse distribution is itself noisy.

### Why add noise back at every step except the last?

The reverse process is a probability distribution $p(x_{t-1} \mid x_t)$, and a distribution has spread. If we always returned the mean without any noise, we'd collapse the diversity of generated samples — all outputs would look similar. The stochastic term $\sigma \cdot z$ preserves the stochasticity that makes each generated image unique.

**Except at $t = 0$**: At the final step, we want the cleanest possible output, so we return just the mean ($z = 0$). Adding noise here would just corrupt our final image pointlessly.

---

## Putting it all together: the generation flow

```
  x_T      x_{T−1}    x_{T−2}    ...    x_1      x_0
  ██░░░░   ██░░░░     ██░░░      ...    █▓░      █
  ██░░░░   ██░░░░     ██▓░░             █▓       █
  ██░░░░   ██░▓░      █▓▓░              ██       █
  ██░░░░   █▓░░░      █▓░░              █▓       █
  ░░░░░░   ░▓░░░      ░░▓░              ░░       ·
  ░░░░░░   ░░░░░      ░░░░              ░░       ·
  pure     static     faint             digit    clean
  static   + hint     digit             + noise  digit

  ←──────  T network calls, each shrinking the noise  ──────→
```

Each reverse step calls the network, uses its noise prediction to compute a posterior mean, and adds a small stochastic kick (except at the last step). With $T = 10$ that's 10 network calls per generated image. The real DDPM paper runs $T = 1000$ — which is why diffusion sampling is considered "slow" compared to GANs or VAEs (one call each). DDIM, consistency models, and later variants cut this down dramatically.


---

## How this maps to code

A typical DDPM trainer implements:

- **`forward_diffusion(x_0, t)`** → $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\varepsilon$, returns $(x_t, \varepsilon)$. Math in [03_forward_process.md](03_forward_process.md).
- **Training loop** → sample random $t$, call `forward_diffusion`, feed $x_t$ to the UNet, MSE against $\varepsilon$, backprop.
- **`sample_timestep(x, t)`** → one reverse step with the mean formula above, add noise if $t > 0$.
- **`sample()`** → reverse loop from $T-1$ down to 0, returning $x_0$.

The network is typically a UNet that takes $(x_t, t)$ and outputs $\varepsilon_\theta(x_t, t)$. The math is almost line-for-line PyTorch.

For a symbol-to-code cheat sheet, see [08_glossary.md](08_glossary.md).

---

## Takeaway

- The forward process defines a target to learn: predict the noise given a corrupted image.
- The network is trained with plain MSE — no adversarial games, no KL weighting.
- Sampling is an iterative denoising loop that runs the network $T$ times.
- Each reverse step computes a mean (from the predicted noise) and adds small variance, except at the final step.
- The reverse process transforms pure noise into a clean image by a sequence of small corrections.

Next note: [07_markov_vs_rnn_lstm_transformer.md](07_markov_vs_rnn_lstm_transformer.md) — a meta comparison of how DDPM's Markov chain relates to (and differs from) RNNs, LSTMs, and Transformers.
