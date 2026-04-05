# 04. Visualizing One Noising Step

## Why this note exists

[03_forward_process.md](03_forward_process.md) throws this formula at you:

$$
x_t = \sqrt{1 - \beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \mathbf{I})
$$

This is the heart of the DDPM forward process. Until you can *see* what it does, the closed-form teleport equation on top of it will feel like black magic. This note slows down to one step and stares at it until it's obvious.

---

## Step 1: Read the formula out loud

Before any arithmetic, translate the formula into English:

> **"The new image $x_t$ equals a shrink-factor times the old image $x_{t-1}$, plus a noise-factor times a fresh Gaussian noise sample."**

That's it. Two terms being added. The first term is a scaled copy of the previous image; the second term is scaled random noise. Everything else is bookkeeping.

---

## Step 2: The two-knob interpretation

Think of $\beta_t$ as a single dial between 0 and 1 that controls how much noise gets mixed into the image at this step. From that one dial, two coefficients pop out:

$$
\begin{aligned}
\text{signal coefficient} &= \sqrt{1 - \beta_t} &\quad \leftarrow \text{the "keep the old image" knob} \\
\text{noise coefficient} &= \sqrt{\beta_t} &\quad \leftarrow \text{the "mix in random noise" knob}
\end{aligned}
$$

These two knobs always trade off. Square them and add:

$$
\left(\sqrt{1 - \beta_t}\right)^2 + \left(\sqrt{\beta_t}\right)^2 = (1 - \beta_t) + \beta_t = 1
$$

So the squared coefficients always sum to 1. In Gaussian-land, the squared coefficient is the variance contribution of each term. This means **the total variance stays normalized at 1** — the signal variance decreases, the noise variance increases, but their sum is conserved.

This is not an accident. The $\beta$ schedule was designed this way on purpose so the image magnitudes don't blow up or collapse as $t$ increases.

---

## Step 3: A tiny 1D walk-through

Let's strip everything down to a single number and run three steps by hand.

**Setup:**

- $x_0 = 1.0$ (our "image" is a single scalar pixel, starting at 1.0)
- $\beta_1 = 0.1$, $\beta_2 = 0.1$, $\beta_3 = 0.1$ (constant schedule for simplicity)
- $\varepsilon$ samples (drawn from $\mathcal{N}(0, 1)$): $\varepsilon_1 = 0.3$, $\varepsilon_2 = -1.2$, $\varepsilon_3 = 0.5$

**Precompute the knobs:**

- signal coefficient $= \sqrt{1 - 0.1} = \sqrt{0.9} \approx 0.9487$
- noise coefficient $= \sqrt{0.1} \approx 0.3162$

**Step 1: $x_0 \to x_1$**

$$
\begin{aligned}
x_1 &= 0.9487 \cdot 1.0 + 0.3162 \cdot 0.3 \\
    &= 0.9487 + 0.0949 \\
    &= 1.0436
\end{aligned}
$$

Reading: start at 1.0, shrink it slightly to 0.9487, then add a tiny noise kick of $+0.0949$. We end up slightly above where we started — the noise happened to push us up.

**Step 2: $x_1 \to x_2$**

$$
\begin{aligned}
x_2 &= 0.9487 \cdot 1.0436 + 0.3162 \cdot (-1.2) \\
    &= 0.9901 + (-0.3794) \\
    &= 0.6107
\end{aligned}
$$

Reading: shrink 1.0436 to 0.9901, then add a noise kick of $-0.3794$ (a big downward draw this time). We drop to 0.6107.

**Step 3: $x_2 \to x_3$**

$$
\begin{aligned}
x_3 &= 0.9487 \cdot 0.6107 + 0.3162 \cdot 0.5 \\
    &= 0.5794 + 0.1581 \\
    &= 0.7375
\end{aligned}
$$

Reading: shrink 0.6107 to 0.5794, add $+0.1581$, land at 0.7375.

After three steps, the "pixel" has drifted from 1.0 to 0.7375 and picked up a noisy wiggle along the way. Running this for hundreds of steps with growing $\beta$ would send it toward pure random noise centered at 0.

---

## Step 4: Extending to a 4-pixel image

Now imagine the "image" is a tiny 4-pixel row: $[1.0, 0.5, -0.2, 0.8]$. One noising step with $\beta = 0.1$:

**Setup:**

- signal coefficient $= 0.9487$
- noise coefficient $= 0.3162$
- Sampled noise $\varepsilon = [0.3, -0.1, 0.7, 0.2]$ (four independent draws from $\mathcal{N}(0, 1)$)

**Compute each pixel independently** (because $\varepsilon$ has identity covariance — each pixel's noise is independent of the others):

$$
\begin{aligned}
\text{pixel 0:}\quad &0.9487 \cdot 1.0   + 0.3162 \cdot 0.3  =  0.9487 + 0.0949 =  1.0436 \\
\text{pixel 1:}\quad &0.9487 \cdot 0.5   + 0.3162 \cdot (-0.1) =  0.4744 - 0.0316 =  0.4428 \\
\text{pixel 2:}\quad &0.9487 \cdot (-0.2) + 0.3162 \cdot 0.7  = -0.1897 + 0.2213 =  0.0316 \\
\text{pixel 3:}\quad &0.9487 \cdot 0.8   + 0.3162 \cdot 0.2  =  0.7590 + 0.0632 =  0.8222
\end{aligned}
$$

**Before → After:**

$$
\begin{aligned}
x_{t-1} &= [ 1.0000,\ 0.5000,\ -0.2000,\ 0.8000 ] \\
x_t     &= [ 1.0436,\ 0.4428,\  0.0316,\ 0.8222 ]
\end{aligned}
$$

Pixel 2 is the most visibly changed because its original magnitude ($-0.2$) was small relative to the injected noise ($0.7$). Pixels 0 and 3 barely moved because their original magnitudes were large. This is exactly the right intuition for why early steps (small $\beta$) barely touch strong features but gradually wash out weak ones.

Scale this reasoning up from 4 pixels to 784 (MNIST) or 65,536 (256×256) and you have the full picture: every pixel is updated independently via the same two-knob formula, with its own fresh noise sample.

---

## Step 5: Why the square roots?

This trips everyone up on first encounter. Short answer: **because $\beta_t$ is a variance, and to scale a Gaussian's standard deviation you multiply by $\sqrt{\text{variance}}$, not variance.**

Long version:

- We want the variance of the noise term to be $\beta_t$. That's the definition of the variance schedule.
- If $\varepsilon \sim \mathcal{N}(0, 1)$, then $\varepsilon$ has variance 1.
- When you multiply a random variable by a constant $c$, its variance scales by $c^2$. So $c \cdot \varepsilon$ has variance $c^2$.
- To get variance $\beta_t$, we need $c^2 = \beta_t$, which means $c = \sqrt{\beta_t}$.

Same argument for the signal term: we want the variance contribution of $c \cdot x_{t-1}$ to be $(1 - \beta_t)$, so $c = \sqrt{1 - \beta_t}$.

This is the **reparameterization trick** — the same idea Kingma and Welling introduced in the 2014 VAE paper. Instead of sampling from some parameterized distribution directly, you sample one *standard* Gaussian $\varepsilon$ and then deterministically transform it. The benefits:

1. **Differentiability**: gradients can flow through deterministic arithmetic, not through a random sampling operation.
2. **Implementation simplicity**: every framework gives you $\mathcal{N}(0, 1)$ samples; you don't need a specialized sampler for each variance.
3. **Numerical stability**: scaling by a pre-computed $\sqrt{\beta_t}$ is more stable than re-parameterizing a distribution per step.

DDPM uses this trick twice: once in each per-step formula (this note) and once in the closed-form teleport equation in [03_forward_process.md](03_forward_process.md).

---

## Step 6: The storyboard — how things evolve as $t$ grows

With a small constant $\beta$ (say, 0.02) the signal knob starts near 1 and shrinks slowly, while the noise knob starts near 0 and grows. Here's a qualitative storyboard:

```
t = 0   [digit clearly visible, zero noise]
         signal knob ≈ 1.0000,  noise knob = 0

t = 2   [digit clearly visible, faintest grain]
         signal knob ≈ 0.9900,  noise knob ≈ 0.1414

t = 5   [digit still recognizable, visible noise grain]
         signal knob ≈ 0.9750,  noise knob ≈ 0.2236

t = 9   [digit partly washed out, heavy noise]
         signal knob ≈ 0.9556,  noise knob ≈ 0.3000

t = T   [pure static, digit completely gone]
         signal knob → 0      ,  noise knob → 1
```

For the assignment config ($T = 10$, $\beta$ from 0.0001 to 0.02), we don't actually reach pure static — see the table in [05_numeric_example.md](05_numeric_example.md). But the *shape* of the storyboard is the same: start clean, end noisy, smooth interpolation in between.

For the original paper ($T = 1000$, same $\beta$ endpoints), the signal knob really does go to near-zero and the final $x_T$ is genuinely indistinguishable from a draw from $\mathcal{N}(0, \mathbf{I})$.

---

## Step 7: Connection to the closed form

[03_forward_process.md](03_forward_process.md) has a magic equation that lets you skip all $T$ steps and jump directly from $x_0$ to any $x_t$:

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\varepsilon
$$

Where does this come from? It's just the per-step formula applied $T$ times and simplified. The derivation is in [03_forward_process.md](03_forward_process.md) Part 6 — you iteratively substitute $x_{t-1}$, then $x_{t-2}$, and so on, using the fact that sums of independent Gaussians combine into a single Gaussian with added variances.

The key shift in perspective:

- **Per-step formula (this note)**: zoomed in on one transition. Uses $\beta_t$ and its one-step knobs.
- **Closed-form (note 03)**: zoomed out to the whole chain. Uses $\bar{\alpha}_t$ (cumulative signal retention) and its $T$-step knobs.

Both describe the same process. The per-step version is how you'd simulate the chain step-by-step (which you rarely need to do); the closed-form is how training actually generates $x_t$ for a random timestep in a single operation.

---

## Step 8: Resources to search for (no guessed URLs)

If you want to see animated visualizations of the forward and reverse processes, there are several well-known blog posts with built-in diagrams. I won't guess URLs — you can search these names directly:

- **Lilian Weng — "What are Diffusion Models?"** A canonical blog post from OpenAI's former head of applied research. Has step-by-step diagrams of the forward and reverse processes.
- **HuggingFace — "The Annotated Diffusion Model"** A tutorial-style walk-through of the DDPM paper with code and visualizations inline.
- **Yang Song — "Generative Modeling by Estimating Gradients of the Data Distribution"** Covers score-based models, which are the continuous-time cousin of DDPMs, with excellent animations.
- **Jay Alammar — "The Illustrated Stable Diffusion"** Targets Stable Diffusion specifically but includes beautifully illustrated explanations of the underlying diffusion process.
- **3Blue1Brown** General math and neural-network visualization videos on YouTube. No DDPM video at the moment, but the style is exactly what "watching the math" should look like.

Search those titles and you'll find the material.

---

## History/lore: the reparameterization trick

The reparameterization trick didn't originate with DDPM. It was popularized by **Diederik Kingma and Max Welling** in their 2014 paper "Auto-Encoding Variational Bayes" (the VAE paper). The problem they were solving: how do you train a model that has a sampling step inside it, so gradients can still flow?

Their insight: push the randomness outside the network. Instead of sampling $z \sim \mathcal{N}(\mu_\theta(x), \sigma^2_\theta(x))$ directly, write:

$$
z = \mu_\theta(x) + \sigma_\theta(x) \cdot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, 1)
$$

Now $\varepsilon$ is a *fixed* random input, and the sampling is replaced by deterministic arithmetic on neural-network outputs. Gradients can flow back through $\mu_\theta$ and $\sigma_\theta$ as if $\varepsilon$ were just another input.

DDPM inherits this trick wholesale. Every time you see $\sqrt{\text{something}} \cdot x + \sqrt{\text{other thing}} \cdot \varepsilon$ in a DDPM formula, it's the same reparameterization trick applied to a pre-designed variance schedule instead of a learned one. Kingma later became one of the authors of the diffusion-model lineage papers (Variational Diffusion Models, 2021), closing the conceptual loop between VAE and DDPM.

---

## Takeaway

- The per-step noising formula is just "shrink the old image a little, add a little fresh Gaussian noise."
- The two coefficients are $\sqrt{1 - \beta_t}$ (signal knob) and $\sqrt{\beta_t}$ (noise knob), and their squares sum to 1 so total variance stays normalized.
- Square roots show up because $\beta_t$ is a *variance*, and to scale a Gaussian's standard deviation you multiply by $\sqrt{\text{variance}}$.
- Every pixel is updated independently with its own noise sample, because the covariance is $\mathbf{I}$ (identity).
- The closed-form teleport in [03_forward_process.md](03_forward_process.md) is this per-step formula applied $T$ times and simplified.
- The whole machinery is a reparameterization trick from the 2014 VAE paper, reused for diffusion.

Run the 1D example by hand once (Step 3 above) and the formula will stop feeling abstract.
