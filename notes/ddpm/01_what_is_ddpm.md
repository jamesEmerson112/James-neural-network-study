# 01. What is DDPM? The Big Picture

## One-sentence summary
A **Denoising Diffusion Probabilistic Model (DDPM)** learns to generate images by training a neural network to *undo* a gradual noise-adding process, one small step at a time.

> **Study path**: If the $\mathcal{N}(\mu, \sigma^2)$ notation feels fuzzy, read [02_what_is_a_gaussian.md](02_what_is_a_gaussian.md) before [03_forward_process.md](03_forward_process.md). Everything from note 03 onward depends on being comfortable with Gaussians. Notes 04–06 are the core math. Note 07 is a meta comparison to RNNs/LSTMs/Transformers. Note 08 is a symbol cheat sheet.

---

## Who made it

- **Original seed idea**: Jascha Sohl-Dickstein et al., 2015, "Deep Unsupervised Learning using Nonequilibrium Thermodynamics". He borrowed directly from statistical physics: if you drop a drop of ink into water, it spreads out (diffuses) into a uniform fog. The math that describes this process is well understood. Sohl-Dickstein asked: what if we ran that process *backwards* to generate images?
- **The breakthrough paper**: Jonathan Ho, Ajay Jain, Pieter Abbeel at UC Berkeley, 2020, "Denoising Diffusion Probabilistic Models" (the paper we now call DDPM). They figured out the right training objective and architecture to make the 2015 idea actually work at a competitive level.
- **Why it matters**: Everything generative you see today — Stable Diffusion, DALL-E 2/3, Midjourney, Sora — traces its lineage back to this 2020 paper.

---

## The problem DDPM solves

Before DDPM, there were two main ways to generate images:

1. **GANs** — fast and sharp, but notoriously unstable to train, prone to "mode collapse" (generator only learns a few variations), and no direct measure of how well the model fits the data.
2. **VAEs** — stable to train and mathematically clean, but samples tend to look blurry because of the Gaussian reconstruction loss.

DDPM offered a third path: **stable training like a VAE, but sample quality that matches or beats GANs**. The trade-off is that sampling is slow (you have to run the network many times to produce one image), but the fidelity is worth it.

---

## The two-process idea

DDPM has two processes that work in opposite directions:

### 1. Forward process (fixed, no learning)
Start with a real image $x_0$. Gradually add a tiny bit of Gaussian noise at each step: $x_0 \to x_1 \to x_2 \to \ldots \to x_T$. After enough steps ($T$ is usually 1000 in the real paper, or 10 in this assignment), the image is essentially pure random noise — indistinguishable from static.

**Key point**: This process is completely predetermined. No neural network is involved. It's just math.

### 2. Reverse process (learned)
Train a neural network to look at a noisy image $x_t$ and predict what noise was added to get there. If it can predict the noise accurately, it can subtract it — producing a slightly less noisy image $x_{t-1}$. Repeat this $T$ times, and you turn pure random noise into a clean image.

**Key point**: This is where learning happens. The entire intelligence of the model lives in one thing: predicting the noise at each step.

---

## The ink-in-water analogy

Imagine dropping a single drop of black ink into a glass of clear water. Over time, the ink spreads out until the whole glass is a uniform gray fog. That's the **forward process** — easy, predictable, and irreversible in the real world.

Now imagine you had a tiny demon who could look at the gray fog and, at each moment, push each ink molecule back slightly toward where it came from. After enough of these tiny pushes, the ink would re-collect into a drop. That demon is the neural network. The "push back slightly" is one denoising step.

The reason this works is that each push is *small and local*. The demon doesn't need to solve the whole problem at once — it only needs to answer "what's the right tiny nudge for this specific moment?" Breaking an impossible task (reassembling ink from fog) into a million easy subtasks (nudge slightly) is the core insight of DDPM.

---

## Why "gradual" is the key

A natural question: why not just train a single neural network that takes pure noise and outputs an image in one shot? Because that's essentially an impossible regression problem — the network would have to hallucinate the entire structure of an image from nothing, and the training signal is too weak.

By breaking generation into T tiny denoising steps, each step becomes a well-defined regression problem: *"Given this slightly-noisy image, what's the noise?"* The network only has to make a small correction at each step. This is the same reason gradient descent works better than trying to solve an optimization problem in one step.

---

## Why it's called "probabilistic"

Both the forward and reverse processes are defined as **probability distributions**, not deterministic functions. The forward process says "given $x_0$, the noisy version $x_t$ is distributed as a Gaussian with this mean and variance." The reverse process similarly outputs a distribution over "what was the image at the previous step?"

In practice, sampling from these distributions is what introduces randomness into generation — each time you run the model starting from different random noise, you get a different image.

---

## What you'll see in the assignment

The assignment has you implement DDPM with **$T = 10$ timesteps** (way fewer than the real paper's 1000) on MNIST/FashionMNIST. This makes it feasible to train on a laptop. The specific things you'll implement:

1. The **noise schedule** ($\beta_t$, $\alpha_t$, $\bar{\alpha}_t$) — covered in [03_forward_process.md](03_forward_process.md)
2. The **forward diffusion** formula (closed-form jump from $x_0$ to $x_t$)
3. The **training loop** (sample a random $t$, noise the image, train the network to predict the noise)
4. The **reverse sampling** (turn random noise back into an image)

Once you get the concepts in the next few notes, all of this becomes very concrete.
