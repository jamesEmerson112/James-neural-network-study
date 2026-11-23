# 05. Working the Math by Hand

Plugging actual numbers into the formulas from [03_forward_process.md](03_forward_process.md). Nothing makes the symbols concrete like running the arithmetic once.

---

## Example A: A small three-step walkthrough

Consider a small-scale diffusion model with variance schedule $\beta_1 = 0.1$, $\beta_2 = 0.2$, $\beta_3 = 0.3$. Compute $\bar{\alpha}_3$, then the mean and variance of $q(x_3 \mid x_0)$.

### Step 1: Compute $\alpha$ values

Recall $\alpha_t = 1 - \beta_t$.

| $t$ | $\beta_t$ | $\alpha_t = 1 - \beta_t$ |
|---|---|---|
| 1 | 0.1 | **0.9** |
| 2 | 0.2 | **0.8** |
| 3 | 0.3 | **0.7** |

### Step 2: Compute $\bar{\alpha}_t$ (cumulative products)

Recall $\bar{\alpha}_t = \alpha_1 \cdot \alpha_2 \cdots \alpha_t$.

| $t$ | Computation | $\bar{\alpha}_t$ |
|---|---|---|
| 1 | $0.9$ | **0.9** |
| 2 | $0.9 \times 0.8$ | **0.72** |
| 3 | $0.9 \times 0.8 \times 0.7$ | **0.504** |

**Result:** $\bar{\alpha}_3 = \mathbf{0.504}$

### Step 3: Write out $q(x_3 \mid x_0)$

From the closed-form formula:

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar{\alpha}_t}\,x_0,\ (1 - \bar{\alpha}_t)\mathbf{I}\right)
$$

Plugging in $\bar{\alpha}_3 = 0.504$:

- **Mean** $= \sqrt{0.504}\,x_0 = \mathbf{0.7099\,x_0}$ (approximately)
- **Variance** $= 1 - 0.504 = \mathbf{0.496}$ (applied identically to every pixel, which is what "$\cdot \mathbf{I}$" means)

**Result:**

$$
q(x_3 \mid x_0) = \mathcal{N}\!\left(0.7099\,x_0,\ 0.496\,\mathbf{I}\right)
$$

### Interpreting this answer

- After 3 noisy steps, only about 71% of the original pixel magnitudes remain (the mean is a shrunken version of $x_0$).
- The accumulated noise variance is about 0.5 — roughly equal parts signal and noise.
- A single sample $x_3$ can be drawn as $x_3 = 0.7099\,x_0 + \sqrt{0.496}\,\varepsilon$ where $\varepsilon \sim \mathcal{N}(0, \mathbf{I})$.
- Visually, if $x_0$ is a clean MNIST digit, $x_3$ would be a recognizable-but-very-noisy digit. The "3 steps with aggressive $\beta$ values" approach is unusually harsh — a more realistic schedule uses much milder $\beta$ values (see Example B below).

---

## Example B: A realistic 10-step configuration

A more realistic (but still small) config:

- $T = 10$ timesteps
- $\beta_{\text{start}} = 0.0001$
- $\beta_{\text{end}} = 0.02$

So $\beta$ is a linear schedule with 10 values from 0.0001 to 0.02.

### Step 1: $\beta$ schedule (linspace from 0.0001 to 0.02 over 10 points)

| $t$ | $\beta_t$ |
|---|---|
| 0 | 0.00010 |
| 1 | 0.00231 |
| 2 | 0.00452 |
| 3 | 0.00673 |
| 4 | 0.00894 |
| 5 | 0.01116 |
| 6 | 0.01337 |
| 7 | 0.01558 |
| 8 | 0.01779 |
| 9 | 0.02000 |

Note: this assignment uses **0-indexed** timesteps ($t = 0, 1, \ldots, 9$), which is standard in code but differs from the paper's 1-indexed convention ($t = 1, 2, \ldots, T$). Conceptually identical.

### Step 2: $\alpha_t = 1 - \beta_t$

| $t$ | $\alpha_t$ |
|---|---|
| 0 | 0.9999 |
| 1 | 0.9977 |
| 2 | 0.9955 |
| 3 | 0.9933 |
| 4 | 0.9911 |
| 5 | 0.9888 |
| 6 | 0.9866 |
| 7 | 0.9844 |
| 8 | 0.9822 |
| 9 | 0.9800 |

### Step 3: $\bar{\alpha}_t$ = cumulative product

| $t$ | $\bar{\alpha}_t$ | $\sqrt{\bar{\alpha}_t}$ | $1 - \bar{\alpha}_t$ | $\sqrt{1 - \bar{\alpha}_t}$ |
|---|---|---|---|---|
| 0 | 0.9999 | 0.9999 | 0.0001 | 0.0100 |
| 1 | 0.9976 | 0.9988 | 0.0024 | 0.0487 |
| 2 | 0.9931 | 0.9965 | 0.0069 | 0.0829 |
| 3 | 0.9864 | 0.9932 | 0.0136 | 0.1167 |
| 4 | 0.9776 | 0.9887 | 0.0224 | 0.1498 |
| 5 | 0.9667 | 0.9832 | 0.0333 | 0.1826 |
| 6 | 0.9537 | 0.9766 | 0.0463 | 0.2151 |
| 7 | 0.9389 | 0.9690 | 0.0611 | 0.2472 |
| 8 | 0.9222 | 0.9603 | 0.0778 | 0.2789 |
| 9 | 0.9037 | 0.9507 | 0.0963 | 0.3103 |

### What to notice

1. **Signal barely shrinks**: Even at $t = 9$ (the final step), $\sqrt{\bar{\alpha}} = 0.95$. So the "clean image" component is still 95% of its original magnitude.
2. **Noise stays small**: The standard deviation of the accumulated noise at $t = 9$ is only $\sqrt{0.0963} \approx 0.31$. That's a modest amount of noise — you'd still recognize the digit.
3. **This is on purpose**: With only 10 timesteps, you can't do what the real paper does (drive the signal all the way to pure noise), so a 10-step schedule keeps the total noise level modest. The network is training on relatively gentle corruptions.

```
  Evolution over the 10-step schedule

  1.00 ┤●●●●●●●●●●     √ᾱ (signal surviving)
  0.95 ┤         ○
  0.90 ┤
  0.50 ┤
  0.31 ┤                                       ▲
  0.20 ┤                                 ▲
  0.10 ┤                           ▲
  0.05 ┤                    ▲
  0.01 ┤▲▲▲▲▲             √(1−ᾱ) (noise std)
  0.00 ┤
       └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─→ t
          0  1  2  3  4  5  6  7  8  9
```

The 10-step schedule is deliberately gentle: even at $t = 9$ the signal is still at 95% and the noise std is only 0.31. Contrast with the real paper ($T = 1000$, same endpoints) where by $t = 1000$ the signal has been driven to near-zero and the final $x_T$ really is pure Gaussian noise. More steps = finer trajectory from clean to noisy = easier to reverse.

Compare this to DDPM's real configuration: $T = 1000$, $\beta_{\text{start}} = 0.0001$, $\beta_{\text{end}} = 0.02$. Same endpoints, but 100× more steps, so $\bar{\alpha}_T \approx 0.00004$ — practically zero, meaning $x_T$ is essentially pure Gaussian noise with almost none of the original signal left.

---

## Example C: One training step (optional walkthrough)

Let's say you pick a random timestep $t = 5$ for a training batch. Here's what happens:

1. **Pull $\bar{\alpha}_5$ from the table**: $\bar{\alpha}_5 = 0.9667$, so $\sqrt{\bar{\alpha}_5} = 0.9832$ and $\sqrt{1 - \bar{\alpha}_5} = 0.1826$.

2. **Sample noise**: $\varepsilon \sim \mathcal{N}(0, \mathbf{I})$, same shape as the input image (e.g., $[\text{batch}, 1, 28, 28]$ for MNIST).

3. **Produce $x_5$**:

    $$
    x_5 = 0.9832\,x_0 + 0.1826\,\varepsilon
    $$

4. **Feed $(x_5, t=5)$ to the network**: $\text{noise\_pred} = \text{net}(x_5, 5)$.

5. **Compute loss**: $\text{loss} = \text{MSE}(\text{noise\_pred}, \varepsilon)$ — the network's job is to predict the exact noise that was added.

6. **Backprop and update**.

That's one training step. You do this millions of times with random $t$ in $[0, T)$, random images, and random noise, and the network slowly learns to recognize noise patterns at every corruption level.

---

## Takeaway

Given any $\beta$ schedule, you can crank out $\alpha$, $\bar{\alpha}$, $\sqrt{\bar{\alpha}}$, and $\sqrt{1 - \bar{\alpha}}$ with a few multiplications. These four tables are all you need to run the forward process. Once you've done it by hand a few times, the formulas stop feeling abstract — they're just bookkeeping for "how much signal" and "how much noise" at each timestep.

Next note: [06_reverse_process.md](06_reverse_process.md) — how the reverse process actually undoes all this corruption to produce generated images.
