# 08. DDPM Glossary — Symbol to Code Mapping

Quick-reference cheat sheet — symbols, code variables, and the formulas that tie them together.

---

## Core quantities

| Symbol | Name | Meaning | Code variable | Location |
|---|---|---|---|---|
| $x_0$ | Clean image | Original training sample | `x_0`, `data` | Input to `forward_diffusion()` |
| $x_t$ | Noisy image at step $t$ | Partially corrupted version | `x_t` | Output of `forward_diffusion()` |
| $x_T$ | Fully noisy image | Pure Gaussian noise | starting point of sampling | `generate()` / `sample()` |
| $t$ | Timestep | Integer in $[0, T)$ | `t` | Function argument |
| $T$ | Total timesteps | How many diffusion steps | `self.timesteps` | `DiffusionTrainer.__init__` |
| $\varepsilon$ (epsilon) | Ground-truth noise | The noise actually added | `noise` | Returned by `forward_diffusion()` |
| $\varepsilon_\theta$ | Predicted noise | Network output | `out`, `predicted_noise` | Output of `self.net(x_t, t)` |
| $\beta_t$ (beta) | Variance schedule | Noise added at step $t$ | `self.beta` | `DiffusionTrainer.__init__` |
| $\alpha_t$ (alpha) | $1 - \beta_t$, signal retention | How much signal survives one step | `self.alpha` | `DiffusionTrainer.__init__` |
| $\bar{\alpha}_t$ (alpha-bar) | Cumulative product of $\alpha$ | Signal surviving after $t$ steps | `self.alphas_bar` | `DiffusionTrainer.__init__` |

---

## Config parameters (from `configs/config_diffusion.yaml`)

| YAML key | Value | What it controls |
|---|---|---|
| `diffusion.timesteps` | 10 | $T$ — number of forward/reverse steps |
| `diffusion.noise_start` | 0.0001 | $\beta_0$ — starting variance |
| `diffusion.noise_end` | 0.02 | $\beta_{T-1}$ — ending variance |

---

## Key formulas (with code equivalents)

### Noise schedule setup

Math:

$$
\beta = \text{linspace}(\text{noise\_start}, \text{noise\_end}, T), \qquad \alpha_t = 1 - \beta_t, \qquad \bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i
$$

Code sketch:

```python
self.beta = torch.linspace(self.noise_start, self.noise_end, self.timesteps, device=self.device)
self.alpha = 1.0 - self.beta
self.alphas_bar = torch.cumprod(self.alpha, dim=0)
```

### Forward diffusion (closed form)

Math:

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \mathbf{I})
$$

Code sketch:

```python
noise = torch.randn_like(x_0)
alpha_bar_t = self.alphas_bar[t]            # index into schedule
sqrt_ab = torch.sqrt(alpha_bar_t)           # reshape for broadcasting
sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar_t)
x_t = sqrt_ab * x_0 + sqrt_one_minus_ab * noise
```

### Training loss

Math:

$$
\mathcal{L} = \text{MSE}\!\left(\varepsilon,\ \varepsilon_\theta(x_t, t)\right)
$$

Code sketch:

```python
t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
x_t, noise = self.forward_diffusion(data, t)
predicted_noise = self.net(x_t, t)
loss = self.criterion(predicted_noise, noise)   # self.criterion is already nn.MSELoss
```

### Reverse step mean

Math:

$$
\mu = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\varepsilon_\theta(x_t, t) \right)
$$

Code sketch:

```python
predicted_noise = self.net(x, t)
alpha_t = self.alpha[t]
alpha_bar_t = self.alphas_bar[t]
beta_t = self.beta[t]
coef1 = 1.0 / torch.sqrt(alpha_t)
coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
mean = coef1 * (x - coef2 * predicted_noise)
```

### Adding variance ($t > 0$ only)

Math:

$$
\begin{aligned}
x_{t-1} &= \mu + \sqrt{\beta_t}\,z, \qquad z \sim \mathcal{N}(0, \mathbf{I}) \\
x_0     &= \mu \quad \text{(at } t = 0\text{, no noise added)}
\end{aligned}
$$

Code sketch:

```python
if t > 0:
    z = torch.randn_like(x)
    x_prev = mean + torch.sqrt(beta_t) * z
else:
    x_prev = mean
```

---

## Gotchas and reminders

1. **Device placement**: `self.beta`, `self.alpha`, `self.alphas_bar` must all live on `self.device`. Use `device=self.device` when creating them, or `.to(self.device)` afterwards.

2. **Broadcasting shape**: when you index `self.alphas_bar[t]` with a batch of timesteps, the result has shape `(batch,)`. You need to reshape it to `(batch, 1, 1, 1)` to broadcast against a 4D image tensor `(batch, channels, H, W)`. Use `.view(-1, 1, 1, 1)` or similar.

3. **Timestep input to the network**: the UNet expects `t` as a tensor, not a Python int. Your training loop passes `t` directly.

4. **$t=0$ edge case in sampling**: Don't add noise at the final step, or your output will be unnecessarily noisy.

5. **Reverse loop direction**: sampling iterates `for i in reversed(range(self.timesteps))` — from high $t$ down to 0.

6. **`torch.no_grad()`** is applied (via decorator) to `sample()` and `sample_timestep()` because we're not training during generation.

7. **Data normalization**: inside the diffusion trainer, images are rescaled from $[0, 1]$ to $[-1, 1]$ before being noised. This is important because the Gaussian noise distribution is symmetric around 0, so the clean data should be centered around 0 too.

---

## Reading the paper?

When you see the DDPM paper (Ho et al. 2020), here's the symbol dictionary you need beyond this glossary:

| Paper symbol | What it is |
|---|---|
| $q$ | Forward process distribution (fixed) |
| $p_\theta$ | Reverse process distribution (learned) |
| $\mu_\theta, \Sigma_\theta$ | Learned mean and variance of reverse step |
| $L_{\text{simple}}$ | The simplified MSE loss (what we actually use for training) |
| $L_{\text{vlb}}$ | The full variational lower bound (derived but not used directly) |

The $L_{\text{simple}}$ equation (Eq. 14 in the paper) is just MSE between true and predicted noise — that's the loss actually used in the training loop. The full ELBO derivation earlier in the paper motivates it but isn't needed to implement the trainer.

---

## Where to look next

The full study order for this folder:

- **[01_what_is_ddpm.md](01_what_is_ddpm.md)** — what DDPM is, who made it, why it exists, the ink-in-water analogy.
- **[02_what_is_a_gaussian.md](02_what_is_a_gaussian.md)** — bell curves, $\mathcal{N}(\mu, \sigma^2)$ notation, and the three properties DDPM relies on.
- **[03_forward_process.md](03_forward_process.md)** — the fixed Markov chain that corrupts images into noise, with the closed-form derivation.
- **[04_visualizing_one_noising_step.md](04_visualizing_one_noising_step.md)** — a zoomed-in walkthrough of the per-step noising formula, with hand-computed examples.
- **[05_numeric_example.md](05_numeric_example.md)** — numeric worked examples with a concrete schedule.
- **[06_reverse_process.md](06_reverse_process.md)** — the learned denoising process, training loop, and sampling loop.
- **[07_markov_vs_rnn_lstm_transformer.md](07_markov_vs_rnn_lstm_transformer.md)** — where DDPM sits among sequence-modeling architectures.
- **[08_glossary.md](08_glossary.md)** — (this file) symbol ↔ code cheat sheet.
- **[09_gan_vs_vae_vs_ddpm.md](09_gan_vs_vae_vs_ddpm.md)** — where DDPM sits among generative-model families (the headline comparison).

When coming back to implement the code, this glossary and [03_forward_process.md](03_forward_process.md) side by side cover most of what's needed. The math translates almost line-for-line into PyTorch.
