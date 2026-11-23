# KL Divergence

> Companion note to `03_kl_divergence_video.mp4`. See [the overview](00_overview.md), and the two videos it builds on: [Logs, EM and Gradient Descent](01_logs_em_and_gradient_descent.md) and [L1 and L2 Regularization, and Weight Decay](../04_regularization/01_l1_l2_regularization_and_weight_decay.md).

The video is a Manim explainer at 1080p30 about the Kullback–Leibler divergence. It builds KL from surprise and entropy, then follows it through three jobs in training a language model:

- the pretraining loss;
- the gap under the ELBO from the EM video;
- the penalty that keeps an RLHF-tuned model close to the model it started from.

All data are synthetic. Every number on screen is computed by the script, which asserts the claims the captions make. Among them:

- Cross-entropy equals entropy plus KL.
- Each shown subtraction adds up after rounding.
- The training loss reaches the data's entropy.
- The ELBO gap equals KL to the posterior.
- The RLHF closed form beats 20,000 random policies.

Render it from this folder (Manim Community through uv; no LaTeX needed):

```
uvx --from manim manim -qh --fps 30 --disable_caching 03_kl_divergence_video.py KLDivergence
```

With Manim installed, plain `manim` in place of `uvx --from manim manim` does the same. Setting `KL_ONLY=elbo_gap,rlhf` renders only the named sections (`intro`, `surprise`, `kl_gap`, `pretraining`, `directions`, `elbo_gap`, `rlhf`, `summary`). Every section starts and ends on an empty frame, so sections rendered separately can be joined with ffmpeg's concat demuxer. The published video was made that way: each section was rendered with the command above plus `KL_ONLY`, in parallel, and the parts were joined without re-encoding.

## Timeline

The video runs 6 minutes 46 seconds.

| Time | Section |
|---|---|
| 0:00 | Title; the next-word example, with the true distribution P beside a model Q |
| 0:30 | Surprise, −log p, and entropy as the total area of rectangles |
| 1:16 | Cross-entropy, the grown and shrunk areas that make up KL, and the swapped direction |
| 2:12 | Job 1: the pretraining loss as the data's entropy plus KL |
| 3:05 | Covering versus hiding: one bell curve fit to two humps in each direction |
| 3:58 | Job 2: the ELBO gap as KL to the posterior, and the VAE form |
| 4:55 | Job 3: the KL penalty in RLHF, swept over β |
| 6:13 | Summary table, and where the name comes from |

## Symbols

| symbol | meaning |
|---|---|
| $y$ | the model's output: one next word in the first half, a whole response in the RLHF section |
| $P$, $Q$ | the true (or target) distribution, and the model or approximation |
| $H(P)$, $H(P, Q)$ | entropy, and cross-entropy |
| $\mathrm{KL}(P \,\|\, Q)$ | the KL divergence from $Q$ to $P$; the distribution on the left weights the average |
| $x$, $z$, $q$ | in the ELBO section: a data point, the hidden curve label, and a guess for the label's probabilities |
| $\pi$, $\pi_{\text{ref}}$, $R$, $\beta$ | in the RLHF section: the model being tuned, the model before tuning, the reward, and the KL penalty's strength |

All logs are natural, so every quantity is in nats. With base-2 logs the same quantities come out in bits.

## Surprise and entropy

An outcome with probability $p$ has surprise $-\log p$. A sure thing has surprise $0$, and a one-in-a-hundred event has $4.6$ nats. Entropy is the average surprise when outcomes come from $P$:

$$H(P) = -\sum_y P(y)\log P(y).$$

The example is the next word after "The robot arm picked up the". Over box, cup, bolt, wrench and cable, $P = (0.40, 0.25, 0.15, 0.12, 0.08)$, so the surprises run from $0.92$ nats for box to $2.53$ for cable, and $H(P) = 1.454$ nats. The video draws entropy as an area. Each word gets a rectangle of width $P(y)$ and height $-\log P(y)$, so the total area is the weighted average.

## Cross-entropy and KL

If the words come from $P$ but you are surprised according to a model $Q$, the average surprise is the cross-entropy

$$H(P, Q) = -\sum_y P(y)\log Q(y).$$

With $Q = (0.30, 0.35, 0.02, 0.23, 0.10)$, $H(P, Q) = 1.691$ nats. In the area picture the rectangles keep their widths and take the heights $-\log Q(y)$. Bolt's rectangle grows the most, from $1.90$ to $3.91$, because $Q$ gave it $0.02$ where $P$ gives $0.15$. The growth adds $0.417$ of area, the shrinking removes $0.180$, and the net is the KL divergence:

$$\mathrm{KL}(P \,\|\, Q) = H(P, Q) - H(P) = \sum_y P(y)\log\frac{P(y)}{Q(y)} = 0.237 \text{ nats}.$$

Two facts make KL useful:

- **It is never negative, and it is 0 only when $Q = P$.** This is Gibbs' inequality. It follows from Jensen's inequality, the same inequality that makes EM's lower bound work.
- **It is not symmetric.** Swapping the arguments gives $\mathrm{KL}(Q \,\|\, P) = 0.163$. The average is weighted by whichever distribution sits on the left, so $\mathrm{KL}(P \,\|\, Q)$ punishes $Q$ hardest for missing words that $P$ uses.

## Job 1: the pretraining loss

Training never sees $P$, only examples. The script draws 2,000 next words from $P$ with `default_rng(7)`. Their frequencies are $\hat P = (0.405, 0.243, 0.1495, 0.1255, 0.077)$, with entropy $H(\hat P) = 1.452$ nats. On one example the loss is the model's surprise at the actual next word, $-\log Q(y_i)$, the negative log-likelihood. Averaged over the examples it is a cross-entropy, which splits into two parts:

$$\frac{1}{n}\sum_{i=1}^{n} -\log Q(y_i) = H(\hat P, Q) = H(\hat P) + \mathrm{KL}(\hat P \,\|\, Q).$$

The entropy is fixed by the data, so minimizing the loss is minimizing the KL from the model to the data. Maximum-likelihood training is minimum-KL training. The video starts from the model $Q$ above, with scores $\log Q$. It then runs 60 steps of gradient descent with step size 1.0, using the softmax cross-entropy gradient $\operatorname{softmax}(\text{scores}) - \hat P$. The loss falls at every step, from $1.689$ to $1.452$, and the KL falls to $5 \times 10^{-7}$.

## The order matters: covering versus hiding

The video fits one bell curve $Q = \mathcal{N}(\mu, \sigma^2)$ to a two-hump distribution $P = 0.6\,\mathcal{N}(-2, 0.6^2) + 0.4\,\mathcal{N}(2, 0.6^2)$. The method is gradient descent on $(\mu, \log\sigma)$, run for 4,000 steps of size 0.05, with central-difference gradients of the KL integrals on a grid of 8,001 points over $[-12, 12]$. Both runs start at $\mu = -1$, $\sigma = 1$.

- **Covering.** Minimizing $\mathrm{KL}(P \,\|\, Q)$ ends at $\mu = -0.40$, $\sigma = 2.05$, which is $P$'s own mean and standard deviation. For a Gaussian $Q$, this direction matches moments, and the fit spreads over both humps and the empty valley between them.
- **Hiding.** Minimizing $\mathrm{KL}(Q \,\|\, P)$ ends at $\mu = -2.00$, $\sigma = 0.60$, inside the bigger hump, with $\mathrm{KL} = 0.510$. Started at $\mu = 1$ instead, it settles inside the smaller hump at $0.915$, a worse local optimum.

The two behaviours are often called mass-covering and mode-seeking. Pretraining uses the covering direction. The ELBO and RLHF use the hiding direction.

## Job 2: the gap under the ELBO

For a model with a hidden variable $z$ and any distribution $q(z)$, one identity holds exactly:

$$\log p(x) = \mathrm{ELBO}(q) + \mathrm{KL}\big(q \,\|\, p(z \mid x)\big), \qquad \mathrm{ELBO}(q) = \sum_z q(z)\log\frac{p(z)\,p(x \mid z)}{q(z)}.$$

It follows in one line from $p(z)\,p(x\mid z) = p(x)\,p(z\mid x)$. The video uses the two-curve mixture from the EM video at its true settings: weights $0.6$ and $0.4$, centres $-1.6$ and $1.9$, widths $0.75$ and $1.0$. At $x = 0$, $p(x) = 0.0590$ and $\log p(x) = -2.830$. The posterior gives the left curve $0.555$ and the right curve $0.445$. As $q$ sweeps from $0.1$ to $0.9$, the gap traces $\mathrm{KL}(q \,\|\, p(z \mid x))$, and it closes only at the posterior. EM's E-step sets $q$ to the posterior, which is why the staircase of bounds in the EM video touched the log-likelihood at every step.

Rearranged, the same bound reads

$$\mathrm{ELBO}(q) = \mathbb{E}_{q}\big[\log p(x \mid z)\big] - \mathrm{KL}\big(q \,\|\, p(z)\big),$$

a fit term minus a KL to the prior. This is the formula in the VAE frame of the EM video. A variational autoencoder's encoder outputs $q$ (Kingma and Welling, 2013), and the KL term keeps it near the prior. The script checks both forms against each other.

## Job 3: the leash in RLHF

RLHF tunes a pretrained model $\pi_{\text{ref}}$ toward responses that a reward model scores highly. The standard objective, as written in the DPO paper (Eq. 3), is

$$\max_\pi\ \mathbb{E}_{y \sim \pi}\big[R(y)\big] - \beta\,\mathrm{KL}\big(\pi \,\|\, \pi_{\text{ref}}\big).$$

Its maximizer has a closed form (DPO Eq. 4):

$$\pi^*(y) = \frac{1}{Z}\,\pi_{\text{ref}}(y)\,e^{R(y)/\beta}, \qquad Z = \sum_y \pi_{\text{ref}}(y)\,e^{R(y)/\beta}.$$

The objective itself is a KL in disguise:

$$\mathbb{E}_\pi[R] - \beta\,\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}}) = \beta\log Z - \beta\,\mathrm{KL}(\pi \,\|\, \pi^*).$$

So the best $\pi$ is the one at zero KL from $\pi^*$.

The video's toy problem has five responses with $\pi_{\text{ref}} = (0.30, 0.30, 0.20, 0.15, 0.05)$ and rewards $R = (1.0, 1.2, 0.8, 0.0, 3.0)$. The sweep shows three points:

| $\beta$ | KL | expected reward |
|---|---|---|
| 10 | 0.002 nats | 1.008 |
| 1 | 0.350 nats | 1.577 |
| 0.1 | 2.996 nats | 3.000 |

At $\beta = 10$ the tuned model barely moves. At $\beta = 0.1$ almost all probability sits on $y_5$, a response the reference model writes only 5% of the time. The KL there is close to its maximum, $-\log 0.05 = 3.00$. Chasing one score that hard is reward over-optimization. InstructGPT added "a per-token KL penalty from the SFT model at each token to mitigate over-optimization of the reward model" (Ouyang et al., 2022). Its objective subtracts $\beta\log\big(\pi^{RL}(y \mid x)/\pi^{SFT}(y \mid x)\big)$ from the reward. Averaged over the policy's own responses, that log-ratio is this KL.

DPO (Rafailov et al., 2023) rearranges the closed form into $R(y) = \beta\log\frac{\pi^*(y)}{\pi_{\text{ref}}(y)} + \beta\log Z$. That lets it train on pairs of preferred and rejected responses directly, without a separate reward model.

The penalty uses the hiding direction. $\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$ is infinite if $\pi$ puts probability where $\pi_{\text{ref}}$ puts none, so the tuned model stays where the original already writes.

## Summary

| job | the KL | direction | what it does |
|---|---|---|---|
| pretraining loss | $H(\text{data}) + \mathrm{KL}(\text{data} \,\|\, Q)$ | covering | the model must cover every word the data uses |
| EM and VAEs | $\log p(x) - \mathrm{ELBO} = \mathrm{KL}(q \,\|\, p(z \mid x))$ | hiding | the bound touches when $q$ is the posterior |
| RLHF | $\beta\,\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$ | hiding | the tuned model stays near what it already writes |

## History

Claude Shannon defined entropy, $H = -\sum p \log p$, in 1948. Solomon Kullback and Richard Leibler introduced the one-way quantity in 1951, as "the mean information for discrimination" between two hypotheses per observation. They wrote it $I(1:2)$. In their paper the word "divergence" meant the two-way sum $J(1,2) = I(1:2) + I(2:1)$, which Harold Jeffreys had already defined and used in 1948. Today the one-way quantity is the one called KL divergence, or relative entropy. The naming history here follows the etymology section of Wikipedia's article, which cites the 1951 paper.

## Sources

- C. E. Shannon, "A Mathematical Theory of Communication," *Bell System Technical Journal* 27, 379–423 and 623–656, 1948. [doi:10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
- S. Kullback and R. A. Leibler, "On Information and Sufficiency," *Annals of Mathematical Statistics* 22(1), 79–86, 1951. [doi:10.1214/aoms/1177729694](https://doi.org/10.1214/aoms/1177729694)
- D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," 2013. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- L. Ouyang et al., "Training language models to follow instructions with human feedback," 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- R. Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Wikipedia, "Kullback–Leibler divergence," etymology section, for the 1951 naming and Jeffreys' symmetrized divergence. [en.wikipedia.org](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
