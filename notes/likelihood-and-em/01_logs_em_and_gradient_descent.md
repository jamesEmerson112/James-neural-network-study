# Logs, EM and Gradient Descent

> Companion note to `01_logs_em_and_gradient_descent_video.mp4`. See [the overview](00_overview.md).

The video is a Manim explainer at 1080p30 about how a model is fit to data by maximizing its log-likelihood. It covers three ways of doing that. The log itself makes the objective usable. EM's E-step and M-step climb it for a Gaussian mixture. Today's gradient training climbs it with autograd. Every number on screen is computed by the script from one seeded dataset of 120 points, and the script asserts each claim a caption makes. Examples are that EM's log-likelihood never decreases and that the bound in the staircase never rises above the curve.

Render it from this folder (Manim Community; no LaTeX needed):

```
manim --resolution 1920,1080 --frame_rate 30 01_logs_em_and_gradient_descent_video.py LogsEMGradient
```

Setting `EMV_ONLY=em_steps,race` renders only the named sections, which is useful when changing one part. Every section starts and ends on an empty frame, so sections rendered separately can be joined with ffmpeg's concat demuxer.

## Timeline

The video runs 7 minutes 34 seconds.

| Time | Section |
|---|---|
| 0:00 | Title |
| 0:08 | Likelihood as a product of 120 heights, and why the product underflows |
| 0:41 | What the log does: a product becomes a sum, the peak stays put, and the log undoes exp |
| 1:48 | A Gaussian mixture with hidden labels, and the log of a sum |
| 2:24 | EM's E-step and M-step, iterated 21 times |
| 3:28 | Why $\ell$ never goes down: Jensen's inequality and the staircase of lower bounds |
| 4:37 | Today: autograd, `logsumexp`, the responsibilities inside the gradient, and EM as a gradient step |
| 5:27 | EM against gradient descent, and why neural networks still train with gradients |
| 6:35 | EM's ideas in today's models, from the comparison table to the VAE's ELBO |
| 6:57 | Summary |

## The data

The script draws 72 points from $\mathcal{N}(-1.6,\ 0.75^2)$ and 48 from $\mathcal{N}(1.9,\ 1.0^2)$ with NumPy's `default_rng(20261122)`. It then rounds each point to three decimals. On screen the points are stacked into a dot histogram with bins $0.25$ wide.

## Likelihood and what the log does

A model assigns each data point a density $p(x_i \mid \theta)$. The likelihood is the product of those densities over the data:

$$L(\theta) = \prod_{i=1}^{N} p(x_i \mid \theta).$$

For the best-fitting two-curve mixture, the product of the 120 densities is about $2.4 \times 10^{-98}$. Float64 can hold that number. Float32 cannot: its smallest positive value is about $1.4 \times 10^{-45}$, so the product rounds to exactly $0$. The 16-bit formats that large models train in, bfloat16 and float16, underflow even sooner. The points average about $10^{-0.81}$ each. A thousand points like them would therefore give about $10^{-813}$, far below float64's smallest positive value of about $4.9 \times 10^{-324}$.

The video covers three properties of the log here, and a fourth comes back in the EM section.

1. **A product becomes a sum.** Since $\log(ab) = \log a + \log b$, the log-likelihood is $\ell(\theta) = \sum_i \log p(x_i \mid \theta)$. For the fitted mixture, $\ell = -224.8$, an ordinary number. A sum over examples can be estimated from a random subset, which is what minibatch training relies on. Over all 120 points the average log-density is $-1.873$, and over one random batch of 30 it is $-2.013$.
2. **The peak stays where it was.** The log is increasing, so it never changes which parameter value is best. Seven heads in ten coin flips give $L(\theta) = \theta^7(1-\theta)^3$ and $\log L(\theta) = 7\log\theta + 3\log(1-\theta)$, and both peak at $\theta = 0.7$.
3. **The log undoes exp.** For a Gaussian,
   $$-\log \mathcal{N}(x \mid \mu, \sigma^2) = \frac{(x-\mu)^2}{2\sigma^2} + \log\sigma + \log\sqrt{2\pi},$$
   so the best $\mu$ for one bell curve minimizes a sum of squares, and the answer is the sample mean $\mu = \frac{1}{n}\sum_i x_i$. The same cancellation turns softmax's negative log-likelihood into cross-entropy, $-\log \mathrm{softmax}(z)_y = -z_y + \log\sum_j e^{z_j}$.

## Hidden labels: the Gaussian mixture

If we knew which bell curve produced each point, fact 3 would finish the job. Each curve's mean would be the average of its own points, and its width the spread of those points. In a mixture the labels are hidden, so each point's density is a weighted sum,

$$p(x) = \pi_1\,\mathcal{N}(x \mid \mu_1, \sigma_1^2) + \pi_2\,\mathcal{N}(x \mid \mu_2, \sigma_2^2), \qquad \ell = \sum_i \log\big(\pi_1 \mathcal{N}_1(x_i) + \pi_2 \mathcal{N}_2(x_i)\big).$$

Now the log sits outside a sum and cannot split it. Setting $\partial\ell/\partial\mu_k = 0$ gives $\mu_k = \sum_i r_{ik} x_i / \sum_i r_{ik}$, a weighted mean. Its weights $r_{ik}$ depend on $\mu_k$ itself, so the equation has no closed-form solution.

## EM: the E-step and the M-step

Dempster, Laird and Rubin named and generalized the expectation-maximization algorithm in 1977. Special cases were older. Baum and colleagues' 1970 algorithm for hidden Markov models is one, and Sundberg's work on exponential families in the early 1970s is another. EM breaks the circular dependence by freezing the weights, solving, and repeating.

- **E-step.** Each point's responsibilities are the share of its density that each curve supplies:
  $$r_{ik} = \frac{\pi_k\,\mathcal{N}(x_i \mid \mu_k, \sigma_k^2)}{\sum_j \pi_j\,\mathcal{N}(x_i \mid \mu_j, \sigma_j^2)}.$$
  At the starting guess, the point $x = 2.53$ gets $0.021$ of its density from the left curve and $0.056$ from the right one. Its responsibility toward the right curve is therefore $0.73$.
- **M-step.** Treat the responsibilities as soft labels and refit each curve in closed form:
  $$N_k = \sum_i r_{ik}, \quad \pi_k = \frac{N_k}{N}, \quad \mu_k = \frac{1}{N_k}\sum_i r_{ik}\,x_i, \quad \sigma_k^2 = \frac{1}{N_k}\sum_i r_{ik}\,(x_i - \mu_k)^2.$$

The video starts from two wide curves, $\mu = (-0.5,\ 0.5)$, $\sigma = (1.6,\ 1.6)$ and $\pi = (0.5,\ 0.5)$. The log-likelihood climbs from $-249.54$ to $-244.24$, then $-243.48$, and so on. After 21 iterations it is within $0.01$ of where EM settles, at $\ell = -224.77$. There $\pi = (0.613,\ 0.387)$, $\mu = (-1.652,\ 1.750)$ and $\sigma = (0.780,\ 1.009)$.

### Why $\ell$ never goes down

The fourth property of the log is that it is concave: it bends downward. By Jensen's inequality (1906), the log of a weighted average is at least the weighted average of the logs. Applied to one point's term with any weights $q_k \ge 0$ that add to 1:

$$\log \sum_k \pi_k \mathcal{N}_k(x) = \log \sum_k q_k \frac{\pi_k \mathcal{N}_k(x)}{q_k} \;\ge\; \sum_k q_k \log \frac{\pi_k \mathcal{N}_k(x)}{q_k}.$$

The right side has the log inside the sum, so it can be maximized in closed form, and the M-step does exactly that. When $q$ equals the responsibilities, the two sides are equal. So each E-step builds a lower bound on $\ell$ that touches it at the current guess, and each M-step jumps to that bound's peak. At the new point, $\ell$ is at least the bound, and the bound is at least the old $\ell$. Wu (1983) gave the careful convergence analysis, correcting a flaw in the 1977 paper's argument.

The video draws this as a staircase for one parameter, $\mu_2$, with the other parameters held at their converged values. Starting from $\mu_2 = -1$, the rounds give $\ell = -371.36$, $-246.28$, $-226.52$, $-224.84$ and $-224.77$.

## Today: gradient steps on the loss

A modern implementation hands the same mixture to autograd:

```python
for step in range(1000):
    log_pi = torch.log_softmax(w, dim=0)
    log_px = Normal(mu, log_sd.exp()).log_prob(x[:, None]) + log_pi
    loss = -torch.logsumexp(log_px, dim=1).mean()   # NLL
    opt.zero_grad(); loss.backward(); opt.step()
```

`logsumexp` computes $\log\sum_k e^{a_k}$ as $m + \log\sum_k e^{a_k - m}$, where $m = \max_k a_k$, so nothing underflows. Mixture-model code computes EM's responsibilities the same way, in log space.

### The E-step is inside the gradient

$$\frac{\partial\,\mathrm{loss}}{\partial \mu_k} = -\frac{1}{N}\sum_i r_{ik}\,\frac{x_i - \mu_k}{\sigma_k^2}.$$

Differentiating the log of a sum produces the responsibilities, so backprop computes the E-step's soft labels as a by-product. For the means, EM's M-step is a gradient step with a particular step size:

$$\mu_k \leftarrow \mu_k - \frac{N\sigma_k^2}{N_k}\cdot\frac{\partial\,\mathrm{loss}}{\partial \mu_k}.$$

Xu and Jordan (1996) showed that EM for Gaussian mixtures is gradient ascent with a parameter-dependent matrix in front of the gradient, and this is that result for the means. Near the answer the step comes to about $1.0$ for the left curve and $2.6$ for the right one. Plain gradient descent instead uses one $\eta$ for every parameter.

### The race

Both methods start from the same guess. Gradient descent works on the mean log-likelihood, and its parameters are $\operatorname{logit}\pi_1$, $\mu_1$, $\mu_2$, $\log\sigma_1$ and $\log\sigma_2$. The count is the number of iterations until $\ell$ is within $0.01$ of EM's converged value, on this dataset and from this start:

| Method | Iterations |
|---|---|
| EM | 21 |
| Gradient descent, $\eta = 1$ | 64 |
| Gradient descent, $\eta = 0.3$ | 211 |
| Gradient descent, $\eta = 0.1$ | 632 |
| Gradient descent, $\eta = 3$ | never settles in 1000 steps |

The largest curvature of the mean log-likelihood at the answer is about $1.085$, so plain gradient descent is stable only for $\eta < 2/1.085 \approx 1.84$. The stiffest direction is almost entirely the left curve's mean and log-width. That is why, at $\eta = 3$, those two parameters keep bouncing.

### Why neural networks still train with gradients

EM wins this small race, but it needs things deep networks don't offer:

1. **A closed-form M-step.** A network's best weights have no weighted-mean formula, while autograd gives a gradient for any differentiable model.
2. **The whole dataset per step.** Each EM step reads all the data. A gradient step reads a minibatch, which works because $\ell$ is a sum.
3. **Hidden variables to infer.** A classifier's labels are in the data, so its log-likelihood is just cross-entropy and there is no E-step at all.
4. **An exact E-step.** When the hidden part is rich, like an image's latent code, the exact posterior is out of reach.

## EM's ideas in today's models

| | EM (1977) | Gradient training (today) |
|---|---|---|
| One step | closed-form refit | small step along the gradient |
| Step size | built in | $\eta$, by hand or adapted (Adam) |
| Data per step | the whole dataset | a minibatch |
| Works for | closed-form E- and M-steps | any differentiable model |
| Guarantee | $\ell$ never decreases | none for one minibatch step |

Neal and Hinton (1998) recast EM as maximizing a single function $F(q, \theta)$. The E-step raises $F$ over $q$, and the M-step raises it over $\theta$. That function is the lower bound above. Variational autoencoders (Kingma and Welling, 2013) maximize the same bound, now called the evidence lower bound (ELBO):

$$\mathrm{ELBO} = \mathbb{E}_{q(z\mid x)}\big[\log p(x \mid z)\big] - \mathrm{KL}\big(q(z \mid x)\,\|\,p(z)\big) \;\le\; \log p(x).$$

The encoder network predicts $q(z \mid x)$ in one forward pass, standing in for the E-step. The decoder is the model. Both are trained together by gradient steps on minibatches.

## Sources

- A. P. Dempster, N. M. Laird and D. B. Rubin, "Maximum Likelihood from Incomplete Data via the EM Algorithm," *Journal of the Royal Statistical Society, Series B* 39(1), 1977.
- J. L. W. V. Jensen, "Sur les fonctions convexes et les inégalités entre les valeurs moyennes," *Acta Mathematica* 30, 1906.
- C. F. J. Wu, "On the Convergence Properties of the EM Algorithm," *Annals of Statistics* 11(1), 1983.
- L. Xu and M. I. Jordan, "On Convergence Properties of the EM Algorithm for Gaussian Mixtures," *Neural Computation* 8(1), 1996.
- R. M. Neal and G. E. Hinton, "A View of the EM Algorithm that Justifies Incremental, Sparse, and Other Variants," in *Learning in Graphical Models*, M. I. Jordan (ed.), 1998.
- D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," arXiv:1312.6114, 2013.
