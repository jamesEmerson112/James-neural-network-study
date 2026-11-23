# L1 and L2 Regularization, and Weight Decay

> Companion note to `01_l1_l2_regularization_and_weight_decay_video.mp4`. See [the overview](00_overview.md).

The video is a Manim explainer at 1080p30 about the two standard penalties on a model's weights, and about how weight decay relates to them and to the learning rate. The worked example predicts a delivery drone's battery drain from ten logged flight features, using synthetic data. Every number on screen is computed by the script from one seeded dataset. The script asserts the claims the captions make, among them:

- Plain least squares overfits.
- Ridge's total squared weight falls as λ grows.
- Lasso's best fit keeps exactly the four real features.
- Gradient descent with weight decay lands on the ridge weights.
- AdamW's average weight follows $(1-\eta\lambda)^t$.

Render it from this folder (Manim Community through uv; no LaTeX needed):

```
uvx --from manim manim -qh --fps 30 --disable_caching 01_l1_l2_regularization_and_weight_decay_video.py Regularization
```

With Manim installed, plain `manim` in place of `uvx --from manim manim` does the same. `--disable_caching` matters here. Without it, Manim hashes every object on screen before each animation, and the live bar charts make that very slow. Setting `REG_ONLY=ridge_l2,lasso_l1` renders only the named sections (`intro`, `overfit`, `ridge_l2`, `lasso_l1`, `why_zero`, `geometry`, `decay`, `adam`, `summary`). Every section starts and ends on an empty frame, so sections rendered separately can be joined with ffmpeg's concat demuxer. The published video was made that way: each section was rendered with the command above plus `REG_ONLY`, in parallel, and the nine parts were joined without re-encoding.

## Timeline

The video runs 9 minutes 19 seconds.

| Time | Section |
|---|---|
| 0:00 | Title; L1 and L2 loss beside L1 and L2 regularization |
| 0:42 | The drone data, standardization, and plain least squares overfitting |
| 1:49 | L2 regularization (ridge): the λ sweep and the U-shaped test error |
| 2:59 | L1 regularization (lasso): weights reaching exactly 0, and the best λ |
| 4:06 | Why L1 gives exact zeros: one weight, the V-shaped penalty, and the two shrink rules |
| 5:27 | The same idea with two weights: a disk and a diamond as budgets |
| 6:22 | Weight decay versus the learning rate, on a toy loss and on the drone data |
| 7:29 | Adam with an L2 penalty versus AdamW |
| 8:50 | Summary table, and the prior each penalty corresponds to |

## Loss versus regularization

L1 and L2 loss measure how wrong the predictions are. L1 and L2 regularization apply the same two measures to the weights instead:

| | measures | formula |
|---|---|---|
| L2 loss | the errors, squared | $\sum_i (y_i - \hat y_i)^2$ |
| L1 loss | the errors, absolute | $\sum_i \lvert y_i - \hat y_i\rvert$ |
| L2 regularization | the weights, squared | $\frac{\lambda}{2}\sum_j w_j^2$ |
| L1 regularization | the weights, absolute | $\lambda\sum_j \lvert w_j\rvert$ |

Here $y_i$ is the measured value for example $i$, $\hat y_i$ is the model's prediction, $w_j$ is the weight on feature $j$, and $\lambda$ is a number you choose. Training minimizes the loss plus the penalty. The loss pulls the weights toward fitting the training data, and the penalty pulls them toward $0$.

The video uses one convention throughout. The fit term is half the mean squared error, the L2 penalty is $\frac{\lambda}{2}\sum_j w_j^2$, and the L1 penalty is $\lambda\sum_j \lvert w_j\rvert$. The halves only rescale $\lambda$. They make the L2 penalty's gradient exactly $\lambda w$ and the one-weight rules below come out clean. Values of $\lambda$ for L1 and for L2 sit on different scales, so the best $\lambda$ for one says nothing about the other.

## The drone data

A delivery company logs 14 flights. For each flight it records ten features and the battery drain in percent per minute. The script draws them with NumPy's `default_rng(85)`. A further 500 flights from the same generator measure test error.

- **Real features.** The drain is $0.9 + 0.30\,\text{payload (kg)} + 0.06\,\text{wind (m/s)} + 0.25\,\text{climb rate (m/s)} - 0.02\,\text{temperature (°C)}$, plus Gaussian noise with standard deviation $0.08$.
- **Useless features.** Hour of day, landing pad number, day of week, GPS satellites in view, firmware build and operator ID. The generator ignores all six.
- **One trap.** Wind rises with the hour of day, so a useless feature is correlated with a real one. In the 14 training flights their correlation is $0.75$.

Each feature is standardized with the training flights' mean and standard deviation, so every column has mean $0$ and mean square $1$. A weight then means "change in drain per standard deviation of the feature". One $\lambda$ treats all features alike, and the true effects become $0.30 \times \mathrm{sd}(\text{payload})$ and so on. The intercept is the mean drain, and it is not penalized.

The seed was chosen to show the lasso's behaviour cleanly. A search over the first 120 seeds found only three where the lasso kept exactly the four real features at its best $\lambda$. With this few flights, the lasso usually keeps a useless feature or drops a real one.

## The fitted weights

| feature | true effect | least squares | ridge, $\lambda = 0.0733$ | lasso, $\lambda = 0.0124$ |
|---|---|---|---|---|
| payload mass | 0.198 | 0.215 | 0.175 | 0.178 |
| wind speed | 0.123 | 0.200 | 0.141 | 0.134 |
| climb rate | 0.225 | 0.270 | 0.212 | 0.226 |
| air temperature | −0.185 | −0.165 | −0.165 | −0.176 |
| hour of day | 0 | 0.017 | −0.001 | 0 |
| landing pad number | 0 | −0.007 | 0.026 | 0 |
| day of week | 0 | −0.051 | −0.017 | 0 |
| GPS satellites | 0 | −0.014 | −0.014 | 0 |
| firmware build | 0 | 0.064 | 0.016 | 0 |
| operator ID | 0 | −0.017 | −0.017 | 0 |
| **training RMSE** | | 0.017 | | |
| **test RMSE** | | 0.186 | 0.102 | 0.084 |

RMSE is the root-mean-square error, the typical size of a miss, in percent per minute.

**Least squares overfits.** It fits ten weights and an intercept, eleven numbers, to 14 flights. That leaves three degrees of freedom for the noise. Its training error is $0.017$, a fifth of the noise level, and its test error is $0.186$, about 11 times larger. Firmware build gets $0.064$ and day of week $-0.051$. Wind's weight is inflated to $0.200$ against a true $0.123$, because hour of day soaks up and gives back part of its signal.

## L2 regularization: ridge

$$J(w) = \frac{1}{2n}\sum_{i=1}^{n}\big(y_i - b - w^\top z_i\big)^2 + \frac{\lambda}{2}\sum_j w_j^2, \qquad w = \Big(\tfrac{1}{n}Z^\top Z + \lambda I\Big)^{-1}\tfrac{1}{n}Z^\top (y - b).$$

Here $n = 14$, $z_i$ is flight $i$'s standardized features, and $b$ is the mean drain. The video sweeps $\lambda$ over 801 values from $10^{-4}$ to $1$. The script asserts four things about that sweep.

- The total $\sum_j w_j^2$ falls at every step.
- Training error rises at every step.
- Test error is U-shaped, with its minimum, $0.102$, at $\lambda = 0.0733$.
- No weight settles at $0$.

Individual weights do not shrink monotonically. Landing pad number goes from $-0.007$ under least squares to $+0.026$ at the best $\lambda$. Hour of day crosses $0$ near $\lambda = 0.068$ on its way from positive to negative, and it keeps moving.

Arthur Hoerl and Robert Kennard published ridge regression in *Technometrics* in 1970, building on Hoerl's "ridge analysis" of 1962. The idea is to add small positive quantities to the diagonal of $X^\top X$. That trades a little bias for a large cut in variance when the features overlap. Their paper also introduced the ridge trace, a plot of each coefficient against $\lambda$. The bar chart's sweep in the video is a ridge trace drawn as bars.

## L1 regularization: the lasso

$$J(w) = \frac{1}{2n}\sum_{i=1}^{n}\big(y_i - b - w^\top z_i\big)^2 + \lambda\sum_j \lvert w_j\rvert.$$

There is no closed form, so the script uses coordinate descent. Each pass updates one weight at a time with the soft threshold:

$$w_j \leftarrow \operatorname{sign}(\rho_j)\,\max(\lvert\rho_j\rvert - \lambda,\ 0), \qquad \rho_j = \tfrac{1}{n}\,z_j^\top\big(r + z_j w_j\big),$$

where $r$ is the current residual. Passes repeat until no weight moves by more than $10^{-14}$. The script warm-starts each $\lambda$ from the next larger one and then checks the optimality conditions at all 801 values of $\lambda$:

- At a zero weight, $\lvert \tfrac{1}{n} z_j^\top r\rvert \le \lambda$.
- At a nonzero weight, $\tfrac{1}{n} z_j^\top r = \lambda\,\operatorname{sign}(w_j)$.

What the sweep shows:

- **Order of zeros.** The six useless weights reach exactly $0$ one at a time, all between $\lambda = 0.0016$ and $\lambda = 0.0081$. The first real weight drops out at $\lambda = 0.146$.
- **Best fit.** At $\lambda = 0.0124$ the lasso keeps exactly the four real features, and its test error, $0.084$, beats ridge's $0.102$.
- **All zero.** Above $\lambda_{\max} = \max_j \lvert z_j^\top (y-b)\rvert / n = 0.182$, every weight is $0$ and the model predicts the mean drain.
- **A weight comes back.** Lasso paths need not be monotone. Landing pad number leaves $0$ again between $\lambda = 0.094$ and $0.155$, with a weight under $0.007$, while the weights around it shrink. The video's caption mentions this.

Robert Tibshirani introduced the lasso in the *Journal of the Royal Statistical Society, Series B* in 1996. The paper compares it with Leo Breiman's nonnegative garrote of 1995, a related shrinkage method that starts from the least-squares estimates.

## Why L1 gives exact zeros

With one weight and the fit term $\frac12(w-a)^2$, where $a$ is the weight the data alone would choose, the two penalties give closed-form rules:

$$\text{L2:}\quad w = \frac{a}{1+\lambda}, \qquad\qquad \text{L1:}\quad w = \operatorname{sign}(a)\,\max(\lvert a\rvert - \lambda,\ 0).$$

L2 divides and L1 subtracts. With $\lambda = 0.5$ and $a = 0.3$, L2 gives $0.2$ and L1 gives exactly $0$. With $a = 0.9$, L2 gives $0.6$ and L1 gives $0.4$.

The reason is the shape of each penalty at $0$. L2's penalty is a smooth bowl whose slope at $0$ is $0$, so near $0$ it hardly pulls. L1's penalty is a V whose slope is $\pm\lambda$ right up to $0$. If the data's pull at $0$, which is $\lvert a\rvert$, is weaker than $\lambda$, the sum's lowest point is the corner itself. The script checks both rules against a brute-force minimum over 300,001 grid points. The lasso's coordinate descent applies exactly this L1 rule to one weight at a time, with $a$ replaced by $\rho_j$.

## Two weights: a budget

For convex problems like these, adding $\lambda$ times a penalty gives the same solution as capping the penalty at some budget $t$. A larger $\lambda$ means a smaller $t$. With two weights the budget is a region you can draw: a disk for L2 and a diamond for L1. The best weights within budget sit where the smallest contour of the training loss first touches the region.

The video uses the loss $\frac12(w-w^*)^\top A\,(w-w^*)$ with $A = \begin{pmatrix}1 & 0.5\\ 0.5 & 0.7\end{pmatrix}$, $w^* = (1.5,\ 1.0)$ and $t = 1$. The disk is touched at $(0.815,\ 0.580)$, where both weights are nonzero. The diamond is touched at its corner $(1,\ 0)$, where $w_2$ is exactly $0$. The script confirms both points by drawing 200,000 random points from the square $[-1, 1]^2$ and checking that none of those inside a region has a lower loss than its touching point. Not every loss touches the diamond at a corner, and in exploration two of three tried settings touched an edge instead. In many dimensions, though, the L1 ball has many corners and low-dimensional edges, so solutions with exact zeros are common.

## Weight decay is not the learning rate

Gradient descent with weight decay updates each weight as

$$w \leftarrow w - \eta\,\nabla L(w) - \eta\lambda w = (1-\eta\lambda)\,w - \eta\,\nabla L(w).$$

The learning rate $\eta$ sets the step size. The decay strength $\lambda$ sets an extra shrink factor per step. With $\eta = 0.1$ and $\lambda = 0.25$ that factor is $0.975$. A resting point satisfies $\nabla L(w) + \lambda w = 0$, which has no $\eta$ in it. That is also the minimum of $L(w) + \frac{\lambda}{2}\lVert w\rVert^2$. So for plain gradient descent, weight decay is L2 regularization, and $\eta$ changes how fast the weights get there, not where they settle.

The video shows this with the loss $\frac12(w-2)^2$, four runs of 80 steps from $w = 0$:

- $\eta = 0.1$ and $\eta = 0.3$ with $\lambda = 0$ both end at $2$.
- With $\lambda = 0.25$, both end at $2/1.25 = 1.6$.

On the drone data, gradient descent with decay $\lambda = 0.0733$ runs 40,000 steps at $\eta = 0.1$ and again at $\eta = 0.3$. Both runs end within $10^{-9}$ of the ridge weights, and both rates are below the stability limit $2/(h_{\max} + \lambda)$, where $h_{\max}$ is the largest eigenvalue of $\frac1n Z^\top Z$.

Anders Krogh and John Hertz analyzed weight decay for neural networks in "A Simple Weight Decay Can Improve Generalization" (NIPS 4, 1991). They showed that in a linear network it suppresses the irrelevant components of the weight vector and, with the right strength, some of the effect of noise in the targets.

## Under Adam, the two differ

PyTorch's `torch.optim.Adam` with `weight_decay=λ` adds $\lambda\theta$ to the gradient before anything else, which is an L2 penalty. Its documentation calls the argument "weight decay (L2 penalty)". Then

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \quad \theta_t = \theta_{t-1} - \gamma\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon},$$

with $\hat m_t$ and $\hat v_t$ the bias-corrected averages. The penalty's share of the step is divided by $\sqrt{\hat v_t}$, the running size of the gradient. A weight whose gradients are large is therefore shrunk less. `torch.optim.AdamW` instead applies $\theta_t \leftarrow \theta_{t-1} - \gamma\lambda\theta_{t-1}$ directly and runs the Adam step on the loss gradient alone. Its default `weight_decay` is $0.01$. Newer PyTorch versions also give `Adam` a `decoupled_weight_decay` flag that turns it into AdamW.

**The demo.** Two weights start at $1$, and their loss gradients are pure noise: $\mathcal N(0, 10^2)$ for weight A and $\mathcal N(0, 0.1^2)$ for weight B. The settings are $\eta = 0.01$, $\lambda = 0.1$, $\beta_1 = 0.9$, $\beta_2 = 0.999$ and $\epsilon = 10^{-8}$. The script runs 2,000 steps, averaged over 1,000 runs, with both methods seeing the same noise.

- Under AdamW both averages follow $(1-\eta\lambda)^t$, which ends at $0.135$. The script asserts they stay within $0.05$ of it throughout.
- Under Adam with the L2 penalty, weight A keeps $0.81$ of its size and weight B falls to about $0$.

Single runs spread around these averages with a standard deviation between $0.07$ and $0.41$ at step 2,000, so the averaging matters.

Ilya Loshchilov and Frank Hutter made this point in "Decoupled Weight Decay Regularization", first posted to arXiv in November 2017 as "Fixing Weight Decay Regularization in Adam" and published at ICLR 2019. They showed that L2 regularization and weight decay are equivalent for SGD but not for adaptive methods like Adam. Under the L2 version, weights with large historic gradient amplitudes are regularized less, and decoupling fixes it. The update in the video follows PyTorch's AdamW documentation.

## A probabilistic reading

Regularized least squares is maximum a posteriori estimation. With Gaussian noise of variance $\sigma^2$, the negative log-posterior divided by $n/\sigma^2$ is

$$\frac{1}{2n}\sum_i (y_i-\hat y_i)^2 + \frac{\sigma^2}{n}\big(-\log p(w)\big).$$

A Gaussian prior $w_j \sim \mathcal N(0, \tau^2)$ gives the L2 penalty with $\lambda = \sigma^2/(n\tau^2)$. A Laplace prior $p(w_j) \propto e^{-\lvert w_j\rvert/s}$, which has a sharp peak at $0$, gives the L1 penalty with $\lambda = \sigma^2/(ns)$. The loss side follows the same pattern. L2 loss is the negative log-likelihood under Gaussian noise, and L1 loss is the negative log-likelihood under Laplace noise. That links this video to [the log-likelihood video](../02_likelihood-and-em/01_logs_em_and_gradient_descent.md).

## Sources

- A. E. Hoerl and R. W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems," *Technometrics* 12(1), 55–67, 1970. [doi:10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
- R. Tibshirani, "Regression Shrinkage and Selection via the Lasso," *Journal of the Royal Statistical Society, Series B* 58(1), 267–288, 1996. [doi:10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
- L. Breiman, "Better Subset Regression Using the Nonnegative Garrote," *Technometrics* 37(4), 1995. [PDF](https://www.stat.cmu.edu/technometrics/90-00/vol-37-04/v3704373.pdf)
- A. Krogh and J. A. Hertz, "A Simple Weight Decay Can Improve Generalization," *Advances in Neural Information Processing Systems 4*, 950–957, 1991. [NeurIPS proceedings](https://papers.nips.cc/paper/1991/hash/8eefcfdf5990e441f0fb6f3fad709e21-Abstract.html)
- I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- PyTorch documentation: [torch.optim.Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html) and [torch.optim.AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html), read for version 2.15.
