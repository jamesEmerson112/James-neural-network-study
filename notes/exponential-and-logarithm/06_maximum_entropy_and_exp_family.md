# 06. Maximum Entropy and the Exponential Family

## The unifying claim

Notes 04 and 05 gave two separate "why does exp appear here" arguments — Herschel's theorem for the Gaussian, and shift-invariance uniqueness for softmax. Each had its own symmetries and its own uniqueness argument. They felt like two coincidences.

**They are not coincidences.** They are special cases of a single, much deeper theorem:

> **Among all probability distributions satisfying a given set of linear constraints on expected values, the one with maximum entropy is always of the form $p(x) \propto \exp(\sum_i \lambda_i f_i(x))$ — an exponential family.**

The Gaussian, the Boltzmann distribution, the exponential distribution, the Poisson distribution, the geometric distribution, and softmax are all special cases of this one theorem. Each one corresponds to a different choice of constraints, and in every case the exponential is forced by the same Lagrangian argument.

This note walks through why that is true, which will make the exponential's ubiquity in probability stop feeling like a series of accidents and start feeling like a law of nature.

---

## The maximum entropy principle

Suppose you are trying to assign a probability distribution to something — say, the value of a random variable $X$ — and you have partial information about it. Specifically, you know the expected values of some functions:

$$
\mathbb{E}[f_1(X)] = \mu_1, \quad \mathbb{E}[f_2(X)] = \mu_2, \quad \ldots, \quad \mathbb{E}[f_k(X)] = \mu_k
$$

For example, you might know the mean ($f_1(x) = x$), the variance ($f_2(x) = x^2$), the average energy ($f_1(x) = E(x)$), or the average number of events in a time interval. These are "moment constraints" — conditions on expected values of specific functions.

Infinitely many probability distributions satisfy any given set of moment constraints. Which one should you choose?

**The maximum entropy principle** (Jaynes, 1957) says: choose the distribution with the **largest entropy** among all distributions satisfying the constraints.

The rationale is that entropy is a measure of "how uncommitted" a distribution is. A high-entropy distribution is as spread-out as it can be. Choosing the max-entropy distribution means **making no assumptions beyond what the constraints force**. Any lower-entropy distribution would encode assumptions you don't actually have evidence for.

This is the "principle of insufficient reason" elevated to a mathematical tool. It says: if your information is $\{$constraint 1, constraint 2, ..., constraint $k\}$, the only honest distribution to assume is the one that is exactly as informative as your constraints — no more, no less.

**Shannon entropy**, from information theory, is defined as:

$$
H(p) = -\sum_x p(x) \log p(x) \quad \text{(discrete)}
$$

or

$$
H(p) = -\int p(x) \log p(x) \, dx \quad \text{(continuous)}
$$

The log in Shannon entropy is the logarithm from [02_the_logarithm.md](02_the_logarithm.md). It is there for exactly the same reason logs appear everywhere in probability: **entropy is the expected log of a probability, and logs turn products of probabilities into sums of log-probabilities**, which is what makes entropy additive over independent subsystems.

---

## The theorem: max entropy forces exponential families

**Theorem.** Let $p$ be a probability distribution (discrete or continuous) that maximizes Shannon entropy $H(p) = -\sum_x p(x) \log p(x)$ subject to:

1. $\sum_x p(x) = 1$ (normalization)
2. $\mathbb{E}_p[f_i(X)] = \mu_i$ for $i = 1, 2, \ldots, k$ (moment constraints)

Then $p$ has the form:

$$
\boxed{\; p(x) = \frac{1}{Z} \exp\!\left(\sum_{i=1}^{k} \lambda_i f_i(x)\right) \;}
$$

where $\lambda_1, \ldots, \lambda_k$ are Lagrange multipliers determined by the constraints and $Z = \sum_x \exp(\sum_i \lambda_i f_i(x))$ is the **partition function** (normalizing constant).

**This is the exponential family.** Every distribution in modern probability that has a clean closed-form density is a special case of this formula. The specific $f_i$ functions determine which distribution you get:

| Constraints | Max-entropy distribution |
|---|---|
| None (just normalization) | Uniform distribution |
| $\mathbb{E}[X] = \mu$, $X \in [0, \infty)$ | Exponential: $p(x) = \lambda e^{-\lambda x}$ |
| $\mathbb{E}[X] = \mu$, $\mathbb{E}[X^2] = \mu^2 + \sigma^2$, $X \in \mathbb{R}$ | Gaussian: $p(x) \propto \exp(-(x - \mu)^2/(2\sigma^2))$ |
| Expected number of events $= \lambda$, counts in $\{0, 1, 2, \ldots\}$ | Poisson: $p(k) = \frac{\lambda^k e^{-\lambda}}{k!}$ |
| $\mathbb{E}[E(X)] = \bar{E}$, over discrete states | Boltzmann: $p(x) \propto e^{-\beta E(x)}$ |
| $\mathbb{E}[\text{class indicator}_i] = p_i$ for $i = 1, \ldots, K$, over finite states | Categorical (softmax): $p_i \propto e^{\lambda_i}$ |

**Every named distribution you meet in probability and statistics is the max-entropy answer to some specific question.** The exponential in each one comes from the same theorem.

### Proof sketch

The proof is a Lagrange multiplier calculation. You want to maximize:

$$
H(p) = -\sum_x p(x) \log p(x)
$$

subject to the $k + 1$ constraints $\sum_x p(x) = 1$ and $\sum_x p(x) f_i(x) = \mu_i$ for $i = 1, \ldots, k$.

The Lagrangian is:

$$
\mathcal{L}(p, \lambda_0, \lambda_1, \ldots, \lambda_k) = -\sum_x p(x) \log p(x) - \lambda_0\!\left(\sum_x p(x) - 1\right) - \sum_{i=1}^{k} \lambda_i\!\left(\sum_x p(x) f_i(x) - \mu_i\right)
$$

Take the partial derivative with respect to $p(x)$ for each $x$ and set it to zero:

$$
\frac{\partial \mathcal{L}}{\partial p(x)} = -\log p(x) - 1 - \lambda_0 - \sum_{i=1}^{k} \lambda_i f_i(x) = 0
$$

Solving for $p(x)$:

$$
\log p(x) = -1 - \lambda_0 - \sum_{i=1}^{k} \lambda_i f_i(x)
$$

Exponentiating both sides:

$$
p(x) = \exp\!\left(-1 - \lambda_0\right) \cdot \exp\!\left(-\sum_i \lambda_i f_i(x)\right)
$$

The first factor is a constant (it doesn't depend on $x$), and it is exactly the normalizing constant $1/Z$. So:

$$
p(x) = \frac{1}{Z} \exp\!\left(-\sum_i \lambda_i f_i(x)\right)
$$

(I've absorbed the sign into $\lambda_i$; whether you write it as positive or negative is a convention.) $\blacksquare$

### Why the exponential falls out

Look at what the proof is doing. The entropy $-\sum p \log p$ has a logarithm in it. When you differentiate $-\log p$ with respect to $p$, you get $-1/p$, which rearranges to "$\log p$ equals a linear function of $x$," which exponentiates to "$p$ equals exp of a linear function of $x$."

**The log inside entropy is what forces the exp in the answer.** Entropy asks a question in log-space ("how uncertain am I, in nats?"), and the answer comes out in exp-space (a distribution with exponential form). The entropy-exponential duality is the log-exp duality from [03_the_functional_equation.md](03_the_functional_equation.md) in its most consequential form.

This is why **every max-entropy problem gives you an exponential family, regardless of what constraints you choose**. The exponential is forced by the structure of entropy, not by the specific form of the constraints. Different constraints give different exponential families — Gaussian, Poisson, Boltzmann — but they are all members of the same family.

---

## The Gaussian as a max-entropy distribution

As a concrete example, let me derive the Gaussian from the max-entropy principle, to see how note 04's result falls out as a special case.

**Constraints**: $X \in \mathbb{R}$ with known mean $\mathbb{E}[X] = \mu$ and known variance $\mathbb{E}[(X - \mu)^2] = \sigma^2$.

**Functions**: $f_1(x) = x$ (to fix the mean), $f_2(x) = (x - \mu)^2$ (to fix the variance).

By the theorem, the max-entropy distribution has the form:

$$
p(x) \propto \exp(\lambda_1 \cdot x + \lambda_2 \cdot (x - \mu)^2)
$$

The $\lambda_1 \cdot x$ term can be absorbed by shifting the mean, so the essential shape is $\exp(\lambda_2 (x - \mu)^2)$. For the distribution to be normalizable (to integrate to 1), $\lambda_2$ must be negative — call it $-\alpha$. Then:

$$
p(x) \propto \exp(-\alpha (x - \mu)^2)
$$

Solving for $\alpha$ using the variance constraint gives $\alpha = 1/(2\sigma^2)$, which produces:

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

**The Gaussian.**

Note that this is a *different* derivation from Herschel's theorem in [04_why_the_gaussian_has_exp.md](04_why_the_gaussian_has_exp.md). Herschel's theorem gave you the Gaussian from independence + rotational symmetry. Max entropy gives you the Gaussian from "fix the mean and variance and be maximally honest." **These are two entirely different justifications for the same formula**, and the fact that both arguments lead to the Gaussian is part of why it is the most important distribution in science. It is not just forced from one direction — it is forced from multiple independent directions.

The Gaussian is to probability what $e^x$ is to calculus: a fixed point that many different questions converge on.

---

## Boltzmann's distribution as a max-entropy distribution

**Constraints**: A physical system with states indexed by $x$, each state having an energy $E(x)$. We know only the average energy $\mathbb{E}[E(X)] = \bar{E}$.

**Function**: $f_1(x) = E(x)$.

By the theorem:

$$
p(x) \propto \exp(-\beta E(x))
$$

where $\beta$ is the Lagrange multiplier for the energy constraint. Physicists write $\beta = 1/(k_B T)$, where $k_B$ is Boltzmann's constant and $T$ is temperature. So:

$$
p(x) = \frac{1}{Z} \exp(-E(x) / k_B T)
$$

**This is the Boltzmann distribution** from 1872. Note that we did not assume "energies are additive" or "independent systems have multiplicative probabilities." Those facts are consequences of the Boltzmann form, not inputs to the derivation. The only input was "maximum entropy given a fixed average energy."

**Boltzmann did not know this max-entropy derivation.** He derived his distribution from much more involved physical arguments about molecular collisions and detailed balance. It took until 1957 for **E. T. Jaynes** to publish *Information Theory and Statistical Mechanics*, which showed that Boltzmann's distribution can be re-derived from Shannon's maximum entropy principle, without any physics at all. **Statistical mechanics, Jaynes argued, is not really a theory about atoms — it is a theory about how to reason under uncertainty, and the fact that it works for gas molecules is just one application of a much more general principle.**

Jaynes's paper is one of the most consequential in 20th-century statistics. It unified thermodynamics and information theory under a single umbrella. And it explained why the exponential appears in so many otherwise-unrelated contexts: anywhere you apply the maximum entropy principle with moment constraints, you get an exponential family.

---

## Softmax as a max-entropy distribution

**Constraints**: A discrete random variable $X$ taking values in $\{1, 2, \ldots, K\}$, with $\mathbb{E}[\mathbf{1}[X = i]] = p_i$ for each $i$.

Wait — that's trivially "fix each probability to be what you want it to be," which has a trivial unique answer and isn't really an interesting constraint.

A more interesting version: constrain the expectation of $K$ feature functions $f_i(x)$ — one per class. Require $\mathbb{E}[f_i(X)] = \mu_i$ for each $i$. The max-entropy distribution is:

$$
p(x) = \frac{1}{Z} \exp\!\left(\sum_i \lambda_i f_i(x)\right)
$$

If $f_i(x) = \mathbf{1}[x = i]$ (indicator for class $i$), this simplifies to:

$$
p(x = i) = \frac{e^{\lambda_i}}{\sum_j e^{\lambda_j}}
$$

**This is softmax.** With the logits $z_i = \lambda_i$, it is exactly the formula from [05_why_softmax_uses_exp.md](05_why_softmax_uses_exp.md).

So softmax is not just "the only shift-invariant normalization of logits" (that was the uniqueness argument in note 05). It is also "the max-entropy distribution over $K$ classes given linear constraints on class-indicator expectations." Two independent derivations, same formula. Again, the Gaussian pattern: forced from multiple directions.

This connection between softmax and max entropy is the deep reason softmax is the natural output layer for multi-class classification. When you train a classifier with cross-entropy loss, you are implicitly assuming a max-entropy model of the output distribution conditioned on the inputs, with the hidden layer activations playing the role of the $\lambda_i$ parameters.

---

## The "exponential family" in statistics

Statisticians have formalized this observation into a named class of distributions: the **exponential family**. A distribution is in the exponential family if its density can be written as:

$$
p(x ; \theta) = h(x) \cdot \exp\!\left(\eta(\theta)^\top T(x) - A(\theta)\right)
$$

where:

- $h(x)$ is a base measure (often 1 or $1/\sqrt{2\pi}$)
- $T(x) = (T_1(x), T_2(x), \ldots)$ is a vector of **sufficient statistics** (the feature functions $f_i$ from above)
- $\eta(\theta)$ is a vector of **natural parameters** (the Lagrange multipliers $\lambda_i$)
- $A(\theta) = \log Z$ is the **log-partition function** (the normalizer)

Members of the exponential family:

- Gaussian (both univariate and multivariate)
- Bernoulli, binomial, multinomial
- Poisson
- Exponential (the distribution, not the function)
- Gamma, beta
- Chi-squared
- Boltzmann / Gibbs
- Categorical / softmax
- Dirichlet
- Wishart
- Many more

**The exponential family is, in a practical sense, "all the distributions that are analytically tractable."** It is the class of distributions with clean closed-form densities, clean conjugate priors in Bayesian inference, clean maximum likelihood estimators, and clean sufficient statistics. Generalized linear models (logistic regression, Poisson regression, linear regression) are all built on exponential family distributions.

And by the max-entropy theorem above, **every exponential family distribution is the max-entropy answer to some moment-constraint problem**. The exponential family is not a random collection of distributions — it is precisely the set of "most honest" distributions for different kinds of prior information.

---

## Why this pattern is inescapable

The max-entropy theorem explains something that otherwise feels magical: why the *same* function ($e^x$) shows up in so many otherwise-unrelated contexts. It shows up in the Gaussian, in softmax, in Boltzmann distributions, in Poisson distributions, in Bayesian priors, in generalized linear models. Each one looks different on the surface, but they are all applications of the same principle:

> **If you want the most honest probability distribution given some linear constraints on expected values, the answer has an exponential in it.**

And the exponential is there because entropy — the thing being maximized — has a log in it. The log and the exp are the unique pair of functions that convert between additive and multiplicative structures (see [03_the_functional_equation.md](03_the_functional_equation.md)). Entropy is an additive quantity (entropy of independent systems adds); probability is a multiplicative quantity (probabilities of independent events multiply); the exp/log pair is what lets you go between them without loss.

When this pattern repeats in your notes — from the Gaussian in [../ddpm/02_what_is_a_gaussian.md](../ddpm/02_what_is_a_gaussian.md) to softmax in [../22_softmax.md](../22_softmax.md) to Boltzmann machines in [../10b_boltzmann_machines_energy_and_randomness.md](../10b_boltzmann_machines_energy_and_randomness.md) — you are seeing the same theorem being applied with different constraints.

---

## Historical thread: Boltzmann → Gibbs → Shannon → Jaynes

**1872 — Ludwig Boltzmann** publishes his paper on the "H-theorem" in statistical mechanics. He derives the Boltzmann distribution $p \propto e^{-E/k_B T}$ from arguments about molecular collisions and the tendency of physical systems to evolve toward equilibrium. Boltzmann did not use the word "entropy" in its modern information-theoretic sense, but he defined a quantity $H$ (the "H-theorem" function) which turned out to be the negative of Shannon entropy. Boltzmann was driven to suicide in 1906, partly because many of his contemporaries refused to believe in the existence of atoms — Boltzmann's entire statistical mechanics program presupposed atomic matter, and this was not fully accepted until Einstein's 1905 paper on Brownian motion.

**1902 — J. Willard Gibbs** publishes *Elementary Principles in Statistical Mechanics*, generalizing Boltzmann's distribution to arbitrary physical systems (not just gases). Gibbs introduces the concept of an "ensemble" of microstates and derives the **Gibbs distribution** $p \propto e^{-\beta H}$ where $H$ is any Hamiltonian (energy function). The Gibbs distribution is the modern form of the Boltzmann distribution, used in essentially all of statistical physics since.

**1948 — Claude Shannon** publishes *A Mathematical Theory of Communication*. Shannon defines entropy as $H(p) = -\sum p \log p$ and shows that it measures the average "information content" of a random variable, in bits (if log base 2) or nats (if natural log). Shannon was aware of the similarity to Boltzmann's H-function — he even discussed the connection with John von Neumann, who reportedly suggested he call his quantity "entropy" because "nobody knows what entropy is, so in a debate you will always have the advantage." (This quote is possibly apocryphal but widely repeated.) Shannon entropy is the same mathematical object as Boltzmann entropy; Shannon just applied it to communication channels instead of gas molecules.

**1957 — Edwin Thompson Jaynes** publishes *Information Theory and Statistical Mechanics*, which unifies Boltzmann and Shannon under the maximum entropy principle. Jaynes's insight was that **Boltzmann's distribution is not a consequence of physics per se — it is a consequence of the max-entropy principle applied to physical systems**. Anywhere you apply the max-entropy principle, you get an exponential family distribution. Statistical mechanics is one application; statistical inference is another; machine learning is another still. Jaynes made it explicit that the same theorem underlies all three.

**1985 — Geoffrey Hinton and Terry Sejnowski** introduce **Boltzmann machines** — neural networks whose stochastic dynamics follow exactly Boltzmann's distribution. See [../10b_boltzmann_machines_energy_and_randomness.md](../10b_boltzmann_machines_energy_and_randomness.md) for the full story. Boltzmann machines are an explicit homage to the statistical mechanics lineage: Hinton chose the name to emphasize that the network's probability distribution is exactly the one from Boltzmann 1872.

**1989 — John Bridle** gives softmax its modern name and derives it from the Gibbs distribution (see [05_why_softmax_uses_exp.md](05_why_softmax_uses_exp.md)). This closes the loop: the exponential in softmax is the exponential in Boltzmann's 1872 gas distribution is the exponential in Jaynes's 1957 max-entropy principle is the exponential in every Transformer attention score computed today.

**2017 onward — every LLM** uses softmax in attention and in output layers, running the exponential family machinery billions of times per second across global inference fleets. None of this infrastructure would work without the max-entropy theorem.

The thread is unbroken: Boltzmann 1872 → Gibbs 1902 → Shannon 1948 → Jaynes 1957 → Hinton 1985 → Bridle 1989 → Vaswani 2017 → Claude 4.5 / GPT-5 / Gemini 3 in 2026. Every node in that chain has the exponential in the same place, and the reason is always the same: it is the max-entropy answer to a question about partial information.

---

## Takeaway

- **Max entropy under linear moment constraints forces exponential-family distributions.** This is a theorem, not a modeling convenience.
- **The exponential falls out of the Lagrangian because entropy has a log in it.** Differentiate $-\log p$, set to zero, exponentiate — and the exp appears. The log-exp duality from [03_the_functional_equation.md](03_the_functional_equation.md) is what drives the result.
- **Named distributions are all special cases.** Gaussian (fix mean + variance), exponential distribution (fix mean), Poisson (fix rate), Boltzmann (fix average energy), softmax / categorical (fix class expectations). Same theorem, different constraints.
- **Jaynes 1957** unified statistical mechanics and information theory under max entropy. This is why physics and ML share so much notation — they are both applying the same theorem.
- **The exponential family is, practically speaking, "all the tractable distributions."** Every closed-form density in probability is an application of the max-entropy theorem with some specific constraints.
- **The thread runs from Boltzmann 1872 → Shannon 1948 → Jaynes 1957 → modern deep learning**, with the exponential playing the same structural role at every step.

Next: [07_eulers_formula_and_rotation.md](07_eulers_formula_and_rotation.md) — the complex exponential, which turns the exp from "growth" into "rotation" and connects it to Fourier analysis, wave equations, and Transformer positional encodings.
