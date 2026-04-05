# 03. The Functional Equation — $e^{a+b} = e^a \cdot e^b$

## The single most important fact about the exponential

$$
\boxed{\;e^{a+b} = e^a \cdot e^b\;}
$$

This identity is the crown jewel of the exponential function. It says: **the exponential turns addition into multiplication.** And via its inverse, the logarithm, it turns multiplication back into addition:

$$
\boxed{\;\ln(xy) = \ln x + \ln y\;}
$$

These are two expressions of the same underlying isomorphism: the exponential is a continuous group homomorphism from $(\mathbb{R}, +)$ — the real numbers under addition — to $(\mathbb{R}_{>0}, \times)$ — the positive real numbers under multiplication. And the logarithm is the inverse homomorphism going the other way.

Everything deep about exp and log in probability, statistics, machine learning, and physics is a consequence of this identity. Not "mostly" a consequence — *every* use. Let me prove this strong claim by walking through the uniqueness theorem first, then the consequences.

---

## The uniqueness theorem

Here is a deep fact that elevates the exponential from "a useful function" to "a mathematical law":

**Theorem.** Let $f : \mathbb{R} \to \mathbb{R}$ be continuous and satisfy $f(a + b) = f(a) \cdot f(b)$ for all real $a, b$. Then either $f \equiv 0$ or there exists a constant $c$ such that $f(x) = e^{cx}$.

**In plain English:** among continuous functions, the exponentials $e^{cx}$ are the *only* ones that turn addition into multiplication. The exponential is forced.

### Proof sketch

Suppose $f$ satisfies $f(a+b) = f(a)f(b)$ and is not identically zero. I'll show $f$ must be an exponential.

**Step 1.** Setting $a = b = 0$ gives $f(0) = f(0)^2$, so $f(0) = 0$ or $f(0) = 1$. If $f(0) = 0$, then for any $x$ we have $f(x) = f(x + 0) = f(x)f(0) = 0$, so $f \equiv 0$, contradicting our assumption. Therefore $f(0) = 1$.

**Step 2.** Setting $b = -a$ gives $f(0) = f(a)f(-a)$, so $f(a)f(-a) = 1$, meaning $f(-a) = 1/f(a)$. In particular, $f$ is never zero, and $f(a)$ and $f(-a)$ have the same sign. Combined with continuity and $f(0) = 1 > 0$, we get $f(x) > 0$ for all $x$.

**Step 3.** Now I can take the logarithm. Define $g(x) = \ln f(x)$. Then $g$ is continuous, and the functional equation becomes:

$$
g(a + b) = \ln f(a+b) = \ln(f(a)f(b)) = \ln f(a) + \ln f(b) = g(a) + g(b)
$$

So $g$ satisfies **Cauchy's functional equation** $g(a+b) = g(a) + g(b)$, with $g$ continuous.

**Step 4.** The continuous solutions to Cauchy's functional equation are exactly the linear functions $g(x) = cx$ for some constant $c$. This is a classical result: additivity plus continuity forces linearity. (Proof: from additivity alone you can show $g(qx) = qg(x)$ for every rational $q$; continuity then forces the same for all real scalars, which means $g$ is linear.)

**Step 5.** So $g(x) = cx$, which means $\ln f(x) = cx$, which means $f(x) = e^{cx}$. $\blacksquare$

### Why the "continuity" hypothesis matters

Without continuity, Cauchy's functional equation $g(a+b) = g(a) + g(b)$ admits pathological solutions. Using the axiom of choice, you can construct functions that are additive but not linear — they take values wildly discontinuously, density-wise, on every interval. These monsters exist as formal mathematical objects, but they cannot be written down explicitly, they cannot be computed, and they do not arise in any natural modeling context. **For every physical, probabilistic, or computational purpose, the continuous solutions are the only ones that matter, which means the exponential is the only solution that matters.**

The upshot: if you ask "what function turns addition into multiplication?" and you require the answer to be computable / measurable / physically reasonable, the answer is unique. It is the exponential.

---

## Consequence 1: Independent events multiply, log-probs add

In probability, two events $A$ and $B$ are **independent** if $P(A \cap B) = P(A) \cdot P(B)$. Independence means **multiplication**.

Take logs:

$$
\log P(A \cap B) = \log P(A) + \log P(B)
$$

So in log-space, independence becomes **addition**. This is not a triviality — it is the reason log-likelihoods exist as a tool, and the reason every probabilistic ML model is trained by maximizing a *sum* of log-probabilities rather than a *product* of probabilities.

**Why the log version is strictly better for computation:**

1. **Products of small probabilities underflow.** A language model assigning probability $10^{-3}$ to each of 1000 independent words produces a sequence probability of $10^{-3000}$ — a number that is zero in any floating-point representation. But the sum of logs is $-3000 \cdot \ln(10) \approx -6908$, a perfectly ordinary double-precision number.

2. **Derivatives of sums are easier than derivatives of products.** The gradient of $\sum_i \log p_\theta(x_i)$ with respect to $\theta$ is $\sum_i \nabla_\theta \log p_\theta(x_i)$ — a sum of gradients. The gradient of $\prod_i p_\theta(x_i)$ with respect to $\theta$ involves the product rule expanded across thousands of terms, which is intractable.

3. **Concavity.** Log-likelihoods are often concave in $\theta$, making maximum-likelihood estimation a well-behaved convex optimization problem. The raw likelihood is almost never concave.

**Every maximum-likelihood procedure in machine learning — and that includes essentially all of modern deep learning, since the softmax cross-entropy loss *is* a negative log-likelihood — lives on the identity $\log(xy) = \log x + \log y$.** Without this identity, you could not train models on large datasets.

---

## Consequence 2: Multivariate Gaussians factor cleanly

The multivariate Gaussian density (in the independent-coordinates case) is:

$$
p(x_1, x_2, \ldots, x_d) = \prod_{i=1}^{d} \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{x_i^2}{2}\right)
$$

Using $e^{a+b} = e^a e^b$ in reverse, this can be rewritten as:

$$
p(x_1, \ldots, x_d) = \frac{1}{(2\pi)^{d/2}} \exp\!\left(-\frac{1}{2}\sum_{i=1}^{d} x_i^2\right) = \frac{1}{(2\pi)^{d/2}} \exp\!\left(-\frac{1}{2}\|x\|^2\right)
$$

The sum in the exponent on the right becomes a product of exponentials when you pull the sum out of the exponent, and this product is exactly the factorization on the left. **Multiplying independent Gaussian densities is trivial because the exponential turns the sum of quadratics in the exponent into a product of densities.**

This is why the multivariate Gaussian with identity covariance has the clean closed form $\exp(-\tfrac{1}{2}\|x\|^2)$: the independence across coordinates corresponds to a sum in the exponent, and the functional equation turns that sum back into a product of 1D densities.

More generally, the full multivariate Gaussian with covariance $\Sigma$ is:

$$
p(x) \propto \exp\!\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1}(x - \mu)\right)
$$

The quadratic form in the exponent encodes all the correlation structure. When $\Sigma$ is diagonal, the quadratic form is a sum of independent squared terms, and the density factors into a product of independent Gaussians. This only works because exp converts sums to products.

---

## Consequence 3: Boltzmann distributions are consistent with additive energy

In statistical mechanics, the **Boltzmann distribution** gives the probability of a system being in a state with energy $E$:

$$
p(\text{state}) \propto \exp(-E(\text{state}) / kT)
$$

where $k$ is Boltzmann's constant and $T$ is temperature.

Now consider two independent physical subsystems $A$ and $B$. The total energy of the combined system is additive: $E_{AB} = E_A + E_B$. And the probability of the combined state should be multiplicative (because the subsystems are independent): $p_{AB} = p_A \cdot p_B$.

For both to hold simultaneously, we need:

$$
\exp(-E_{AB}/kT) \;\;\overset{?}{=}\;\; \exp(-E_A/kT) \cdot \exp(-E_B/kT)
$$

This is exactly the functional equation $e^{a+b} = e^a \cdot e^b$ with $a = -E_A/kT$ and $b = -E_B/kT$. It holds automatically. **The Boltzmann distribution has the form $\exp(-E/kT)$ because that is the only form consistent with "energies add, independent probabilities multiply."** Any other distribution would violate one of these two basic physical requirements.

This is the mathematical core of why exponentials show up all over statistical physics. Given the two physical facts (energies are additive, independent probabilities are multiplicative), the exponential is forced by the uniqueness theorem above. It is not a modeling choice. It is the only shape consistent with the physics.

And this is why, when Hinton introduced **Boltzmann machines** in 1985 as neural network models, they inherited the exponential from physics directly. See [../10b_boltzmann_machines_energy_and_randomness.md](../10b_boltzmann_machines_energy_and_randomness.md).

---

## Consequence 4: KL divergence, cross-entropy, and entropy

The **Kullback-Leibler divergence** between two distributions $p$ and $q$ is:

$$
D_{KL}(p \| q) = \sum_x p(x) \log\!\frac{p(x)}{q(x)}
$$

The log here is doing two jobs simultaneously, both of which depend on the functional equation:

1. **Converting the ratio to a difference.** $\log(p/q) = \log p - \log q$. This is the functional equation applied to division: if $p$ and $q$ were independent probabilities of something, their ratio's log would be the difference of log-probabilities — a measure of how many "nats of information" $p$ provides over $q$.

2. **Making expectations tractable.** $D_{KL}$ is the expectation under $p$ of the log-likelihood ratio $\log(p/q)$. Expectations of logs of products decompose into sums — a recurring theme in every information-theoretic quantity.

**Cross-entropy** $H(p, q) = -\sum_x p(x) \log q(x)$ is the same machinery: the log turns the product of $q$ over many independent samples into a sum, and the expectation under $p$ is what we actually observe in the limit of many samples. **Shannon entropy** $H(p) = -\sum_x p(x) \log p(x)$ is the special case where $p = q$.

All three — entropy, cross-entropy, KL divergence — are built on the observation that **log-probabilities add where probabilities multiply**, which is the functional equation in its dual form. The log is the bridge between "multiplicative structure of independent events" and "additive structure of information content."

---

## Consequence 5: The softmax formula is forced

As a preview of [05_why_softmax_uses_exp.md](05_why_softmax_uses_exp.md): the softmax function

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

uses the exponential specifically because it is the unique continuous function that makes logits (log-probabilities up to an additive constant) convert to probabilities in a shift-invariant way. Any function $\phi$ used in place of $e$ must satisfy $\phi(z + c) = g(c)\phi(z)$, which is a variant of the functional equation, and the only continuous solutions are exponentials. More in note 05.

---

## The big picture: exp and log are a group isomorphism

Here is the most general way to say what we have been circling around.

The real numbers under addition, $(\mathbb{R}, +)$, form a group. The positive real numbers under multiplication, $(\mathbb{R}_{>0}, \times)$, also form a group. **The exponential function $e^x$ is a continuous isomorphism between these two groups**, and the logarithm is the inverse isomorphism.

What does "isomorphism" mean here? It means the two groups have *exactly the same structure* — every statement you can make about addition in $(\mathbb{R}, +)$ has a translated statement about multiplication in $(\mathbb{R}_{>0}, \times)$, and vice versa, via the functional equation.

| In $(\mathbb{R}, +)$ | In $(\mathbb{R}_{>0}, \times)$ |
|---|---|
| Identity element: $0$ | Identity element: $1$ |
| Inverse of $a$: $-a$ | Inverse of $y$: $1/y$ |
| $a + b$ | $y \cdot z$ |
| $a + (-a) = 0$ | $y \cdot (1/y) = 1$ |
| Linear function $cx$ | Power function $y^c$ |

The top row corresponds via $e^0 = 1$. The second row via $e^{-a} = 1/e^a$. The third via $e^{a+b} = e^a \cdot e^b$ (the functional equation). The fourth is implied by the previous ones. The fifth via $e^{cx} = (e^x)^c$.

**This is a deep fact about the structure of real numbers.** The additive and multiplicative structures of $\mathbb{R}$ are not "two different things" — they are the same structure viewed in two different coordinate systems, and the exponential is the map between the two coordinate systems.

Every time you see a conversion between "additive thinking" and "multiplicative thinking" in probability or physics — log-likelihoods, entropy, information, decibels, orders of magnitude, compound interest, exponential families — what you are really seeing is this isomorphism at work.

---

## One more beautiful consequence: $\ln$ as an integral

I mentioned this in [02_the_logarithm.md](02_the_logarithm.md) but want to close this note with it because it is an elegant corollary of everything above.

Since $\ln$ is a continuous function satisfying $\ln(xy) = \ln x + \ln y$ and $\ln 1 = 0$, its derivative must satisfy a specific relationship. Differentiating both sides of $\ln(xy) = \ln x + \ln y$ with respect to $x$ (holding $y$ fixed) gives:

$$
\frac{y}{xy} = \frac{d \ln x}{dx}
$$

(The $y$ in the numerator comes from the chain rule on the left-hand side.) Simplifying:

$$
\frac{d \ln x}{dx} = \frac{1}{x}
$$

So the logarithm is the unique antiderivative of $1/x$ that vanishes at $x = 1$:

$$
\ln x = \int_1^x \frac{dt}{t}
$$

This is Euler's 1748 analytic definition of the natural logarithm. It is not a separate definition — it is what the functional equation forces once you ask for a continuous, differentiable solution.

---

## Takeaway

- **$e^{a+b} = e^a \cdot e^b$** — the exponential turns addition into multiplication. Its dual $\ln(xy) = \ln x + \ln y$ turns multiplication back into addition.
- **Uniqueness theorem**: among continuous functions, the only solutions to $f(a+b) = f(a)f(b)$ are exponentials $f(x) = e^{cx}$. This means "turn addition into multiplication" is a problem with a forced answer.
- **Consequences in ML and physics**: log-likelihoods (sums, not products), Gaussian factorization, Boltzmann distributions, KL divergence, cross-entropy, softmax, energy-based models. **Every one is an application of the functional equation.**
- **Exp and log are a group isomorphism** between $(\mathbb{R}, +)$ and $(\mathbb{R}_{>0}, \times)$. The additive and multiplicative structures of the real numbers are the same structure viewed in two different coordinate systems, and $e^x$ / $\ln$ is the map between them.
- **Every time your math needs to turn sums into products or vice versa, $e^x$ is what does it — and by the uniqueness theorem, it is the *only* continuous function that can.**

Next: [04_why_the_gaussian_has_exp.md](04_why_the_gaussian_has_exp.md) — the first specific application of the functional equation, Herschel's 1850 theorem that forces the bell curve.
