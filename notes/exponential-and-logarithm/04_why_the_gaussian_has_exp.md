# 04. Why the Gaussian Has $\exp$ — Herschel's Theorem

## The question

In [../ddpm/02_what_is_a_gaussian.md](../ddpm/02_what_is_a_gaussian.md) you saw the Gaussian probability density function:

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

Why does this function contain $\exp(-\text{quadratic})$? Not "because it makes a nice bell shape." Not "because someone in the 1800s chose it." **The Gaussian has this specific form because it is the unique shape forced by two very simple requirements, via an elegant 1850 theorem.** This note walks through the argument.

The argument matters because it is the paradigm for understanding why exponentials appear in probability more generally. The Gaussian is not a special case — it is the first case of a pattern that repeats itself across softmax, Boltzmann distributions, and the entire exponential family. See [06_maximum_entropy_and_exp_family.md](06_maximum_entropy_and_exp_family.md) for the general version.

---

## The two requirements

Consider a probability density $p(x, y)$ over the 2D plane. Require two things:

1. **Independence between components.** The density factorizes: $p(x, y) = f(x) \cdot f(y)$ for some univariate density $f$. In plain English: the $x$ and $y$ coordinates are independent and identically distributed. Knowing $x$ tells you nothing about $y$.

2. **Rotational symmetry.** The density depends only on the distance from the origin, $r = \sqrt{x^2 + y^2}$. In plain English: rotating the coordinate frame by any angle leaves the density unchanged. There is no preferred direction.

**Herschel's theorem (1850):** The only density $p(x, y)$ satisfying both requirements is of the form

$$
p(x, y) = c \cdot \exp\!\left(-\alpha(x^2 + y^2)\right)
$$

for some constants $c > 0$ and $\alpha > 0$.

In other words: **independence + rotational symmetry forces a Gaussian.** The exponential of a negative quadratic is not chosen. It is the *only* shape that satisfies these two natural conditions simultaneously.

This is a remarkable theorem. You give me two symmetry requirements that, individually, are extremely weak, and together they pin down the distribution to within two free parameters ($c$ and $\alpha$). No other probability distribution on the plane has this property.

---

## The proof

### Setup

By requirement (1), independence, $p(x, y) = f(x) f(y)$ for some function $f$.

By requirement (2), rotational symmetry, $p(x, y)$ depends only on $r^2 = x^2 + y^2$. So there is some function $g$ such that:

$$
p(x, y) = g(x^2 + y^2)
$$

Combining (1) and (2):

$$
f(x) f(y) = g(x^2 + y^2)
$$

This is the core functional equation we need to solve.

### Taking logs

Take the natural log of both sides (we can do this because a probability density is positive):

$$
\ln f(x) + \ln f(y) = \ln g(x^2 + y^2)
$$

Define $F(x) = \ln f(x)$ and $G(u) = \ln g(u)$ (where $u = x^2 + y^2$). The equation becomes:

$$
F(x) + F(y) = G(x^2 + y^2)
$$

The log turned the product on the left-hand side into a sum. This is the functional equation from [03_the_functional_equation.md](03_the_functional_equation.md) doing structural work: we needed to turn "independence gives a product" into "additivity of something."

### Differentiating to extract structure

Differentiate both sides with respect to $x$, holding $y$ fixed:

$$
F'(x) = 2x \cdot G'(x^2 + y^2)
$$

Rearrange to isolate the $G'$ term:

$$
\frac{F'(x)}{2x} = G'(x^2 + y^2)
$$

Now here is the key observation: **the left-hand side depends only on $x$**. But the right-hand side, $G'(x^2 + y^2)$, depends on both $x$ and $y$ through the argument $x^2 + y^2$.

For this equation to hold for *all* $x$ and $y$ simultaneously, the right-hand side cannot actually depend on $y$. Which means $G'(x^2 + y^2)$ must be the same for every choice of $y$ (holding $x$ fixed). But as you vary $y$, the argument $x^2 + y^2$ takes on every value $\geq x^2$. So $G'(u)$ must be constant — the same for every $u$ in its domain.

Call this constant $-\alpha$ (the negative sign will make the final form have a negative quadratic, which gives a proper density that decays at infinity). Then:

$$
G'(u) = -\alpha
$$

Integrating:

$$
G(u) = -\alpha u + C_G
$$

for some constant $C_G$.

### Back-substituting

Recall $G(u) = \ln g(u)$, so:

$$
\ln g(u) = -\alpha u + C_G \quad \Longrightarrow \quad g(u) = e^{C_G} e^{-\alpha u}
$$

And $p(x, y) = g(x^2 + y^2) = e^{C_G} \cdot e^{-\alpha(x^2 + y^2)}$. Writing $c = e^{C_G}$:

$$
\boxed{\;p(x, y) = c \cdot \exp(-\alpha(x^2 + y^2))\;}
$$

This is the Gaussian. $\blacksquare$

The constant $c$ is determined by normalization (the total probability must be 1), and $\alpha$ is determined by the variance (one parameter, one symmetry). Everything else in the formula is forced by the two requirements.

### Three lines, compressed

The entire argument fits in three lines:

1. Independence + rotational symmetry give the functional equation $f(x)f(y) = g(x^2 + y^2)$.
2. Take logs. Differentiate in $x$. The result has a left-hand side depending only on $x$ and a right-hand side depending on both $x$ and $y$ — so both sides must be constant.
3. Constant derivative means $G(u) = -\alpha u + C$, which means $p = c \exp(-\alpha(x^2 + y^2))$.

What makes this argument feel magical on first encounter is how little you put in and how much comes out. You ask for independence and rotational symmetry — two requirements that any reasonable person would write down for a "neutral" 2D distribution — and the exponential of a quadratic falls out as the unique answer. **The Gaussian is not a modeling assumption you should scrutinize. It is a theorem about what "neutral and independent" even means in 2D.**

---

## Why this proof works: the functional equation again

Look at what did the structural work here. The step "take logs" is what turned the multiplication $f(x)f(y)$ into the addition $F(x) + F(y)$, which is exactly the functional equation from [03_the_functional_equation.md](03_the_functional_equation.md). The logarithm's defining property $\ln(xy) = \ln x + \ln y$ is the hinge of the whole argument.

Without that identity — without the fact that exp and log exchange addition and multiplication — there would be no clean way to combine "independence gives a product" with "rotational symmetry gives a function of $x^2 + y^2$." The Gaussian's functional form exists because the exponential exists as an isomorphism between addition and multiplication.

This is the pattern to watch for: **every time a probability distribution has a clean closed form with $\exp$ in it, somewhere in the derivation you will find a step that turns independence (multiplication) into additivity (sum) via the log.** The Gaussian is the cleanest example of this pattern, which is why it is the most important distribution in science.

---

## Maxwell 1860: Applying Herschel to physics

Eleven years after Herschel proved his theorem, **James Clerk Maxwell** used it to solve one of the most important problems in 19th-century physics: **what is the distribution of velocities of molecules in a gas?**

Maxwell's reasoning, in 1860, went exactly like this:

1. Consider the distribution of molecular velocities $(v_x, v_y, v_z)$ in a gas at thermal equilibrium.
2. **Independence of components.** The $x$, $y$, and $z$ components of velocity should be statistically independent — no one component "knows" about the others. (This is an assumption, but a plausible one for an isotropic gas: there is no mechanism by which the $x$-velocity of a molecule could be correlated with its $y$-velocity.)
3. **Rotational symmetry.** The distribution should be the same in every direction — there is no preferred axis in a box of gas. Rotating the coordinate frame leaves the physics unchanged.

These are precisely Herschel's two requirements, applied to velocity space instead of position space. **Herschel's theorem immediately forces the distribution to be Gaussian:**

$$
p(v_x, v_y, v_z) \propto \exp(-\alpha(v_x^2 + v_y^2 + v_z^2))
$$

The single constant $\alpha$ is determined by the average kinetic energy (equivalently, the temperature) of the gas. Maxwell worked out this identification and arrived at:

$$
p(v) \propto \exp\!\left(-\frac{m \|v\|^2}{2 k_B T}\right)
$$

where $m$ is the molecular mass, $k_B$ is Boltzmann's constant (not yet named "Boltzmann's constant" in 1860 — Boltzmann was still in school), and $T$ is the temperature. This is the **Maxwell-Boltzmann velocity distribution**, the founding result of the kinetic theory of gases.

Everything about why gas molecules obey an exponential distribution — from room-temperature air to the interior of stars — traces back to Herschel's 1850 argument about independence plus rotational symmetry forcing exponentials.

Maxwell is often remembered for his equations of electromagnetism (published 1865), but historians of physics consider his 1860 paper on gas molecules to be an equally important breakthrough. It was the first time statistical reasoning was applied rigorously to a physical system, and it launched the field of **statistical mechanics** as a rigorous discipline. The path from Maxwell's velocity distribution in 1860 to Boltzmann's more general distribution $p \propto e^{-E/k_B T}$ in 1872 (see [06_maximum_entropy_and_exp_family.md](06_maximum_entropy_and_exp_family.md)) is direct: once Maxwell had shown that velocities are Gaussian-distributed, Boltzmann recognized that the underlying principle applied to any energy function, not just the kinetic energy $\tfrac{1}{2}m\|v\|^2$.

---

## Why the multivariate Gaussian has $\exp(-\tfrac{1}{2}x^\top \Sigma^{-1} x)$

The same Herschel argument, suitably generalized, also tells us why the *correlated* multivariate Gaussian has the form it does:

$$
p(x) \propto \exp\!\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1}(x - \mu)\right)
$$

The argument is: if you want a distribution that is a product of independent components *after a suitable linear transformation*, and you want it to have the independence + rotational symmetry properties in the transformed coordinates, then in the original coordinates it must have the form $\exp(-\text{quadratic form})$. The quadratic form $(x - \mu)^\top \Sigma^{-1}(x - \mu)$ is the most general quadratic that allows for a mean shift ($\mu$) and an arbitrary covariance structure ($\Sigma$).

Concretely: apply a change of variables $y = \Sigma^{-1/2}(x - \mu)$ that "whitens" the distribution (makes it have identity covariance and zero mean). In $y$-coordinates, the distribution is the standard multivariate Gaussian with independent components and rotational symmetry, so by Herschel it must be $\propto \exp(-\tfrac{1}{2}\|y\|^2)$. Transform back to $x$-coordinates and you get $\propto \exp(-\tfrac{1}{2}(x - \mu)^\top \Sigma^{-1}(x - \mu))$.

The exponential and the quadratic form show up in the multivariate Gaussian for exactly the same reason they show up in the univariate case — because the distribution can be decomposed into independent, rotationally-symmetric components in a suitable coordinate system.

---

## The Central Limit Theorem gives a second, independent reason

The argument above explains why the Gaussian is the unique independent-and-rotationally-symmetric distribution. There is a *second* reason the Gaussian shows up everywhere in nature, which is the **Central Limit Theorem (CLT)**: the sum of many independent random variables (under mild conditions) converges to a Gaussian, regardless of the underlying distributions.

The CLT is the reason measurement errors in experiments look Gaussian — they are the sum of many small, independent, unrelated perturbations — and the reason heights, weights, and IQ scores look roughly Gaussian (they are the sum of many small genetic and environmental factors).

**Herschel's theorem and the CLT are two completely different justifications for the Gaussian's ubiquity.** Herschel says: "if you want independence and rotational symmetry, the Gaussian is forced." The CLT says: "if you add up enough independent things, the Gaussian shows up whether you wanted it or not." Both arguments are correct, and they give different intuitions about why nature and mathematics conspire to put Gaussians everywhere.

In DDPM specifically, the Gaussian noise in the forward process is chosen for the Herschel reason (independence + rotational symmetry in pixel space), not for the CLT reason. The CLT would be relevant if you were modeling camera sensor noise or measurement uncertainty; in DDPM, you are choosing a noise distribution for its *mathematical* properties, and Herschel gives you exactly the properties you need.

---

## Takeaway

- **The Gaussian's $\exp(-\tfrac{1}{2}x^2)$ form is forced, not chosen.** Any probability density on $\mathbb{R}^2$ that is (a) independent across components and (b) rotationally symmetric must be Gaussian. This is **Herschel's theorem, 1850.**
- **The proof takes three lines.** Turn the multiplicative functional equation $f(x)f(y) = g(x^2 + y^2)$ into an additive one via $\log$, differentiate, observe that the only consistent solution has constant derivative, exponentiate back. The functional equation from [03_the_functional_equation.md](03_the_functional_equation.md) is what does the work.
- **Maxwell 1860** applied this to gas molecules, producing the Maxwell-Boltzmann velocity distribution and launching statistical mechanics.
- **The multivariate Gaussian with covariance $\Sigma$** is the Herschel solution in a whitened coordinate system, transformed back to the original coordinates. Same theorem, generalized.
- **The Central Limit Theorem is a second, independent reason** Gaussians appear everywhere. Herschel says the Gaussian is the unique symmetric distribution; the CLT says sums of independent things become Gaussian regardless.

Next: [05_why_softmax_uses_exp.md](05_why_softmax_uses_exp.md) — another uniqueness theorem that forces the exponential, this time for the softmax function.
