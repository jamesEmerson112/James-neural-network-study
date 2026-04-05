# 01. The Exponential Function

## The one-sentence claim

The exponential function $e^x$ is the unique smooth function $f : \mathbb{R} \to \mathbb{R}$ satisfying:

$$
f'(x) = f(x), \qquad f(0) = 1
$$

That's the whole definition. Everything else in this folder — the Taylor series, the limit definition, the functional equation $e^{a+b} = e^a e^b$, the appearance of exp inside the Gaussian PDF, inside softmax, inside Boltzmann distributions — is a consequence of this single line.

Say it out loud once: *"The exponential is the function whose derivative equals itself."*

That sentence sounds like a curiosity. It is actually the most structurally powerful fact in quantitative modeling.

---

## Why "equals its own derivative" is profound

The derivative $f'(x)$ is "the rate at which $f$ is changing at the point $x$." Saying $f'(x) = f(x)$ means **the rate of change at any point equals the value at that point**.

Make it concrete. If $f(x) = 7$, then $f'(x) = 7$ — the function is growing at rate 7 when its value is 7. If $f(x) = 100$, then $f'(x) = 100$ — the function is growing at rate 100 when its value is 100. **The bigger it gets, the faster it grows.** And in the exact same proportion as its current size.

This is the defining shape of every compounding process in nature:

- **Population growth**: a population that reproduces at a constant per-capita rate grows at a rate proportional to its current size.
- **Compound interest**: money in a bank account earning $r$% per period grows at a rate proportional to the current balance.
- **Radioactive decay**: with a negative sign, this is $f' = -\lambda f$ — the decay rate is proportional to how much radioactive material is left.
- **Chemical reactions of first order**: the reaction rate is proportional to reactant concentration.
- **Bacterial populations**: doubling-time dynamics.

Any time you see the phrase "grows at a rate proportional to its current value" (or "shrinks at a rate proportional to its current value"), you are looking at an exponential. The ODE $dy/dx = ky$ has solution $y = Ce^{kx}$, and every first-order linear process with constant coefficients is governed by this.

The reason the exponential shows up everywhere in physics, finance, biology, and machine learning is that **most processes that change in proportion to their current state are everywhere**. The exponential is the mathematical face of "compounding."

---

## Three equivalent definitions

Mathematicians define $e^x$ in at least three ways. You should internalize all three because each one unlocks a different intuition about what the exponential *is*. They are not three definitions of three different things — they are three views of the same object, from algebra, analysis, and differential equations respectively.

### (a) The Taylor series

$$
e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \frac{x^4}{24} + \frac{x^5}{120} + \cdots
$$

Read this as: *"the exponential is every polynomial stacked on top of every other polynomial, with factorial denominators keeping the higher-order terms in check."* It is the "infinite polynomial" that captures linear, quadratic, cubic, quartic, and all higher-order contributions simultaneously.

Two things to notice about this series:

1. **Every term is positive when $x > 0$.** That's why the exponential grows so fast — all the contributions pile on in the same direction.
2. **Differentiating the series term-by-term gives back the same series.** Apply $\frac{d}{dx}$ to each term:

$$
\frac{d}{dx}\!\left[\sum_{n=0}^{\infty} \frac{x^n}{n!}\right] = \sum_{n=1}^{\infty} \frac{n \cdot x^{n-1}}{n!} = \sum_{n=1}^{\infty} \frac{x^{n-1}}{(n-1)!} = \sum_{m=0}^{\infty} \frac{x^m}{m!}
$$

The last equality is just re-indexing with $m = n - 1$. You end up back where you started. This is the condition $f' = f$ in disguise — it's baked into the structure of the series through the factorials.

The factorials are doing real work here. They're what makes the series converge for all $x$. Without them you'd get a geometric series that blows up; with them, the $n!$ in the denominator grows faster than any polynomial numerator, so every term eventually becomes negligible and the sum is finite for every real input.

### (b) The limit (continuous compounding)

$$
e^x = \lim_{n \to \infty} \left(1 + \frac{x}{n}\right)^n
$$

Read this as: *"take a tiny gain of $x/n$, apply it $n$ independent times, and let $n \to \infty$."* This is **continuous compounding**.

**Jacob Bernoulli discovered this in 1683**, and the question that led him there is a beautiful one: *if you have $\$1$ at 100% annual interest, what happens if you compound more and more frequently?*

- Compounded once per year: $\$1 \times (1 + 1) = \$2$.
- Compounded twice per year (6-month periods at 50%): $\$1 \times (1 + 0.5)^2 = \$2.25$.
- Compounded 4 times: $(1 + 0.25)^4 = 2.4414\ldots$
- Compounded 12 times (monthly): $(1 + 1/12)^{12} = 2.6130\ldots$
- Compounded 365 times (daily): $(1 + 1/365)^{365} = 2.7146\ldots$
- Compounded 10,000 times: $\approx 2.7181\ldots$
- In the limit $n \to \infty$: $\approx 2.71828\ldots$

Bernoulli recognized that this sequence converges to a finite number, not to infinity. He didn't know what the number *was* — that had to wait for Euler — but he knew it was bounded. **The constant $e \approx 2.71828\ldots$ is literally named after a question about banking.**

More generally, for any $x$ (not just $x = 1$), the same limit produces $e^x$: take $x/n$ as the per-period rate and $n$ as the number of periods.

This definition makes intuitive sense of why $e^x$ governs compounding processes: it is *literally* the result of infinite compounding.

### (c) The differential equation

$$
\frac{dy}{dx} = y, \qquad y(0) = 1
$$

The unique solution is $y(x) = e^x$.

This is the **growth equation**. Read it as: *"the rate of change is equal to the current value, and the initial value is 1."* Any time a physical or economic system obeys this law, the exponential is forced to appear.

This definition is the most abstract of the three, but it is the one that makes the exponential's universality obvious. If you have any system where the rate of change is proportional to the current state, you're solving $dy/dx = ky$, whose solution is $y = Ce^{kx}$. The exponential is not a modeling choice — it is the mathematical answer to a very common question.

### The three are one

These three definitions are not similar. They are identical.

- Starting from the Taylor series, you can prove that $e^x = \lim (1 + x/n)^n$ by expanding $(1 + x/n)^n$ via the binomial theorem and taking $n \to \infty$.
- Starting from the ODE $dy/dx = y$, you can prove that the solution must have the Taylor series form by plugging in a generic power series $y = \sum a_n x^n$, matching coefficients, and finding $a_n = 1/n!$.
- Starting from the limit definition, you can prove that the result satisfies $dy/dx = y$ by differentiating the limit.

All three roads lead to the same function. When you understand that the infinite polynomial, the compound-interest limit, and the self-referential derivative are all describing a single mathematical object, you have "understood" the exponential at a structural level.

---

## The constant $e$

The number $e = 2.71828182845904523536\ldots$ is what you get when you evaluate $e^x$ at $x = 1$:

$$
e = e^1 = \sum_{n=0}^{\infty} \frac{1}{n!} = 1 + 1 + \frac{1}{2} + \frac{1}{6} + \frac{1}{24} + \frac{1}{120} + \cdots = \lim_{n \to \infty}\left(1 + \frac{1}{n}\right)^n
$$

Like $\pi$, it is **irrational** (Euler proved this in 1737 using continued fractions) and **transcendental** (Charles Hermite proved this in 1873, meaning $e$ is not a root of any polynomial with rational coefficients — it is a genuinely "deeper" number than algebraic irrationals like $\sqrt{2}$).

Four digits of $e$ are enough for almost everything: $e \approx 2.7183$. Memorize that much and you can sanity-check any exponential calculation you see.

The notation "$e$" for this constant was introduced by **Leonhard Euler** in a private manuscript in 1727 or 1728 and published in his 1736 book *Mechanica*. There is an enduring rumor that Euler used the letter "e" because it was the first letter of his own name, but historians generally think this is coincidence — the letter "e" was simply the next unused vowel after $a$ (which Euler was already using for other quantities). Euler's deeper contribution was the 1748 masterpiece *Introductio in Analysin Infinitorum*, which gave the Taylor series for $e^x$, proved Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$, and essentially founded modern analysis around the exponential.

---

## A small beautiful identity: $e^x$ is its own integral (up to a constant)

Since $\frac{d}{dx} e^x = e^x$, running the relationship backwards gives:

$$
\int e^x \, dx = e^x + C
$$

The exponential is the unique function (up to a multiplicative constant) that is simultaneously its own derivative *and* its own antiderivative. In calculus class, this is usually presented as a convenient fact. In reality, it is a restatement of the defining property: any function whose derivative is itself must also be its own antiderivative, because differentiation and integration are inverse operations.

Put another way: the exponential is a **fixed point** of the operator $\frac{d}{dx}$. Most functions change when you differentiate them. The exponential doesn't.

---

## Why the exponential is the protagonist of mathematics

A brief philosophical detour. Consider the kinds of functions you meet in first-year calculus:

- **Polynomials** ($x, x^2, x^3, \ldots$): differentiate one and you drop a degree. They converge to zero under repeated differentiation.
- **Trigonometric functions** ($\sin, \cos$): differentiate and they rotate among themselves in a 4-cycle: $\sin \to \cos \to -\sin \to -\cos \to \sin$.
- **The exponential** ($e^x$): differentiate and it stays exactly the same.

The exponential is the only elementary function that is a **fixed point** of differentiation. Every other smooth function either moves toward zero (polynomials) or cycles through some finite orbit (trig functions) when you differentiate it repeatedly. The exponential sits still. That stillness under differentiation is what makes it uniquely suited to describing compounding and self-reinforcing processes.

There is a deep sense in which the exponential is the "simplest" non-trivial function: it is the one that tells you "I am what I am." When physicists and statisticians reach for a function to describe a system whose behavior depends on its own current state, they reach for the exponential because there is nothing simpler that does the job.

---

## History in a paragraph

**Jacob Bernoulli, 1683** — discovers the constant $e$ while computing the limit of compound interest. He does not name it.

**Gottfried Wilhelm Leibniz, around 1690** — uses the letter "b" for this same constant in correspondence with Huygens. The letter did not stick.

**Leonhard Euler, 1727 (or 1728)** — introduces the letter $e$ for the constant in an unpublished manuscript. He also gives the modern definition of $e^x$ as a power series.

**Leonhard Euler, 1737** — proves $e$ is irrational using its continued fraction expansion.

**Leonhard Euler, 1748** — publishes *Introductio in Analysin Infinitorum*, the book that essentially founds modern analysis. It contains the Taylor series for $e^x$, the formula $e^{i\theta} = \cos\theta + i\sin\theta$ (see [07_eulers_formula_and_rotation.md](07_eulers_formula_and_rotation.md)), and the identity $e^{i\pi} + 1 = 0$ in its modern form. Between Bernoulli's banking curiosity in 1683 and Euler's finished treatment in 1748, the exponential went from "a weird number that shows up in compound interest" to "the central object of mathematical analysis."

**Charles Hermite, 1873** — proves $e$ is transcendental. This was the first number proven transcendental that hadn't been constructed *to be* transcendental (unlike Liouville's artificial example from 1844).

The pattern: mathematicians keep finding the exponential in contexts where they weren't looking for it. That is the signature of a fundamental object.

---

## Takeaway

- **$e^x$ is the unique smooth function with $f' = f$ and $f(0) = 1$.** Everything else follows.
- **Three equivalent definitions**: Taylor series ($\sum x^n/n!$), continuous-compounding limit ($\lim (1 + x/n)^n$), and differential equation ($dy/dx = y$). Same object, three views.
- **The constant $e \approx 2.71828$** is the value $e^1$. Bernoulli discovered it in 1683 from a banking question; Euler named it in 1727.
- **The exponential is the mathematical face of compounding** — any system whose rate of change is proportional to its current value is an exponential in disguise.

Next: [02_the_logarithm.md](02_the_logarithm.md) — the inverse of the exponential, and the function humans actually invented *first*.
