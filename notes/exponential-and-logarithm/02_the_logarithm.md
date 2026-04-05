# 02. The Logarithm

## The one-sentence claim

The **logarithm** is the inverse of the exponential. If $y = e^x$, then $x = \ln y$. That's it.

$$
\ln(e^x) = x, \qquad e^{\ln y} = y \quad (\text{for } y > 0)
$$

Where the exponential turns addition into multiplication, the logarithm turns multiplication back into addition. They are two sides of the same identity, and every deep use of one implies a use of the other.

---

## The defining property (the mirror of exp's functional equation)

From the exponential's functional equation $e^{a+b} = e^a \cdot e^b$, apply $\ln$ to both sides and you get the **logarithm's defining property**:

$$
\boxed{\;\ln(xy) = \ln x + \ln y\;}
$$

Read this as: **the logarithm turns multiplication into addition.**

This is the mirror image of exp's identity, and it is the single most important thing about the logarithm. Every deep application of the log in probability, statistics, information theory, and physics is a case of "we had a product, we wanted a sum, we took a log."

Two immediate consequences:

1. $\ln(x^n) = n \ln x$ — raising to a power becomes multiplying the log by that power.
2. $\ln(x/y) = \ln x - \ln y$ — division becomes subtraction.

These are not separate rules to memorize. They all flow from the single identity $\ln(xy) = \ln x + \ln y$.

---

## Why humans invented the logarithm *first*

Here is an irony of mathematical history. We usually teach the exponential first and introduce the logarithm as its inverse — it is the "derived" concept in the presentation order. **But historically, humans invented the logarithm over a century before they understood the exponential function.**

**John Napier, 1614.** A Scottish laird with a side interest in mathematics publishes *Mirifici Logarithmorum Canonis Descriptio* ("A Description of the Wonderful Canon of Logarithms"). His motivation is not abstract — it is astronomical. In the early 1600s, computing planetary orbits required multiplying together huge numbers, and doing this by hand was brutally slow and error-prone. Multiplying two seven-digit numbers by hand takes minutes; multiplying dozens of them takes hours; chaining them takes days.

Napier had a brilliant idea: *what if I could turn every multiplication into an addition?* That would reduce multi-day calculations to minutes. He spent **twenty years** (roughly 1594–1614) constructing a table that, for each number you wanted to multiply, gave you a corresponding "logarithm" such that adding the logarithms was equivalent to multiplying the original numbers.

Napier's logarithms were not quite the modern logarithm (he used a base close to $1/e$, not $e$ or 10, and his construction was geometric rather than analytic), but the core idea was exactly what we now call the logarithm. His insight was expressed in the very title of his book: *mirifici* — "wonderful" — because the ability to turn multiplications into additions was genuinely miraculous for working astronomers.

**Johannes Kepler** picked up Napier's tables almost immediately. He had spent decades computing the orbit of Mars by hand, culminating in his 1609 publication of the first two laws of planetary motion. When he got hold of Napier's logarithms in 1616, he adopted them enthusiastically and used them to compute the *Rudolphine Tables* of planetary positions (published 1627), which were the most accurate astronomical tables of their era. **Kepler's third law of planetary motion and his precise ephemerides would have been drastically harder to derive without Napier's logarithms.** This is not an exaggeration: logarithms were the computational infrastructure that made 17th-century quantitative astronomy possible.

**Henry Briggs, 1617–1624.** A colleague of Napier's who argued — correctly — that the logarithm would be more useful if its base were 10 rather than Napier's awkward geometric base. Briggs constructed the first tables of base-10 logarithms, which dominated scientific computation until the invention of electronic calculators in the 1960s.

**Leonhard Euler, 1748.** In the same *Introductio in Analysin Infinitorum* where Euler laid out the modern theory of the exponential function, he showed that $\ln x$ could be defined analytically as the integral

$$
\ln x = \int_1^x \frac{dt}{t}
$$

and that it is the inverse of the exponential $e^x$. This is where the logarithm stopped being just "a lookup table" and became a mathematical object in its own right, fully connected to the exponential via the functional equation.

**Slide rules, 1622–1970s.** William Oughtred invented the slide rule in 1622, a mechanical device that uses logarithmic scales to perform multiplication by physically sliding one ruler against another — literally turning multiplication into addition via sliding. For nearly 350 years, the slide rule was the primary computational tool for engineers, scientists, and navigators. Every Apollo engineer who put humans on the Moon in 1969 had a slide rule on their desk. The slide rule only fell out of use when electronic calculators became cheap in the early 1970s.

The lineage: **Napier 1614 → Kepler's orbit of Mars → Briggs's base-10 tables → Oughtred's slide rule → Euler's analytic definition → every engineering desk for 350 years → electronic calculators → log-likelihood in ML → cross-entropy → KL divergence.** "Log turns products into sums" is not a modern mathematical convenience. It is one of the most consequential ideas in the history of quantitative science.

---

## Different bases, same function

Logarithms come in different "bases." The three you will encounter in ML and physics are:

- **Natural log**, base $e$: written $\ln x$ or sometimes $\log x$ in math contexts. This is the inverse of $e^x$ and the default in calculus.
- **Log base 2**: written $\log_2 x$. Used in information theory (bits), computer science (binary), and the analysis of algorithms.
- **Log base 10** (common log): written $\log_{10} x$ or sometimes $\log x$ in engineering contexts. Used in pH, decibels, the Richter scale, and scientific notation.

**These are not three different functions.** They are the same function scaled by a constant. The change-of-base formula is:

$$
\log_a x = \frac{\ln x}{\ln a}
$$

So $\log_2 x = \ln x / \ln 2 \approx \ln x / 0.693$, and $\log_{10} x = \ln x / \ln 10 \approx \ln x / 2.303$. They all have the same *shape* — same derivatives (up to a constant), same functional equation $\log(xy) = \log x + \log y$, same asymptotic behavior — they just scale differently along the vertical axis.

**In ML, the natural log $\ln$ is the default** because its derivative is simpler:

$$
\frac{d}{dx} \ln x = \frac{1}{x}
$$

Any other base gives you $\frac{1}{x \ln a}$, which adds an annoying constant factor to every gradient computation. Base $e$ is what you use when you want the math to be as clean as possible.

**Information theory, however, uses base 2** because it matches the bit as a natural unit of information. When Shannon defined entropy as $H = -\sum p_i \log_2 p_i$, he was measuring "bits of surprise per symbol." Convert to nats (base $e$) and the formula becomes $H = -\sum p_i \ln p_i$, which is the version you see in ML cross-entropy losses. Same object, different unit — like centimeters versus inches.

---

## Why the logarithm is how humans quantify "orders of magnitude"

Human perception and measurement systems are overwhelmingly logarithmic, because log-scales are the natural way to compare quantities that span many orders of magnitude.

- **Scientific notation**: $6.022 \times 10^{23}$ is easier to read than writing out 24 digits. The "$10^{23}$" is literally the log-base-10 of the number, isolated and displayed.
- **Decibels** (sound intensity): measured on a log scale because human hearing perceives loudness roughly logarithmically. A 10 dB increase means 10× the sound power; a 20 dB increase means 100×. Without the log, you'd need scientific notation to express the range from a whisper (around $10^{-12}$ W/m²) to a jet engine (around $1$ W/m²).
- **pH** (acidity): $\text{pH} = -\log_{10}[\text{H}^+]$. pH 3 is ten times more acidic than pH 4, not "one unit" more. Chemistry spans 14 orders of magnitude of hydrogen ion concentration, so the log scale is essential.
- **Richter scale** (earthquakes): magnitude $M = \log_{10}(\text{amplitude})$. A magnitude-7 earthquake releases ~32× the energy of a magnitude-6 earthquake.
- **Stellar magnitudes** (astronomy): brightness measured on a log scale because the range from the faintest visible star to the Sun is ~10 orders of magnitude.
- **Musical pitch**: octaves are doublings of frequency, so pitch perception is logarithmic in frequency. Middle C is 261.6 Hz; C an octave up is 523.2 Hz; two octaves up is 1046.4 Hz. Each step is the same *ratio*, not the same arithmetic distance.

**The common thread**: whenever you have a quantity that varies over many orders of magnitude, you represent it on a log scale so that the representation is compact and the human eye can compare ratios visually. The logarithm is the mathematical technology that makes "orders of magnitude" a workable concept.

In ML, the same principle applies to probabilities. A typical language model assigns probabilities like $10^{-100}$ or $10^{-500}$ to rare sequences. Trying to compute with these numbers directly is numerically impossible — they underflow to zero in floating-point arithmetic. But their logs (log-probabilities) are manageable numbers like $-230$ or $-1150$, and because products of probabilities become sums of log-probabilities, entire computations can be carried out in log-space without ever touching the numerically-dangerous raw probabilities. **Log-probabilities are not a mathematical abstraction. They are the only way to compute with very small probabilities on real hardware.**

---

## The logarithm in ML and probability

A quick preview of where you will see the log in later notes. (The full treatment is in [03_the_functional_equation.md](03_the_functional_equation.md) and [08_watchlist.md](08_watchlist.md).)

- **Log-likelihood.** The likelihood of i.i.d. data is a product: $L(\theta) = \prod_i p_\theta(x_i)$. Taking the log turns this into a sum: $\log L = \sum_i \log p_\theta(x_i)$. Sums are easier to differentiate than products, and log-likelihoods don't underflow.
- **Cross-entropy loss.** $-\sum_i y_i \log p_i$ is the standard classification loss. The log there is what makes the loss sensitive to very small predicted probabilities — if your model assigns probability $10^{-20}$ to the correct class, the loss is $20$ (in natural log units), whereas if it assigned $0.5$ the loss is $0.693$. The log penalizes bad predictions on a logarithmic scale, which matches the exponential behavior of neural network output layers.
- **KL divergence.** $D_{KL}(p \| q) = \sum_i p_i \log(p_i / q_i)$. The log here comes from the expectation of a log-likelihood ratio — a measure of how many nats of extra information you need to code samples from $p$ using a code optimized for $q$.
- **Entropy.** $H(p) = -\sum_i p_i \log p_i$. This is Shannon's 1948 definition. The log (base 2 for bits, base $e$ for nats) is what converts probabilities into "expected surprise per symbol." More on this in [06_maximum_entropy_and_exp_family.md](06_maximum_entropy_and_exp_family.md).
- **Log-sum-exp trick.** A numerically stable way to compute $\log(\sum_i e^{x_i})$ when the $x_i$ are large. You factor out the maximum: $\log(\sum_i e^{x_i}) = x^* + \log(\sum_i e^{x_i - x^*})$, where $x^* = \max_i x_i$. This prevents overflow in softmax and is used in every deep learning framework.

Every one of these is an application of the single identity $\log(xy) = \log x + \log y$ (often in the form $\log \prod = \sum \log$).

---

## A small beautiful identity: the derivative of log

$$
\frac{d}{dx} \ln x = \frac{1}{x}
$$

This is the cleanest possible derivative. The logarithm is the function whose rate of change at any point is the reciprocal of the point.

Euler's definition of the natural logarithm in 1748 was precisely the integral that makes this true:

$$
\ln x = \int_1^x \frac{dt}{t}
$$

Read this as: *"the natural logarithm of $x$ is the area under the curve $1/t$ from $t = 1$ to $t = x$."* Before Euler, logarithms were tables of numbers constructed by brute force. After Euler, they were geometric objects: areas under a hyperbola. This is where the logarithm became a "mathematical object" rather than a "computational trick."

---

## Takeaway

- **The logarithm is the inverse of the exponential.** $\ln(e^x) = x$, $e^{\ln y} = y$.
- **Its defining property is $\ln(xy) = \ln x + \ln y$** — the mirror of exp's functional equation. This is why the log shows up every time products need to become sums.
- **Napier invented logarithms in 1614** to save astronomical calculations. Kepler used them to compute the orbit of Mars. Slide rules used them for 350 years. Log-likelihoods use them today.
- **Different bases are the same function scaled by a constant.** ML uses natural log ($\ln$, base $e$) for clean derivatives. Information theory uses base 2 for bits. Engineering uses base 10 for orders of magnitude.
- **The log is how humans represent quantities that span many orders of magnitude** (pH, decibels, Richter, magnitudes, log-probabilities) because ratios become visual distances on a log scale.

Next: [03_the_functional_equation.md](03_the_functional_equation.md) — the single identity that makes both exp and log universal.
