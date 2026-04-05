# 07. Euler's Formula and Rotation — The Complex Exponential

## The formula

$$
\boxed{\; e^{i\theta} = \cos\theta + i \sin\theta \;}
$$

This is **Euler's formula** (published 1748). It extends the exponential function from the real numbers to the complex numbers, and in doing so it transforms the exponential from "a function that describes growth" into "a function that describes rotation."

The formula says: if you plug in an imaginary number $i\theta$ as the exponent, the output is a complex number whose real part is $\cos\theta$ and whose imaginary part is $\sin\theta$. Plotted in the complex plane (where the horizontal axis is real and the vertical axis is imaginary), this is a point on the **unit circle** at angle $\theta$ from the positive real axis.

As $\theta$ varies from $0$ to $2\pi$, the point $e^{i\theta}$ traces out the unit circle exactly once. **Multiplication by $e^{i\theta}$ is rotation by angle $\theta$.** This one fact — that the exponential in the imaginary direction is rotation — connects the exponential to Fourier transforms, wave equations, signal processing, and, most recently, Transformer positional encodings.

---

## Why the formula is true

There are several ways to prove Euler's formula, all of which require extending $e^x$ to complex arguments. The cleanest proof uses the Taylor series from [01_the_exponential.md](01_the_exponential.md).

### Starting from the Taylor series

The Taylor series for $e^x$ is:

$$
e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \frac{x^4}{4!} + \cdots
$$

This series converges for all real $x$. But it also makes sense for complex $x$: you just plug in a complex number and compute each term. So we can *define* $e^z$ for complex $z$ by the same series:

$$
e^z = \sum_{n=0}^{\infty} \frac{z^n}{n!}
$$

Now substitute $z = i\theta$, where $\theta$ is real and $i$ is the imaginary unit ($i^2 = -1$):

$$
e^{i\theta} = 1 + i\theta + \frac{(i\theta)^2}{2!} + \frac{(i\theta)^3}{3!} + \frac{(i\theta)^4}{4!} + \frac{(i\theta)^5}{5!} + \cdots
$$

Now compute the powers of $i$:

- $i^0 = 1$
- $i^1 = i$
- $i^2 = -1$
- $i^3 = i \cdot i^2 = -i$
- $i^4 = (i^2)^2 = 1$ — and the pattern repeats with period 4

So the series becomes:

$$
e^{i\theta} = 1 + i\theta - \frac{\theta^2}{2!} - i\frac{\theta^3}{3!} + \frac{\theta^4}{4!} + i\frac{\theta^5}{5!} - \frac{\theta^6}{6!} - \cdots
$$

Group the terms by whether they have an $i$:

$$
e^{i\theta} = \underbrace{\left(1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \frac{\theta^6}{6!} + \cdots\right)}_{\text{real part}} + i \underbrace{\left(\theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \frac{\theta^7}{7!} + \cdots\right)}_{\text{imaginary part}}
$$

Look at those two series. The first is the Taylor series for $\cos\theta$:

$$
\cos\theta = 1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \frac{\theta^6}{6!} + \cdots
$$

And the second is the Taylor series for $\sin\theta$:

$$
\sin\theta = \theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \frac{\theta^7}{7!} + \cdots
$$

So $e^{i\theta} = \cos\theta + i\sin\theta$. $\blacksquare$

### Why this proof is magical

Look at what happened. Three apparently unrelated functions — the exponential, the cosine, and the sine — have Taylor series that are, up to signs, the same series cut into three interleaved pieces. The exponential's series runs through all the integer powers; the cosine's series picks out the even powers with alternating signs; the sine's series picks out the odd powers with alternating signs. When you plug in $i\theta$, the powers of $i$ (which cycle through $1, i, -1, -i$) distribute the terms exactly right — the real parts land in the cosine series and the imaginary parts land in the sine series.

The Taylor series for $e^x$, $\cos x$, and $\sin x$ were discovered (or at least formalized) separately in the 17th and 18th centuries. It took **Euler** to notice that they were related by this identity. The relationship is not a coincidence — it reflects the fact that all three functions are solutions to linear ODEs with constant coefficients, and their behaviors are tightly constrained by the same underlying equation ($y'' = -y$ for sine/cosine and $y' = y$ for exp, which are closely related).

**Euler's formula is the bridge that unifies exponentials and trigonometry.** Before 1748, these were two separate branches of mathematics. After 1748, they were two aspects of the same thing.

---

## The geometric meaning: rotation in the complex plane

Consider the complex plane, where complex numbers $a + bi$ are plotted as points $(a, b)$ — real part on the horizontal axis, imaginary part on the vertical axis. In this plane:

- The number $1$ is at $(1, 0)$ on the positive real axis.
- The number $i$ is at $(0, 1)$, one unit up the imaginary axis.
- The number $-1$ is at $(-1, 0)$ on the negative real axis.
- The number $-i$ is at $(0, -1)$, one unit down.

These four points are on the **unit circle** — the circle of radius 1 centered at the origin.

Now plot $e^{i\theta}$ for varying $\theta$. By Euler's formula, this is the point $(\cos\theta, \sin\theta)$. As $\theta$ varies, the point traces out the unit circle:

- $\theta = 0$: $e^{i \cdot 0} = 1$, point at $(1, 0)$.
- $\theta = \pi/2$: $e^{i\pi/2} = \cos(\pi/2) + i\sin(\pi/2) = 0 + i = i$, point at $(0, 1)$.
- $\theta = \pi$: $e^{i\pi} = \cos\pi + i\sin\pi = -1 + 0 = -1$, point at $(-1, 0)$.
- $\theta = 3\pi/2$: $e^{i \cdot 3\pi/2} = 0 - i = -i$, point at $(0, -1)$.
- $\theta = 2\pi$: $e^{i \cdot 2\pi} = 1$, back to start.

The function $\theta \mapsto e^{i\theta}$ is a **parameterization of the unit circle**. Angle in → point on the unit circle out.

### Multiplication by $e^{i\theta}$ is rotation

Here is the geometric punchline. Suppose you have a complex number $w = re^{i\varphi}$ (written in polar form, where $r = |w|$ is the magnitude and $\varphi$ is the angle). Multiply it by $e^{i\theta}$:

$$
w \cdot e^{i\theta} = r e^{i\varphi} \cdot e^{i\theta} = r e^{i(\varphi + \theta)}
$$

The magnitude $r$ is unchanged, but the angle shifts from $\varphi$ to $\varphi + \theta$. **Multiplying by $e^{i\theta}$ rotates the complex number by $\theta$ radians counterclockwise around the origin.**

This is one of the most beautiful geometric facts in mathematics. The exponential function, which started life as a description of growth (Bernoulli's 1683 compound interest), becomes a description of rotation when you let its argument be imaginary. The same function does both — "grow" when the argument is real, "rotate" when the argument is imaginary, and combinations of the two when the argument is a general complex number $a + bi$ (grow with rate $a$ while rotating with angular velocity $b$).

### Why addition-to-multiplication gives rotation-to-rotation

Recall the functional equation $e^{a+b} = e^a \cdot e^b$ from [03_the_functional_equation.md](03_the_functional_equation.md). Apply it to two imaginary arguments:

$$
e^{i\theta_1} \cdot e^{i\theta_2} = e^{i(\theta_1 + \theta_2)}
$$

This says: **multiplying two unit-complex-numbers rotates by the sum of their angles.** The functional equation of the exponential becomes the "angles add under composition of rotations" law of trigonometry. Two rotations composed in sequence give a rotation by the sum of the angles.

From this identity you can *immediately* derive the angle-addition formulas for sine and cosine. Expand both sides:

$$
(\cos\theta_1 + i\sin\theta_1)(\cos\theta_2 + i\sin\theta_2) = \cos(\theta_1 + \theta_2) + i\sin(\theta_1 + \theta_2)
$$

Multiply out the left-hand side:

$$
(\cos\theta_1 \cos\theta_2 - \sin\theta_1 \sin\theta_2) + i(\sin\theta_1 \cos\theta_2 + \cos\theta_1 \sin\theta_2)
$$

Matching real and imaginary parts with the right-hand side gives:

$$
\cos(\theta_1 + \theta_2) = \cos\theta_1 \cos\theta_2 - \sin\theta_1 \sin\theta_2
$$

$$
\sin(\theta_1 + \theta_2) = \sin\theta_1 \cos\theta_2 + \cos\theta_1 \sin\theta_2
$$

These are the angle-addition formulas that high-school students memorize. With Euler's formula, they are not separate identities to memorize — they are a single statement ($e^{i\theta_1} \cdot e^{i\theta_2} = e^{i(\theta_1 + \theta_2)}$) that encompasses both. **The functional equation of the exponential is the same thing as the angle-addition formulas of trigonometry.**

---

## Euler's identity: $e^{i\pi} + 1 = 0$

The most famous equation in mathematics. Set $\theta = \pi$ in Euler's formula:

$$
e^{i\pi} = \cos\pi + i\sin\pi = -1 + 0 = -1
$$

Rearranging:

$$
\boxed{\; e^{i\pi} + 1 = 0 \;}
$$

This identity connects five of the most important constants in mathematics in a single line:

- $e$ (from calculus, the base of the natural exponential)
- $i$ (from algebra, $\sqrt{-1}$)
- $\pi$ (from geometry, the ratio of a circle's circumference to its diameter)
- $1$ (arithmetic, the multiplicative identity)
- $0$ (arithmetic, the additive identity)

Plus three of the most important operations: exponentiation, multiplication (implicit between $i$ and $\pi$), and addition.

**Richard Feynman** called it "the most remarkable formula in mathematics." **Carl Friedrich Gauss** reportedly said that if a student could not immediately see why this identity was true, they would never be a first-rate mathematician. (Gauss was famously demanding.) A 1990 poll of readers of *The Mathematical Intelligencer* voted it the most beautiful theorem in mathematics.

The reason it feels so profound is that it connects five constants from five different branches of mathematics that had no apparent relationship. $e$ came from compound interest in banking. $i$ came from solving cubic equations. $\pi$ came from geometry. $1$ and $0$ came from counting. Each one was discovered or defined in a completely different context. And yet when you combine them in the specific form $e^{i\pi}$, you get exactly $-1$. **There is no way this is a coincidence.** The identity is a window into the deep unity of mathematics — a sign that all these apparently-separate fields are actually facets of a single underlying structure.

The specific explanation: $e^{i\pi} = -1$ because "rotating by $\pi$ radians (180°) around the unit circle takes $+1$ to $-1$." Which, once you see it geometrically, becomes obvious. But the fact that the algebraic object $e^{i\pi}$ — defined via Taylor series or compound interest — turns out to equal the geometric object "rotate by 180°" is not obvious at all until you know Euler's formula.

---

## Why this matters for ML: Fourier and positional encodings

The complex exponential is the heart of **Fourier analysis**. A Fourier series represents a periodic function as a sum of complex exponentials:

$$
f(x) = \sum_{n=-\infty}^{\infty} c_n e^{i n x}
$$

where the $c_n$ are complex coefficients. The Fourier transform, the discrete Fourier transform, the fast Fourier transform, wavelet transforms, and essentially all of signal processing are built on this one construction: decomposing functions into sums of complex exponentials at different frequencies.

The reason complex exponentials are the natural basis is **they are eigenfunctions of the derivative operator**: $\frac{d}{dx} e^{i\omega x} = i\omega \cdot e^{i\omega x}$. When you differentiate a complex exponential, you get back the same function multiplied by a constant. This makes them the "natural coordinate system" for linear differential equations — they diagonalize the derivative.

### Transformer positional encodings

Here is a direct connection to modern machine learning. In the original Transformer paper (Vaswani et al. 2017, see [../19_transformers.md](../19_transformers.md)), positional encodings are added to token embeddings to give the model information about position. The encoding for position $\text{pos}$ and dimension $i$ is:

$$
\text{PE}(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)
$$

$$
\text{PE}(\text{pos}, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)
$$

The positional encoding is a pair of $\sin$ and $\cos$ terms for each even-odd dimension pair, with different frequencies chosen geometrically. **These are the real and imaginary parts of a complex exponential** $e^{i \cdot \text{pos} \cdot \omega_i}$ for different angular frequencies $\omega_i = 10000^{-2i/d}$.

Why? Because the Vaswani team wanted the positional encodings to have a specific property: *the relative position $\text{pos}_2 - \text{pos}_1$ should be easy for the attention mechanism to extract from the sum of two positional encodings.* The Fourier-like sine/cosine encoding has exactly this property because of the angle-addition formulas that fall out of Euler's formula:

$$
e^{i\text{pos}_2 \cdot \omega} = e^{i(\text{pos}_1 + \Delta) \cdot \omega} = e^{i\text{pos}_1 \cdot \omega} \cdot e^{i\Delta \cdot \omega}
$$

The encoding for position $\text{pos}_2$ is the encoding for $\text{pos}_1$ *multiplied by a rotation that depends only on $\Delta = \text{pos}_2 - \text{pos}_1$*. This means a linear layer in the network can learn to extract $\Delta$ by looking for this rotational structure. Relative position is encoded as a rotation in the frequency dimensions of the embedding, and extracting it is a matter of undoing the rotation.

**Positional encodings work because of the functional equation of the exponential, specialized to imaginary arguments.** The 2017 Transformer architecture used Euler's 1748 formula as a design principle, whether the authors thought about it that way or not.

Later architectures like **RoPE (Rotary Position Embedding)** make the connection even more explicit: they multiply the query and key vectors by a rotation matrix derived from the complex exponential, baking position information directly into the attention computation as a rotation. RoPE is used in LLaMA, Mistral, and many other modern LLMs. It is Euler's formula, barely disguised.

---

## History in a paragraph

**1543 — Rafael Bombelli** (Italian mathematician) is forced to deal with square roots of negative numbers while solving cubic equations. He is the first to formalize operations on what would later be called imaginary numbers, though he does not know what they "mean."

**1637 — René Descartes** coins the term "imaginary" for numbers like $\sqrt{-1}$, in the pejorative sense: he thought they were mathematical fictions with no real meaning. The name stuck even after mathematicians realized they were perfectly legitimate.

**1714 — Roger Cotes** discovers an early form of Euler's formula in his work on logarithms of complex numbers: $\log(\cos\theta + i\sin\theta) = i\theta$. This is equivalent to Euler's formula but stated in terms of the log rather than the exp. Cotes died in 1716 at age 33, before he could develop the idea fully. Newton famously said of Cotes: "Had Cotes lived, we might have known something."

**1748 — Leonhard Euler** publishes *Introductio in Analysin Infinitorum*, which contains the formula $e^{i\theta} = \cos\theta + i\sin\theta$ in its modern form. Euler derives it via the Taylor series manipulation shown above. He also writes down the identity $e^{i\pi} + 1 = 0$ as a consequence.

**1799 — Caspar Wessel** (Norwegian surveyor, not a professional mathematician) publishes a paper that gives the geometric interpretation of complex numbers as points in the plane. His paper is ignored at the time.

**1806 — Jean-Robert Argand** (Swiss, also not a professional mathematician) independently publishes the same geometric interpretation. This is why the complex plane is often called the "Argand diagram."

**1831 — Carl Friedrich Gauss** publishes his own version of the geometric interpretation, lending his enormous prestige to the idea. Imaginary numbers finally become respectable.

**Mid-1800s — Augustin-Louis Cauchy and Bernhard Riemann** develop complex analysis as a full mathematical theory. The complex exponential becomes the central object of the field.

**Early 1800s onward — Joseph Fourier** develops Fourier analysis, using complex exponentials as the basic building blocks of periodic functions. This eventually becomes the foundation of signal processing, quantum mechanics, and nearly all of applied mathematics.

**20th century** — Complex exponentials become ubiquitous in physics (quantum mechanics uses them for wave functions), engineering (electrical engineering uses them for AC circuits), and pure mathematics (complex analysis, number theory).

**2017 — Vaswani et al.** use sine and cosine positional encodings in the Transformer paper, bringing Euler's formula into the heart of modern deep learning. **2021 — RoPE** makes the rotational interpretation of positional encodings explicit, and by 2024 most open-source LLMs use RoPE. Euler's 1748 formula is running inside every LLM in 2026, whether or not the engineers using it think of it that way.

---

## Takeaway

- **$e^{i\theta} = \cos\theta + i\sin\theta$.** Euler's formula. Extends the exponential from real to complex arguments and turns "growth" into "rotation."
- **The proof falls out of the Taylor series** of $e^x$, $\cos x$, and $\sin x$. The powers of $i$ distribute the terms so that real parts land in cosine and imaginary parts land in sine.
- **Multiplying by $e^{i\theta}$ is rotation by $\theta$** in the complex plane. The functional equation $e^{a+b} = e^a e^b$ becomes "angles add under composition of rotations."
- **$e^{i\pi} + 1 = 0$** — the most beautiful equation in mathematics. Connects $e$, $i$, $\pi$, $1$, and $0$ in five symbols.
- **Complex exponentials are the basis of Fourier analysis** and therefore of all signal processing, quantum mechanics, and modern wave-based physics.
- **Transformer positional encodings use $\sin$ and $\cos$** — the real and imaginary parts of $e^{i\theta}$ — because the functional equation makes relative positions easy to extract as rotations in the frequency dimensions. See [../19_transformers.md](../19_transformers.md).
- **RoPE (Rotary Position Embedding)** makes the connection explicit: it applies complex-exponential rotations directly to query and key vectors in attention. Used in LLaMA, Mistral, and most modern LLMs.

Next: [08_watchlist.md](08_watchlist.md) — the practical reference. Every place $e^x$ or $\log$ appears in ML, with a one-line explanation of what structural job it is doing.
