# 05. Why Softmax Uses $\exp$ — Shift-Invariance Uniqueness

## The question

In [../22_softmax.md](../22_softmax.md) you saw the softmax function:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

This formula is used everywhere in modern machine learning — classifier output layers, attention scores ([../23_scaled_dot_product_attention.md](../23_scaled_dot_product_attention.md)), reinforcement learning policies, language model vocabularies. It is the standard way to convert a vector of real-valued "logits" $z \in \mathbb{R}^K$ into a probability distribution over $K$ classes.

**Why does softmax use $e^{z_i}$ instead of, say, $z_i^2$, $|z_i|$, $(z_i)_+$, or any other positive-valued function?** Any function that produces positive outputs and then gets divided by its sum will produce a valid probability distribution. So why is the exponential the one that stuck?

The answer is that softmax is **forced** by a uniqueness theorem analogous to Herschel's theorem for the Gaussian. Given one very natural requirement — shift-invariance of logits — the exponential is the only continuous function that works. This note walks through the argument.

The argument matters because it explains why the same function (softmax) shows up in so many otherwise-unrelated contexts: classification, attention, policy selection, Boltzmann machines, energy-based models. They are all different applications of the same forced mathematical answer to the same forced mathematical question.

---

## Setup: what is a "logit"?

Start with the right mental model of what a logit is.

You have $K$ classes and some way of scoring each one. The scores are real numbers $z_1, z_2, \ldots, z_K$. They are called "logits" — a term that comes from logistic regression, where the logit is the log-odds of a binary outcome.

**Crucial property of logits: they are only meaningful up to an additive constant.**

Here is why. The probabilities $p_i$ must sum to 1: $\sum_i p_i = 1$. So the probabilities have only $K - 1$ degrees of freedom, not $K$. But the logit vector $z = (z_1, \ldots, z_K)$ has $K$ degrees of freedom. There is a mismatch: one degree of freedom in the logits is "redundant" and doesn't correspond to any difference in the probability distribution.

The redundancy is specifically an **additive shift**: if you add the same constant $c$ to every logit, the probability distribution should be unchanged. In symbols:

$$
(z_1, z_2, \ldots, z_K) \quad \text{and} \quad (z_1 + c, z_2 + c, \ldots, z_K + c)
$$

should produce the *same* probability distribution for any $c \in \mathbb{R}$. The absolute values of the logits carry no information — only their *differences* do.

This is analogous to how log-probabilities are only defined up to an additive constant (because probabilities are only defined up to normalization, i.e., multiplication by a constant). In log-space, the multiplicative normalization becomes an additive shift. Logits are "log-probabilities up to normalization."

**Therefore, any function that converts logits to probabilities must be invariant to adding a constant to all logits.** This is a hard constraint, not a soft preference. Any function that fails this constraint would make the probabilities depend on the arbitrary additive reference point, which makes no sense.

---

## The uniqueness theorem

**Theorem.** Let $\phi : \mathbb{R} \to \mathbb{R}_{>0}$ be a continuous function, and define

$$
p_i(z) = \frac{\phi(z_i)}{\sum_{j=1}^{K} \phi(z_j)}
$$

Suppose $p_i(z + c \cdot \mathbf{1}) = p_i(z)$ for all $z \in \mathbb{R}^K$, all $c \in \mathbb{R}$, and all $i \in \{1, \ldots, K\}$ (where $\mathbf{1}$ is the all-ones vector). Then $\phi(z) = a \cdot e^{bz}$ for some constants $a > 0$ and $b \in \mathbb{R}$.

**In plain English:** the only continuous function $\phi$ that gives shift-invariant probabilities via the "divide by the sum" construction is the exponential (up to the free constants $a$ and $b$, which cancel in the final formula anyway).

So softmax is not one of many possible shift-invariant normalizations. It is the *only* one.

### Proof

Start with the shift-invariance condition:

$$
\frac{\phi(z_i + c)}{\sum_j \phi(z_j + c)} = \frac{\phi(z_i)}{\sum_j \phi(z_j)}
$$

for all $z$, $c$, and $i$. Cross-multiply:

$$
\phi(z_i + c) \cdot \sum_j \phi(z_j) = \phi(z_i) \cdot \sum_j \phi(z_j + c)
$$

Rearrange:

$$
\frac{\phi(z_i + c)}{\phi(z_i)} = \frac{\sum_j \phi(z_j + c)}{\sum_j \phi(z_j)}
$$

The right-hand side does not depend on $i$ (it is the same for every choice of $i$, since summing over $j$ doesn't see $i$). So the left-hand side, $\phi(z_i + c)/\phi(z_i)$, must also not depend on $i$ — which means it is a function of $c$ alone. Call this function $g(c)$:

$$
\frac{\phi(z + c)}{\phi(z)} = g(c) \quad \text{for all } z, c \in \mathbb{R}
$$

Equivalently:

$$
\phi(z + c) = g(c) \cdot \phi(z)
$$

This is a functional equation for $\phi$. Setting $z = 0$:

$$
\phi(c) = g(c) \cdot \phi(0)
$$

So $g(c) = \phi(c) / \phi(0)$. Substituting back:

$$
\phi(z + c) = \frac{\phi(c)}{\phi(0)} \cdot \phi(z)
$$

Multiply both sides by $\phi(0)$:

$$
\phi(0) \cdot \phi(z + c) = \phi(z) \cdot \phi(c)
$$

Define $\psi(z) = \phi(z) / \phi(0)$. Then:

$$
\psi(z + c) = \psi(z) \cdot \psi(c)
$$

This is the **functional equation from [03_the_functional_equation.md](03_the_functional_equation.md)**: $f(a + b) = f(a) f(b)$. By the uniqueness theorem proved there, the only continuous solutions are $\psi(z) = e^{bz}$ for some constant $b$.

Therefore $\phi(z) = \phi(0) \cdot e^{bz} = a \cdot e^{bz}$ where $a = \phi(0)$. $\blacksquare$

### Why the constants $a$ and $b$ drop out

Plugging $\phi(z) = a e^{bz}$ back into the softmax formula:

$$
p_i = \frac{a e^{b z_i}}{\sum_j a e^{b z_j}} = \frac{e^{b z_i}}{\sum_j e^{b z_j}}
$$

The constant $a$ cancels immediately (numerator and denominator).

The constant $b$ does not cancel, but it is equivalent to rescaling the logits: if you replace $z_i$ with $b z_i$, the formula becomes the standard softmax with $b = 1$. In practice, $b$ is absorbed into an additional hyperparameter called **temperature**:

$$
\text{softmax}_T(z)_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

where $T > 0$ is the temperature. Low temperature (small $T$) makes the distribution sharp (concentrated on the largest logit); high temperature makes it flat (close to uniform). This is the same "temperature" that appears in Boltzmann distributions in physics, and the connection is not a coincidence — see [06_maximum_entropy_and_exp_family.md](06_maximum_entropy_and_exp_family.md).

---

## What this uniqueness theorem rules out

To appreciate how strong the theorem is, consider what would happen if you tried to use a different function in place of $e^{z_i}$. Any of these would produce a valid probability distribution (positive, sums to 1), but none would be shift-invariant:

| Candidate $\phi(z)$ | Shift-invariant? | Why it fails |
|---|---|---|
| $\phi(z) = z^2$ | No | Adding a constant changes the squared values non-uniformly |
| $\phi(z) = z^2 + 1$ | No | Same problem |
| $\phi(z) = \|z\|$ | No | Adding a large constant swamps the logit differences |
| $\phi(z) = \max(z, 0)$ | No | Shifting can make all logits negative, zeroing the distribution |
| $\phi(z) = (z - \min_j z_j)$ | No (also not continuous as a function of $z_i$ alone) | Not purely a function of $z_i$ |
| $\phi(z) = \exp(z)$ | **Yes** | Forced by the theorem |

The only continuous function that plays nicely with shift-invariance is the exponential. **Softmax uses exp because the constraint "logits are only meaningful up to an additive shift" forces it.**

This is a structural fact about classification. It is not a design choice made by a specific ML researcher in the 1980s. Anyone starting from "I have $K$ logits and I want to convert them to a probability distribution in a way that respects the inherent shift-invariance of log-probabilities" will end up at softmax, because softmax is the only continuous function that works.

---

## The log-sum-exp connection

Softmax's denominator, $\sum_j e^{z_j}$, is the exponential of a quantity called **log-sum-exp**:

$$
\text{LSE}(z) = \log \sum_j e^{z_j}
$$

The log-sum-exp function has its own beautiful properties:

1. **It is a smooth approximation to the max function.** As the logit gaps grow, $\text{LSE}(z) \to \max_j z_j$. As the gaps shrink to zero, $\text{LSE}(z) \to \log K + \text{avg}(z)$. So LSE smoothly interpolates between "take the max" and "take the average," depending on how peaked the distribution is.
2. **It is the log-partition function of the Boltzmann distribution.** $Z = \sum_j e^{z_j}$ is called the "partition function" in physics. Its log is a free energy.
3. **It is the gradient of softmax has a clean form.** $\nabla_{z_i} \text{LSE}(z) = p_i$, the $i$-th softmax probability.

The log-sum-exp trick is the standard numerical technique for computing softmax stably:

$$
\text{LSE}(z) = z^* + \log \sum_j e^{z_j - z^*}, \quad z^* = \max_j z_j
$$

Subtracting $z^*$ before exponentiating prevents overflow when some $z_j$ is very large. Every deep learning framework implements softmax this way internally. The justification is shift-invariance itself: subtracting the max from all logits is a shift by $-z^*$, which by the uniqueness theorem does not change the probabilities, but it makes the numerics safe.

**Shift-invariance of softmax is not just a theoretical property — it is actively exploited in every real implementation to keep the computation numerically stable.** The same theorem that forces the exponential also enables the numerical trick.

---

## Historical note: Bridle 1989

The softmax function was used informally in various forms before 1989 — the logistic regression community had been using the binary case ($K = 2$), which reduces to the sigmoid $\sigma(x) = 1/(1 + e^{-x})$, since the 1940s. Softmax-like constructions also appeared in statistical mechanics going back to Boltzmann's 1872 distribution.

The name "softmax" and its modern treatment as a neural network output layer are due to **John Bridle**, who published a paper in 1989 titled *Probabilistic Interpretation of Feedforward Classification Network Outputs, with Relationships to Statistical Pattern Recognition*. Bridle's paper did two things:

1. **It named the function "softmax."** The name is meant to suggest "a smooth (differentiable) version of the argmax function" — a function that tells you which class has the highest logit, but in a soft, probability-weighted way rather than a hard, deterministic way.

2. **It derived softmax from the Gibbs distribution.** Bridle explicitly noted that softmax is the Boltzmann distribution $p \propto e^{-E/T}$ applied to $-z_i$ as the "energy" of class $i$. This made the physics-to-ML lineage unmistakable: the exponential in softmax is the same exponential in Boltzmann's 1872 velocity distribution, applied to a different energy function.

Bridle was not the first to use the function in a neural network — it had appeared in Hopfield networks, Boltzmann machines (Hinton & Sejnowski, 1985), and competitive learning algorithms throughout the 1980s. But Bridle was the first to give it the name and the rigorous probabilistic interpretation that made it the standard output layer for classification networks going forward.

Today every language model, every image classifier, every attention mechanism, every policy network in reinforcement learning uses softmax in essentially the form Bridle gave it in 1989. **The exponential in that formula is doing the same structural work Maxwell exploited in 1860 and Boltzmann formalized in 1872 — it is the unique continuous function consistent with the underlying symmetries of the problem.**

---

## Softmax inside attention

A quick preview of the most consequential modern application. In the Transformer's attention mechanism (see [../23_scaled_dot_product_attention.md](../23_scaled_dot_product_attention.md)):

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

The scaled dot products $QK^\top/\sqrt{d_k}$ are logits. Softmax converts them into a probability distribution over past positions (for each query). Then this probability distribution weights the value vectors.

Why softmax here specifically? Because the dot products are real numbers with arbitrary additive offsets — shifting them all by the same constant does not change the "who attends to whom" relationships. Softmax is the forced shift-invariant normalization. Every attention head in every transformer uses softmax for exactly this reason.

The attention mechanism, which powers every LLM on the planet as of 2026, is running the same uniqueness argument every time it computes a forward pass. **There are approximately $10^{26}$ softmax operations performed per day across all of OpenAI, Anthropic, Google, and Meta's inference fleets in 2026.** Every one of them is exploiting the theorem in this note.

---

## Takeaway

- **Softmax uses $e^{z_i}$ because the exponential is the unique continuous function consistent with shift-invariance of logits.** Logits are log-probabilities up to an additive constant, and the normalization must respect this symmetry. The uniqueness argument is a corollary of the functional-equation theorem from [03_the_functional_equation.md](03_the_functional_equation.md).
- **The constants $a$ and $b$ drop out** of the final softmax formula — $a$ cancels in numerator and denominator, and $b$ becomes the temperature hyperparameter.
- **Log-sum-exp is softmax's "partition function,"** and subtracting the max before exponentiating is the standard numerical trick for stable implementation. The trick itself is an application of shift-invariance.
- **Bridle 1989** gave softmax its name and derived it from Boltzmann's 1872 distribution, completing a direct lineage from 19th-century statistical mechanics to modern neural networks.
- **Softmax is not a design choice — it is forced by symmetry.** Same way the Gaussian is forced by Herschel's theorem, softmax is forced by shift-invariance.

Next: [06_maximum_entropy_and_exp_family.md](06_maximum_entropy_and_exp_family.md) — the general principle behind both the Gaussian and softmax: maximum entropy forces exponential families.
