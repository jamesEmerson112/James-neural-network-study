# 08. Watchlist — Where $\exp$ and $\log$ Appear in ML

This is the practical reference. No history, no derivations, no proofs. For every place the exponential or logarithm shows up in machine learning, here is a one-line statement of what structural job it is doing.

Every entry falls into one of **three categories**, which are the only three jobs the exponential ever does:

- **(a) Max entropy under a constraint.** The exp appears because you're computing (or using) the max-entropy distribution for some set of constraints. See [06_maximum_entropy_and_exp_family.md](06_maximum_entropy_and_exp_family.md).
- **(b) Preserving independence under combination.** The exp appears because independent things multiply and additive things add, and the exp is the unique bridge between the two. See [03_the_functional_equation.md](03_the_functional_equation.md).
- **(c) Rotating or compounding continuously.** The exp appears because the process grows at a rate proportional to its current state, or because something is rotating in the complex plane. See [01_the_exponential.md](01_the_exponential.md) and [07_eulers_formula_and_rotation.md](07_eulers_formula_and_rotation.md).

When you see $e^x$ or $\log$ in a paper, stop for three seconds and identify which of the three categories it belongs to. After a few weeks of doing this, the structural joints of the entire field become visible.

---

## Probability distributions

**Gaussian PDF.** $p(x) \propto \exp(-(x-\mu)^2/(2\sigma^2))$. Category **(a)**: max-entropy distribution with fixed mean and variance, equivalently forced by Herschel's independence+rotational-symmetry theorem. See [../ddpm/02_what_is_a_gaussian.md](../ddpm/02_what_is_a_gaussian.md) and [04_why_the_gaussian_has_exp.md](04_why_the_gaussian_has_exp.md).

**Multivariate Gaussian.** $p(x) \propto \exp(-\tfrac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu))$. Category **(a)**: same as above, generalized to multiple dimensions with a covariance matrix. The exp lets the density factor into a product across whitened coordinates — category **(b)** is also at work.

**Exponential distribution.** $p(x) = \lambda e^{-\lambda x}$ on $[0, \infty)$. Category **(a)**: max-entropy distribution on the positive reals with fixed mean. Models waiting times between Poisson events.

**Poisson distribution.** $p(k) = \lambda^k e^{-\lambda}/k!$. Category **(a)**: max-entropy distribution for counts with fixed rate. Models the number of rare independent events in a fixed interval.

**Boltzmann distribution.** $p(\text{state}) \propto e^{-E(\text{state})/k_B T}$. Category **(a)**: max-entropy distribution with fixed average energy. The archetypal "physics" distribution. See [../10b_boltzmann_machines_energy_and_randomness.md](../10b_boltzmann_machines_energy_and_randomness.md).

**Softmax / categorical distribution.** $p_i = e^{z_i}/\sum_j e^{z_j}$. Category **(a)**: max-entropy distribution over $K$ classes, also uniquely forced by shift-invariance of logits. See [../22_softmax.md](../22_softmax.md) and [05_why_softmax_uses_exp.md](05_why_softmax_uses_exp.md).

**Bernoulli / logistic.** $p(y=1 | x) = \sigma(w^\top x) = 1/(1 + e^{-w^\top x})$. Category **(a)**: the two-class case of softmax.

**Dirichlet.** $p(x) \propto \prod_i x_i^{\alpha_i - 1}$. Strictly speaking this is not "exp of something" — but the log turns it into $\sum_i (\alpha_i - 1) \log x_i$, which is the exp of a linear function of $\log x_i$. The Dirichlet is in the exponential family via log-space coordinates.

---

## Neural network activations

**Sigmoid.** $\sigma(x) = 1/(1 + e^{-x})$. Category **(a)**: the two-class softmax, which is max-entropy for a binary classification constraint. Historically the most common activation function in pre-2012 neural networks; still used in output layers for binary classification and inside LSTM/GRU gates.

**Tanh.** $\tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})$. A rescaled sigmoid: $\tanh(x) = 2\sigma(2x) - 1$. Category **(a)** via the sigmoid. Used in RNN hidden states, LSTM outputs, and as a general "squashing" function. See [../13_vanishing_gradient_and_tanh.md](../13_vanishing_gradient_and_tanh.md).

**GELU.** $\text{GELU}(x) = x \cdot \Phi(x)$ where $\Phi$ is the Gaussian CDF, which contains $\int e^{-t^2/2} dt$ inside it. Category **(a)** via the Gaussian. The default activation in modern Transformers (BERT, GPT-2, GPT-3, and later).

**Swish / SiLU.** $\text{swish}(x) = x \cdot \sigma(x)$. Category **(a)** via the sigmoid. A smoother alternative to ReLU, used in EfficientNet and some Transformer variants.

**Softplus.** $\text{softplus}(x) = \log(1 + e^x)$. Category **(b)**: a smooth approximation to ReLU, where the log-exp pair converts a "max with zero" operation into a differentiable function.

---

## Losses

**Cross-entropy loss.** $\mathcal{L} = -\sum_i y_i \log p_i$. Category **(b)**: the log is turning the product of predicted probabilities over independent samples into a sum. Also category **(a)**: cross-entropy is the negative log-likelihood of a categorical distribution, which is a max-entropy object.

**Binary cross-entropy.** $\mathcal{L} = -[y \log p + (1-y) \log(1-p)]$. Category **(b)**: same as cross-entropy but for the two-class case.

**Negative log-likelihood (NLL).** $\mathcal{L}(\theta) = -\sum_i \log p_\theta(x_i)$. Category **(b)**: the log turns the product-of-independent-likelihoods into a sum that can be summed over batches and differentiated term-by-term. Every maximum-likelihood method in ML is running this.

**KL divergence.** $D_{KL}(p \| q) = \sum_i p_i \log(p_i / q_i)$. Category **(b)**: the log of the ratio $p/q$ is the difference $\log p - \log q$, which decomposes the KL into the expectation of a log-likelihood ratio. Used in variational inference, policy gradient methods, distillation losses.

**Entropy regularization.** $\mathcal{L} + \beta H(p_\theta)$ where $H(p) = -\sum p \log p$. Category **(a)**: encourages the policy or distribution to stay close to the max-entropy distribution (spreading probability mass uniformly). Used in reinforcement learning to encourage exploration.

**Focal loss.** $\mathcal{L} = -(1 - p_t)^\gamma \log(p_t)$. Category **(b)** via the $\log$. A modification of cross-entropy that down-weights easy examples; used in object detection (RetinaNet).

---

## Attention and Transformers

**Scaled dot-product attention.** $\text{Attention}(Q, K, V) = \text{softmax}(QK^\top/\sqrt{d_k})V$. Category **(a)** via the softmax, which is the max-entropy distribution over past positions given the dot-product scores. See [../23_scaled_dot_product_attention.md](../23_scaled_dot_product_attention.md).

**Sinusoidal positional encodings.** $\text{PE}(\text{pos}, 2i) = \sin(\text{pos}/10000^{2i/d})$, $\text{PE}(\text{pos}, 2i+1) = \cos(\text{pos}/10000^{2i/d})$. Category **(c)**: real and imaginary parts of the complex exponential $e^{i \cdot \text{pos} \cdot \omega_i}$. The functional equation makes relative positions extractable as rotations. See [07_eulers_formula_and_rotation.md](07_eulers_formula_and_rotation.md) and [../19_transformers.md](../19_transformers.md).

**Rotary Position Embeddings (RoPE).** Rotates query and key vectors by an angle proportional to their position before computing attention. Category **(c)**: directly uses the complex-exponential rotation from Euler's formula. Used in LLaMA, Mistral, Gemma, and most open-source LLMs from 2023 onward.

**Temperature-scaled softmax.** $\text{softmax}_T(z)_i = e^{z_i/T}/\sum_j e^{z_j/T}$. Category **(a)**: the Boltzmann distribution with temperature $T$ as the Lagrange multiplier for the energy constraint. Low $T$ makes the distribution sharp; high $T$ makes it flat. Used in knowledge distillation, language model sampling, and reinforcement learning policies.

---

## Reinforcement learning

**Discounted returns.** $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$, where $\gamma^k = e^{k \ln \gamma}$. Category **(c)**: exponential decay of future rewards, controlled by the discount factor $\gamma \in (0, 1)$.

**Boltzmann exploration (softmax policy).** $\pi(a | s) = e^{Q(s,a)/T}/\sum_{a'} e^{Q(s,a')/T}$. Category **(a)**: max-entropy policy given the Q-values as pseudo-energies. See [../25_reinforcement_learning_overview.md](../25_reinforcement_learning_overview.md).

**Maximum entropy RL (SAC, PPO with entropy bonus).** Adds an entropy term $\alpha H(\pi)$ to the reward. Category **(a)**: encourages the policy to remain close to uniform, for exploration and robustness.

**Policy gradient with log-derivative trick.** $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot R]$. Category **(b)**: the log-derivative identity $\nabla \log p = \nabla p / p$ is what lets you sample from $p$ and still compute an unbiased gradient. Without the log, REINFORCE doesn't work.

---

## Generative models

**Variational Autoencoder (VAE) ELBO.** $\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$. Category **(b)** via the logs and the KL term. The log-likelihood term is the reconstruction error (often Gaussian, hence exp-in-disguise); the KL term regularizes the encoder toward a Gaussian prior.

**Reparameterization trick.** $z = \mu + \sigma \cdot \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, I)$. Category **(a)**: exploits the Gaussian's exponential form to sample through a differentiable operation. See [../ddpm/04_visualizing_one_noising_step.md](../ddpm/04_visualizing_one_noising_step.md).

**Diffusion forward process.** $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1}, \beta_t \mathbf{I})$. Category **(a)**: Gaussian noise at every step. See [../ddpm/03_forward_process.md](../ddpm/03_forward_process.md).

**Diffusion noise schedules.** Cosine schedules (e.g., in improved DDPM) use $\bar{\alpha}_t = \cos^2(\ldots)$, which are related to the complex exponential via Euler's formula. Linear and geometric schedules use exponential decay in the variance. Category **(c)**: exponential or cosine-based decay.

**Energy-based models.** $p(x) \propto \exp(-E_\theta(x))$. Category **(a)**: the Gibbs distribution with a learned energy function. The exponential is forced by the Boltzmann form. Covers Boltzmann machines, restricted Boltzmann machines, and modern score-based / diffusion models (which are energy-based models in disguise). See [../24_generative_models_taxonomy.md](../24_generative_models_taxonomy.md).

**Normalizing flows.** $\log p_X(x) = \log p_Z(f(x)) + \log|\det \partial f/\partial x|$. Category **(b)**: the change-of-variables formula involves a log-determinant of the Jacobian, which comes from the log of the absolute value of the determinant of a matrix. Every flow model trains on this log-determinant.

**Score matching.** $\nabla_x \log p(x)$ is the "score function." Category **(b)**: the gradient of the log-density, which is the derivative of the log (hence the identity $\nabla \log p = \nabla p / p$). Score-based generative models (Yang Song's work) learn the score function, from which the exponential density can be reconstructed.

**Langevin dynamics.** $x_{t+1} = x_t + \eta \nabla \log p(x_t) + \sqrt{2\eta}\,\varepsilon$. Category **(c)**: a stochastic process whose stationary distribution is $p(x) \propto e^{-E(x)}$ for $E = -\log p$. Used in score-based generative models and MCMC sampling.

---

## Natural language and embeddings

**Word2vec softmax.** The output of the word2vec model is a softmax over the vocabulary: $p(w_c | w) = e^{v_c^\top u_w}/\sum_{c'} e^{v_{c'}^\top u_w}$. Category **(a)**: max-entropy distribution over vocabulary conditioned on context. Since the denominator sums over the entire vocabulary (hundreds of thousands of words), training is expensive, which motivates negative sampling as an approximation.

**Negative sampling (in word2vec, noise contrastive estimation).** Replaces the full softmax with a binary classification: "is this a true context-word pair or a random one?" The loss involves $\log \sigma(v_c^\top u_w) + \sum_{i} \log \sigma(-v_{n_i}^\top u_w)$. Category **(b)**: logs of sigmoids, which themselves contain exponentials.

**Language model perplexity.** $\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_i \log p_\theta(x_i)\right)$. Category **(a)**: the exponential of the negative average log-likelihood, which converts a log-space quantity (cross-entropy) back to a multiplicative quantity (an effective vocabulary size). Perplexity of $n$ means "the model is as uncertain as if it were choosing uniformly among $n$ words."

---

## Numerical / algorithmic

**Log-sum-exp trick.** $\log \sum_i e^{x_i} = x^* + \log \sum_i e^{x_i - x^*}$ where $x^* = \max_i x_i$. Category **(b)**: a numerical technique for computing the log-partition function stably. The trick is an application of shift-invariance, which in turn comes from the functional equation. Every deep learning framework implements softmax via log-sum-exp internally.

**Log-probabilities throughout LLM inference.** Every modern LLM does its internal computations in log-probability space to avoid numerical underflow. Sampling, beam search, and speculative decoding all operate on logits / log-probs. Category **(b)**: logs turn multiplication of small probabilities into addition of manageable numbers.

**Gumbel-max trick.** To sample from a categorical distribution, add Gumbel noise to the logits and take the argmax: $i^* = \arg\max_i (z_i + g_i)$ where $g_i \sim \text{Gumbel}(0, 1)$. Category **(a)**: because the Gumbel distribution has $p(g) = e^{-(g + e^{-g})}$ — exponential in the exponent. The resulting distribution over $i^*$ is exactly the softmax.

---

## The three-job compression

Here is the whole note in one sentence:

> **Every time you see $e^x$ or $\log$ in a machine learning paper, it is doing one of three jobs: (a) expressing a max-entropy distribution, (b) converting products to sums or vice versa via the functional equation, or (c) describing a compounding or rotating process.**

Specifically:

- **If the exp is inside a probability density or normalization**, it is job (a): max entropy.
- **If the exp or log is inside a loss function or a derivative trick**, it is job (b): functional equation.
- **If the exp is controlling dynamics over time or a rotation**, it is job (c): compounding or rotating.

You will sometimes see an exp doing two jobs at once (e.g., the Gaussian in a VAE ELBO is max-entropy AND the log of it turns products into sums in the loss), but every instance boils down to this three-way categorization.

---

## What you should now be able to do

Read any ML paper and, when you see $\exp$ or $\log$:

1. **Stop for three seconds.**
2. **Identify which of the three jobs the exp/log is doing.**
3. **Recognize which specific mechanism from notes 04, 05, 06, or 07 is at play.**
4. **Proceed.**

That three-second pause is the difference between reading a paper and *understanding* a paper. Without the pause, the exp looks like arbitrary notation. With the pause, it looks like the inevitable consequence of a structural requirement — and you know what that requirement is.

---

## Takeaway

- **Three jobs.** Every exp and log in ML is either (a) max entropy, (b) functional equation, or (c) compounding/rotating.
- **The Gaussian, softmax, sigmoid, GELU, cross-entropy, KL, attention scores, positional encodings, discount factors, ELBO, energy-based models, normalizing flows, and Langevin dynamics** are all applications of one or more of these three jobs.
- **The watchlist is finite and closes here.** You have now seen essentially every place exp and log appear in modern machine learning. When you encounter a new paper, the exp will fit into one of these categories.
- **The three-second pause** — stopping to identify which job the exp is doing — is the habit that turns paper-reading from pattern matching into structural understanding.
