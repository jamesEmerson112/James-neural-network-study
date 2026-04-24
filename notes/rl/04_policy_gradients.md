# 04. Policy Gradient Methods

*Source: UW CSE 579, Policy Gradients lecture notes*

## What this note unpacks

Value-based methods (Q-Learning, DQN) learn $Q^*$ and act greedily. Policy gradient methods skip $Q$ entirely and optimize the policy $\pi_\theta$ directly by gradient ascent on expected return. The question: how do you differentiate through a stochastic trajectory?

This note covers:
- The objective: expected total reward $J(\theta)$
- The likelihood ratio trick — turning an intractable gradient into a tractable expectation
- Why you don't need to know the environment dynamics
- REINFORCE algorithm
- Variance reduction: reward-to-go, baselines, and the advantage function
- The Policy Gradient Theorem (state-distribution form)
- Actor-Critic methods
- Natural Policy Gradient: Fisher Information Metric, KL-divergence, and why vanilla gradients fail with correlated features

---

## The objective

A trajectory $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T, a_T)$ under policy $\pi_\theta$ has total reward:

$$R(\tau) = \sum_{t=0}^{T-1} r(s_t, a_t)$$

The objective is to maximize expected total reward:

$$J(\theta) = \mathbb{E}_{p(\tau|\theta)}\!\left[R(\tau)\right] = \sum_{\tau} p(\tau|\theta)\, R(\tau)$$

We want $\nabla_\theta J(\theta)$ so we can do gradient ascent: $\theta \leftarrow \theta + \alpha\, \nabla_\theta J$.

---

## The likelihood ratio trick

### The problem

$$\nabla_\theta J = \sum_{\tau} \nabla_\theta p(\tau|\theta)\, R(\tau)$$

This sum is over all possible trajectories — exponentially many. You can't estimate it by sampling because $\nabla_\theta p(\tau|\theta)$ is not an expectation under $p(\tau|\theta)$.

### The trick

Multiply and divide by $p(\tau|\theta)$:

$$\nabla_\theta J = \sum_{\tau} p(\tau|\theta) \cdot \frac{\nabla_\theta p(\tau|\theta)}{p(\tau|\theta)} \cdot R(\tau)$$

Recognize $\frac{\nabla_\theta p}{p} = \nabla_\theta \log p$ (chain rule of logarithm):

$$\boxed{\nabla_\theta J = \mathbb{E}_{p(\tau|\theta)}\!\left[\nabla_\theta \log p(\tau|\theta) \cdot R(\tau)\right]}$$

Now it's an expectation — estimate it by sampling trajectories and averaging.

**Interpretation**: increase the log-probability of trajectories with high reward, decrease the log-probability of trajectories with low reward. "Do more of what works."

---

## The dynamics cancel out

A trajectory's probability factorizes (Markov property):

$$p(\tau|\theta) = p(s_0) \prod_{t=0}^{T-2} p(s_{t+1}|s_t, a_t) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t)$$

Taking the log:

$$\log p(\tau|\theta) = \log p(s_0) + \sum_{t=0}^{T-2} \log p(s_{t+1}|s_t, a_t) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t)$$

Taking $\nabla_\theta$: the first two terms don't depend on $\theta$, so their gradients are zero.

$$\nabla_\theta \log p(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

**You don't need to know the transition model.** The gradient depends only on $\nabla_\theta \log \pi_\theta$ — the gradient of your own policy. This is computable via backprop.

Final policy gradient:

$$\nabla_\theta J = \mathbb{E}_{p(\tau|\theta)}\!\left[\left(\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\right) R(\tau)\right]$$

Estimated from $N$ sampled trajectories:

$$\widetilde{\nabla}_\theta J = \frac{1}{N}\sum_{i=1}^{N}\!\left[\left(\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)})\right) R(\tau^{(i)})\right]$$

---

## REINFORCE

```
Initialize policy parameters θ
repeat:
    1. Collect N trajectories {τ⁽ⁱ⁾} by running π_θ in the environment
    2. Compute estimated gradient:
       ∇̃J = (1/N) Σᵢ [ (Σₜ ∇_θ log π_θ(aₜ⁽ⁱ⁾|sₜ⁽ⁱ⁾)) R(τ⁽ⁱ⁾) ]
    3. Update: θ ← θ + α ∇̃J
until converged (or bored)
```

This is an unbiased estimate of the true policy gradient (by the law of large numbers). But the variance is brutal — more on that below.

---

## Worked example — Tetris with Boltzmann policy

Features: $f_1$ = number of holes after placement, $f_2$ = height of tallest column after placement, etc.

**Boltzmann (softmax) policy:**

$$\pi_\theta(a|s) = \frac{\exp(\theta^\top f(s,a))}{\sum_{a'} \exp(\theta^\top f(s,a'))}$$

**Gradient of log-policy** (analytically tractable for this form):

$$\nabla_\theta \log \pi_\theta(a|s) = f(s,a) - \mathbb{E}_{\pi_\theta(a'|s)}\!\left[f(s,a')\right]$$

Reading: the gradient is the difference between the feature of the chosen action and the expected feature over all actions. If the chosen action's feature is above average and the trajectory gets high reward, the gradient pushes $\theta$ to increase the weight on that feature.

### Numerical example

State $s$, two actions $a_0$ (bad) and $a_1$ (good). Features: $f(s, a_0) = 3$, $f(s, a_1) = 1$. Current $\theta = 1$.

Probabilities:

$$\pi_\theta(a_0|s) = \frac{e^3}{e^3 + e} = \frac{e^2}{e^2 + 1} \approx 0.88$$

$$\pi_\theta(a_1|s) = \frac{e}{e^3 + e} = \frac{1}{e^2 + 1} \approx 0.12$$

The policy heavily favors $a_0$ (the bad action) because its feature value is larger.

Expected feature: $\mathbb{E}[f] \approx 0.88 \times 3 + 0.12 \times 1 = 2.76$

Q-values from critic: $Q^\pi(s, a_0) = 1$, $Q^\pi(s, a_1) = 100$.

Gradient estimate:

$$\widetilde{\nabla}_\theta J = 0.88 \times (3 - 2.76) \times 1 + 0.12 \times (1 - 2.76) \times 100 \approx -20.79$$

The gradient is negative — it says **decrease $\theta$**. This makes the probability of $a_1$ (the good action with lower feature) go up, which is correct. The algorithm discovered that high feature values correlate with low reward.

---

## Variance reduction

### The problem

REINFORCE is unbiased but has enormous variance. Imagine all sampled trajectories have positive reward (some just higher than others). The gradient pushes up the probability of *all* of them. With finite samples, the estimated gradient points in random directions each iteration.

### Fix 1: Reward-to-go (causality)

Actions at time $t$ can't affect past rewards. Replace $R(\tau)$ with future reward:

$$\nabla_\theta J = \mathbb{E}\!\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \left(\sum_{t'=t}^{T-1} r(s_{t'}, a_{t'})\right)\right]$$

The inner sum $\sum_{t'=t}^{T-1} r(s_{t'}, a_{t'})$ is the **reward-to-go** from time $t$. This removes irrelevant past rewards from the gradient, reducing variance without introducing bias.

### Fix 2: Baselines

Subtract a baseline $b(s_t)$ from the reward-to-go:

$$\nabla_\theta J = \mathbb{E}\!\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \left(\sum_{t'=t}^{T-1} r(s_{t'}, a_{t'}) - b(s_t)\right)\right]$$

**This is still unbiased.** Proof: $b$ doesn't depend on $\theta$, so:

$$\mathbb{E}\!\left[\nabla_\theta \log p(\tau|\theta) \cdot b\right] = \sum_\tau \nabla_\theta p(\tau|\theta) \cdot b = b \cdot \nabla_\theta\!\left(\sum_\tau p(\tau|\theta)\right) = b \cdot \nabla_\theta 1 = 0$$

The baseline must not depend on $\theta$ (but it can depend on the state). A good baseline approximates the average performance — then the gradient only responds to *relative* reward, not absolute level.

A common choice: $b(s_t) = V^\pi(s_t)$, or an estimate of it.

---

## The Policy Gradient Theorem

Replace the trajectory-based reward estimate with the Q-function:

$$\nabla_\theta J = \mathbb{E}_{s \sim d_{\pi_\theta}(s),\, a \sim \pi_\theta(a|s)}\!\left[\nabla_\theta \log \pi_\theta(a|s)\, Q^{\pi_\theta}(s, a)\right]$$

where $d_{\pi_\theta}(s) = \frac{1}{T}\sum_{t=0}^{T-1} p_{\pi_\theta}(s, t)$ is the **state distribution** under $\pi_\theta$ — the fraction of time spent in each state.

### The advantage function

Since $V^{\pi_\theta}(s)$ is only a function of state (not action), it can serve as a baseline:

$$\mathbb{E}_{\pi_\theta(a|s)}\!\left[\nabla_\theta \log \pi_\theta(a|s) \cdot V^{\pi_\theta}(s)\right] = V^{\pi_\theta}(s) \cdot \underbrace{\mathbb{E}_{\pi_\theta(a|s)}\!\left[\nabla_\theta \log \pi_\theta(a|s)\right]}_{= 0}$$

The last step: $\sum_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) = \sum_a \nabla_\theta \pi_\theta(a|s) = \nabla_\theta \sum_a \pi_\theta(a|s) = \nabla_\theta 1 = 0$.

So we can subtract $V$ from $Q$ for free:

$$\boxed{\nabla_\theta J = \mathbb{E}_{d_{\pi_\theta},\, \pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A^{\pi_\theta}(s, a)\right]}$$

where:

$$A^{\pi_\theta}(s, a) = Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)$$

is the **advantage** — how much better action $a$ is compared to the average action from state $s$. Positive advantage = better than average, negative = worse.

The gradient says: make actions with positive advantage more likely, negative advantage less likely.

---

## Actor-Critic methods

The Policy Gradient Theorem connects the gradient to estimating $Q^\pi$ or $A^\pi$. If we approximate these with a learned function, we get **Actor-Critic**:

- **Actor**: the policy $\pi_\theta$ — chooses actions
- **Critic**: an estimate $\hat{A}^\pi_\phi(s, a)$ or $\hat{Q}^\pi_\phi(s, a)$ — evaluates how good those actions were

Gradient estimate:

$$\widetilde{\nabla}_\theta J = \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta \log \pi_\theta(a_i|s_i)\, \hat{A}^\pi_\phi(s_i, a_i)$$

By introducing the critic, variance is further reduced compared to pure REINFORCE (which estimates $Q$ from raw trajectory returns). Most modern policy gradient methods — TRPO, PPO, DDPG — are Actor-Critic architectures.

---

## Natural Policy Gradient

### The problem with vanilla gradient descent

Gradient descent measures "small step" by the L2 norm in parameter space: $\|\Delta\theta\|_2 \leq \epsilon$. But this depends on the parameterization.

**Example**: two parameterizations of Tetris:
- Parameterization 1: $f_1$ = holes, $f_2$ = height. Parameters: $\theta_1, \theta_2$.
- Parameterization 2: $g_1 = g_2 = \cdots = g_{100}$ = holes, $g_{101}$ = height. Parameters: $\phi_1, \ldots, \phi_{101}$.

Both represent the same policy space. But $\nabla_{\phi_1} J = \cdots = \nabla_{\phi_{100}} J = \nabla_{\theta_1} J$, so Parameterization 2 takes a step **100× larger** in the "holes" direction. The gradient ascent algorithm's behavior changes just because of how we numbered the parameters — that's a problem.

### The fix: measure distance in policy space, not parameter space

Replace the L2 norm $\Delta\theta^\top \Delta\theta$ with a metric $\Delta\theta^\top G(\theta) \Delta\theta$ where $G(\theta)$ captures how much the **policy** (not just the parameters) changes.

The steepest ascent problem becomes:

$$\max_{\Delta\theta}\, J(\theta + \Delta\theta) \quad \text{s.t.} \quad \Delta\theta^\top G(\theta)\, \Delta\theta \leq \epsilon$$

Solving via Lagrange multipliers with first-order Taylor approximation of $J$:

$$\Delta\theta = \frac{1}{2\lambda}\, G^{-1}(\theta)\, \nabla_\theta J$$

### What metric to use? The Fisher Information Metric.

**Chentsov's theorem** says there is a unique metric on parametric probability distributions that is invariant to reparameterization. It's the **Fisher Information Metric**:

$$G(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\nabla_\theta \log \pi_\theta \cdot (\nabla_\theta \log \pi_\theta)^\top\right]$$

Equivalently, $G(\theta)$ is the second-order approximation of the KL divergence between $\pi_\theta$ and $\pi_{\theta+\Delta\theta}$:

$$\text{KL}(\pi_{\theta+\Delta\theta} \| \pi_\theta) \approx \Delta\theta^\top G(\theta)\, \Delta\theta$$

So the constraint $\Delta\theta^\top G(\theta)\, \Delta\theta \leq \epsilon$ means: "the new policy must be close to the old policy in KL divergence."

### The Natural Policy Gradient update

For the policy gradient setting, the Fisher Information is computed over the state-action distribution:

$$G(\theta) = \mathbb{E}_{d_{\pi_\theta}(s),\, \pi_\theta(a|s)}\!\left[\nabla_\theta \log \pi_\theta(a|s)\, (\nabla_\theta \log \pi_\theta(a|s))^\top\right]$$

Estimated from samples:

$$\tilde{G}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\!\left[\nabla_\theta \log \pi_\theta(a_i|s_i)\, (\nabla_\theta \log \pi_\theta(a_i|s_i))^\top\right]$$

Update:

$$\Delta\theta = \frac{1}{2\lambda}\, \tilde{G}^{-1}(\theta)\, \widetilde{\nabla}_\theta J$$

### Handling singular $G$

If features are correlated (e.g., two identical features), $G$ is singular. Changes in the null space of $G$ don't affect the policy at all. Use the **pseudo-inverse** $G^\dagger$ instead of $G^{-1}$ — project away directions that don't matter.

For large parameter counts, inverting $G$ is expensive ($O(d^3)$ for $d$ parameters). In practice, solve $G \Delta\theta = \nabla J$ iteratively using Conjugate Gradient and terminate early for an approximate solution.

### Impact

The natural gradient makes an "enormous difference" (lecture notes' words) in algorithms like REINFORCE. It's the foundation for TRPO (Trust Region Policy Optimization), which formalizes the KL constraint, and PPO (Proximal Policy Optimization), which approximates it cheaply.

---

## Conservative Policy Iteration

REINFORCE is essentially a "soft" Policy Iteration — it nudges action probabilities toward actions with high Q-values via small gradient steps. Unlike hard Policy Iteration (which can make large policy changes per step and diverge with approximation), REINFORCE's small steps keep the distribution over trajectories stable.

**Conservative Policy Iteration** formalizes this: at each step, follow the old policy with probability $\alpha$ and take the greedy action $\arg\max_a \tilde{Q}(s, a)$ with probability $1 - \alpha$. This guarantees small changes to the trajectory distribution while still moving in the steepest uphill direction.

---

## Quick-fire self-test

1. Write the likelihood ratio policy gradient. *($\nabla_\theta J = \mathbb{E}[\nabla_\theta \log p(\tau|\theta) \cdot R(\tau)]$)*
2. Why don't you need to know the transition model? *(The $\log p(s_{t+1}|s_t,a_t)$ terms don't depend on $\theta$, so their gradients are zero — only $\nabla_\theta \log \pi_\theta$ remains)*
3. What is the advantage function? *($A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ — how much better action $a$ is than average)*
4. Why does subtracting a baseline $b(s)$ preserve unbiasedness? *($\mathbb{E}[\nabla_\theta \log p(\tau|\theta) \cdot b] = b \cdot \nabla_\theta 1 = 0$)*
5. What is the Actor-Critic architecture? *(Actor = learned policy $\pi_\theta$; Critic = learned value estimator $\hat{Q}_\phi$ or $\hat{A}_\phi$. Critic bootstraps the reward estimate to reduce variance.)*
6. Why does vanilla gradient descent fail with correlated features? *(L2 norm in parameter space doesn't reflect true policy change — redundant features cause disproportionately large steps)*
7. What metric does the Natural Policy Gradient use? *(Fisher Information Metric — the unique reparameterization-invariant metric on probability distributions, equivalent to second-order KL divergence)*
8. Name three variance reduction techniques for REINFORCE. *(Reward-to-go / causality, baselines, advantage function via critic)*
9. What algorithms build on the Natural Policy Gradient idea? *(TRPO formalizes the KL constraint, PPO approximates it cheaply)*
