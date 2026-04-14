# 04. Policy Gradients and REINFORCE

## What this note unpacks

Two big families of RL algorithms: **value-based** (learn $Q$, act greedy — that's DQN in the next note) and **policy-based** (learn the policy $\pi_\theta$ directly). This note is about the policy-based side.

The question is: how do you compute a gradient of "expected return" with respect to policy parameters $\theta$, when the return is the result of running a whole stochastic trajectory through a stochastic environment? You can't naively apply chain rule because the sampling step in the middle isn't differentiable.

The answer is the **log-derivative trick** — a simple algebraic identity that turns the intractable gradient into a tractable expectation. Apply it once and you get the **policy gradient theorem**. Apply the theorem to Monte Carlo sampled trajectories and you get **REINFORCE**.

This note covers:
- The objective: expected return under the policy
- The log-derivative trick
- The policy gradient theorem — full derivation in 6 lines
- REINFORCE algorithm pseudocode
- The variance problem and how baselines fix it
- The advantage function as the best baseline
- Williams 1992 and the birth of policy gradients

---

## The objective

The agent's goal is to maximize expected total return under its own policy. Formally:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ R(\tau) \right]
$$

where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T, r_T)$ is a **trajectory** sampled by running the policy $\pi_\theta$ in the environment, and $R(\tau) = \sum_t \gamma^t r_t$ is the total discounted return of the trajectory.

**We want to compute $\nabla_\theta J(\theta)$** so we can do gradient ascent: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$.

**The problem**: $J(\theta)$ is an expectation over $\tau$, and the distribution over $\tau$ depends on $\theta$ (because the agent's actions are sampled from $\pi_\theta$). So you can't just move the gradient inside the expectation — the thing you're averaging over is itself changing as $\theta$ changes.

Writing the expectation explicitly:

$$
J(\theta) = \sum_\tau P_\theta(\tau) R(\tau)
$$

where $P_\theta(\tau)$ is the probability of trajectory $\tau$ under the policy. Now the gradient is:

$$
\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta P_\theta(\tau) \cdot R(\tau)
$$

The problem: $\nabla_\theta P_\theta(\tau)$ is not an expectation anymore. You can't estimate it by sampling, because sampling gives you draws from $P_\theta$, not the gradient of $P_\theta$. If you tried to evaluate this naively, you'd need to sum over every possible trajectory, which is exponential in trajectory length.

The **log-derivative trick** fixes this.

---

## The log-derivative trick

The identity:

$$
\nabla_\theta P_\theta(\tau) = P_\theta(\tau) \cdot \nabla_\theta \log P_\theta(\tau)
$$

This is just the chain rule applied to $\log$:

$$
\nabla_\theta \log P_\theta(\tau) = \frac{\nabla_\theta P_\theta(\tau)}{P_\theta(\tau)} \quad\Longrightarrow\quad \nabla_\theta P_\theta(\tau) = P_\theta(\tau) \cdot \nabla_\theta \log P_\theta(\tau)
$$

It looks trivial. But it has a dramatic consequence: it lets you rewrite a gradient of a density as that density times a gradient of the log. And **the density is what you're already sampling from**, so the product can be reinterpreted as an expectation.

Substituting back:

$$
\nabla_\theta J(\theta) = \sum_\tau P_\theta(\tau) \cdot \nabla_\theta \log P_\theta(\tau) \cdot R(\tau) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \nabla_\theta \log P_\theta(\tau) \cdot R(\tau) \right]
$$

**This is the policy gradient in expectation form.** It's estimable by Monte Carlo: run the policy, collect trajectories, average $\nabla_\theta \log P_\theta(\tau) \cdot R(\tau)$ over them. Each sample gives you an unbiased gradient estimate.

The log is doing the critical work. Without it, you can't move the gradient inside the expectation. With it, the gradient *is* an expectation, which is exactly what you can estimate by sampling.

---

## Simplifying $\log P_\theta(\tau)$

The trajectory probability decomposes using the Markov property:

$$
P_\theta(\tau) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t \mid s_t) \cdot P(s_{t+1} \mid s_t, a_t)
$$

where $p(s_0)$ is the initial state distribution, $\pi_\theta$ is the policy, and $P$ is the environment transition. Taking the log:

$$
\log P_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T} \log P(s_{t+1} \mid s_t, a_t)
$$

Now the gradient with respect to $\theta$:

$$
\nabla_\theta \log P_\theta(\tau) = \underbrace{\nabla_\theta \log p(s_0)}_{= 0} + \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) + \underbrace{\sum_{t=0}^{T} \nabla_\theta \log P(s_{t+1} \mid s_t, a_t)}_{= 0}
$$

**The first and third terms vanish** because $p(s_0)$ and $P(s_{t+1} \mid s_t, a_t)$ don't depend on $\theta$ — they're properties of the environment, not the policy. Only the policy term has $\theta$ in it.

So:

$$
\nabla_\theta \log P_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

**This is the "magic" of the log-derivative trick applied to RL**: the environment dynamics $P$ drop out of the gradient *even though* the trajectory depends on them. The agent can compute the policy gradient without knowing the transition function. This is why policy gradient methods are model-free.

---

## The policy gradient theorem

Putting it all together:

$$
\boxed{
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R(\tau) \right]
}
$$

This is the **policy gradient theorem** (Williams 1992, Sutton et al. 2000). It says:

> The gradient of expected return with respect to policy parameters equals the expectation, over trajectories, of the sum over time of $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ times the trajectory return.

### Reading it intuitively

Look at one term: $\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R(\tau)$.

- $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ is the direction in parameter space that would *increase* the probability of taking action $a_t$ in state $s_t$.
- $R(\tau)$ is how good the trajectory that action was part of turned out to be.

Multiplying them: **if the trajectory was good ($R > 0$), take a step in the direction that makes $a_t$ more likely in $s_t$. If the trajectory was bad ($R < 0$), take a step that makes $a_t$ less likely.**

That's the one-line summary of REINFORCE: **push up the log-probability of actions that led to good returns, push down actions that led to bad returns, weighted by how good.**

```
  trajectory: s₀ → a₀ → r₀ → s₁ → a₁ → r₁ → ... → s_T → a_T → r_T

  R(τ) = sum of rewards  (how good overall?)

  for each step t:
    push θ in direction ∇_θ log π_θ(a_t|s_t) · R(τ)
    
    if R(τ) high:  "a_t was probably part of a good trajectory → make it more likely"
    if R(τ) low:   "a_t was probably part of a bad trajectory → make it less likely"
```

---

## REINFORCE — the Monte Carlo policy gradient algorithm

The simplest application of the policy gradient theorem. "REINFORCE" is Williams' name, and it stands for "**RE**ward **I**ncrement = **N**onnegative **F**actor × **O**ffset **R**einforcement × **C**haracteristic **E**ligibility" — that was the actual acronym in the 1992 paper. Nobody ever uses the expansion. Just say "REINFORCE."

```
algorithm: REINFORCE
  initialize policy π_θ with random parameters θ
  loop forever:
    # sample one trajectory
    τ ← run π_θ in the environment for T steps,
        collecting (s_0, a_0, r_0, ..., s_T, a_T, r_T)
    
    # compute returns (one per timestep, not per trajectory)
    for t = 0, 1, ..., T:
      G_t ← Σ_{k=t}^{T} γ^{k-t} r_k          # discounted return from step t
    
    # gradient ascent on every timestep
    for t = 0, 1, ..., T:
      θ ← θ + α · ∇_θ log π_θ(a_t|s_t) · G_t
```

**Key detail:** the update for action $a_t$ uses $G_t$ (the return *from step $t$ onward*), not $R(\tau)$ (the full trajectory return). This is a refinement of the bare policy gradient theorem: rewards that came before action $a_t$ can't possibly have been caused by $a_t$, so they should be dropped from the gradient for that action. The refined version is:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]
$$

This is still unbiased but has lower variance than using $R(\tau)$ everywhere. It's the form you'll usually see in modern references.

---

## The variance problem

REINFORCE is **unbiased** (the sample average converges to the true gradient) but **high variance** (individual samples can be wildly different). Here's why:

- Trajectories can be long and stochastic. A single bad action early might lead to a great trajectory by luck, or a good action might be followed by bad ones.
- The return $G_t$ depends on dozens or hundreds of random choices. Its variance grows with the horizon.
- A noisy gradient means slow, unstable learning.

**Example:** imagine a 1000-step episode where you usually get total return in $[90, 110]$. The return of a single trajectory might be 91 or 107 — about 20% fluctuation. That 20% is multiplicative noise on every gradient step, and it swamps the signal.

Fixing this is the main concern of *modern* policy gradient methods (TRPO, PPO, A2C, SAC, ...). The simplest fix — and the one you should know for the quiz — is the **baseline**.

---

## Baselines — subtracting a state value

**Claim:** if $b(s)$ is any function of the state (not of the action), then for any such $b$:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b(s_t)) \right]
$$

Subtracting a state-dependent baseline does **not change the expected gradient** but **reduces variance**, often dramatically.

### Proof that baselines don't bias the gradient

Look at the extra term introduced by subtracting $b(s_t)$:

$$
\mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot b(s_t) \right]
$$

Factor the inner expectation conditional on state $s_t$:

$$
= \mathbb{E}_{s_t}\!\left[ b(s_t) \cdot \mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)}\!\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] \right]
$$

The inner expectation over actions is:

$$
\mathbb{E}_{a_t \sim \pi_\theta}\!\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] = \sum_a \pi_\theta(a \mid s_t) \cdot \nabla_\theta \log \pi_\theta(a \mid s_t) = \sum_a \nabla_\theta \pi_\theta(a \mid s_t) = \nabla_\theta \underbrace{\sum_a \pi_\theta(a \mid s_t)}_{= 1} = 0
$$

The second-to-last step uses the log-derivative trick *in reverse*: $\pi \cdot \nabla \log \pi = \nabla \pi$. And since $\pi_\theta(\cdot \mid s_t)$ is a probability distribution, it sums to 1, whose gradient is 0.

So the baseline term adds 0 to the expectation — **unbiased**. $\blacksquare$

### Why it reduces variance

Intuitively: if $G_t$ usually equals, say, 100, and you're nudging $\theta$ proportional to $G_t$ whether $G_t$ is 95 or 105, most of the gradient is saying "all my actions were good" even though the relative difference between actions is tiny. By subtracting a baseline close to the average return ($b(s_t) \approx 100$), you get $G_t - b(s_t)$ in the range $[-5, +5]$, and the gradient focuses on the *difference* between actions, not the absolute level.

Mathematically, the variance of the gradient estimator is minimized when the baseline is chosen to be the conditional expectation — which brings us to the next idea.

---

## The advantage function

**Claim:** the best choice of baseline is the state-value function $V^\pi(s)$. Subtracting it gives the **advantage function**:

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

Reading: "how much better is action $a$ than the average action in state $s$, under policy $\pi$?" If $A > 0$, action $a$ is above average; if $A < 0$, below average.

With this baseline, the policy gradient becomes:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A^\pi(s_t, a_t) \right]
$$

The sign of the gradient is now sharp: **push up actions that are above average, push down actions that are below average**, rather than "push up everything that happened in a good trajectory."

In practice, you don't know $V^\pi$ exactly — you learn an approximation $V_\phi(s)$ simultaneously with $\pi_\theta$. This is the idea behind **actor-critic** methods:

- **Actor**: the policy $\pi_\theta$, trained via policy gradient
- **Critic**: the value function $V_\phi$, trained via regression on observed returns or TD error

A2C (Advantage Actor-Critic), A3C (Asynchronous A2C), and PPO are all variants of this actor-critic pattern with different details for stability and parallelism.

---

## Why the log: the intuition distilled

The quiz will probably test you on "why does the log appear?" The answer in one sentence:

> **Because the identity $\nabla_\theta p_\theta(x) = p_\theta(x) \cdot \nabla_\theta \log p_\theta(x)$ lets you rewrite a gradient of a density as a density times a gradient of log, which turns the gradient of an expectation into an expectation (of the gradient of log times the reward) — and expectations are what Monte Carlo sampling can estimate.**

Three things this is **not**:
1. It's not for numerical stability, though logs *also* help with underflow
2. It's not because the loss is cross-entropy (that's unrelated)
3. It's not about turning multiplication into addition, though that's another use of log

It's specifically the **score function trick**: $\nabla \log p$ is called the **score function** of $p$, and the identity lets you express gradients of expectations as expectations of scores.

---

## History/lore

- **1987 — Ronald J. Williams** (Northeastern) introduces the core REINFORCE idea in a technical report on "A class of gradient-estimating algorithms for reinforcement learning in neural networks."
- **1992 — Williams** publishes *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* in *Machine Learning* 8, the canonical REINFORCE paper. The acronym REINFORCE appears here for the first time. Williams proved that the REINFORCE gradient estimator is unbiased in expectation, even though each sample is high variance.
- **2000 — Sutton, McAllester, Singh, Mansour** publish *Policy Gradient Methods for Reinforcement Learning with Function Approximation* at NIPS. This paper formalized the **policy gradient theorem** for function-approximated policies and value functions, bridging REINFORCE to modern deep RL. Richard Sutton is the same Sutton who later coined "the bitter lesson."
- **2002 — Sham Kakade** publishes *A Natural Policy Gradient* at NIPS, introducing the natural policy gradient — a variant that uses the Fisher information matrix to make updates invariant to policy parameterization. This is the theoretical ancestor of TRPO and PPO.
- **2015 — Schulman et al.** publish *Trust Region Policy Optimization* at ICML, fixing the step-size problem in vanilla policy gradient with a KL-divergence constraint. TRPO is the first policy gradient method to reliably train deep neural network policies on hard control problems.
- **2017 — Schulman et al.** publish *Proximal Policy Optimization* (PPO) — a simplified TRPO that uses a clipped surrogate objective instead of the KL constraint. PPO became the default RL algorithm for the next decade, powering OpenAI Five (Dota 2), ChatGPT's RLHF, and most of deep RL research.

The direct lineage: **Williams 1987 → Williams 1992 (REINFORCE) → Sutton et al. 2000 (policy gradient theorem) → Kakade 2002 (natural gradient) → Schulman 2015 (TRPO) → Schulman 2017 (PPO) → Christiano et al. 2017 (RLHF) → ChatGPT 2022**.

Every modern RL-for-language-models system is downstream of Williams' 1992 trick.

---

## Takeaway

- **Objective**: $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$ — expected return under the policy.
- **Log-derivative trick**: $\nabla_\theta p_\theta = p_\theta \cdot \nabla_\theta \log p_\theta$ — the identity that turns gradients of expectations into expectations of gradients.
- **Policy gradient theorem**: $\nabla_\theta J(\theta) = \mathbb{E}_\tau \!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t\right]$. Environment dynamics drop out — that's why PG is model-free.
- **REINFORCE**: sample a trajectory, compute $G_t$ at each step, take a gradient step on $\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t$. Simple, unbiased, high variance.
- **Baseline**: subtract any state-dependent $b(s_t)$ from $G_t$ to reduce variance without changing the expected gradient. The best choice is $V^\pi(s_t)$, which gives the **advantage function** $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$.
- **Intuition**: push up log-probs of actions that turned out better than average, push down actions worse than average.

Next note: [quiz_5_07_learning_paradigms_comparison.md](quiz_5_07_learning_paradigms_comparison.md) — moving from RL (Lesson 17) into Lesson 18's territory of supervised / unsupervised / semi-supervised / few-shot / self-supervised learning.

For the value-based cousin of REINFORCE, see [quiz_5_05_dqn_deep_dive.md](quiz_5_05_dqn_deep_dive.md).
