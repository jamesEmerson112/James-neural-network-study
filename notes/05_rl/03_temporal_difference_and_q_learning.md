# 03. Temporal Difference Learning and Q-Learning

*Source: UW CSE 579, TD Learning lecture notes*

## What this note unpacks

Value Iteration and Policy Iteration require knowing the transition model $P(x' \mid x, a)$. What if you don't? You have to learn from experience — from actual $(x, a, r, x')$ transitions as the agent interacts with the world.

**Batch methods** (like Fitted Q-Iteration) collect a dataset offline and fit a value function. They're data-efficient but memory-hungry and computationally expensive.

**Online methods** update the value function after every single transition. They're cheap and lightweight, but less data-efficient.

This note covers:
- TD Learning — online policy evaluation, the TD error
- SARSA — TD extended to the Q-function (on-policy)
- Q-Learning — the off-policy version that learns $Q^*$ directly
- Fitted Q-Learning — Q-Learning with function approximation
- Exploration: $\epsilon$-greedy and Boltzmann
- Experience replay — bridging online and batch

---

## Setup: value and Q-functions (infinite horizon)

For a fixed policy $\pi$:

**Value function:**

$$V^\pi(x) = \mathbb{E}\!\left[ \sum_{t=0}^{\infty} \gamma^t\, r(x_t, \pi(x_t)) \right], \quad x_0 = x$$

**Action-value (Q) function:**

$$Q^\pi(x, a) = r(x, a) + \mathbb{E}\!\left[ \sum_{t=1}^{\infty} \gamma^t\, r(x_t, \pi(x_t)) \right], \quad x_0 = x$$

**Bellman equations:**

$$V^\pi(x) = r(x, \pi(x)) + \gamma\, \mathbb{E}_{p(x'|x,\pi(x))}\!\left[V^\pi(x')\right]$$

$$Q^\pi(x, a) = r(x, a) + \gamma\, \mathbb{E}_{p(x'|x,a)}\!\left[Q^\pi(x', \pi(x'))\right]$$

**Optimal Bellman equations:**

$$V^*(x) = \max_{a}\!\left[ r(x, a) + \gamma\, \mathbb{E}_{p(x'|x,a)}\!\left[V^*(x')\right] \right]$$

$$Q^*(x, a) = r(x, a) + \gamma\, \mathbb{E}_{p(x'|x,a)}\!\left[\max_{a'} Q^*(x', a')\right]$$

---

## TD Learning — online policy evaluation

TD Learning estimates $V^\pi$ by updating after every single experience $(x, \pi(x), r, x')$.

### The update rule

$$\tilde{V}^\pi(x) \leftarrow (1 - \alpha)\, \tilde{V}^\pi(x) + \alpha\!\left[ r + \gamma\, \tilde{V}^\pi(x') \right]$$

Equivalently:

$$\tilde{V}^\pi(x) \leftarrow \tilde{V}^\pi(x) + \alpha\!\left[ r + \gamma\, \tilde{V}^\pi(x') - \tilde{V}^\pi(x) \right]$$

The term in brackets is the **TD error**:

$$\delta = r + \gamma\, \tilde{V}^\pi(x') - \tilde{V}^\pi(x)$$

Reading: "the actual reward $r$ plus the discounted estimated future, minus what we currently think this state is worth." If $\delta > 0$, the state was better than expected — nudge the estimate up. If $\delta < 0$, worse than expected — nudge it down.

### TD as exponential moving average

The first form of the update shows that TD is an exponential moving average: the new estimate is $(1-\alpha)$ of the old estimate plus $\alpha$ of the one-step bootstrapped target $r + \gamma \tilde{V}^\pi(x')$.

### The loss being minimized

$$L = \frac{1}{2}\!\left(V^\pi(x) - \tilde{V}^\pi(x)\right)^2$$

Since we don't know $V^\pi(x)$, we approximate it as $y = r + \gamma\, \tilde{V}^\pi(x')$, giving:

$$L_{\text{approx}} = \frac{1}{2}\!\left(y - \tilde{V}^\pi(x)\right)^2$$

The gradient (treating $y$ as constant) gives the TD update rule.

### Algorithm: TD Learning

```
Initialize V̄(x) for all x
repeat:
    Initialize x to starting state
    while x is not terminal:
        a ← π(x)
        Take action a, observe reward r and next state x'
        V̄(x) ← (1 - α) V̄(x) + α [r + γ V̄(x')]
        x ← x'
```

---

## Worked example — grid world TD

4×4 grid, start at (0,0), goal at (3,3). Reward +1 at goal, 0 elsewhere. Policy: go right until wall, then go down. Initialize $\tilde{V}^\pi(x) = 0$ for all $x$.

**Iteration 1:** Robot follows the policy right→right→right→down→down→down to goal.

- Most transitions: $r = 0$, both $\tilde{V}^\pi(x) = 0$ and $\tilde{V}^\pi(x') = 0$, so TD error $= 0$, no update.
- At cell (3,2) → goal: $r = 1$, TD error $= 1 + \gamma \cdot 0 - 0 = 1$.

$$\tilde{V}^\pi((3,2)) \leftarrow (1-\alpha) \cdot 0 + \alpha \cdot (1 + 0) = \alpha$$

**Iteration 2:**

- At (3,2) → goal again: $\tilde{V}^\pi((3,2)) \leftarrow (1-\alpha) \cdot \alpha + \alpha \cdot 1 = \alpha + \alpha(1-\alpha)$
- At (3,1) → (3,2): $r = 0$, TD error $= 0 + \gamma \cdot \alpha - 0 = \gamma\alpha$

$$\tilde{V}^\pi((3,1)) \leftarrow \alpha^2 \gamma$$

Values propagate backward one cell per iteration. This is slow — you run the entire policy trajectory just to update the next cell upstream.

---

## SARSA — on-policy Q-function learning

SARSA extends TD to the action-value function $Q^\pi(x, a)$. The name comes from the quintuple used per update: $(S, A, R, S', A')$.

### Update rule

$$\tilde{Q}^\pi(x, a) \leftarrow (1 - \alpha)\, \tilde{Q}^\pi(x, a) + \alpha\!\left[ r(x,a) + \gamma\, \tilde{Q}^\pi(x', \pi(x')) \right]$$

### Algorithm: SARSA

```
Initialize Q̃(x, a) for all x, a
repeat:
    Initialize x to starting state
    while x is not terminal:
        a ← π(x)
        Take action a, observe (x, a, r, x')
        a' ← π(x')
        Q̃(x,a) ← (1-α) Q̃(x,a) + α [r + γ Q̃(x', a')]
        x ← x'
```

SARSA and TD are both **on-policy** — they can only learn about the policy currently being executed, because the update uses $\pi(x')$ (the action the current policy would take in the next state).

---

## Q-Learning — off-policy optimal Q

Q-Learning learns $Q^*$ directly, without needing to evaluate a specific policy first.

### Key difference from SARSA

SARSA bootstraps from $\tilde{Q}(x', \pi(x'))$ — the value of the action the current policy would pick.

Q-Learning bootstraps from $\max_{a'} \tilde{Q}^*(x', a')$ — the value of the **best** action, regardless of what the agent actually did:

$$\tilde{Q}^*(x, a) \leftarrow \alpha\!\left[ r + \gamma \max_{a'} \tilde{Q}^*(x', a') \right] + (1-\alpha)\, \tilde{Q}^*(x, a)$$

### Why this is off-policy

The update doesn't reference the current policy $\pi$ at all. The agent can follow any exploration policy (random, $\epsilon$-greedy, whatever) to generate experiences, and Q-Learning will still converge to $Q^*$. This is the fundamental advantage over SARSA.

### Convergence conditions

Q-Learning converges $\tilde{Q}^* \to Q^*$ as $k \to \infty$ if:

1. Every state-action pair is visited infinitely often
2. $\sum_{k=0}^{\infty} \alpha_k = \infty$ (learning rates sum to infinity — don't stop learning too early)
3. $\sum_{k=0}^{\infty} \alpha_k^2 < \infty$ (learning rates are square-summable — you do eventually slow down)

These two conditions together mean: start with large $\alpha$, anneal it toward zero. Learn fast early, refine slowly later. A common choice satisfying both: $\alpha_k = 1/k$.

---

## On-policy vs off-policy — summary

| | TD / SARSA (on-policy) | Q-Learning (off-policy) |
|--|----------------------|------------------------|
| What it learns | $V^\pi$ or $Q^\pi$ for the current policy | $Q^*$ for the optimal policy |
| Update uses | $\pi(x')$ — action the current policy would take | $\max_{a'} Q^*(x', a')$ — the best action |
| Can use old data? | No — samples must come from $\pi$ | Yes — any experience works |
| Exploration | Trapped by the current policy | Free to explore however you want |

---

## Fitted Q-Learning — function approximation

For large or continuous state spaces, represent $Q^*$ with a parameterized function $Q_\theta(x, a)$.

### Loss

$$L = \frac{1}{2}\!\left(y - Q_\theta(x, a)\right)^2, \quad y = r + \gamma \max_{a'} Q_\theta(x', a')$$

### Gradient (semi-gradient, treating $y$ as constant)

$$\widetilde{\nabla}_\theta L = -\!\left(y - Q_\theta(x, a)\right) \nabla_\theta Q_\theta(x, a)$$

### Update rule

$$\theta \leftarrow \theta + \alpha\!\left[ r + \gamma\, Q_\theta(x', a^*) - Q_\theta(x, a) \right] \nabla_\theta Q_\theta(x, a)$$

where $a^* = \arg\max_{a'} Q_\theta(x', a')$.

### The semi-gradient problem

The true gradient includes a term $\gamma \nabla_\theta Q_\theta(x', a^*)$ from differentiating through $y$. Q-Learning drops this term (treats $y$ as constant). This makes it **not true gradient descent** — convergence to a local minimum is not guaranteed in general.

### Bellman residual method

The alternative: keep the full gradient including the $\nabla_\theta Q_\theta(x', a^*)$ term. This gives true gradient descent, but requires **two independent samples** of the next state for unbiased estimation. Often impractical with real systems (you'd need to visit the same state-action pair twice with independent randomness).

---

## Exploration policies

Q-Learning can learn from any exploration policy, but it still needs to visit every state-action pair. Two standard approaches:

### $\epsilon$-Greedy

$$a = \begin{cases} \arg\max_a \tilde{Q}(x, a) & \text{with probability } 1 - \epsilon \\ \text{random action} & \text{with probability } \epsilon \end{cases}$$

Start with $\epsilon \approx 1$ (mostly random), decay $\epsilon \to 0$ over training (increasingly greedy).

### Boltzmann exploration

$$p(a \mid x) = \frac{\exp[\beta\, Q(x, a)]}{\sum_{a'} \exp[\beta\, Q(x, a')]}$$

- $\beta = 0$: uniform random (maximum exploration)
- $\beta \to \infty$: greedy policy (no exploration)

Start with small $\beta$, increase over training. Boltzmann exploration is "softer" than $\epsilon$-greedy — it preferentially explores actions with higher estimated value rather than choosing uniformly at random.

---

## Experience replay

Q-Learning and SARSA use each experience exactly once, which is data-inefficient. Experience replay bridges the gap between online and batch:

1. Collect experiences $(x, a, r, x')$ into a **replay buffer** $\mathcal{D}$
2. Periodically sample a random mini-batch from $\mathcal{D}$
3. Perform Q-Learning updates on the batch

### Why it helps

- **Data reuse**: each experience can be used many times
- **Breaks correlation**: consecutive experiences from the same trajectory are highly correlated, which causes overfitting. Random sampling from the buffer decorrelates the training data.

Since Q-Learning is off-policy, old experiences from previous policies are still valid for updating $Q^*$. SARSA cannot use experience replay (it's on-policy — old experiences came from a different policy).

---

## Quick-fire self-test

1. Write the TD update rule and identify the TD error. *($\tilde{V}(x) \leftarrow \tilde{V}(x) + \alpha[r + \gamma \tilde{V}(x') - \tilde{V}(x)]$; the bracketed term is the TD error)*
2. What makes SARSA "on-policy"? *(The update uses $Q(x', \pi(x'))$ — the action chosen by the current policy)*
3. How does Q-Learning's update differ from SARSA's? *(Uses $\max_{a'} Q(x', a')$ instead of $Q(x', \pi(x'))$ — learns $Q^*$ regardless of the behavior policy)*
4. State the two learning rate conditions for Q-Learning convergence. *($\sum \alpha_k = \infty$ and $\sum \alpha_k^2 < \infty$)*
5. Why is Fitted Q-Learning not true gradient descent? *(It treats the target $y = r + \gamma \max Q(x', a')$ as constant, dropping the gradient through $y$)*
6. What problem does experience replay solve? *(Data inefficiency and correlated samples from sequential trajectories)*
7. Can SARSA use experience replay? Why or why not? *(No — SARSA is on-policy; old experiences came from a different policy, so they're invalid for updating the current policy's Q-function)*
8. Compare $\epsilon$-greedy vs Boltzmann exploration. *($\epsilon$-greedy: random with probability $\epsilon$, greedy otherwise. Boltzmann: softmax over Q-values, preferentially explores higher-value actions)*
