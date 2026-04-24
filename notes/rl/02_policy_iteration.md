# 02. Policy Iteration

*Source: UW CSE 579, Policy Improvement lecture notes*

## What this note unpacks

Value Iteration finds $V^*$ by repeatedly applying the Bellman operator. It works, but it's slow — in later rounds, the implied policy barely changes even though the value function is still updating. Policy Iteration exploits this: instead of iterating on values, iterate on policies directly.

This note covers:
- Policy evaluation — computing $V^\pi$ for a fixed policy (as a linear system)
- Policy improvement — greedily improving a policy using the Q-function
- The full Policy Iteration algorithm
- Convergence proof: monotonic improvement + no local optima
- Access models (what level of simulator access do different RL settings give you?)
- Modified Policy Iteration (what practitioners actually use)

---

## Policy evaluation — how good is a given policy?

Given a fixed stationary policy $\pi$, its value function satisfies:

$$V^\pi(x) = c(x, \pi(x)) + \gamma \sum_{x'} p(x' \mid x, \pi(x))\, V^\pi(x')$$

This is a **linear equation** in $V^\pi$ — one equation per state, all coupled. We can write it in matrix form.

Let $\vec{c}_\pi$ be the vector of costs under policy $\pi$ (one entry per state), and $P_\pi$ be the row-stochastic transition matrix under $\pi$:

$$(P_\pi)_{ij} = p(x_j \mid x_i, \pi(x_i))$$

Then:

$$\vec{V}^\pi = \vec{c}_\pi + \gamma\, P_\pi\, \vec{V}^\pi$$

Rearranging:

$$\vec{V}^\pi - \gamma\, P_\pi\, \vec{V}^\pi = \vec{c}_\pi$$

$$(I - \gamma\, P_\pi)\, \vec{V}^\pi = \vec{c}_\pi$$

$$\boxed{\vec{V}^\pi = (I - \gamma\, P_\pi)^{-1}\, \vec{c}_\pi}$$

For $\gamma < 1$, the eigenvalues of $P_\pi$ are all $\leq 1$ (it's row-stochastic), so $I - \gamma P_\pi$ is always invertible. This gives an exact closed-form solution.

For large state spaces, matrix inversion is $O(|\mathcal{X}|^3)$ which is expensive, so iterative methods (running Value Iteration with the fixed policy) are used instead.

---

## Policy improvement — making it better

Given the value function $V^\pi$ of a policy $\pi$, define the **Q-function** (action-value function):

$$Q^\pi(x, a) = c(x, a) + \gamma\, \mathbb{E}_{p(x'|x,a)}\!\left[ V^\pi(x') \right]$$

$Q^\pi(x, a)$ is the cost of taking action $a$ in state $x$ and then following $\pi$ forever after.

The improved policy picks the action that minimizes the Q-function at every state:

$$\pi'(x) = \arg\min_a\, Q^\pi(x, a) = \arg\min_a\!\left[ c(x, a) + \gamma \sum_{x'} p(x' \mid x, a)\, V^\pi(x') \right]$$

**Guarantee**: $\pi'$ is at least as good as $\pi$. For every state $x$: $V^{\pi'}(x) \leq V^\pi(x)$.

---

## The Policy Iteration algorithm

```
Initialize: arbitrary policy π₀

repeat:
    1. Policy Evaluation:  compute V^{πₖ}
       (solve the linear system or run iterative evaluation)

    2. Policy Improvement:  for each x ∈ X:
       πₖ₊₁(x) = argmin_a [ c(x,a) + γ Σ_{x'} p(x'|x,a) V^{πₖ}(x') ]

    k ← k + 1

until πₖ == πₖ₋₁  (policy stopped changing)

return πₖ
```

Each iteration does two expensive things: evaluating the current policy (solving a linear system or running many Bellman updates) and then sweeping over all states to improve the policy. But the number of outer iterations is typically much smaller than Value Iteration.

---

## Convergence — why it works

Two things need to be true:

### 1. Monotonic improvement

Each new policy $\pi_{k+1}$ is at least as good as $\pi_k$. The improvement step only changes the action at state $x$ if the new action has lower Q-value than the current one. So:

$$V^{\pi_{k+1}}(x) \leq V^{\pi_k}(x) \quad \forall x$$

The improvement at step 1 (from state $x_0$) is:

$$\gamma\, \mathbb{E}_{p(x_0)}\!\left[ \mathbb{E}_{p(x' \mid x_0, \pi_{k+1}(x_0))}\!\left[V^{\pi_k}(x')\right] - \mathbb{E}_{p(x' \mid x_0, \pi_k(x_0))}\!\left[V^{\pi_k}(x')\right] \right]$$

We only switch actions when this quantity is non-negative.

### 2. No local optima

When Policy Iteration stops improving ($\pi_k = \pi_{k+1}$), we've reached a fixed point. At this fixed point:

$$V^{\pi'}(x) = \min_a\!\left[ c(x, a) + \gamma\, \mathbb{E}\!\left[V^{\pi'}(x')\right] \right]$$

This is exactly the **Bellman optimality equation**. Since the Bellman equation has a unique solution (for $\gamma < 1$), the fixed point must be $V^* = V^{\pi^*}$. There are no local optima — the algorithm always reaches the global optimum.

### Performance Difference Lemma

$$V^{\pi'}(x_0) - V^\pi(x_0) = \sum_{t=0}^{\infty} \gamma^t\, \mathbb{E}_{\rho_t}\!\left[ V^{\pi', \pi_{t+1}, \pi_{t+2}, \ldots}(x) - V^{\pi, \ldots}(x) \right]$$

where $\rho_t = \Pr[x_t = x \mid x_0, \pi']$. This telescoping sum shows that if each single-step improvement is non-negative, the total improvement is non-negative.

---

## Value Iteration vs Policy Iteration

| | Value Iteration | Policy Iteration |
|--|----------------|------------------|
| What it iterates on | Value function $V$ | Policy $\pi$ |
| Per-iteration cost | One Bellman backup per state | Full policy evaluation + one improvement sweep |
| Number of iterations | More iterations | Fewer iterations (value gap shrinks exponentially) |
| Convergence | $V$ converges gradually | $\pi$ often converges in very few iterations |

The value gap $|V^{\pi_k}(x) - V^*(x)|$ decreases **exponentially** with each Policy Iteration step. So Policy Iteration typically needs far fewer outer iterations than Value Iteration, but each iteration is more expensive.

---

## Access models — what do you have access to?

Different RL problems give you different levels of access to the environment:

| Level | What you get | Example |
|-------|-------------|---------|
| **Full probabilistic** | Exact $P(x' \mid x, a)$ and $c(x,a)$ for all states/actions | Textbook MDPs, solved analytically |
| **Deterministic generative** | $f(x, a) \to x'$ deterministically; can control the random seed | Tetris simulator |
| **Generative** | Can put the system in any state and simulate forward | Programmable physics simulator |
| **Reset** | Can run episodes and reset to a known initial state/distribution | Robot in a lab |
| **Trace** | One continuous trajectory, no resets | Real-world deployment — "life is like a solo violin performance where you're learning how to play the violin" |

As you go down the table, less access = harder problem = need more sophisticated algorithms.

---

## Modified Policy Iteration — what people actually use

The expensive part of Policy Iteration is the policy evaluation step (solving the linear system or running iterative evaluation to convergence). **Modified Policy Iteration** cuts this cost:

1. **Warm-start** the evaluation with the value function from the previous iteration
2. Run only **one iteration** of policy evaluation (one Bellman backup), not full convergence
3. Immediately do the improvement step

This is a hybrid between Value Iteration (which does one backup with a greedy policy each round) and full Policy Iteration (which evaluates the policy to convergence). In practice, Modified Policy Iteration converges nearly as fast as full PI but is much cheaper per iteration.

---

## Function approximation for large state spaces

When $|\mathcal{X}|$ is too large for a table (which is almost always in real problems):

- **Linear function approximation**: $V(x) = \mathbf{w}^\top \phi(x)$ where $\phi(x)$ is a feature vector
- **Nearest neighbor**: for query state $x$, find the closest sampled state $x'$ and return its stored value

These convert the exact DP into an approximate DP. The approximation introduces error, but makes the problem tractable.

---

## Value Iteration vs Policy Iteration — comparison

| | Value Iteration | Policy Iteration |
|---|---|---|
| What it updates | Value function $V$ | Policy $\pi$ (with $V^\pi$ recomputed each round) |
| Inner loop | One Bellman optimality backup per state per iter | Full policy evaluation (possibly many sweeps) per iter |
| Convergence | Asymptotic — needs a tolerance check | Exact — terminates when policy stops changing |
| Iterations | Many (each is cheap) | Few (each is expensive) |
| In practice | Simpler, often faster wall-clock | Fewer iterations but each is costly |

**Mnemonic:** Bellman **expectation** → $\sum_a \pi(a \mid s)$ (average over policy). Bellman **optimality** → $\max_a$ (pick the best).

Modified Policy Iteration is the spectrum between them: run evaluation for just a few sweeps instead of full convergence. Value Iteration is the extreme where you do one sweep per improvement.

---

## Worked example — 4-cell gridworld (Value Iteration)

```
   ┌─────┬─────┐
   │  s₁ │ s₂=G│   goal s₂ gives +10 and terminates
   ├─────┼─────┤
   │ s₀=S│  s₃ │   start at s₀, each step = 0 reward
   └─────┴─────┘

   actions: {up, down, left, right}
   deterministic transitions (off-grid = stay in place)
   γ = 0.9
```

**Iteration 0:** $V = [0, 0, 0, 0]$

**Iteration 1:** For each non-goal state, compute $\max_a [R + \gamma V(\text{next})]$:
- $V(s_0)$: all neighbors have $V = 0$, so $0 + 0.9 \cdot 0 = 0$. → $V(s_0) = 0$
- $V(s_1)$: "right" → $s_2$ (goal): $10 + 0.9 \cdot 0 = 10$. → $V(s_1) = 10$
- $V(s_3)$: "up" → $s_2$ (goal): $10 + 0.9 \cdot 0 = 10$. → $V(s_3) = 10$

$V = [0, 10, 0, 10]$

**Iteration 2:**
- $V(s_0)$: "up" → $s_1$: $0 + 0.9 \cdot 10 = 9$. "right" → $s_3$: $0 + 0.9 \cdot 10 = 9$. → $V(s_0) = 9$

$V = [9, 10, 0, 10]$

**Iteration 3:** $V = [9, 10, 0, 10]$ — **converged.**

**Greedy policy:**
```
   ┌─────┬─────┐
   │  →  │  G  │   s₁: go right to goal
   ├─────┼─────┤
   │ ↑/→ │  ↑  │   s₀: up or right (tied), s₃: up to goal
   └─────┴─────┘
```

---

## Why convergence is guaranteed — contraction mapping

The Bellman optimality operator $\mathcal{T}$:

$$(\mathcal{T} V)(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a) + \gamma V(s') \right]$$

is a **$\gamma$-contraction** under the sup norm:

$$\|\mathcal{T} V - \mathcal{T} U\|_\infty \leq \gamma \|V - U\|_\infty$$

Applying $\mathcal{T}$ shrinks the gap between any two value functions by factor $\gamma$ each iteration. By the **Banach fixed-point theorem**, this guarantees convergence to a unique fixed point $V^* = \mathcal{T} V^*$ from any starting point.

Policy Iteration converges because each improvement step is strictly better (or equal, at the optimum), and there are finitely many policies.

---

## History

- **1957** — Bellman introduces the Bellman equation in *Dynamic Programming*
- **1960** — Ronald Howard formalizes policy iteration in *Dynamic Programming and Markov Processes* (MIT PhD thesis)
- **1989** — Christopher Watkins proves Q-learning convergence (Cambridge PhD) — the first model-free algorithm with a convergence guarantee
- **1996** — Bertsekas & Tsitsiklis publish *Neuro-Dynamic Programming* — theoretical foundation for deep RL
- **2013** — Mnih et al. show Q-learning + CNN + experience replay beats humans at Atari — DP's grand comeback with 60 years of GPU progress

---

## Quick-fire self-test

1. Write the matrix equation for policy evaluation. *($\vec{V}^\pi = (I - \gamma P_\pi)^{-1} \vec{c}_\pi$)*
2. What does the policy improvement step do? *(At each state, pick the action that minimizes $Q^\pi(x,a)$ — act greedily w.r.t. the current value function)*
3. Why are there no local optima in Policy Iteration? *(At a fixed point, the value function satisfies the Bellman optimality equation, which has a unique solution)*
4. How does Policy Iteration's convergence rate compare to Value Iteration's? *(The value gap shrinks exponentially per PI step — fewer iterations needed, but each is more expensive)*
5. What is Modified Policy Iteration? *(Warm-start evaluation from the previous value function and do only one Bellman backup instead of evaluating to convergence)*
6. Name the 5 access models, from most to least access. *(Full probabilistic → Deterministic generative → Generative → Reset → Trace)*
7. What is the "trace model" analogy? *("Life is like a solo violin performance where you're learning how to play the violin" — one continuous trajectory, no resets)*
8. What is the Bellman optimality operator, and why does it guarantee convergence? *($\mathcal{T}$ is a $\gamma$-contraction — Banach fixed-point theorem gives unique fixed point $V^*$)*
9. In the 4-cell gridworld, why does $s_0$ converge to $V = 9$? *(Best neighbor has $V = 10$, and $\gamma \cdot 10 = 9$)*
