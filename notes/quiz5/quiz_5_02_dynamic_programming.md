# 02. Dynamic Programming — Solving an MDP When You Know the Rules

## What this note unpacks

If you know the MDP's transition probability $P$ and reward function $R$ (i.e., you have a *model* of the environment), you can solve the MDP exactly via **dynamic programming (DP)**. The two canonical algorithms are **value iteration** and **policy iteration**. Both compute the optimal value function and optimal policy, just by different routes.

This note covers:
- The Bellman expectation and optimality equations (the mathematical heart of DP)
- Value iteration: apply the Bellman optimality operator until convergence
- Policy iteration: alternate policy evaluation and policy improvement
- A worked 4-cell gridworld example
- Why both converge (contraction mapping intuition)

DP is the "easy case" of RL — you have the full model, so you can just compute. Real RL (Q-learning, DQN, REINFORCE) is about what to do when you *don't* have $P$ and $R$, and those methods are all approximations of DP.

---

## The Bellman equations

### Bellman expectation equation (for a fixed policy $\pi$)

The value of a state under policy $\pi$ equals the expected reward of following $\pi$ for one step, plus the discounted value of the resulting state:

$$
V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a) + \gamma V^\pi(s') \right]
$$

Reading: "take the weighted average over all actions (weighted by how often $\pi$ picks each), and for each action, take the weighted average over all possible next states (weighted by transition probabilities), of the immediate reward plus the discounted future value."

This is a **system of linear equations** — one equation per state, with $V^\pi(s)$ as the unknowns. For small MDPs you can solve it directly by matrix inversion. For large MDPs, iterative methods (below) are faster.

The Q-function version:

$$
Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a) + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a') \right]
$$

### Bellman optimality equation (for the optimal policy)

The optimal policy always picks the action that maximizes expected value. So the optimal value function satisfies:

$$
V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a) + \gamma V^*(s') \right]
$$

And for Q-values:

$$
Q^*(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a) + \gamma \max_{a'} Q^*(s', a') \right]
$$

The only difference from the Bellman expectation equation is **expectation** ($\sum_a \pi(a \mid s) \cdot$) has been replaced with **max** ($\max_a$). That single change makes the equation **nonlinear**, which is why you can't just invert a matrix to solve it — you need iterative algorithms.

**Mnemonic for remembering which is which:**
- Bellman **expectation** → $\sum_a \pi(a \mid s)$ (average over what $\pi$ does)
- Bellman **optimality** → $\max_a$ (best possible action)

---

## Value iteration

**Idea:** repeatedly apply the Bellman optimality operator until $V$ stops changing.

```
algorithm: value_iteration
  initialize V(s) ← 0 for all s
  repeat:
    Δ ← 0
    for each state s:
      v_old ← V(s)
      V(s) ← max_a Σ_{s'} P(s'|s, a) [R(s, a) + γ · V(s')]
      Δ ← max(Δ, |v_old − V(s)|)
    if Δ < θ:   # small tolerance
      break
  return V

  # extract the greedy policy at the end
  for each state s:
    π(s) ← argmax_a Σ_{s'} P(s'|s, a) [R(s, a) + γ · V(s')]
  return π
```

**What's happening:**
- Start with any initial value estimate (zeros are fine)
- At each iteration, update every state's value to be the value of the best action, assuming the current value estimate for future states
- The updates are a contraction under the sup norm — every iteration brings $V$ closer to $V^*$ by a factor of $\gamma$
- Once $V$ has converged, extract the policy by taking $\arg\max_a$ at each state

**Convergence rate:** after $k$ iterations, $\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$. With $\gamma = 0.9$, each iteration shrinks the error by 10%. Exponential convergence.

---

## Policy iteration

**Idea:** alternate two steps — evaluate the current policy, then improve it greedily — until the policy stops changing.

```
algorithm: policy_iteration
  initialize π arbitrarily (e.g., all "up")
  repeat:
    # --- policy evaluation ---
    # solve V^π for the current π by iterating:
    repeat:
      for each state s:
        V(s) ← Σ_a π(a|s) Σ_{s'} P(s'|s, a) [R(s, a) + γ · V(s')]
      until V converges

    # --- policy improvement ---
    policy_stable ← True
    for each state s:
      a_old ← π(s)
      π(s) ← argmax_a Σ_{s'} P(s'|s, a) [R(s, a) + γ · V(s')]
      if π(s) ≠ a_old:
        policy_stable ← False

    if policy_stable:
      break
  return π, V
```

**What's happening:**
- **Policy evaluation** solves the Bellman *expectation* equation for the current $\pi$. This gives $V^\pi$, the value of the current policy.
- **Policy improvement** acts greedily w.r.t. $V^\pi$ — at each state, pick the action that maximizes one-step lookahead. The policy improvement theorem says this new policy is strictly better (or equal) to the old one.
- Alternate until the greedy policy matches the current policy (i.e., no state wants to switch actions). At that point you've reached $\pi^*$.

**Convergence:** guaranteed in a finite number of iterations for finite MDPs (each iteration either strictly improves the policy or terminates, and there are only finitely many policies).

---

## Value iteration vs policy iteration

| | Value iteration | Policy iteration |
|---|---|---|
| What it updates | Value function $V$ | Policy $\pi$ (with $V^\pi$ recomputed each round) |
| Inner loop | One Bellman optimality backup per state per iter | Full policy evaluation (possibly many sweeps) per iter |
| Convergence | Asymptotic — needs a tolerance check | Exact — terminates when policy stops changing |
| Iterations | Many (each is cheap) | Few (each is expensive) |
| In practice | Simpler, often faster wall-clock | Fewer iterations but each is costly |

**A middle ground:** *modified policy iteration* runs policy evaluation for just a few sweeps (not until full convergence) before improving the policy. Value iteration is literally this with one sweep per improvement. So the two algorithms are points on a spectrum.

---

## Worked example — 4-cell gridworld

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

### Iteration 0: $V = [0, 0, 0, 0]$ (for $s_0, s_1, s_2, s_3$)

### Iteration 1 of value iteration

For each non-goal state, compute $\max_a [R + \gamma V(\text{next})]$:

- $V(s_0)$: best action is... all neighbors have $V = 0$, so any action gives $0 + 0.9 \cdot 0 = 0$. $V(s_0) \leftarrow 0$.
- $V(s_1)$: "right" lands in $s_2$ (goal) → $10 + 0.9 \cdot 0 = 10$. $V(s_1) \leftarrow 10$.
- $V(s_2) = 0$ (terminal).
- $V(s_3)$: "up" lands in $s_2$ (goal) → $10 + 0.9 \cdot 0 = 10$. $V(s_3) \leftarrow 10$.

$V = [0, 10, 0, 10]$

### Iteration 2

- $V(s_0)$: "up" lands in $s_1$ → $0 + 0.9 \cdot 10 = 9$. "right" lands in $s_3$ → $0 + 0.9 \cdot 10 = 9$. Max is 9. $V(s_0) \leftarrow 9$.
- $V(s_1)$: "right" lands in $s_2$ → $10 + 0.9 \cdot 0 = 10$. $V(s_1) \leftarrow 10$.
- $V(s_3)$: "up" lands in $s_2$ → $10 + 0.9 \cdot 0 = 10$. $V(s_3) \leftarrow 10$.

$V = [9, 10, 0, 10]$

### Iteration 3

- $V(s_0)$: "up" lands in $s_1$ → $0 + 0.9 \cdot 10 = 9$. "right" lands in $s_3$ → $0 + 0.9 \cdot 10 = 9$. Max is 9. $V(s_0) \leftarrow 9$.

$V = [9, 10, 0, 10]$ — **converged.**

### Extract the greedy policy

```
   ┌─────┬─────┐
   │  ↑  │  →  │   s₁: go right to reach goal
   ├─────┼─────┤   s₀: go up OR right (tied)
   │ ↑/→ │  ↑  │   s₃: go up to reach goal
   └─────┴─────┘
```

With values $V^* = [9, 10, 0, 10]$, and the optimal policy reaching the goal in at most 2 steps from anywhere.

---

## Why both algorithms converge — contraction mapping intuition

Define the **Bellman optimality operator** $\mathcal{T}$ that takes a value function and returns a new one:

$$
(\mathcal{T} V)(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a) + \gamma V(s') \right]
$$

$\mathcal{T}$ is a **$\gamma$-contraction** under the sup norm. That is, for any two value functions $V$ and $U$:

$$
\|\mathcal{T} V - \mathcal{T} U\|_\infty \leq \gamma \|V - U\|_\infty
$$

Reading: "applying $\mathcal{T}$ to two different value functions shrinks the gap between them by a factor of $\gamma$."

By the **Banach fixed-point theorem**, any contraction mapping on a complete metric space has a **unique fixed point** that iteration converges to exponentially. The fixed point of $\mathcal{T}$ is exactly $V^*$ (by definition of the Bellman optimality equation: $V^* = \mathcal{T} V^*$). So value iteration converges to $V^*$ from any starting point.

Policy iteration converges because each policy improvement step produces a strictly better policy (or an equal one, at which point you're at the optimum), and there are only finitely many policies in a finite MDP.

---

## When DP doesn't apply

DP requires two things:
1. **Known model**: you need $P$ and $R$ explicitly.
2. **Tractable state space**: you need to be able to sweep over all states.

Real RL problems violate at least one:
- **Unknown model** → use **Q-learning**, which estimates $Q^*$ directly from experience without needing $P$.
- **Huge state space** → use **function approximation** (e.g., a neural network for $V_\theta$ or $Q_\theta$), which generalizes across states instead of tabulating them.

Deep RL (DQN, REINFORCE, actor-critic) is what you do when both assumptions fail. But the math underneath is still the Bellman equations — deep RL is approximate DP with sampled transitions and learned value functions.

---

## History/lore

- **1957 — Bellman** introduces the Bellman equation in *Dynamic Programming*. See [quiz_5_01_mdp_formal_definition.md](quiz_5_01_mdp_formal_definition.md) for the full history of why "dynamic programming" has that funny name.
- **1960 — Ronald Howard** publishes *Dynamic Programming and Markov Processes* based on his MIT PhD thesis, formalizing policy iteration as a practical algorithm. Howard's treatment made DP concrete for engineers and economists (not just RAND operations researchers).
- **1989 — Christopher Watkins** proves the convergence of Q-learning in his PhD thesis at Cambridge. This is the first model-free algorithm with a convergence guarantee — the bridge from DP to modern RL.
- **1996 — Bertsekas & Tsitsiklis** publish *Neuro-Dynamic Programming*, the first rigorous treatment of DP with function approximation. This is the theoretical foundation for what would later become deep RL.
- **2013 — Mnih et al.** publish *Playing Atari with Deep Reinforcement Learning* at NeurIPS workshops, showing that Q-learning + CNN function approximation + experience replay could beat humans at Atari games. This was DP's grand comeback — the same equations Bellman wrote in 1957, now with 60 years of GPU progress behind them.

The lesson: **the equations are ancient, the approximations are new.** Every deep RL algorithm is a way of approximating DP when you can't do the exact version.

---

## Takeaway

- **Bellman expectation** ($\sum_a \pi(a \mid s)$) computes the value of a specific policy. **Bellman optimality** ($\max_a$) computes the value of the best policy.
- **Value iteration**: repeatedly apply $\mathcal{T}$ until $V$ converges, then extract the greedy policy. Each iteration is cheap, but many are needed.
- **Policy iteration**: alternate full policy evaluation and greedy improvement until the policy is stable. Fewer iterations, each is expensive.
- **Both converge** because the Bellman operator is a $\gamma$-contraction (exponential convergence to the unique fixed point $V^*$).
- **DP assumes you have the model.** Real RL (Q-learning, DQN, REINFORCE) is approximate DP for when you don't.

Next note: [quiz_5_03_exploration_vs_exploitation.md](quiz_5_03_exploration_vs_exploitation.md) — how the agent decides what to try when it doesn't yet know what's good.
