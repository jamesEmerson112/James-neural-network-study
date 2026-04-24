# 01. Markov Decision Processes

*Source: UW CSE 579, MDP lecture notes*

## What this note unpacks

A Markov Decision Process (MDP) is the mathematical framework underneath all of reinforcement learning. Before you can talk about Q-learning, policy gradients, or DQN, you need this object nailed down — it defines what "optimal behavior" even means.

This note covers:
- The 9 components of an MDP (state, action, transition, reward, horizon, discount, policy, value function, objective)
- The Markov property — why it matters and what it buys you
- Tetris as a concrete MDP
- Dynamic programming for deterministic MDPs (backward induction)
- Value Iteration for stochastic MDPs
- Infinite horizon problems, the Bellman equation, and discount factors
- Convergence: when and why the value function stabilizes

---

## The big picture — what an MDP looks like

```
                         THE MDP LOOP
                         ============

        ┌─────────────────────────────────────────────┐
        │                 ENVIRONMENT                  │
        │                                             │
        │   States: x₀, x₁, x₂, ...                 │
        │   Transition: x' ~ T(x, a)                 │
        │   Reward: r = R(x, a)                       │
        │                                             │
        └────────┬───────────────────────▲────────────┘
                 │                       │
           state xₜ                action aₜ
           reward rₜ                     │
                 │                       │
                 ▼                       │
        ┌────────────────────────────────┴────────────┐
        │                   AGENT                      │
        │                                             │
        │   Policy: π(x) → a                          │
        │   "what action to take in each state"       │
        │                                             │
        └─────────────────────────────────────────────┘

    At each time step t:
      1. Agent observes state xₜ
      2. Agent picks action aₜ = π(xₜ)
      3. Environment returns reward rₜ and next state xₜ₊₁
      4. Repeat
```

### One step of an MDP, expanded

```
    state xₜ ──── action aₜ ────► TRANSITION MODEL T(xₜ, aₜ)
       │                                    │
       │                          ┌─────────┼─────────┐
       │                          ▼         ▼         ▼
       │                        x'₁       x'₂       x'₃    (possible next states)
       │                       p=0.7     p=0.2     p=0.1    (transition probabilities)
       │
       └──► REWARD: r(xₜ, aₜ) = immediate payoff
```

### Stochastic vs deterministic

```
    DETERMINISTIC                    STOCHASTIC
    ═════════════                    ══════════

    x ──(a)──► x'                   x ──(a)──┬──► x'₁  (p=0.6)
                                             ├──► x'₂  (p=0.3)
    one outcome,                             └──► x'₃  (p=0.1)
    known in advance
                                    multiple possible outcomes,
                                    weighted by probability
```

### A 3-state MDP example

```
    ┌───────────────────────────────────────────────────────┐
    │                                                       │
    │        a=right, r=-1         a=right, r=+10           │
    │   ┌───┐  ──────────►  ┌───┐  ──────────►  ┌───┐     │
    │   │ A │                │ B │                │ G │     │
    │   │   │  ◄──────────  │   │                │ ☆ │     │
    │   └───┘  a=left, r=-1 └───┘                └───┘     │
    │     │                   │                    ▲        │
    │     │  a=stay, r=-1     │  a=stay, r=-1      │        │
    │     └───────┐           └───────┐            │        │
    │             ▼                   ▼            │        │
    │           (self)              (self)     ABSORBING    │
    │                                          (terminal)   │
    └───────────────────────────────────────────────────────┘

    States:  {A, B, G}
    Actions: {left, right, stay}
    Rewards: -1 per step, +10 for entering G
    Goal G is absorbing: once you enter, you stay forever with r=0

    Optimal policy: π*(A) = right, π*(B) = right
    (get to G as fast as possible to stop accumulating -1's)
```

### The value function — what each state is "worth"

```
    With γ = 0.9 and optimal policy (always go right):

    V*(A) = -1 + γ·V*(B)         V*(G) = 0   (terminal, no more cost)
    V*(B) = +10 + γ·V*(G) = +10

    V*(A) = -1 + 0.9·(10) = 8.0
    V*(B) = 10
    V*(G) = 0

    ┌───────┐     ┌───────┐     ┌───────┐
    │ A     │     │ B     │     │ G     │
    │ V=8.0 │ ──► │ V=10  │ ──► │ V=0   │
    └───────┘     └───────┘     └───────┘
```

### The Bellman equation — one picture

```
    V*(x) = min  [ c(x,a)  +  γ · Σ p(x'|x,a) · V*(x') ]
             a       ▲              ▲         ▲
             │       │              │         │
        try all   immediate     discount   expected future
        actions   cost now      factor     value of where
                                           you end up
```

### Discount factor γ — "how far ahead do you care?"

```
    γ = 0.0: only care about immediate reward (greedy)
    ├─ r₀ ─┤

    γ = 0.5: nearby rewards matter, distant ones fade fast
    ├─ r₀ ── 0.5·r₁ ── 0.25·r₂ ── 0.125·r₃ ── ... ─┤

    γ = 0.9: care a lot about the future (effective horizon ~10 steps)
    ├─ r₀ ── 0.9·r₁ ── 0.81·r₂ ── 0.73·r₃ ── ... ── 0.35·r₁₀ ─┤

    γ = 1.0: all rewards equally important (may diverge!)
    ├─ r₀ ── r₁ ── r₂ ── r₃ ── ... ── r₁₀₀ ── r₁₀₀₀ ── ∞? ─┤

    Effective horizon ≈ 1/(1-γ):
      γ=0.9  → ~10 steps     γ=0.99 → ~100 steps
      γ=0.5  → ~2 steps      γ=1.0  → ∞ (danger)
```

### Value Iteration — the algorithm in pictures

```
    t=T-1 (last step):   V*(x) = min_a c(x,a)     ← just the immediate cost

    t=T-2:               V*(x) = min_a [c(x,a) + V*(next state at T-1)]
                                        └──────────────────────────────┘
                                         one-step lookahead

    t=T-3:               V*(x) = min_a [c(x,a) + V*(next state at T-2)]
                                        └──────────────────────────────┘
                                         two-step lookahead

    ...working backward until t=0...

    ═══════════════════════════════════════════════════
    After T iterations: V*(x, 0) = optimal value from any state
                        π*(x, 0) = optimal action from any state
```

---

## The 9 components of an MDP

### 1. State space $\mathcal{X}$

A complete description of the system — like position and velocity in classical mechanics. Knowing the state makes the past irrelevant for predicting the future. Denoted $x \in \mathcal{X}$ (or $s \in \mathcal{S}$).

Examples: board configuration in Tetris, pose of a robot arm, grid cell of a rover.

There's typically an initial state $x_0$ and possibly a terminal (absorbing) state that ends the episode.

### 2. Action space $\mathcal{A}$

The set of things the agent can do. Denoted $a \in \mathcal{A}$ (or $u \in \mathcal{U}$, often called the "control space" in control theory).

Examples: torques on a joint, moving to an adjacent grid cell, choosing a column and rotation in Tetris.

### 3. Transition model $T$

How the world responds to actions. For stochastic systems:

$$x' \sim T(x, a)$$

where $T$ is a probability distribution over next states. For deterministic systems: $x' = T(x, a)$.

The key MDP property: $p(x' \mid x, a, \text{all previous } x\text{'s and } a\text{'s}) = T(x, a)$. The next state depends only on the current state and action — not the history. This is the **Markov property**.

### 4. Reward / Cost function

$r(x, a)$ or $c(x, a)$ — the immediate payoff of taking action $a$ in state $x$. Can also depend on the next state: $r(x, a, x')$, or even time: $r(x, a, x', t)$.

Reward and cost are interchangeable: negate the reward and switch max to min.

### 5. Horizon $T$

The number of steps the problem runs for. $T \in \mathbb{N}$.

- $T = 1$: optimal control reduces to greedy search
- $0 < T < \infty$: must reason $T$ steps ahead; optimal policy is typically time-dependent
- $T = \infty$: need a discount factor $\gamma$ to ensure convergence

### 6. Discount factor $\gamma$

$0 \leq \gamma \leq 1$. A reward received $n$ steps in the future is worth $\gamma^n$ of its face value today.

Two interpretations:
- **Impatience**: rewards sooner are worth more
- **Probability of death**: at each step, the agent survives with probability $\gamma$ and dies (enters an absorbing zero-reward state) with probability $1 - \gamma$. This gives an effective horizon of $O\!\left(\frac{1}{1-\gamma}\right)$.

### 7. Policy $\pi$

$\pi: \mathcal{X} \times \{0, \ldots, T-1\} \to \mathcal{A}$. A function that maps states (and optionally time) to actions. Tells the agent what to do in every state.

If the problem is Markov, the optimal policy needs only state and time — not the full history.

### 8. Value function $V^\pi(x, t)$

The expected discounted sum of rewards from following policy $\pi$ starting at state $x$ at time $t$. The optimal value function $V^*(x, t)$ is the value function of the best policy $\pi^*$.

### 9. Objective function

What we're optimizing. The standard choice is **expected cumulative reward**:

$$J = \mathbb{E}\!\left[ \sum_{t=0}^{T-1} \gamma^t\, r(x_t, a_t) \right]$$

Other options include expected infinite discounted reward ($T = \infty$) or immediate reward ($T = 1$).

---

## The Markov property

The defining property: the future is conditionally independent of the past given the present state.

$$p(x_{t+1} \mid x_t, a_t, x_{t-1}, a_{t-1}, \ldots, x_0, a_0) = p(x_{t+1} \mid x_t, a_t) = T(x_t, a_t)$$

This is what makes DP and value-based methods tractable. Without it, the optimal policy would need to condition on the entire history, which is exponentially larger.

---

## Tetris as an MDP

| Component | Tetris mapping |
|-----------|---------------|
| States | Board configuration ($2^k$ for $k$ cells) $\times$ 7 piece types |
| Actions | ~40 (column $\times$ orientation), not all valid for every piece |
| Transition | Deterministic board update + random next piece |
| Cost/Reward | Options: +1 per line cleared, free rows at top, +1 for surviving |
| Horizon | Until the board fills up (game over = terminal state) |

---

## Solving deterministic MDPs — backward induction

When $T(x, a)$ is deterministic, we can compute the optimal policy by working backwards from the last time step.

**Time $T-1$ (last step):**

$$\pi^*(x, T-1) = \arg\min_a\, c(x, a)$$

$$V^*(x, T-1) = \min_a\, c(x, a)$$

**Time $T-2$:**

$$\pi^*(x, T-2) = \arg\min_a\!\left[ c(x, a) + V^*(T(x, a),\, T-1) \right]$$

$$V^*(x, T-2) = \min_a\!\left[ c(x, a) + V^*(T(x, a),\, T-1) \right]$$

**General recursion (any $t \leq T-2$):**

$$V^*(x, t) = \min_a\!\left[ c(x, a) + V^*(T(x, a),\, t+1) \right]$$

$$\pi^*(x, t) = \arg\min_a\!\left[ c(x, a) + V^*(T(x, a),\, t+1) \right]$$

### Key insight: value determines policy

If you have $V^*$, you never need to store $\pi^*$ explicitly. Just act greedily with respect to $V^*$:

$$\pi^*(x, t) = \arg\min_a\!\left[ c(x, a) + V^*(T(x, a),\, t+1) \right]$$

### Complexity

The naive recursive algorithm has complexity $O(|\mathcal{X}||\mathcal{A}|T^2)$ due to redundant computation. With memoization (or explicit backward induction / DP), this drops to $O(|\mathcal{X}||\mathcal{A}|T)$.

---

## Value Iteration for stochastic MDPs

When transitions are stochastic, we take expectations over next states:

$$V^*(x, t) = \min_a\!\left[ c(x, a) + \sum_{x' \in \mathcal{X}} p(x' \mid x, a)\, V^*(x',\, t+1) \right]$$

$$\pi^*(x, t) = \arg\min_a\!\left[ c(x, a) + \sum_{x' \in \mathcal{X}} p(x' \mid x, a)\, V^*(x',\, t+1) \right]$$

### Algorithm: Value Iteration (finite horizon)

```
for t = T-1, ..., 0:
    for each x in X:
        if t == T-1:
            V(x, t) = min_a c(x, a)
        else:
            V(x, t) = min_a [ c(x, a) + sum_{x'} p(x'|x,a) V(x', t+1) ]
```

**Complexity:** $O(|\mathcal{X}|^2 |\mathcal{A}| T)$ in the worst case. In practice, most states have few reachable neighbors, so it reduces to $O(k|\mathcal{X}||\mathcal{A}|T)$ where $k$ is the average branching factor. For deterministic problems, $k = 1$.

---

## Infinite horizon problems

### Why time drops out

For finite horizon, both $V^*$ and $\pi^*$ depend on time (a hockey team trailing with 10 seconds left plays differently than at the start). But as $T \to \infty$, the value function converges to a stationary solution — no time dependence.

Analogy: with no time limit, you'd never change your strategy based on the clock. The optimal move from state $x$ is always the same.

### The Bellman equation

The fixed-point equation that $V^*$ satisfies when $T \to \infty$:

$$V^*(x) = \min_a\!\left[ c(x, a) + \gamma \sum_{x'} p(x' \mid x, a)\, V^*(x') \right]$$

And the optimal policy:

$$\pi^*(x) = \arg\min_a\!\left[ c(x, a) + \gamma \sum_{x'} p(x' \mid x, a)\, V^*(x') \right]$$

Note: no time index. The Bellman equation says: the optimal value of a state equals the best single-step cost plus the discounted expected optimal value of the next state.

### Discount factor as probability of death

The Bellman equation can be derived by imagining the agent dies with probability $1 - \gamma$ at each step:

$$V^*(x, t) = \min_a\!\left[ c(x, a) + \sum_{x'}\!\left[ \gamma \cdot p(x' \mid a, x)\, V^*(x', t+1) + (1 - \gamma) \cdot 0 \right] \right]$$

The $(1-\gamma) \cdot 0$ term is the "death" outcome with zero future value. Simplifying gives the standard discounted Bellman equation.

### Convergence

If $\gamma < 1$, the discounted sum of rewards is finite with probability 1, and $V^*$ converges. For $\gamma = 1$, convergence is not guaranteed in general (divergence if the goal is unreachable, oscillation in cyclic MDPs).

---

## Two approaches for infinite horizon Value Iteration

### Approach 1: Finite horizon approximation

Run the finite-horizon DP (Algorithm 4) for $T = O\!\left(\log\frac{1}{\epsilon}\right)$ steps with discount factor $\gamma$. Since $\gamma^T = O(\epsilon)$, the error in the value function is $O(\epsilon)$.

- **Pro**: theoretically stronger bounds; the resulting time-varying policy is optimal for the finite-horizon problem
- **Con**: requires $O(T)$ extra memory to store values at every time step

### Approach 2: Iterative Bellman (in-place)

Initialize $V(x)$ arbitrarily. Repeatedly apply the Bellman operator:

$$V_{\text{new}}(x) = \min_a\!\left[ c(x, a) + \gamma \sum_{x'} p(x' \mid x, a)\, V_{\text{old}}(x') \right]$$

Update $V_{\text{old}} \leftarrow V_{\text{new}}$ and repeat until convergence.

- **Pro**: low memory — only stores one value function
- **Con**: slower convergence rate in the worst case

Both approaches converge to the same $V^*$ as iterations $\to \infty$.

---

## Worked example — 2D grid robot

Setup: a robot on a 2D grid. Start at S, goal at G. Obstacles are walls. Cost = 1 for every non-goal state, 0 at the goal. Goal is absorbing. No discount ($\gamma = 1$, finite horizon).

```
┌───┬───┬───┬───┐
│ S │   │   │   │
├───┼───┼───┼───┤
│   │ █ │   │   │
├───┼───┼───┼───┤
│   │   │   │ G │
└───┴───┴───┴───┘
```

**Value Iteration, backward from $T-1$:**

Step 1 ($t = T-1$): $V^*(x, T-1) = \min_a c(x, a)$. Every state has cost 1 except G which has cost 0. So $V^*(G) = 0$, $V^*(\text{everywhere else}) = 1$.

Step 2 ($t = T-2$): Each state's value = cost(1) + $V^*$ of best neighbor at $T-1$. States adjacent to G get value $1 + 0 = 1$. States two steps from G get $1 + 1 = 2$. And so on.

After enough iterations, $V^*(x)$ at each cell equals the shortest-path distance to G (accounting for obstacles). The optimal policy at each cell points toward the neighbor with the lowest value — i.e., toward the shortest path.

---

## Robot vacuum — a complete MDP walkthrough

A 3×3 apartment with a dock and a dirt cell. This example instantiates every MDP concept above in a single concrete system.

```
      col 0    col 1    col 2
   ┌────────┬────────┬────────┐
0  │ DOCK   │        │        │
   │ (0,0)  │ (0,1)  │ (0,2)  │
   ├────────┼────────┼────────┤
1  │        │        │        │
   │ (1,0)  │ (1,1)  │ (1,2)  │
   ├────────┼────────┼────────┤
2  │        │        │ DIRT   │
   │ (2,0)  │ (2,1)  │ (2,2)  │
   └────────┴────────┴────────┘
```

**MDP tuple:**
- $S$ = every (cell, bin status) = $9 \times 2 = 18$ states. E.g., $((1, 2), \text{empty})$
- $A = \{N, S, E, W\}$
- $P$ is deterministic — N from $(1, 1)$ always lands in $(0, 1)$. Bin flips on entering $(2,2)$ (empty→dirty) or entering $(0,0)$ with dirty bin (dirty→empty + reward)
- $R = -1$ per step, $+20$ when entering $(0,0)$ with dirty bin
- $\gamma = 0.9$

**Worked trajectory — optimal path (9 steps):**

| $k$ | $\gamma^k$ | $R_k$ | $\gamma^k R_k$ |
|---|---|---|---|
| 0 | 1.0000 | -1 | -1.0000 |
| 1 | 0.9000 | -1 | -0.9000 |
| 2 | 0.8100 | -1 | -0.8100 |
| 3 | 0.7290 | -1 | -0.7290 |
| 4 | 0.6561 | -1 | -0.6561 |
| 5 | 0.5905 | -1 | -0.5905 |
| 6 | 0.5314 | -1 | -0.5314 |
| 7 | 0.4783 | -1 | -0.4783 |
| 8 | 0.4305 | +20 | +8.6093 |

$$G_0 = -5.6953 + 8.6093 \approx 2.91$$

An 18-step meandering path gives $\approx -4.86$ (negative — step costs outweigh the discounted reward). An infinite random walk gives $G_0 = -10$ (geometric series). This is what $\gamma$ and $R$ balance.

---

## Common confusions

**"MDP = Markov chain"** — No. A Markov chain is just $(S, P)$ — no actions, no rewards, no agent. DDPM's forward process is a Markov chain. Atari is an MDP.

**"Markov property = no memory"** — The Markov property says the *state* contains everything relevant. The agent can have memory (LSTM policy, state augmentation with last 4 frames like DQN). "Markov" is about the state definition, not the agent's brain.

**"$\gamma$ is a physical property"** — $\gamma$ is a modeling choice / hyperparameter. The environment doesn't know what $\gamma$ is. Two agents with different $\gamma$ operate on the same environment but choose different policies.

**"Rewards must be nonnegative"** — Rewards can be any real number. Step penalties ($-1$), crash penalties ($-100$), and cost-of-living terms are all standard.

**"$P(s'|s,a)$ is what the agent learns"** — $P$ is a property of the environment. Model-free RL (Q-learning, DQN, REINFORCE, PPO) never represents $P$. Only model-based RL approximates $P$ as a tool.

**"Value function = reward function"** — $R(s,a)$ is the immediate one-step reward. $V^\pi(s)$ is the expected cumulative discounted future reward. $R$ might be $-1$ at a state where $V$ is $+8$ (because a $+20$ reward is coming soon).

---

## Additional worked examples

### Example A — trivial: 2-cell hallway

```
   ┌───┬───┐
   │ L │ R │        episode ends when agent lands in R
   └───┴───┘
```

$S = \{L, R\}$, $A = \{\text{stay}, \text{move}\}$, $R(L, \text{move}) = +1$, $\gamma = 0.9$. Optimal policy: always move. $V^*(L) = 1$, $V^*(R) = 0$. The smallest non-trivial MDP.

### Example B — slippery gridworld

Same 4-cell grid but with stochastic transitions: "up" lands correctly 80%, slips sideways 10%, stays put 10%. Now the full Bellman expectation equation is needed:

$$V^\pi(s_0) = \sum_a \pi(a \mid s_0) \sum_{s'} P(s' \mid s_0, a) \left[ R(s_0, a) + \gamma V^\pi(s') \right]$$

The optimal policy may prefer some directions based on which slip outcomes are least harmful.

### Example C — non-Markov observations (POMDP)

If the robot vacuum can only see whether its current cell has dirt (a single bit), two different underlying states produce the same observation. The Markov property fails.

**Fixes:** (1) state augmentation — use last $k$ observations + actions as state, or (2) POMDP framework — maintain a belief distribution $b(s)$ over underlying states. In practice, deep RL handles this with recurrent networks (LSTM/GRU/Transformer) whose hidden state implicitly maintains a belief.

---

## Quick-fire self-test

1. Name the 9 components of an MDP. *(State, action, transition, reward/cost, horizon, discount, policy, value function, objective)*
2. What is the Markov property? *(The next state depends only on the current state and action, not the history)*
3. Write the Bellman equation for $V^*$ in the infinite horizon case. *($V^*(x) = \min_a[c(x,a) + \gamma \sum_{x'} p(x'|x,a) V^*(x')]$)*
4. What is the complexity of Value Iteration for stochastic MDPs? *($O(|\mathcal{X}|^2 |\mathcal{A}| T)$, or $O(k|\mathcal{X}||\mathcal{A}|T)$ with sparse transitions)*
5. Give two interpretations of the discount factor $\gamma$. *(Impatience — future rewards worth less; probability of death — agent survives with probability $\gamma$ each step)*
6. Why does the optimal policy become time-independent as $T \to \infty$? *(The value function converges to a fixed point; with no time limit, the best action from state $x$ doesn't depend on when you're there)*
7. What goes wrong if $\gamma = 1$ in infinite horizon? *(Value function may diverge or oscillate — convergence not guaranteed)*
8. In the robot vacuum, is $((0,0), \text{empty})$ the same state as $((0,0), \text{dirty})$? *(No — different bin status means different optimal actions. Ignoring bin status would violate the Markov property.)*
9. What happens if $\gamma = 0$? *(Myopic greedy policy — only immediate reward matters. Usually disastrous because the agent can't plan ahead.)*
