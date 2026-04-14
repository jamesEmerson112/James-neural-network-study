# 01. The Markov Decision Process — Formal Definition

## What this note unpacks

When people say "RL problem," they almost always mean "Markov Decision Process." An MDP is the mathematical object that RL is built on, and almost every RL algorithm is a way of solving or approximately solving one. If you can't recite the MDP tuple from memory and say what each piece means, every downstream concept (Bellman equations, value iteration, Q-learning, policy gradients) will feel like it came from nowhere.

This note defines the MDP formally, walks through each component with a running example (a robot vacuum), states the Markov property precisely, introduces the two functions the agent is trying to compute (the value function and the policy), exposes the common confusions students carry into quiz day, and gives you a glossary + self-test checklist.

---

## Intuition first — why MDPs exist

Before any math: an MDP is **the smallest abstraction you need to describe any situation where some "thing" makes decisions over time and gets feedback**. That's it. Strip the name away and you get:

> *There's a world. The world is in some state. Something (the agent) does things. Doing things changes the state. Changing the state sometimes produces feedback (good or bad). The agent wants the feedback to be good.*

This is the minimal vocabulary for describing a decision-making problem. Chess is this. Self-driving is this. Portfolio management is this. Potty-training a puppy is this. Sending a spacecraft to Mars is this. All of them — wildly different problems — fit inside the same 5-tuple $(S, A, P, R, \gamma)$ once you identify what "state," "action," "transition," "reward," and "discount" mean in their context.

The reason the framework is so universal is that it throws away almost everything. An MDP **doesn't care** what your state actually is (pixels? joint angles? chessboard?). It doesn't care what your actions are (motor torques? stock buy orders? English sentences?). It doesn't care whether the world is physical or virtual. All it insists on is that you can describe:

1. The possible situations (states)
2. The possible choices (actions)
3. What happens when you make each choice (transition probabilities)
4. How good or bad each outcome is (rewards)
5. How much the future matters compared to the present (discount factor)

Once you have these five things, the Bellman equation applies, dynamic programming works, Q-learning converges, and policy gradient theorems hold. The structure comes for free the moment you put your problem in MDP form. **This is why understanding the MDP is a one-time cost that pays off forever** — every other RL topic is just "given an MDP, here's how to solve it."

The **"Markov"** part is the key restriction: the 5-tuple framework works **only if** the current state contains everything the agent needs to predict the future. No hidden variables, no memory of what happened two steps ago, nothing up the environment's sleeve. If you can't describe your problem that way, you either need to redefine your state until you can (state augmentation), or you need a more general framework (POMDPs, partially observable MDPs — out of scope here).

One more framing: you can think of the MDP tuple as the **"rules of the game"** and the agent's policy as the **"strategy."** The rules tell you what's allowed and what the consequences are; the strategy tells you what choices to make given the rules. Learning in RL is learning the strategy, not the rules — the rules are fixed by the environment. This is why you don't need to know $P$ and $R$ to act (model-free RL) but you do need to know them to *plan* (model-based RL and dynamic programming).

---

## The running example — the robot vacuum

Throughout this note we'll use a single concrete example: a **robot vacuum** in a tiny 3×3 apartment, trying to collect dirt and return to its dock. Every abstract MDP concept below will be instantiated in this example so you can see what it looks like in practice.

**The apartment:**

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

- **Dock** at cell $(0, 0)$ — where the robot charges and "finishes"
- **Dirt** at cell $(2, 2)$ — the target to collect
- Robot starts at the dock with an empty dustbin

**The robot's setup:**

- It can see which cell it's in and whether its dustbin is empty or has dirt
- It can move N, S, E, W one cell per step (bumping into a wall = stay in place)
- When it enters $(2, 2)$, it automatically collects the dirt and the dustbin becomes "dirty"
- When it returns to $(0, 0)$ with a dirty bin, it empties the bin at the dock — that's the goal
- Each step costs a little energy (small negative reward)
- Emptying the bin at the dock gives a large positive reward

This is a simple MDP but it has **everything**: discrete states, discrete actions, deterministic transitions, a mixture of small step penalties and a large terminal reward, and a natural episode structure (from "dock, empty bin" to "dock, empty bin again" after collecting dirt). We'll use it at every step.

---

## The formal definition

A **Markov Decision Process** is a 5-tuple:

$$
\text{MDP} = (S, A, P, R, \gamma)
$$

where:

- $S$ is the **state space** — the set of all possible configurations the environment can be in
- $A$ is the **action space** — the set of all actions the agent can take
- $P(s' \mid s, a)$ is the **transition probability** — given you're in state $s$ and take action $a$, the probability of landing in state $s'$
- $R(s, a)$ is the **reward function** — the immediate reward for taking action $a$ in state $s$ (sometimes written $R(s, a, s')$ to depend on the next state too)
- $\gamma \in [0, 1)$ is the **discount factor** — how much future rewards are worth relative to immediate ones

That's the whole definition. Everything in RL is either one of these five things, or a function computed from them.

**What does this look like for the robot vacuum?**

- $S$ = every combination of (cell, bin status) = $9 \times 2 = 18$ states. A state is e.g. $((1, 2), \text{empty})$ = "robot is in cell (1,2), bin is empty."
- $A = \{N, S, E, W\}$ = four movement actions.
- $P((s', \text{bin}') \mid (s, \text{bin}), a)$ is deterministic — N from $(1, 1)$ always lands in $(0, 1)$. The bin status changes only on entering $(2, 2)$ (bin → dirty) or entering $(0, 0)$ with a dirty bin (bin → empty + reward).
- $R((s, \text{bin}), a)$ = $-1$ for every step (small energy cost) **plus** $+20$ when the robot enters $(0, 0)$ with a dirty bin (task completion).
- $\gamma = 0.9$ — the robot cares about the immediate future more than the distant future, but not by much.

Now every downstream concept ("the value of state $((1, 1), \text{dirty})$" or "the optimal policy from state $((2, 1), \text{empty})$") is something concrete you can picture.

---

## A concrete example — the 4-cell gridworld

The robot vacuum has 18 states, which is a lot to visualize all at once. Let's also introduce a **smaller** example for drawing Bellman equations by hand — a $2 \times 2$ grid with just 4 states and no bin:

```
   ┌─────┬─────┐
   │  s₁ │  s₂ │   ← s₂ is the goal, reward +10
   ├─────┼─────┤
   │  s₀ │  s₃ │   ← start at s₀
   └─────┴─────┘
```

- **State space** $S = \{s_0, s_1, s_2, s_3\}$ — four possible positions
- **Action space** $A = \{\text{up}, \text{down}, \text{left}, \text{right}\}$ — four possible moves
- **Transition probability** $P(s' \mid s, a)$ — deterministic in this example: "up" from $s_0$ always lands in $s_1$, "right" from $s_1$ always lands in $s_2$, etc. Moves that go off the grid leave the agent in place.
- **Reward function** $R(s, a)$ — $+10$ for any action that lands in $s_2$ (the goal), $0$ everywhere else. Some gridworlds also use a small negative step penalty (e.g., $-0.04$) to encourage reaching the goal quickly.
- **Discount factor** $\gamma = 0.9$ — a reward received two steps from now is worth $0.9^2 = 0.81$ of a reward received right now.

The **episode** ends when the agent reaches $s_2$. The agent's job is to figure out a policy — a rule for picking actions — that maximizes the total discounted reward over time.

Think of this as the **hand-computation** version of the vacuum example. It's small enough that you can draw out all four Bellman backups on a whiteboard, but it has the same structure.

---

## The five components in detail

### State space $S$

The set of all possible situations the environment can be in. States can be:

- **Discrete and finite**: gridworlds, chess positions, Atari game pixels (technically finite because of screen resolution and color depth)
- **Discrete and infinite**: integer coordinates on an infinite grid
- **Continuous**: the position and velocity of a robot arm, the pixel intensities of a camera image treated as real numbers

The size of $S$ matters enormously for which algorithms work. Tabular methods (value iteration, Q-learning) only work when $|S|$ is small enough to enumerate. Deep RL exists because most interesting problems have states that are too big or too continuous to enumerate.

**In the vacuum:** $|S| = 18$, discrete and finite. Every state is fully described by `(row, col, bin_status)`. We could enumerate all 18 in a lookup table if we wanted to.

### Action space $A$

The set of choices available to the agent at each step. Like states, actions can be:

- **Discrete**: up/down/left/right, fire/don't-fire, buy/sell/hold
- **Continuous**: the torque applied to each joint of a robot, steering angle, throttle

Some MDPs also have **state-dependent action spaces** $A(s)$ — different actions are available in different states. (Chess: you can only make legal moves.) For simplicity, most introductory treatments assume a single global action space.

**In the vacuum:** $A = \{N, S, E, W\}$, discrete, four actions, same set available in every state (invalid moves just leave the robot in place).

### Transition probability $P(s' \mid s, a)$

The dynamics of the environment. It says: "if you're in state $s$ and take action $a$, here's the probability of landing in each possible next state $s'$." Formally:

$$
P: S \times A \times S \to [0, 1], \qquad \sum_{s' \in S} P(s' \mid s, a) = 1
$$

The transition function can be:

- **Deterministic**: $P(s' \mid s, a)$ is 1 for exactly one $s'$ and 0 for all others. Gridworlds and chess are deterministic.
- **Stochastic**: multiple possible next states with non-trivial probabilities. Rolling dice, windy gridworlds, real-world robotics (slippage, measurement noise).

Crucially, the agent **does not necessarily know** $P$. In **model-based RL**, the agent learns or is given $P$ and uses it for planning (e.g., dynamic programming). In **model-free RL** (Q-learning, REINFORCE, DQN), the agent learns a policy or value function without ever explicitly modeling $P$ — it just experiences transitions.

**In the vacuum:** $P$ is deterministic. For any `(cell, bin, action)`, there's exactly one resulting `(cell', bin')`. Concretely:

$$
P\!\left(((0, 1), \text{empty}) \;\middle|\; ((1, 1), \text{empty}),\ N\right) = 1
$$

and $P(s' \mid s, a) = 0$ for every other $s'$. One of the check-your-understanding questions below asks what would change if a battery ran low and N sometimes failed — that's the stochastic version.

### Reward function $R(s, a)$

A scalar number telling the agent how good each action was. Rewards are the only signal the agent ever gets about its goals — there are no labels, no demonstrations, no ground-truth "correct" actions.

The reward function can be written as:

- $R(s, a)$ — reward depends on state and action
- $R(s, a, s')$ — reward depends on state, action, and next state (equivalent in expectation)
- $R(s)$ — reward depends only on the state (common in gridworlds where the goal has intrinsic value)

**Key subtlety:** the reward is typically *scalar*. Not a vector, not a structured object. This forces all the agent's goals into a single number, and reward engineering (choosing a reward function that actually causes the behavior you want) is one of the hardest parts of applied RL.

**In the vacuum:** rewards are a combination of a step penalty and a terminal reward:

$$
R(s, a) = \begin{cases}
+20 & \text{if taking } a \text{ from } s \text{ lands the robot at } (0, 0) \text{ with a dirty bin} \\
-1 & \text{otherwise}
\end{cases}
$$

The $-1$ per step creates pressure to finish quickly. Without it, the robot could meander forever and still eventually collect the $+20$, and $\gamma < 1$ alone might not be enough to rush it. The $+20$ terminal reward is the only positive signal — it's what defines "success."

### Discount factor $\gamma$

A number between 0 and 1 that says how much the agent cares about future rewards vs immediate ones. The discounted return from time step $t$ is:

$$
G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \gamma^3 R_{t+3} + \ldots = \sum_{k=0}^{\infty} \gamma^k R_{t+k}
$$

Three reasons $\gamma$ exists:

1. **Mathematical convergence**: if the episode is infinite and $\gamma = 1$, the sum can blow up. Setting $\gamma < 1$ guarantees the sum is finite (as long as rewards are bounded).
2. **Modeling uncertainty**: the future is uncertain, so future rewards are less trustworthy. Discounting encodes "a bird in the hand is worth two in the bush."
3. **Effective horizon**: with $\gamma = 0.99$, rewards more than ~$\frac{1}{1-\gamma} = 100$ steps away are essentially ignored. This gives the agent an effective planning horizon without needing to explicitly reason about time.

Common values:
- $\gamma = 0$ — myopic, only immediate reward matters
- $\gamma = 0.9$ — short horizon (~10 steps)
- $\gamma = 0.99$ — long horizon (~100 steps) — the default in most deep RL papers
- $\gamma = 1$ — undiscounted, only used for finite episodic tasks

**In the vacuum:** we chose $\gamma = 0.9$. This means if the robot collects the $+20$ reward in 5 steps instead of 8, the discounted return is much higher ($0.9^5 \cdot 20 = 11.81$ vs $0.9^8 \cdot 20 = 8.60$), giving the robot a geometric reason to hurry. Combined with the $-1$ step penalty, this produces strong pressure to take the shortest path.

---

## Worked walkthrough — three steps of the robot vacuum

Let's actually run the robot through three steps and compute the discounted return, so all the symbols above become real numbers.

**Setup:**
- Start state: $s_0 = ((0, 0), \text{empty})$ — robot at dock, bin empty
- $\gamma = 0.9$, step cost $-1$, dock-with-dirty-bin reward $+20$
- Suppose the robot follows a hardcoded policy: always take the shortest path to dirt, then shortest path back

**Step 0 → 1:** from $((0, 0), \text{empty})$, take action S (south).

- New state: $s_1 = ((1, 0), \text{empty})$
- Reward: $R_0 = -1$ (just a step)
- Transition: deterministic — S from $(0, 0)$ lands in $(1, 0)$

**Step 1 → 2:** from $((1, 0), \text{empty})$, take action S.

- New state: $s_2 = ((2, 0), \text{empty})$
- Reward: $R_1 = -1$

**Step 2 → 3:** from $((2, 0), \text{empty})$, take action E (east).

- New state: $s_3 = ((2, 1), \text{empty})$
- Reward: $R_2 = -1$

**Discounted return from $s_0$ (so far, just these 3 steps):**

$$
G_0 = R_0 + \gamma R_1 + \gamma^2 R_2 = -1 + 0.9 \cdot (-1) + 0.81 \cdot (-1) = -2.71
$$

Still negative, because the robot is spending energy without yet collecting any positive reward. Let's keep going.

**Step 3 → 4:** from $((2, 1), \text{empty})$, take action E.

- New state: $s_4 = ((2, 2), \text{dirty})$ — robot enters dirty cell, bin becomes dirty
- Reward: $R_3 = -1$ (the dirt pickup itself has no extra reward; only returning to the dock does)

**Step 4 → 5:** from $((2, 2), \text{dirty})$, take action W.

- New state: $s_5 = ((2, 1), \text{dirty})$
- Reward: $R_4 = -1$

**... and the robot continues West, North, North back to the dock. After 4 more steps, it enters $((0, 0), \text{dirty})$ and gets $+20$, and the bin becomes empty again. Let's say the reward sequence is:**

$$
R_0, R_1, R_2, R_3, R_4, R_5, R_6, R_7, R_8 = -1, -1, -1, -1, -1, -1, -1, -1, +20
$$

**Total discounted return from the start:**

$$
G_0 = \sum_{k=0}^{8} \gamma^k R_k = \sum_{k=0}^{7} 0.9^k \cdot (-1) + 0.9^8 \cdot 20
$$

Computing term by term:

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

Sum:

$$
G_0 = -5.6953 + 8.6093 \approx 2.91
$$

**Reading:** the robot ends up $+2.91$ ahead — the large terminal reward outweighs the accumulated step costs, even after discounting. That's what a "good" trajectory looks like under this reward structure.

**What if the robot took an 18-step path instead?** Then the discounted terminal reward would be $0.9^{17} \cdot 20 \approx 3.57$, and the accumulated step costs would be much larger ($\sum_{k=0}^{16} 0.9^k \cdot (-1) \approx -8.43$). The total would be $\approx -4.86$. **Negative.** The robot loses more to step costs than it gains from the terminal reward. That's why the optimal policy avoids meandering.

**What if the robot took an infinite-length meandering path and never reached the dock?** Then $G_0 = \sum_{k=0}^{\infty} 0.9^k \cdot (-1) = -1 / (1 - 0.9) = -10$. This is the worst case — purely burning energy without ever collecting reward. $\gamma < 1$ is what makes the sum finite (otherwise it would be $-\infty$).

This one walkthrough already demonstrates everything $\gamma$, $R$, and $G_t$ are trying to balance. **The whole rest of RL is finding policies that maximize this number in expectation across trajectories.**

---

## The Markov property

The "M" in MDP. Formally:

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0, a_0) = P(s_{t+1} \mid s_t, a_t)
$$

In English: **the probability of the next state depends only on the current state and action, not on the entire history of states and actions.** The present state summarizes everything relevant from the past.

This is a huge assumption, and it's what makes the math tractable. Without it, the agent would need to remember every past state and action, and the "state" would effectively grow with time. With it, the agent only needs to reason about the current state.

**Why the vacuum example is Markov:** given `(cell, bin_status)`, you know everything relevant about the future. It doesn't matter *how* the robot got to $(1, 1)$ with a dirty bin — whether via $(2, 2) \to (2, 1) \to (1, 1)$ or $(2, 2) \to (1, 2) \to (1, 1)$ — the next step's possibilities are identical. The `(cell, bin_status)` summary is sufficient.

**Why the vacuum example would NOT be Markov if the state were just `cell`:** then $(1, 1)$ could mean "robot is at $(1, 1)$ with no dirt yet" or "robot is at $(1, 1)$ with dirt in the bin." The optimal next action differs between the two. So "cell alone" isn't a sufficient state — it fails the Markov property. Adding `bin_status` fixes it. **That's state augmentation.**

**Not all problems are naturally Markov.** Consider a robot navigating by reading a single sensor reading at a time — the sensor alone doesn't tell you velocity, so the agent can't predict the future from one reading. The fix is to redefine the state to include enough history (e.g., the last 4 frames of Atari, which implicitly encodes velocity). This is called **state augmentation** and it's a universal trick: if a problem isn't Markov in the raw observations, redefine the state until it is.

```
Non-Markov (raw sensor):        Markov (augmented state):

   o₀ → o₁ → o₂ → ...             s_t = (o_{t-3}, o_{t-2}, o_{t-1}, o_t)
    ↑
   single sensor reading           includes enough history
   doesn't capture velocity        to predict the future
```

Cross-reference: [../ddpm/07_markov_vs_rnn_lstm_transformer.md](../ddpm/07_markov_vs_rnn_lstm_transformer.md) covers Markov chains in general — DDPM's forward process is a Markov chain in the physicist sense, and MDPs are Markov chains that also have actions and rewards layered on top.

---

## Policy, value function, and Q-function

The MDP tuple $(S, A, P, R, \gamma)$ defines the *environment*. The *agent* is defined by these three things, which it's trying to learn:

### Policy $\pi(a \mid s)$

A **policy** is a rule for picking actions. It's a function from states to actions (or a distribution over actions):

$$
\pi: S \to A \qquad \text{(deterministic)}
$$

$$
\pi(a \mid s) \in [0, 1], \qquad \sum_a \pi(a \mid s) = 1 \qquad \text{(stochastic)}
$$

The agent's entire job is to find a **good policy** — ideally an **optimal policy** $\pi^*$ that maximizes expected discounted return from every state.

**In the vacuum:** a sensible hardcoded policy is "if bin is empty, take shortest path to $(2, 2)$; if bin is dirty, take shortest path to $(0, 0)$." A learned policy might discover this or something subtly different. An optimal policy is whichever one maximizes expected discounted return from every starting state — which, for this MDP, is exactly the shortest-path policy (because rewards are deterministic and there's no randomness to hedge against).

### State-value function $V^\pi(s)$

Given a policy $\pi$, the **value function** $V^\pi(s)$ is the expected discounted return if you start in state $s$ and follow $\pi$ forever:

$$
V^\pi(s) = \mathbb{E}_\pi \!\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k} \;\middle|\; s_t = s \right]
$$

Reading: "starting from $s$, how much reward do I expect to accumulate (discounted) if I play according to $\pi$?"

The **optimal value function** $V^*(s) = \max_\pi V^\pi(s)$ is the best value achievable from state $s$ under any policy.

**In the vacuum:** we just computed $V^\pi((0, 0), \text{empty}) \approx 2.91$ for the shortest-path policy above. $V^\pi((2, 2), \text{dirty})$ would be the expected return starting with dirt already in the bin and at the dirt cell — probably around $0.9^4 \cdot 20 - \sum_{k=0}^{3} 0.9^k \approx 13.12 - 3.44 \approx 9.68$. States closer to "task complete" have higher value.

### Action-value function $Q^\pi(s, a)$

Same idea, but now conditioning on a specific first action before following $\pi$:

$$
Q^\pi(s, a) = \mathbb{E}_\pi \!\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k} \;\middle|\; s_t = s,\ a_t = a \right]
$$

Reading: "starting from $s$, if I take action $a$ first and then follow $\pi$, how much reward do I expect?"

$Q^\pi$ is usually more useful than $V^\pi$ because it directly tells you the value of each action. Once you have $Q^*$, the optimal policy is just $\pi^*(s) = \arg\max_a Q^*(s, a)$ — no planning needed.

Relationship:

$$
V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)
$$

The state value is the expected action value under the policy.

**In the vacuum:** from state $((1, 1), \text{empty})$ with the shortest-path policy, $Q((1, 1)\text{empty}, \text{S})$ and $Q((1, 1)\text{empty}, \text{E})$ would both be high (both lead toward the dirt), while $Q((1, 1)\text{empty}, \text{N})$ and $Q((1, 1)\text{empty}, \text{W})$ would be lower (they move away from dirt).

---

## The agent-environment loop

```
  ┌──────────────────────────────────────┐
  │                                      │
  │               agent                  │
  │                 │                    │
  │                 │ a_t                │
  │                 ▼                    │
  │        ┌─────────────────┐           │
  │        │   environment   │           │
  │        └─────────────────┘           │
  │                 │                    │
  │                 │ s_{t+1}, r_{t+1}   │
  │                 ▼                    │
  │               agent                  │
  │                                      │
  └──────────────────────────────────────┘

  loop forever:
    1. agent observes state s_t
    2. agent picks action a_t ~ π(·|s_t)
    3. environment transitions to s_{t+1} ~ P(·|s_t, a_t)
    4. environment emits reward r_{t+1} = R(s_t, a_t)
    5. agent updates its policy or value function
```

Everything RL does is a variation on this loop. The differences between RL algorithms are:
- What the agent stores (value function? Q-function? policy directly?)
- How the agent updates it (value iteration, Q-learning, policy gradient, actor-critic, ...)
- How the agent chooses actions (greedy? ε-greedy? sampled from $\pi$? Thompson sampling?)

---

## Common confusions

Real misconceptions students carry into quiz day. Each is stated as the confusion first, then corrected.

**Confusion 1: "An MDP is the same thing as a Markov chain."**

*Reality:* A Markov chain has states and transitions only: $(S, P)$. No actions, no rewards, no agent, no discount. An MDP adds three more things: actions $A$, rewards $R$, and the discount factor $\gamma$. A Markov chain describes a *process* (e.g., the weather tomorrow given the weather today); an MDP describes a *decision problem* (what should the agent choose at each step to maximize long-term reward?). DDPM's forward process is a Markov chain — no agent makes decisions, the noise is fixed. Atari is an MDP — there's an agent and actions.

**Confusion 2: "The Markov property means the agent has no memory."**

*Reality:* The Markov property says the *state* has no memory — i.e., the state is defined to include everything relevant from the past. The agent can absolutely have memory (stored inside the state definition, via state augmentation, or via an LSTM policy network), but the "Markov" part is about how the state relates to history, not how the agent's brain works. If you augment the state to include the last 4 observations (like DQN on Atari), the new state is Markov even though the raw observations aren't.

**Confusion 3: "$\gamma$ is a physical property of the environment."**

*Reality:* $\gamma$ is a modeling choice made by the problem designer. It's a tunable hyperparameter, not a law of nature. The environment doesn't "know" what $\gamma$ is. Two different agents with different $\gamma$ values both operate on the exact same environment — they just choose different policies. Changing $\gamma$ changes the agent's effective planning horizon; it does not change the physics of the world.

**Confusion 4: "The reward function must be nonnegative."**

*Reality:* Rewards can be any real number, positive or negative. Step penalties ($-1$ per step), punishment for wrong moves ($-100$ for crashing), and cost-of-living terms are all common. The only thing required is that you know how to combine them into discounted cumulative return — and that works fine with negative numbers. In the vacuum example, most rewards are $-1$, and the only positive reward comes at task completion.

**Confusion 5: "$P(s' \mid s, a)$ is what the agent is learning."**

*Reality:* $P$ is a property of the **environment**, not the agent. In model-based RL, the agent might try to learn an approximation of $P$ as a tool, but in most modern deep RL (Q-learning, DQN, REINFORCE, actor-critic, PPO), the agent never explicitly represents $P$ at all. Model-free RL learns a **policy** or **value function** directly from experience, bypassing $P$ entirely. When you see "$P$" in an equation, that's the true environment dynamics, not something the agent computed.

**Confusion 6: "Value function and reward function are the same thing."**

*Reality:* $R(s, a)$ is the **immediate** reward for one action. $V^\pi(s)$ is the **expected cumulative discounted future reward** from following $\pi$ starting at $s$. $R$ is what the environment gives you; $V$ is what the agent computes by thinking about the future. A single step of $R$ might be $-1$ (step penalty), but $V$ for the same state could easily be $+8$ if the agent knows a $+20$ reward is coming a few steps ahead.

---

## Check your understanding

Answer each question in your head before expanding. If you can't answer without looking, re-read the relevant section.

<details>
<summary><strong>Q1.</strong> In the robot vacuum example, is the state $((0, 0), \text{empty})$ the same state as $((0, 0), \text{dirty})$? Why does the distinction matter?</summary>

**No, they're different states.** Even though the robot is in the same physical cell in both, the bin status is different. $((0, 0), \text{empty})$ is the initial state (start of episode) or the state right after completing a task (dock, empty bin). $((0, 0), \text{dirty})$ is a *different* situation — it's the moment the robot arrives at the dock carrying dirt, which triggers the $+20$ reward and transitions the bin to empty.

**Why it matters:** the optimal action differs between them. In $((0, 0), \text{empty})$ the robot should go collect dirt (S or E to approach $(2, 2)$). In $((0, 0), \text{dirty})$ the robot has just finished and the episode effectively ends (bin transitions to empty and the $+20$ reward is collected). If the state ignored bin status, the MDP would violate the Markov property — "where should I go next from $(0, 0)$?" would depend on history (did I collect dirt already or not?), which is the exact failure mode the Markov property forbids.
</details>

<details>
<summary><strong>Q2.</strong> Suppose we change the vacuum's movement rules so that action N has a 20% chance of doing nothing (battery glitch) instead of moving. Does this change the state space, the action space, or the transition probability?</summary>

**Only the transition probability changes.** The state space $S$ still has the same 18 states (cell × bin status). The action space $A$ still has the same 4 actions. But now $P$ is **stochastic** for N actions:

$$
P(\text{new cell is North} \mid s, N) = 0.8, \quad P(\text{stay in place} \mid s, N) = 0.2
$$

The environment is still a valid MDP — the Markov property still holds, the equations still apply. It just became stochastic. The optimal policy might also change: now N is less reliable than other directions, so the planner should prefer alternate routes when possible.
</details>

<details>
<summary><strong>Q3.</strong> If $\gamma = 0$, what policy maximizes expected return in the vacuum?</summary>

**A myopic policy that picks whichever action has the highest immediate reward.** With $\gamma = 0$, the return formula becomes $G_t = R_t$ — future rewards are completely ignored. So the optimal action is just $\arg\max_a R(s, a)$.

In the vacuum, this is a disaster: almost every action gives $R = -1$, so the agent can't distinguish between useful actions and useless ones. The only action that gives a different reward is the one that enters $(0, 0)$ with a dirty bin ($+20$). But from any state more than one step away from completion, all actions look identical ($-1$), so a myopic agent would pick randomly.

**Lesson:** $\gamma = 0$ is almost never useful because it destroys the agent's ability to plan. Real RL uses $\gamma$ close to 1 (0.9 to 0.99).
</details>

<details>
<summary><strong>Q4.</strong> Why do we need both $V^\pi$ and $Q^\pi$? Isn't one enough?</summary>

**You can derive one from the other, but in practice both are useful at different times.**

- $V^\pi(s)$ answers "how valuable is state $s$?" It's a function of state only.
- $Q^\pi(s, a)$ answers "how valuable is taking action $a$ in state $s$?" It's a function of state and action.

You can compute $V^\pi$ from $Q^\pi$ via $V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$.

You can go the other way only if you know the transition function: $Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s')$.

**Why use $Q$ in practice:** once you have $Q^*$, selecting the optimal action is $\arg\max_a Q^*(s, a)$ — no model of the environment needed. This is what **Q-learning** and **DQN** do. You can act optimally just by knowing $Q^*$.

**Why use $V$ in practice:** it's simpler, has a smaller output space (one value per state instead of one per state-action pair), and is easier to train. Actor-critic methods use $V$ as a baseline for the policy gradient (we'll see this in [quiz_5_06_policy_gradients_and_reinforce.md](quiz_5_06_policy_gradients_and_reinforce.md)).

So both are used, but in different contexts. $V$ for evaluation and baselines; $Q$ for action selection.
</details>

<details>
<summary><strong>Q5.</strong> The vacuum's state space has 18 states: $9$ cells × $2$ bin statuses. If we added a "battery level" (integer from 0 to 100), how many states would there be? Would the algorithms in the rest of quiz 5 still work?</summary>

**The state space would grow to $9 \times 2 \times 101 = 1818$ states.** Still small enough for tabular methods (Q-learning, value iteration) to enumerate.

But if we made battery continuous (any real number between 0 and 100), the state space becomes **uncountably infinite**. Tabular methods break — you can't store or update a Q-value for every real-number battery level.

**The fix:** use function approximation. Instead of a lookup table $Q(s, a)$, use a neural network $Q_\theta(s, a)$ that takes the state as input and outputs Q-values. This is what DQN does — same algorithm, just with a neural net in place of a table. See [quiz_5_05_dqn_deep_dive.md](quiz_5_05_dqn_deep_dive.md).

**Takeaway:** the MDP framework works for discrete and continuous state spaces equally well. The framework itself has no size limit. What changes is the *algorithm* you use to solve it.
</details>

---

## Additional worked examples

Three more examples at different difficulty levels to build intuition across the full range of MDPs.

### Example A — trivial: a 2-cell "hallway"

```
   ┌───┬───┐
   │ L │ R │        episode ends when agent lands in R
   └───┴───┘
```

- $S = \{L, R\}$
- $A = \{\text{stay}, \text{move}\}$
- $P(R \mid L, \text{move}) = 1$, $P(L \mid L, \text{stay}) = 1$, etc.
- $R(L, \text{move}) = +1$ (landing in R), all other rewards 0
- $\gamma = 0.9$

**Optimal policy:** always move. $V^*(L) = 1$, $V^*(R) = 0$ (terminal). The smallest non-trivial MDP that has any interesting behavior — one action decision, one reward, everything collapses immediately. If you can solve this by hand, you understand the framework.

### Example B — quiz-level: slippery gridworld

Same as the 4-cell gridworld in the existing section, but with **stochastic transitions**:

- "up" from $s_0$ lands in $s_1$ **80% of the time**, slips to $s_3$ 10% of the time, stays in place 10% of the time
- Similar slipping rules for all other actions

Now $P$ is non-trivial. The Bellman expectation equation:

$$
V^\pi(s_0) = \sum_a \pi(a \mid s_0) \sum_{s'} P(s' \mid s_0, a) \left[ R(s_0, a) + \gamma V^\pi(s') \right]
$$

actually requires computing the sum over next states. The optimal policy might prefer some directions over others depending on which slip directions hurt least.

**Why this matters:** real-world RL is stochastic. A "slippery gridworld" is the simplest possible stochastic MDP — easy to draw, easy to compute, and it exposes the full Bellman equation machinery that deterministic gridworlds hide. If you're confident with the deterministic case, the slippery case is the next step.

### Example C — edge case: non-Markov observations

Imagine the robot vacuum **can only see whether its current cell has dirt**, not which cell it's in. The "observation" is a single bit: `{dirt, no dirt}`.

Is this a valid MDP? **No.** The observation $o_t$ is not a sufficient statistic for the future. Two different underlying states ($(0, 0), \text{empty}$ and $(1, 0), \text{empty}$, say) produce the same observation (`no dirt`), but their futures are different (different distances from the dirt cell). The Markov property fails.

**Two fixes:**

1. **State augmentation**: define the state as the last $k$ observations, plus maybe actions. E.g., $s_t = (o_{t-3}, a_{t-3}, o_{t-2}, a_{t-2}, o_{t-1}, a_{t-1}, o_t)$. This gives the agent enough history to implicitly infer which cell it's in.

2. **Use a POMDP framework**: Partially Observable MDP. The agent maintains a **belief state** $b(s)$ — a probability distribution over which underlying state it could be in — and updates the belief using Bayesian inference. POMDPs are strictly more general than MDPs but strictly harder to solve. Out of scope for this quiz; flagged here as the "next level up" if you continue studying RL.

In practice, deep RL handles partial observability by using **recurrent networks** (LSTM, GRU, or transformer) as the policy or value function, which implicitly maintain a belief state through the hidden activations. This is what DeepMind's AlphaStar did for real-time strategy games.

---

## Connections to other topics

How this note's concepts show up in every other quiz 5 note.

- **In [quiz_5_02_dynamic_programming.md](quiz_5_02_dynamic_programming.md)**: dynamic programming solves the MDP defined here when $P$ and $R$ are known. The Bellman equations (expectation and optimality) are recursive relationships between $V^\pi$ or $V^*$ at different states, and value iteration / policy iteration are algorithms that solve them.
- **In [quiz_5_03_exploration_vs_exploitation.md](quiz_5_03_exploration_vs_exploitation.md)**: the agent-environment loop is where exploration matters. At every step, the agent picks an action from $\pi$; exploration strategies (ε-greedy, UCB, etc.) are different ways of picking.
- **In [quiz_5_04_challenges_of_rl.md](quiz_5_04_challenges_of_rl.md)**: the sparse reward challenge is about $R$ being zero almost everywhere. The credit assignment challenge is about figuring out which past action earned a reward many steps later. Both are properties of the MDP's reward structure, not the learning algorithm.
- **In [quiz_5_05_dqn_deep_dive.md](quiz_5_05_dqn_deep_dive.md)**: DQN approximates $Q^*(s, a)$ (the optimal action-value function of the MDP) with a neural network $Q_\theta(s, a)$.
- **In [quiz_5_06_policy_gradients_and_reinforce.md](quiz_5_06_policy_gradients_and_reinforce.md)**: REINFORCE parameterizes the policy $\pi_\theta(a \mid s)$ as a neural network and maximizes expected return $\mathbb{E}_{\tau \sim \pi_\theta}[G_0]$, where the trajectory distribution is determined by the MDP.
- **In [../ddpm/07_markov_vs_rnn_lstm_transformer.md](../ddpm/07_markov_vs_rnn_lstm_transformer.md)**: DDPM's forward process is a Markov chain — states and transitions only, no actions or rewards. It's a pure Markov chain, not an MDP. The "M" is the same; the rest is different.

Everything downstream of this note — every RL algorithm, every deep RL paper, every policy optimization method — assumes this vocabulary. Master the 5-tuple and the agent-environment loop, and you have the foundation for everything else.

---

## History/lore

- **1957 — Richard Bellman** publishes *Dynamic Programming* (Princeton University Press). This book introduces the Bellman equation and the formal framework for sequential decision-making under uncertainty. Bellman was working at RAND Corporation on military operations research; his first applications were to resource allocation and inventory control, not games. The term "dynamic programming" was chosen deliberately to be vague — Bellman wrote in his 1984 autobiography that "programming" was a hot buzzword at the time, and he wanted a name that would get his work funded by a suspicious US Secretary of Defense who reportedly "had a pathological fear and hatred of the word 'research'."
- **1953 — Ronald Howard** publishes *Dynamic Programming and Markov Processes* as his MIT PhD thesis, formalizing policy iteration. Howard's framework made the Bellman equations concrete for practitioners.
- **1988 — Sutton** formalizes temporal difference (TD) learning in *Learning to Predict by the Methods of Temporal Differences*. TD learning is the insight that you don't need to wait for the end of an episode to update your value estimates — you can bootstrap by comparing your prediction at $t$ against your prediction at $t+1$.
- **1989 — Watkins** publishes his PhD thesis *Learning from Delayed Rewards* at King's College Cambridge, introducing **Q-learning**. Q-learning is the first model-free algorithm with a proof of convergence to the optimal Q-function under standard conditions.
- **1998 — Sutton & Barto** publish *Reinforcement Learning: An Introduction* (MIT Press), the definitive textbook on RL. Every RL researcher today learned from this book (the 2018 second edition is even better). Richard Sutton went on to coin "the bitter lesson" (2019): the general-purpose methods that leverage computation always eventually beat the clever domain-specific methods.

The vocabulary ("MDP," "policy," "value function," "Bellman equation") is all from the 1950s–1980s. Deep RL (2013 onward) doesn't change any of these definitions — it just uses neural networks as function approximators for the value function or policy, in place of lookup tables.

---

## Glossary

Every technical term used in this note, alphabetical, one-line definition.

- **Action** — a choice the agent makes at each step; one element of the action space $A$.
- **Action space $A$** — the set of all possible actions the agent can take.
- **Action-value function $Q^\pi(s, a)$** — the expected discounted return from taking action $a$ in state $s$ and then following policy $\pi$.
- **Agent** — the entity that observes states and picks actions. The "learner" in RL.
- **Bellman equation** — a recursive relationship between the value of a state and the values of its successor states. Appears in both expectation form (for a fixed policy) and optimality form (for the optimal policy).
- **Deterministic transition** — $P(s' \mid s, a)$ equals 1 for exactly one $s'$ and 0 for all others. The action always has the same outcome.
- **Discount factor $\gamma$** — a number in $[0, 1)$ that weights future rewards. Smaller means more myopic; larger means longer effective horizon.
- **Environment** — everything outside the agent. Produces next states and rewards in response to actions. Defined by $(S, A, P, R)$.
- **Episode** — a sequence of state-action-reward-state transitions that ends at a terminal state (or runs forever, for continuing tasks).
- **Markov chain** — a sequence of states with transition probabilities, no actions or rewards. $(S, P)$ only.
- **Markov Decision Process (MDP)** — the 5-tuple $(S, A, P, R, \gamma)$ that defines an RL problem.
- **Markov property** — $P(s_{t+1} \mid s_t, a_t, \text{history}) = P(s_{t+1} \mid s_t, a_t)$. The future depends only on the current state and action, not the past.
- **Model-based RL** — RL methods that explicitly learn or use $P$ and $R$ for planning.
- **Model-free RL** — RL methods that learn a policy or value function directly from experience without explicitly modeling $P$ and $R$. Q-learning, DQN, REINFORCE are all model-free.
- **Optimal policy $\pi^*$** — a policy that maximizes expected discounted return from every state.
- **Optimal value function $V^*$** — the value of the best possible policy; $V^*(s) = \max_\pi V^\pi(s)$.
- **Policy $\pi(a \mid s)$** — the agent's rule for picking actions given states. Can be deterministic ($\pi(s) = a$) or stochastic (a distribution over actions).
- **POMDP** — Partially Observable MDP. When the agent doesn't see the full state, just an observation.
- **Return $G_t$** — total discounted future reward from time step $t$: $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k}$.
- **Reward $R(s, a)$** — the immediate scalar feedback the environment gives for taking action $a$ in state $s$.
- **State $s$** — one configuration of the environment. One element of the state space $S$.
- **State augmentation** — redefining the state to include enough history that the Markov property holds.
- **State space $S$** — the set of all possible states.
- **State-value function $V^\pi(s)$** — the expected discounted return starting from $s$ and following $\pi$ thereafter.
- **Stochastic transition** — $P(s' \mid s, a)$ has multiple possible next states with non-zero probability. Actions can have different outcomes.
- **Terminal state** — a state where the episode ends. No further actions are taken.
- **Transition probability $P(s' \mid s, a)$** — the probability of the environment transitioning to state $s'$ when action $a$ is taken in state $s$.

---

## Takeaway

- **An MDP is the 5-tuple $(S, A, P, R, \gamma)$**: state space, action space, transition probability, reward, discount factor.
- **The Markov property** says the next state depends only on the current state and action, not on history — redefine the state until this holds if it doesn't naturally.
- **Policy $\pi(a \mid s)$** is what the agent picks (actions given states). **Value function $V^\pi(s)$** and **Q-function $Q^\pi(s, a)$** are what the agent computes (expected future return). The optimal policy is $\pi^*(s) = \arg\max_a Q^*(s, a)$.
- **The agent-environment loop** is the whole of RL in five lines: observe, act, transition, reward, update.
- **The robot vacuum example** is the mental picture to carry into every other RL topic: 18 states, 4 actions, a step penalty plus a task-completion reward, $\gamma = 0.9$ to encourage fast paths. Every algorithm below solves some version of this kind of problem.
- **History**: Bellman 1957 invented the framework; Sutton & Barto 1998 is the textbook; deep RL (2013+) plugged neural networks into the same old equations.

Next note: [quiz_5_02_dynamic_programming.md](quiz_5_02_dynamic_programming.md) — how to solve an MDP when you know $P$ and $R$.
