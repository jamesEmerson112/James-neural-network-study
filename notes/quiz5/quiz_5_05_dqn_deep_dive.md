# 05. DQN — Deep Q-Network Deep Dive

## What this note unpacks

DQN (Mnih et al. 2013, 2015) is the algorithm that started deep RL. It's **Q-learning + a deep neural network as the Q-function approximator + two stability tricks** (experience replay and target networks). Before DQN, Q-learning with neural networks was notoriously unstable and usually diverged. DQN showed that with the right tricks, it could master Atari games from raw pixels and reach human-level performance.

This note covers:
- Q-learning as a one-line update rule
- Why deep Q-learning is unstable (the deadly triad)
- Experience replay — what it does and why
- Target networks — what they do and why
- The DQN loss function explicitly
- The DQN algorithm in pseudocode
- Mnih 2013 / 2015 history

---

## Starting point: tabular Q-learning

The classic Q-learning update (Watkins 1989) is:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

Reading: "nudge $Q(s, a)$ toward the observed reward plus the discounted best-next-value."

The quantity in square brackets is the **TD error**:

$$
\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)
$$

"What I just observed" ($r + \gamma \max_{a'} Q(s', a')$) minus "what I predicted" ($Q(s, a)$). The update moves $Q(s, a)$ in the direction that reduces this error.

For a **tabular** Q-function (one value per $(s, a)$ pair), this converges to $Q^*$ under mild conditions (all pairs visited infinitely often, decaying step size). For a **deep neural network** Q-function, it doesn't.

---

## Why deep Q-learning is unstable: the deadly triad

Sutton & Barto coined **the deadly triad**: three properties that, when combined, cause value-based RL to diverge:

1. **Function approximation** (e.g., a neural network instead of a lookup table)
2. **Bootstrapping** (the target uses the model's own prediction: $r + \gamma Q_\theta(s', a')$)
3. **Off-policy learning** (the update distribution differs from the policy distribution)

Q-learning has all three. When you plug in a deep net, the three interact badly:

- **Correlated samples.** Consecutive transitions in a single episode are highly correlated (same state, adjacent actions, similar reward). SGD assumes samples are i.i.d. Training on correlated samples biases the gradient.
- **Moving target.** The target $r + \gamma \max_{a'} Q_\theta(s', a')$ depends on $\theta$. As you update $\theta$, the target moves. You're regressing onto a quantity that changes every step.
- **Catastrophic interference.** A neural net trained on correlated data in one region of state space tends to forget everything it learned about other regions.

**Without fixes**, deep Q-learning with a naive loss $\|r + \gamma \max_{a'} Q_\theta(s', a') - Q_\theta(s, a)\|^2$ diverges to infinity on most problems. DQN's two tricks — experience replay and target networks — are the minimum set of hacks that make it stable.

---

## Fix 1: experience replay

**Idea:** store past transitions in a buffer, then sample training batches randomly from the buffer instead of using the most recent transitions.

```
   environment
        │
        │ (s, a, r, s')
        ▼
  ┌─────────────────────┐
  │   replay buffer     │   ← store millions of past transitions
  │   (FIFO, ~1M size)  │
  └─────────────────────┘
        │
        │ random sample
        ▼
   training batch
        │
        ▼
   SGD update on Q_θ
```

**What it buys you:**

1. **Decorrelates samples.** Random sampling from the buffer breaks the temporal correlation within episodes. The training batch looks much more i.i.d. — closer to the assumption SGD relies on.
2. **Sample efficiency.** A transition is used multiple times for updates, instead of seen once and thrown away.
3. **Smoother learning.** The update isn't dominated by whatever happened in the last few steps.

**Implementation details (from the 2015 Nature DQN paper):**
- Buffer size: 1 million transitions
- Batch size: 32
- Sample uniformly at random (later: prioritized experience replay samples with probability proportional to TD error)

---

## Fix 2: target network

**Idea:** use a **lagging copy** of the Q-network to compute the target, and update it only periodically.

Define two networks:
- $Q_\theta$ — the **online network**, which gets gradient updates every step
- $Q_{\theta^-}$ — the **target network**, whose parameters $\theta^-$ are a frozen snapshot of $\theta$

Every $C$ steps (e.g., $C = 10{,}000$), copy $\theta$ into $\theta^-$. Between copies, $\theta^-$ is held constant.

The target in the loss uses $Q_{\theta^-}$, not $Q_\theta$:

$$
\text{target} = r + \gamma \max_{a'} Q_{\theta^-}(s', a')
$$

**What it buys you:**

The target is held fixed for $C$ steps, so $Q_\theta$ is regressing onto something stationary for a while. When the target finally updates, the online network has already gotten closer to it, and the step change isn't catastrophic. This converts the "moving target" problem into a "target that moves in occasional discrete jumps," which is much more stable.

Without the target network:
```
  step 1:   target = r + γ·max Q_θ₁(s')       ← Q_θ₁ update → θ₂
  step 2:   target = r + γ·max Q_θ₂(s')       ← Q_θ₂ update → θ₃
  step 3:   target = r + γ·max Q_θ₃(s')       ← chasing own tail
```

With the target network:
```
  step 1:   target = r + γ·max Q_θ⁻(s')       ← fixed
  ...
  step C:   target = r + γ·max Q_θ⁻(s')       ← still fixed
  step C+1: θ⁻ ← θ                            ← one discrete update
  step C+1: target = r + γ·max Q_θ⁻(s')       ← new fixed value
```

---

## The DQN loss function

Putting both fixes together, the DQN loss is:

$$
L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}}\!\left[ \left( \underbrace{r + \gamma \max_{a'} Q_{\theta^-}(s', a')}_{\text{target (fixed)}} - \underbrace{Q_\theta(s, a)}_{\text{prediction}} \right)^2 \right]
$$

where:
- $\mathcal{D}$ is the replay buffer (decorrelated samples)
- $\theta^-$ is the target network (fixed target)
- $\theta$ is the online network (being updated)
- The square makes it a regression loss (MSE)

**Read it slowly:**
- $(s, a, r, s')$ is a transition: "in state $s$, took action $a$, got reward $r$, ended up in $s'$"
- $\max_{a'} Q_{\theta^-}(s', a')$ is the best value achievable from $s'$ according to the old target network
- $r + \gamma \max_{a'} Q_{\theta^-}(s', a')$ is the Bellman-optimality target
- $Q_\theta(s, a)$ is what the online network currently predicts
- The squared difference is minimized by gradient descent on $\theta$

**Crucially, the gradient only flows through $Q_\theta(s, a)$, not through $Q_{\theta^-}$.** You don't backprop into the target network — it's held fixed by construction.

---

## The DQN algorithm

```
algorithm: DQN
  initialize Q-network Q_θ with random weights
  initialize target network Q_θ⁻ ← Q_θ
  initialize replay buffer D (capacity ~1M)
  
  for episode = 1, 2, ...:
    initialize state s_0
    for t = 0, 1, ..., T:
      # ─── action selection (ε-greedy with decaying ε) ───
      with probability ε:
        a_t ← random action
      else:
        a_t ← argmax_a Q_θ(s_t, a)
      
      # ─── execute and observe ───
      execute a_t, observe r_t and s_{t+1}
      store transition (s_t, a_t, r_t, s_{t+1}) in D
      
      # ─── sample a mini-batch and update ───
      sample mini-batch of transitions (s_j, a_j, r_j, s_{j+1}) from D
      for each transition in the batch:
        if s_{j+1} is terminal:
          y_j ← r_j
        else:
          y_j ← r_j + γ · max_a' Q_θ⁻(s_{j+1}, a')    # target network used here
      
      # ─── gradient step on (y_j − Q_θ(s_j, a_j))² ───
      θ ← θ − α · ∇_θ Σ_j (y_j − Q_θ(s_j, a_j))²
      
      # ─── periodic target network sync ───
      every C steps: θ⁻ ← θ
      
      s_t ← s_{t+1}
```

That's the whole algorithm. Five non-trivial ideas: Q-learning, experience replay, target network, ε-greedy exploration, and a CNN as the function approximator. The clever insight of DQN was seeing that these five ingredients together are enough.

---

## Why DQN was a breakthrough

Before DQN, deep Q-learning didn't work. Gerald Tesauro had trained TD-Gammon in 1992 using a neural network to learn backgammon — but backgammon is a simple 2-player board game, and TD-Gammon needed extensive hand-engineered features. Getting a neural network to learn Atari from raw pixels was considered probably impossible.

The DQN paper showed:
- Q-learning scaled to deep networks **with the right stability tricks**
- A single architecture and single set of hyperparameters could master **49 Atari games** without per-game tuning
- Performance **matched or exceeded human experts** on most of them
- Input was **raw pixels** — no feature engineering

This was the moment the ML community realized deep RL was a real thing. AlphaGo (2016), AlphaStar (2019), OpenAI Five (2019), and RLHF (2022) are all downstream of DQN's proof that deep neural networks plus classical RL tricks could scale.

---

## Comparison: DQN vs REINFORCE

DQN and REINFORCE ([quiz_5_06_policy_gradients_and_reinforce.md](quiz_5_06_policy_gradients_and_reinforce.md)) are the two canonical deep RL algorithms, and they take opposite approaches:

| | DQN (value-based) | REINFORCE (policy-based) |
|---|---|---|
| What it learns | $Q_\theta(s, a)$ (value of each action) | $\pi_\theta(a \mid s)$ (the policy directly) |
| How it acts | Greedy on $Q_\theta$, $\varepsilon$-greedy for exploration | Sample from $\pi_\theta$ |
| Action space | Discrete (argmax over actions) | Discrete or continuous |
| On/off policy | Off-policy (replay buffer) | On-policy (uses fresh trajectories only) |
| Bias/variance | Biased (bootstrapping) but low variance | Unbiased but high variance |
| Sample efficiency | High (replay buffer reuses data) | Low (throws data away) |
| Stability tricks | Experience replay + target network | Baselines + advantage function |
| Landmark applications | Atari (DQN 2015), Rainbow (2017) | Humanoid (2016), OpenAI Five (2019 via PPO) |

**Rule of thumb:** if your action space is discrete and small, use a value-based method (DQN, Rainbow). If it's continuous, use a policy-based method (DDPG, SAC, PPO). If you want the best of both, use actor-critic (A2C, A3C, PPO), which learns both $\pi_\theta$ and $V_\phi$.

---

## History/lore

- **1989 — Chris Watkins** introduces Q-learning in his Cambridge PhD thesis *Learning from Delayed Rewards*. Proves convergence for tabular Q-learning. Watkins later went on to work at DeepMind.
- **1992 — Gerald Tesauro** builds **TD-Gammon** at IBM Watson. A neural network learns backgammon via self-play and temporal difference learning. Reaches expert level and reportedly changes how humans play (some of its opening moves were novel). This is the earliest "deep RL" result, but the network was small by modern standards and backgammon is a simple domain.
- **2013 — Mnih, Kavukcuoglu, Silver, Graves, Antonoglou, Wierstra, Riedmiller** (DeepMind) publish *Playing Atari with Deep Reinforcement Learning* at the NeurIPS Deep Learning workshop. This is the first DQN paper. It introduces experience replay and the basic architecture, and shows good results on 7 Atari games.
- **2014 — DeepMind acquired by Google** for ~$500 million. DQN is a big part of the pitch. Founders Demis Hassabis, Shane Legg, and Mustafa Suleyman become part of Google.
- **2015 — Mnih et al.** publish the full DQN paper *Human-level Control through Deep Reinforcement Learning* in *Nature*. This is the version with the target network (the 2013 paper didn't have it). Results: matches or beats human experts on 49 Atari games using a single architecture and single hyperparameter setting. The paper has >20,000 citations and is one of the most influential ML papers ever.
- **2017 — Hessel et al.** publish *Rainbow* — a combined agent using six improvements on top of DQN (double Q-learning, prioritized experience replay, dueling networks, multi-step learning, distributional RL, noisy nets). Rainbow remains the state of the art on Atari for value-based methods.
- **2018 — Kapturowski et al.** publish R2D2 (*Recurrent Replay Distributed DQN*), scaling DQN to hundreds of actors in parallel on distributed infrastructure.
- **2020 — Badia et al.** publish *Agent57*, the first agent to beat human baselines on all 57 Atari games. Uses adaptive exploration and meta-controller.

The Atari benchmark became the ImageNet of RL — every algorithm since 2013 has been measured against it. Atari is "solved" in the sense that Agent57 beats humans on every game, but sample efficiency is still far from human (Agent57 needs tens of billions of frames; humans need minutes).

---

## Takeaway

- **DQN = Q-learning + deep network + experience replay + target network.** Four ingredients, all necessary.
- **Q-learning loss**: $L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a))^2]$
- **Experience replay** decorrelates samples and improves sample efficiency; buffer holds ~1M transitions, sample randomly.
- **Target network** $Q_{\theta^-}$ is a lagging copy that holds the regression target fixed for $C$ steps, preventing the "chasing own tail" divergence.
- **DQN = value-based**: learns $Q$, acts greedy. Compare to REINFORCE (policy-based, see [quiz_5_06_policy_gradients_and_reinforce.md](quiz_5_06_policy_gradients_and_reinforce.md)).
- **History**: Watkins 1989 (Q-learning) → Tesauro 1992 (TD-Gammon) → Mnih 2013/2015 (DQN) → Rainbow 2017 → Agent57 2020.

Next note: [quiz_5_06_policy_gradients_and_reinforce.md](quiz_5_06_policy_gradients_and_reinforce.md) — the policy-based side of deep RL, directly learning $\pi_\theta$ via the policy gradient theorem.
