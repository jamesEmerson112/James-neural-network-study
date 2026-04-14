# 04. Challenges of Reinforcement Learning

## What this note unpacks

Quiz topic 4 is qualitative: "**Why is RL harder than supervised learning?**" You need to be able to name at least four specific reasons and explain each in a sentence. This note gives you a clean list of the canonical challenges — the ones every RL textbook names — with enough depth to explain them to someone else.

None of these are problems the algorithms of notes 01–06 fully solve. They're the reasons RL research is still very much ongoing.

---

## The comparison table

| Property | Supervised learning | Reinforcement learning |
|---|---|---|
| Feedback signal | Correct label for every input | Scalar reward, often delayed or sparse |
| Data distribution | Fixed — the dataset doesn't change | **Non-stationary** — depends on the policy |
| Sample efficiency | ~1000s of examples | Often millions to billions of interactions |
| Reproducibility | Deterministic given the dataset | Stochastic from environment, exploration, initial conditions |
| Credit assignment | Label points directly at the error | Reward might apply to an action 100 steps earlier |
| Exploration | Not a concept | Fundamental — the agent's own choices affect what it learns from |
| Evaluation | Test on held-out data | Run full episodes in an environment |
| Typical training time | Hours to days | Days to weeks, sometimes months |

Every row is a challenge. The six canonical ones explained below correspond to the rows that are genuinely harder in RL, not just "different."

---

## Challenge 1: Sparse rewards

**The problem:** most actions in most environments produce **zero reward**. Only a tiny fraction of transitions (reaching a goal, scoring a point, winning a game) give any signal at all. The agent has to figure out long chains of "useless-looking" actions that eventually lead to the reward.

**Canonical example:** Montezuma's Revenge (Atari). The agent has to navigate a room, climb a ladder, jump over a skull, grab a key, and exit through a door before getting *any* reward. Vanilla DQN gets stuck on this game for millions of frames because it never stumbles into the reward by chance.

```
  action: ←    →    ↑    ↓    ←    →    ←    ←    ↑    ... (hundreds of steps)
  reward: 0    0    0    0    0    0    0    0    0    ... 100
                                                            ↑
                                                    first reward ever seen
```

Supervised learning doesn't have this problem: every example has a label, no matter how hard. In RL, the agent has to *generate* its own signal, and if it's exploring poorly, it never does.

**Partial fixes:** reward shaping (add intermediate rewards to guide the agent), curiosity / intrinsic motivation (reward the agent for visiting novel states), hierarchical RL (learn sub-goals).

## Challenge 2: Delayed rewards

**The problem:** rewards can be delayed by hundreds or thousands of steps. The action that caused the outcome might have happened a long time ago. The agent has to figure out "which of my past actions was actually responsible?"

**Canonical example:** chess. A bad opening move might not be punished until move 40. By then, which move was the bad one?

```
  move 1:  e4           (good move? bad move? effect unknown)
  move 2:  Nf3          (good move? bad move? effect unknown)
  ...
  move 40: king falls   ← big negative reward
```

The agent sees a single large reward at the end and has to allocate credit back across all 40 moves. This is extremely hard — especially since the "wrong" moves might be early, not late.

## Challenge 3: Credit assignment

**The problem:** closely related to delayed rewards, but general. Given that a trajectory had some total return, which actions were responsible? All of them? Some of them? The lucky one?

The Bellman equation and TD learning are the *canonical solution* to credit assignment — by writing $V(s_t) = r_t + \gamma V(s_{t+1})$, you propagate credit one step at a time from the future back to the past. But TD is only approximate and suffers from its own problems (bootstrap bias, variance).

**Why it's a problem even after TD learning:** TD learning solves credit assignment in expectation, not for individual trajectories. On any given trajectory, a lucky action (good outcome not caused by this action) and an unlucky action (bad outcome not caused by this action) both get credited as if they were responsible. You need a lot of data to average out the noise.

**The mental model:** imagine trying to teach a dog to sit by giving it a treat at the end of a 10-minute walk. The dog has no way to know *which* of its hundreds of behaviors during the walk was rewarded. This is the credit assignment problem in its cleanest form.

## Challenge 4: Non-stationarity (the moving target)

**The problem:** in supervised learning, you're regressing onto a fixed dataset. The labels don't move. In RL, **the data distribution depends on the policy**, and the policy is changing as you train. As the policy improves, it visits different states, generates different trajectories, and faces a different training distribution.

```
  early training:   policy is random → agent wanders → sees one distribution of states
  mid training:     policy has some structure → sees a different distribution
  late training:    policy is good → only visits reachable states near the optimum
```

This is fundamentally different from supervised learning. Every improvement changes the training set. That's why RL is sometimes described as "chasing a moving target."

**In deep RL**, this is compounded by the fact that the value function (e.g., $Q_\theta$) is also used to compute its own training targets (via the Bellman equation). As $\theta$ changes, the target changes too. This is why DQN uses a **target network** — a lagging copy that holds the target fixed for a while. See [quiz_5_05_dqn_deep_dive.md](quiz_5_05_dqn_deep_dive.md).

## Challenge 5: Sample inefficiency

**The problem:** RL needs absurdly more data than supervised learning to solve comparable problems. DQN needed ~200 million frames to master Atari games. AlphaGo needed millions of self-play games. Human experts can learn Atari games in minutes; DQN takes weeks of wall-clock time.

**Why:**
- Each trajectory contains limited information — mostly "this wasn't terrible, keep going"
- Exploration wastes many samples on states the agent will never visit again
- The non-stationary target means old data becomes stale
- High variance in policy gradient estimates means many samples are needed just for stable updates

**Fixes (the whole field of sample-efficient RL):** model-based RL (learn $P$, plan with it), offline RL (train on fixed datasets), meta-RL (learn to learn faster), imitation learning (bootstrap from human demos).

## Challenge 6: Exploration vs exploitation

**The problem:** covered in depth in [quiz_5_03_exploration_vs_exploitation.md](quiz_5_03_exploration_vs_exploitation.md). Every step, the agent must decide: try something new (might discover a better strategy) or stick with what works (guaranteed return). Too much of either is fatal — pure exploitation gets stuck in local optima; pure exploration never cashes in.

**Why it's uniquely an RL problem:** in supervised learning there's no choice to make. The dataset is given. In RL, the agent's own action choices determine which data it collects, so exploration *is* the data collection strategy. A poorly-exploring agent simply never sees the training signal it needs.

---

## Bonus challenges (nice to mention if the quiz asks for more than four)

**Instability and divergence.** Function approximation + off-policy learning + bootstrapping is called "the deadly triad" (Sutton & Barto). Together they can cause value estimates to diverge to infinity rather than converge. DQN partially fixes this with experience replay and target networks, but instability is still common in deep RL.

**Partial observability.** Real environments often don't show the full state — a robot has camera images, not ground-truth object positions. When the observation isn't Markov, you need to augment the state with history (e.g., stack the last 4 frames) or use recurrent networks.

**Reward hacking.** Agents sometimes find ways to maximize reward that don't match the designer's intent. Classic example: a boat racing agent learned to circle in place scoring power-ups instead of finishing the race. The reward function was technically correct but misaligned with the goal.

**Safety during exploration.** A robot learning to walk by trial and error might break itself. An autonomous car can't explore by crashing. This motivates *safe RL*, where the agent has constraints during learning.

**Hyperparameter sensitivity.** RL algorithms are notoriously sensitive to learning rate, discount factor, replay buffer size, etc. A working configuration on one environment often fails on another. This is why RL papers report multiple random seeds — individual runs vary enormously.

---

## One-line summaries for quick recall

| Challenge | One-line |
|---|---|
| Sparse rewards | Most transitions give zero signal; agent must stumble into reward |
| Delayed rewards | Action at step 5 might cause reward at step 1000 |
| Credit assignment | Which action in a long trajectory was responsible for the outcome? |
| Non-stationarity | Data distribution depends on policy; both move together |
| Sample inefficiency | Millions of interactions needed vs thousands for supervised |
| Exploration | Agent chooses its own training data via action selection |

**Minimum quiz answer** for "name four challenges of RL": sparse/delayed rewards, credit assignment, non-stationarity, sample inefficiency. Bonus points for mentioning exploration, the deadly triad, or partial observability.

---

## Why games became the RL testbed

The canonical observation: real-world RL is hard because reality has all six challenges above *plus* things like hardware safety and real-time constraints. Games fix most of these:

| Property | Real world | Video games |
|---|---|---|
| Reward signal | Fuzzy ("how well did the robot grip?") | Clear (score, win/lose) |
| Speed | Real-time, one attempt | Millions of episodes overnight |
| Cost | Hardware breaks | Free (simulation) |
| Reproducibility | Noisy, unpredictable | Same rules every run |
| Safety | Serious concern | Irrelevant |

Games are the RL equivalent of ImageNet — a clean, reproducible testbed where the algorithm is the only variable. That's why DQN (2013), AlphaGo (2016), OpenAI Five (2019), and AlphaStar (2019) are all game-playing agents. The hope is that techniques that work on games will eventually transfer to robotics and real-world control. Progress has been slower than the game milestones suggest.

---

## Takeaway

- **Sparse + delayed rewards + credit assignment**: the signal is rare, late, and hard to attribute.
- **Non-stationarity**: the data distribution moves as the policy changes — supervised learning doesn't have this.
- **Sample inefficiency**: RL needs millions of interactions for problems supervised learning solves with thousands.
- **Exploration vs exploitation**: the agent chooses its own training data via action selection.
- **Games became the default testbed** because they eliminate the practical difficulties (speed, cost, reward clarity) and leave only the algorithmic challenges.

Next note: [quiz_5_05_dqn_deep_dive.md](quiz_5_05_dqn_deep_dive.md) — DQN's specific fixes for the instability caused by deep function approximation + bootstrapping.
