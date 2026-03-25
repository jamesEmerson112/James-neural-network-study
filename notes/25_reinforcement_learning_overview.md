# Reinforcement Learning — Teaching Machines Through Trial and Error

> Part of the [Neural Network Study Timeline](00_timeline.md). See also: [Generative Models Taxonomy](24_generative_models_taxonomy.md), [Softmax](22_softmax.md).

---

## The Core Idea

Supervised learning: "Here's the answer, learn to predict it."
Reinforcement learning: "I won't tell you the answer. Try things, and I'll tell you if it was good or bad."

```
Agent  ──action──→  Environment
  ↑                      │
  └──state + reward──────┘

Loop forever:
  1. Agent observes the state (e.g., pixels on screen, board position)
  2. Agent picks an action (e.g., move left, place stone at D4)
  3. Environment returns a new state + reward (e.g., +1 for scoring, -1 for dying)
  4. Agent updates its strategy to get more reward over time
```

That's it. Everything in RL is a variation of this loop.

---

## The Three Flavors of Machine Learning

| Paradigm | Signal | Example |
|---|---|---|
| **Supervised** | Correct answer for every input | "This image is a cat" → learn to classify |
| **Unsupervised** | No labels at all | "Find patterns in this data" → clustering, generative models |
| **Reinforcement** | Delayed, sparse reward | "You scored 100 points after 1,000 moves" → figure out which moves mattered |

RL is uniquely hard because the reward is **delayed** (you might not know if a move was good until much later) and **sparse** (most actions get no feedback at all).

---

## Why Video Games?

Games became the default RL testbed for practical reasons:

| Property | Games | Real World |
|---|---|---|
| Reward signal | Clear (score, win/lose) | Fuzzy ("how well did the robot grip that cup?") |
| Speed | Millions of episodes overnight | Real-time, one attempt at a time |
| Cost | Free (simulation) | Expensive (hardware breaks, safety risks) |
| Reproducibility | Same rules every time | Noisy, unpredictable |
| Complexity range | Pong → StarCraft | Hard to control difficulty |

Games aren't the goal — they're the **gym** where RL algorithms get strong enough for real problems.

---

## The Historical Pipeline — From Checkers to ChatGPT

### Samuel's Checkers (1959) — The First

**Who**: Arthur Samuel, IBM

The very first program that learned from experience. Used self-play and a value function (estimate how good a board position is). Beat a former Connecticut state champion on live TV in 1956.

This is where the term "machine learning" was coined.

### TD-Gammon (1992) — Neural Nets Enter the Game

**Who**: Gerald Tesauro, IBM

A neural network that learned backgammon entirely through self-play. Used **temporal difference (TD) learning** — update your value estimate based on the *difference* between consecutive predictions, not just the final outcome.

```
TD learning insight:
  Don't wait until the game ends to learn.
  After each move, compare "how good I thought this position was"
  vs "how good the NEXT position turned out to be."
  Update based on that difference.
```

TD-Gammon reached expert-level play and changed how humans played backgammon — some of its discovered strategies were adopted by professional players.

**Why it matters**: proved that neural nets + self-play + RL could reach superhuman performance. But the field wasn't ready to generalize this — it took 20 more years.

### DQN (2013) — One Agent, 49 Games

**Who**: Mnih et al., DeepMind

**Paper**: "Playing Atari with Deep Reinforcement Learning"

The breakthrough: combine a **deep CNN** (to process raw pixels) with **Q-learning** (to estimate action values). One single architecture, with no game-specific engineering, learned to play 49 different Atari games from raw pixels alone.

```
Input:  raw screen pixels (210×160×3)
     ↓
CNN:  extract visual features
     ↓
Q-network:  estimate Q(state, action) for each possible action
     ↓
Output: pick the action with highest Q-value
```

Key innovations:
- **Experience replay**: store past experiences in a buffer, sample randomly to train (breaks correlation between consecutive experiences)
- **Target network**: use a slowly-updated copy of the Q-network for stable training targets

DQN was the paper that put DeepMind on the map and led to Google acquiring them for ~$500M in 2014.

### AlphaGo (2016) — The Cultural Earthquake

**Who**: Silver, Huang et al., DeepMind

Beat **Lee Sedol** (one of the greatest Go players ever) 4-1 in a match watched by 200 million people.

Go was considered the "holy grail" of game AI because:
- Board has 19×19 = 361 positions
- More possible board states than atoms in the universe (~10^170)
- Brute-force search is impossible
- Experts said "at least 10 years away" in 2015

AlphaGo combined:
1. **Supervised learning** — trained on millions of human expert games
2. **Reinforcement learning** — improved by playing against itself
3. **Monte Carlo Tree Search (MCTS)** — smart lookahead search guided by the neural net

**Move 37** in Game 2 became legendary — a move so unconventional that human commentators thought it was a mistake. It wasn't. AlphaGo had discovered strategies that 3,000 years of human Go hadn't found.

### AlphaGo Zero & AlphaZero (2017) — No Human Knowledge

**Who**: Silver et al., DeepMind

AlphaGo Zero removed all human expert data — learned **entirely from self-play**, starting from random play. It surpassed the original AlphaGo in 3 days.

**AlphaZero** generalized the approach to chess, shogi, and Go with the same algorithm. Beat the world's best chess engine (Stockfish) after just 4 hours of self-play training.

### AlphaStar (2019) — Real-Time Strategy

**Who**: Vinyals et al., DeepMind

Mastered **StarCraft II** — a game with:
- Real-time decisions (not turn-based)
- Imperfect information (fog of war — you can't see the whole map)
- Long-term planning over thousands of time steps
- Multiple units to coordinate simultaneously

Reached Grandmaster level, top 0.2% of human players.

### OpenAI Five (2019) — Team Coordination

**Who**: OpenAI

Beat world champion teams in **Dota 2** — a 5v5 team game requiring coordination between five agents. Each game lasts ~45 minutes with ~20,000 time steps.

Trained using **PPO (Proximal Policy Optimization)** — the same algorithm that would later be used for RLHF in ChatGPT.

### RLHF and ChatGPT (2022) — RL Meets Language

**Who**: Ouyang et al., OpenAI

**The connection**: the same RL techniques (specifically PPO) used to train game-playing agents were repurposed to **align language models** with human preferences.

```
Game RL:                          RLHF:
  Agent = game player               Agent = language model
  Action = move in game              Action = generate a response
  Reward = game score                Reward = human preference score
  Environment = game                 Environment = conversation
```

The "reward model" in RLHF is trained on human comparisons ("Response A is better than Response B"), then used to give reward signals to the language model via PPO.

This is how GPT-3 (a raw text predictor) became ChatGPT (a helpful assistant). The RL wasn't teaching it language — it was teaching it **how to be helpful**.

---

## Why RL Is One of the Hardest ML Subfields

### The Math Stack

RL sits at the intersection of multiple hard fields simultaneously:

```
Probability theory          — stochastic processes, Markov chains
Optimization                — gradient methods, convex/non-convex
Dynamic programming         — Bellman equations, value iteration
Statistics                  — estimation, bias-variance
Control theory              — optimal control, stability
Game theory (multi-agent)   — Nash equilibria, minimax
```

Most ML courses need linear algebra + calculus + probability. RL needs all of that **plus** dynamic programming and control theory.

### The Bellman Equation — The Heart of RL

Everything in RL traces back to this one recursive equation:

```
V(s) = max_a [ R(s,a) + γ · V(s') ]

In English:
  The value of a state =
    the best action's immediate reward
    + the discounted value of the next state

γ (gamma) = discount factor (0 to 1)
  "How much do I care about future rewards vs immediate rewards?"
```

This looks simple but it's recursive — the value of a state depends on the value of future states, which depend on the value of *their* future states, and so on. Solving this is what makes RL hard.

### The Instability Problem

In supervised learning, the dataset is fixed. In RL, everything shifts:

| What changes | Why it's a problem |
|---|---|
| Data distribution | You're learning from your own behavior, which keeps changing as you learn |
| Training target | Value estimates update as the model improves |
| Exploration vs exploitation | You need random actions to discover rewards, but also need to use what you've learned |

### The Credit Assignment Problem

```
You played 1,000 moves and won the game.
Which of those 1,000 moves actually mattered?

Move 37 might have been the winning move,
but you won't know until 963 moves later.
```

This is fundamentally harder than supervised learning where every example has an immediate label.

### Debugging is a Nightmare

```
Supervised learning:  loss goes down  → probably working
RL:                   reward goes down → could be exploring
                                       → could be diverging
                                       → could be a bug
                                       → could be a hyperparameter issue
                                       → could be fine and it'll recover
                                         in 10,000 more episodes
```

---

## RL Beyond Games — Real-World Applications

Games are the gym. Here's where RL went after getting strong:

| Application | What RL does | Who |
|---|---|---|
| **LLM alignment (RLHF)** | Teaches language models to be helpful, not just predict text | OpenAI, Anthropic |
| **Robotics** | Teaches robots to walk, grip, manipulate objects | Boston Dynamics, DeepMind |
| **Chip design (AlphaChip)** | Places transistors on chips, beating human engineers | Google DeepMind (2020) |
| **Protein folding (AlphaFold)** | Predicts 3D protein structures from amino acid sequences | DeepMind (2020, Nobel Prize 2024) |
| **Autonomous driving** | Learns driving policies from simulation before real-world deployment | Waymo, Tesla |
| **Energy optimization** | Reduced Google data center cooling costs by 40% | DeepMind (2016) |
| **Drug discovery** | Generates novel molecular structures optimized for target properties | Various pharma |

The PPO algorithm that trained OpenAI Five to play Dota 2 is the **same algorithm** that trained ChatGPT to follow instructions. Game → language model, same math.

---

## Key Concepts Glossary

| Term | Meaning |
|---|---|
| **Agent** | The learner / decision-maker |
| **Environment** | Everything the agent interacts with |
| **State (s)** | Current situation the agent observes |
| **Action (a)** | What the agent can do |
| **Reward (r)** | Scalar feedback signal |
| **Policy (π)** | Agent's strategy — maps states to actions |
| **Value function V(s)** | Expected total future reward from state s |
| **Q-function Q(s,a)** | Expected total future reward from state s after taking action a |
| **Discount factor (γ)** | How much to weight future vs immediate rewards (0–1) |
| **Episode** | One complete run (game start to game over) |
| **Exploration** | Trying random actions to discover new rewards |
| **Exploitation** | Using current knowledge to maximize reward |

---

## Recommended Learning Path

If you decide to climb this mountain:

```
1. David Silver's RL Course (DeepMind, free YouTube)
   — gold standard intro, builds intuition before heavy math

2. Sutton & Barto textbook (free PDF online)
   — the bible, dense but comprehensive. Better as reference alongside a course

3. Sergey Levine's CS 285 (UC Berkeley, free YouTube)
   — deeper, more modern, covers deep RL. Harder.

Prerequisites you already have:
  ✓ Neural network fundamentals
  ✓ Backprop intuition
  ✓ Softmax and probability distributions
  ✓ Loss functions

What you'd need to build:
  → Probability (conditional, Bayes' theorem, expectations)
  → Markov chains (state transitions)
  → Bellman equation
  → Dynamic programming
```

---

## The One-Sentence Summary

**Reinforcement learning teaches machines through trial-and-error reward signals — starting from games as a proving ground, the same algorithms (PPO) now align ChatGPT, fold proteins, and design computer chips.**

---

## Cross-References

- [Generative Models Taxonomy](24_generative_models_taxonomy.md) — GANs use a minimax game (game theory connection)
- [Softmax](22_softmax.md) — Boltzmann distribution appears in RL exploration (Boltzmann exploration policy)
- [Transformers](19_transformers.md) — decision transformers frame RL as sequence prediction
- [Master Timeline](00_timeline.md) — where each RL milestone fits
