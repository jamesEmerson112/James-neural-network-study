# Quiz 5 Study Guide — Module 4 (Lessons 13, 14, 17, 18)

> **Due: April 20, 2026, 8:00 AM ET — NO grace period.**
> Lessons covered: 13 (Generative Models), 14 (Embeddings), 17 (Deep Reinforcement Learning), 18 (Unsupervised & Semi-Supervised Learning).

This is the master index for Quiz 5 study. Every note referenced here either lives in this folder (for self-contained reading) or is a cross-reference to a deep-dive outside the folder. Walk the self-test checklist at the bottom before the quiz.

---

## Focus topic → note map

| # | Topic | Primary note | Supporting notes |
|---|---|---|---|
| 1 | MDPs (states / actions / environment) | [quiz_5_01_mdp_formal_definition.md](quiz_5_01_mdp_formal_definition.md) | [../25_reinforcement_learning_overview.md](../25_reinforcement_learning_overview.md), [../26_three_paradigms_of_learning.md](../26_three_paradigms_of_learning.md) |
| 2 | Dynamic programming for MDPs | [quiz_5_02_dynamic_programming.md](quiz_5_02_dynamic_programming.md) | — |
| 3 | Exploration vs exploitation | [quiz_5_03_exploration_vs_exploitation.md](quiz_5_03_exploration_vs_exploitation.md) | [../22_softmax.md](../22_softmax.md) |
| 4 | Challenges of RL | [quiz_5_04_challenges_of_rl.md](quiz_5_04_challenges_of_rl.md) | [../25_reinforcement_learning_overview.md](../25_reinforcement_learning_overview.md) |
| 5 | DQN | [quiz_5_05_dqn_deep_dive.md](quiz_5_05_dqn_deep_dive.md) | [../25_reinforcement_learning_overview.md](../25_reinforcement_learning_overview.md) |
| 5/6 | REINFORCE + policy gradients derivation | [quiz_5_06_policy_gradients_and_reinforce.md](quiz_5_06_policy_gradients_and_reinforce.md) | [../exponential-and-logarithm/08_watchlist.md](../exponential-and-logarithm/08_watchlist.md) |
| 7 | Semi / few / self-supervised + data assumptions | [quiz_5_07_learning_paradigms_comparison.md](quiz_5_07_learning_paradigms_comparison.md) | [../26_three_paradigms_of_learning.md](../26_three_paradigms_of_learning.md) Part IV |
| 8 | Self-supervised tasks (inputs / outputs / losses) | [quiz_5_08_self_supervised_tasks_catalog.md](quiz_5_08_self_supervised_tasks_catalog.md) | [../20_bert.md](../20_bert.md), [../00c_timeline_2018-now_llm_era.md](../00c_timeline_2018-now_llm_era.md) |
| 9 | GANs and VAEs: training / objectives / losses | [quiz_5_09_gan_and_vae.md](quiz_5_09_gan_and_vae.md) | [../ddpm/09_gan_vs_vae_vs_ddpm.md](../ddpm/09_gan_vs_vae_vs_ddpm.md) for the full comparison |
| 10 | Word2vec | [quiz_5_10_word2vec_deep_dive.md](quiz_5_10_word2vec_deep_dive.md) | [../22_softmax.md](../22_softmax.md), [../exponential-and-logarithm/08_watchlist.md](../exponential-and-logarithm/08_watchlist.md) |

**Practice test:** [quiz_5_00_practice_test.md](quiz_5_00_practice_test.md) — 10 multiple choice questions with answer key. Take it cold, then circle back to weak topics.

---

## Section 1 — RL topics (quiz topics 1–6, Lesson 17)

### 1.1 MDP definitions — [quiz_5_01_mdp_formal_definition.md](quiz_5_01_mdp_formal_definition.md)

**What the quiz tests:** the formal definition of a Markov Decision Process — state space, action space, environment, transition probabilities, rewards, discount factor. The Markov property ("future depends only on present state, not history").

**What to memorize:**
- An MDP is the tuple $(S, A, P, R, \gamma)$: state space, action space, transition probability $P(s' \mid s, a)$, reward function $R(s, a)$, discount factor $\gamma \in [0, 1)$
- **Markov property**: $P(s_{t+1} \mid s_t, a_t, s_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$ — present state summarizes everything relevant from the past
- **Discount factor $\gamma$**: weights future rewards — $\gamma = 0$ myopic, $\gamma = 1$ total return, typical 0.9–0.99
- Agent's goal: find a policy $\pi(a \mid s)$ that maximizes expected cumulative discounted reward

### 1.2 Dynamic programming — [quiz_5_02_dynamic_programming.md](quiz_5_02_dynamic_programming.md)

**What the quiz tests:** value iteration, policy iteration, Bellman expectation equation vs Bellman optimality equation, why these converge. DP assumes the transition and reward functions are known.

**What to memorize:**
- **Bellman expectation** (fixed policy $\pi$): $V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) [R(s, a) + \gamma V^\pi(s')]$
- **Bellman optimality**: $V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) [R(s, a) + \gamma V^*(s')]$ — replace $\sum_a \pi(\cdot)$ with $\max_a$
- **Value iteration**: repeatedly apply Bellman optimality operator until $V$ converges, then extract greedy policy
- **Policy iteration**: alternate policy evaluation + policy improvement until policy stops changing
- Both converge because the Bellman operator is a $\gamma$-contraction mapping (exponential convergence by Banach fixed-point theorem)

### 1.3 Exploration vs exploitation — [quiz_5_03_exploration_vs_exploitation.md](quiz_5_03_exploration_vs_exploitation.md)

**What the quiz tests:** the tradeoff, and specific algorithms — ε-greedy, softmax/Boltzmann, UCB, Thompson sampling.

**What to memorize:**
- **ε-greedy**: with prob $\varepsilon$ explore (random action), else exploit (argmax Q). Common to decay $\varepsilon$ over time.
- **Softmax / Boltzmann**: $\pi(a \mid s) = \exp(Q(s, a)/\tau) / \sum_{a'} \exp(Q(s, a')/\tau)$ — temperature $\tau$ controls explore-exploit smoothly
- **UCB**: $\arg\max_a [Q(s, a) + c \sqrt{\ln t / N(s, a)}]$ — optimistic estimate with uncertainty bonus
- **Thompson sampling**: maintain posterior over Q-values, sample from it, act greedy — Bayesian

### 1.4 Challenges of RL — [quiz_5_04_challenges_of_rl.md](quiz_5_04_challenges_of_rl.md)

**What the quiz tests:** why RL is hard — sparse rewards, delayed rewards, credit assignment, non-stationarity, sample inefficiency, exploration difficulty.

**What to memorize:**
- **Sparse rewards**: most actions get zero feedback
- **Delayed rewards**: reward at step 1000 might be due to action at step 5
- **Credit assignment**: which of the thousand actions leading to a win actually caused it?
- **Non-stationarity**: the data distribution changes as the policy changes
- **Sample inefficiency**: RL needs millions of interactions; supervised learning needs thousands
- **Exploration hardness**: without good exploration, agent may never find high-reward regions

### 1.5 DQN — [quiz_5_05_dqn_deep_dive.md](quiz_5_05_dqn_deep_dive.md)

**What the quiz tests:** the Q-learning loss, why deep Q-learning is unstable, experience replay, target networks.

**What to memorize:**
- **Q-learning loss**: $L(\theta) = \mathbb{E}\!\left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)\right)^2\right]$ — MSE between the Bellman target and the current Q-value
- **Experience replay**: store $(s, a, r, s')$ transitions in a buffer, sample mini-batches randomly → decorrelates samples, improves sample efficiency
- **Target network** $\theta^-$: a lagging copy of $\theta$ updated every $C$ steps → stabilizes the regression target
- **ε-greedy for exploration** with decaying $\varepsilon$ from 1.0 to 0.1
- **DQN = Q-learning + deep net + experience replay + target network**

### 1.6 REINFORCE + policy gradients derivation — [quiz_5_06_policy_gradients_and_reinforce.md](quiz_5_06_policy_gradients_and_reinforce.md)

**What the quiz tests:** the log-derivative trick, the policy gradient theorem derivation, REINFORCE algorithm, baseline variance reduction, advantage function.

**What to memorize:**
- **Objective**: $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$
- **Log-derivative trick**: $\nabla_\theta p_\theta(x) = p_\theta(x) \cdot \nabla_\theta \log p_\theta(x)$
- **Policy gradient theorem**: $\nabla_\theta J(\theta) = \mathbb{E}_\tau\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t\right]$
- Environment dynamics $P$ drop out of the gradient — that's why PG is model-free
- **REINFORCE algo**: sample trajectory, compute $G_t$, update $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t$
- **Baseline $b(s)$**: subtract any state-dependent baseline to reduce variance without biasing the gradient
- **Advantage function**: $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ — the best baseline is $V^\pi$
- **Intuition**: push up log-probs of actions better than average, push down actions worse than average

---

## Section 2 — Learning paradigms (quiz topics 7–8, Lesson 18)

### 2.1 Semi / few / self-supervised learning — [quiz_5_07_learning_paradigms_comparison.md](quiz_5_07_learning_paradigms_comparison.md)

**What the quiz tests:** the differences between paradigms and what data each assumes.

**What to memorize:**
- **Supervised**: every input has a label. Lots of labeled data required.
- **Unsupervised**: no labels. Clustering, density estimation, dimensionality reduction.
- **Semi-supervised**: small labeled set + large unlabeled set. Unlabeled data is used to regularize or propagate labels.
- **Self-supervised**: no human labels; labels come from the data itself. Unlimited data (internet scale).
- **Few-shot**: a few labeled examples per class (typically 1–5), leveraging a large pretrained model. GPT-3 in-context learning.
- **Zero-shot**: no examples of the target class at all — leverage a pretrained model's general knowledge.

### 2.2 Self-supervised tasks — [quiz_5_08_self_supervised_tasks_catalog.md](quiz_5_08_self_supervised_tasks_catalog.md)

**What the quiz tests:** canonical SSL tasks with inputs, outputs, and loss functions.

**What to memorize:**
- **Masked language modeling (BERT)**: input = sentence with ~15% tokens masked, output = predicted tokens, loss = cross-entropy
- **Autoregressive (GPT)**: input = $x_{1:t}$, output = $x_{t+1}$, loss = cross-entropy on next token
- **Contrastive (SimCLR, MoCo)**: input = two augmented views of same image + other images, loss = **NT-Xent**: $-\log[\exp(\text{sim}(x_a, x_b)/\tau) / \sum_i \exp(\text{sim}(x_a, x_i)/\tau)]$
- **Masked image modeling (MAE)**: mask 75% of patches, reconstruct pixels with MSE
- **Denoising autoencoder**: corrupt input with noise, reconstruct clean version with MSE — direct ancestor of DDPM
- **Pretext tasks**: rotation prediction (0/90/180/270°), jigsaw, colorization — classical SSL, now superseded

---

## Section 3 — Generative models (quiz topic 9, Lesson 13)

### 3.1 GANs and VAEs — [quiz_5_09_gan_and_vae.md](quiz_5_09_gan_and_vae.md)

**What the quiz tests:** GAN minimax loss, VAE ELBO, reparameterization trick, training procedures, failure modes.

**What to memorize:**

**GAN:**
- **Objective**: $\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$
- **Two networks**: Generator $G$ maps noise to fake images; Discriminator $D$ classifies real vs fake
- **Training**: alternate updating $D$ (maximize loss) and $G$ (minimize loss)
- **Failure modes**: mode collapse, training instability, no explicit likelihood

**VAE:**
- **Objective (ELBO)**: $\mathcal{L}_{\text{ELBO}}(x) = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{\text{KL}}(q_\phi(z \mid x) \| p(z))$
- **Two networks**: Encoder $q_\phi(z \mid x)$; Decoder $p_\theta(x \mid z)$
- **Reparameterization trick**: $z = \mu_\phi(x) + \sigma_\phi(x) \cdot \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, \mathbf{I})$. Randomness is pushed outside the network, so gradients can flow through $\mu$ and $\sigma$.
- **Two loss terms**: reconstruction (decoder output vs input) + KL (encoder vs prior)
- **Failure modes**: blurry outputs, posterior collapse

---

## Section 4 — Word2vec (quiz topic 10, Lesson 14)

### 4.1 Word2vec — [quiz_5_10_word2vec_deep_dive.md](quiz_5_10_word2vec_deep_dive.md)

**What the quiz tests:** Skip-gram vs CBOW, negative sampling, hierarchical softmax, vector arithmetic.

**What to memorize:**
- **Skip-gram**: given center word, predict context words
- **CBOW**: given context words, predict center word
- **Full softmax is expensive**: $p(w_c \mid w) = \exp(v_c^\top u_w) / \sum_{w' \in V} \exp(v_{w'}^\top u_w)$ — denominator sums over whole vocabulary
- **Negative sampling**: replace softmax with binary classification on positive pair + $k$ random negatives. Loss: $\log \sigma(v_c^\top u_w) + \sum_i \log \sigma(-v_{n_i}^\top u_w)$
- **Hierarchical softmax**: Huffman tree, $\log |V|$ binary decisions instead of $|V|$ softmax terms
- **Vector arithmetic**: $\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})$
- **Distributional hypothesis**: "you shall know a word by the company it keeps" (Firth, 1957)

---

## Self-test checklist

On April 19 (night before the quiz), walk this list. If you can answer each without looking at notes, you're ready.

- [ ] **Topic 1** — I can define an MDP as a 5-tuple and state what each component means.
- [ ] **Topic 1** — I can explain the Markov property in one sentence.
- [ ] **Topic 2** — I can write value iteration and policy iteration as pseudocode.
- [ ] **Topic 2** — I can state both the Bellman expectation and the Bellman optimality equations.
- [ ] **Topic 3** — I can describe ε-greedy, UCB, and Thompson sampling and say when I'd use each.
- [ ] **Topic 4** — I can list at least four reasons RL is harder than supervised learning.
- [ ] **Topic 5** — I can write the DQN loss function and explain what experience replay and target networks do.
- [ ] **Topic 5** — I can write the REINFORCE update rule.
- [ ] **Topic 6** — I can derive the policy gradient theorem using the log-derivative trick.
- [ ] **Topic 6** — I can explain why a baseline reduces variance without changing the expected gradient.
- [ ] **Topic 7** — I can name the data assumptions of semi-supervised, few-shot, and self-supervised learning.
- [ ] **Topic 7** — I can explain why self-supervised learning "unlocks the internet as training data."
- [ ] **Topic 8** — I can describe at least four self-supervised tasks with their inputs, outputs, and loss functions.
- [ ] **Topic 8** — I can write the NT-Xent contrastive loss and explain what it does.
- [ ] **Topic 9** — I can write the GAN minimax loss and explain both players' objectives.
- [ ] **Topic 9** — I can write the VAE ELBO and identify the reconstruction and KL terms.
- [ ] **Topic 9** — I can explain the reparameterization trick and why VAE needs it.
- [ ] **Topic 10** — I can draw the Skip-gram and CBOW architectures side by side.
- [ ] **Topic 10** — I can explain why negative sampling is faster than full softmax, and write its loss.
- [ ] **Topic 10** — I can explain the "king − man + woman ≈ queen" result.

---

## Folder layout

```
notes/quiz5/
├── quiz_5_00_study_guide.md                    ← you are here
├── quiz_5_00_practice_test.md                  ← 10 MC questions
├── quiz_5_01_mdp_formal_definition.md          ← quiz topic 1
├── quiz_5_02_dynamic_programming.md            ← quiz topic 2
├── quiz_5_03_exploration_vs_exploitation.md    ← quiz topic 3
├── quiz_5_04_challenges_of_rl.md               ← quiz topic 4
├── quiz_5_05_dqn_deep_dive.md                  ← quiz topic 5 (DQN half)
├── quiz_5_06_policy_gradients_and_reinforce.md ← quiz topic 5/6 (REINFORCE + derivation)
├── quiz_5_07_learning_paradigms_comparison.md  ← quiz topic 7
├── quiz_5_08_self_supervised_tasks_catalog.md  ← quiz topic 8
├── quiz_5_09_gan_and_vae.md                    ← quiz topic 9
└── quiz_5_10_word2vec_deep_dive.md             ← quiz topic 10
```

Files 01–10 match the quiz focus topic numbers exactly. Post-quiz, this folder will be dissolved and its contents merged back into the main `notes/` tree.

---

*Last updated: 2026-04-10.*
