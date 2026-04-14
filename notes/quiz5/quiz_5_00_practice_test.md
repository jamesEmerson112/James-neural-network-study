# Quiz 5 Practice Test — 10 Multiple Choice Questions

> **Purpose:** self-assessment before the real [Quiz 5](quiz_5_00_study_guide.md) on April 20, 2026.
> **Instructions:** no peeking at notes. Pick one option per question. Answer key at the bottom.
> **Scoring guide at the end.**

---

## Q1

An MDP is formally defined as a tuple $(S, A, P, R, \gamma)$. What does $P$ represent?

A. The policy $\pi(a \mid s)$ — the agent's strategy
B. The probability distribution $P(s' \mid s, a)$ — how the environment transitions between states
C. The prior over the initial state
D. The parameters of the value function approximator

---

## Q2

The **Bellman optimality equation** for the state-value function is:

A. $V^*(s) = \sum_a \pi(a \mid s) [R(s, a) + \gamma V^*(s')]$
B. $V^*(s) = \max_a [R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s')]$
C. $V^*(s) = \mathbb{E}_\pi[R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \ldots]$
D. $V^*(s) = \min_a [R(s, a) + \gamma V^*(s')]$

---

## Q3

Which statement about **value iteration vs policy iteration** is correct?

A. Value iteration requires a model of the MDP; policy iteration does not
B. Policy iteration alternates between policy evaluation and policy improvement until the policy stops changing
C. Value iteration only works for deterministic MDPs
D. Policy iteration is always faster than value iteration in both compute and memory

---

## Q4

An agent using **ε-greedy exploration** with $\varepsilon = 0.1$ will:

A. Always pick the action with the highest Q-value
B. Pick a uniformly random action with probability 0.9 and the argmax action with probability 0.1
C. Pick the argmax action with probability 0.9 and a uniformly random action with probability 0.1
D. Pick actions proportional to their Q-values via softmax

---

## Q5

In DQN, the purpose of the **target network** (with parameters $\theta^-$ held fixed for several updates) is to:

A. Reduce memory usage by not backpropagating through it
B. Stabilize training by preventing the regression target from moving as the online network updates
C. Provide a baseline for variance reduction in policy gradient estimates
D. Allow parallel updates across multiple actor threads

---

## Q6

The REINFORCE update rule is:

$$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t$$

Why does the $\log$ appear in this formula?

A. It's a numerical trick to avoid underflow when probabilities are tiny
B. It converts the probability into a negative log-likelihood for cross-entropy loss
C. Because of the identity $\nabla_\theta p_\theta(x) = p_\theta(x) \cdot \nabla_\theta \log p_\theta(x)$, which lets the policy gradient be written as an expectation that can be estimated by sampling
D. Because the log is the only function that turns multiplication into addition

---

## Q7

You have 100,000 unlabeled images and only 500 labeled images. Which learning paradigm is designed precisely for this data assumption?

A. Few-shot learning
B. Self-supervised learning
C. Semi-supervised learning
D. Zero-shot learning

---

## Q8

In BERT's **masked language modeling** task, the inputs and loss are:

A. Input: a full sentence; output: the next sentence; loss: cosine similarity
B. Input: a sentence with ~15% of tokens replaced by `[MASK]`; output: predicted tokens at masked positions; loss: cross-entropy over masked positions
C. Input: an image split into patches; output: a classification label; loss: hinge loss
D. Input: two augmented views of the same image; output: embeddings; loss: NT-Xent

---

## Q9

The GAN minimax objective is:

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

Which of the following correctly describes what each player is doing?

A. $G$ tries to maximize both terms; $D$ tries to minimize both terms
B. $D$ tries to output 1 for real data and 0 for generated data; $G$ tries to make $D$ output 1 for its generated data
C. $G$ and $D$ are trained jointly via a single loss with no adversarial component
D. $D$ minimizes reconstruction error; $G$ maximizes KL divergence to the prior

---

## Q10

In Word2vec's Skip-gram with **negative sampling**, the reason for replacing the full softmax is:

A. Negative sampling produces more accurate embeddings than the true softmax
B. The full softmax denominator sums over the entire vocabulary ($\sim$millions of words), making each update prohibitively slow
C. Negative sampling is required to compute gradients through discrete tokens
D. The softmax formula doesn't work for variable-length context windows

---

## Answer key

<details>
<summary><strong>Click to reveal answers</strong></summary>

---

**Q1: B** — $P$ is the transition probability $P(s' \mid s, a)$. The policy $\pi$ is what the agent *learns*; $P$ is part of the environment definition. See [quiz_5_00_study_guide.md §1.1](quiz_5_00_study_guide.md).

**Q2: B** — Bellman *optimality* uses $\max_a$ (the optimal policy acts greedily). Option A is Bellman *expectation* (for a fixed policy). Option C is the definition of $V$ as expected return, not the Bellman recursion. Option D is just wrong (min).

**Q3: B** — Policy iteration does exactly that: alternate eval + improvement until the policy is stable. Option A is wrong (both assume a known MDP model). C is wrong (both handle stochastic MDPs). D is wrong (neither is universally faster).

**Q4: C** — ε is the exploration probability. With $\varepsilon = 0.1$, you explore 10% of the time (random action) and exploit 90% of the time (argmax). A common confusion is getting the probabilities backwards (option B).

**Q5: B** — The target network freezes the regression target $r + \gamma \max_{a'} Q_{\theta^-}(s', a')$ for a while so the online Q-network has a stable target to fit. Without it, both sides of the Bellman equation move together and training oscillates or diverges.

**Q6: C** — This is the **log-derivative trick** (also called the "score function trick" or "REINFORCE trick"). The identity lets you bring the gradient inside an expectation, which turns the policy gradient into something you can estimate by sampling trajectories. A and B describe unrelated uses of log. D is true but not why it appears here.

**Q7: C** — Semi-supervised learning is defined by this exact setup: a small labeled set + a large unlabeled set. Few-shot (A) assumes a large pretrained model + 1–5 examples per class. Self-supervised (B) uses *no* human labels. Zero-shot (D) uses no examples of the target class.

**Q8: B** — MLM masks tokens and trains the model to predict them. Option A is the "next sentence prediction" task (BERT's other pretraining objective, but it's not MLM). Option D describes SimCLR/MoCo contrastive learning. Option C is a vanilla classification task, not MLM.

**Q9: B** — $D$ is a binary classifier (real = 1, fake = 0). $G$ wants $D$ to be fooled into outputting 1 on its fakes. The minimax game is adversarial because $G$ and $D$ have opposing objectives.

**Q10: B** — The full softmax has a denominator $\sum_{w' \in V} \exp(v_{w'}^\top u_w)$ over the entire vocabulary, which is $O(|V|)$ per training step. Negative sampling approximates it with a few random "negative" words per positive pair, making each update $O(k)$ where $k$ is small (5–20). Option A is wrong — negative sampling is *less* accurate but much faster. Options C and D are wrong factually.

</details>

---

## Scoring guide

| Score | Interpretation |
|---|---|
| **9–10 correct** | Ready. Review your weakest topic for polish. |
| **7–8 correct** | Solid foundation. Focus extra review on the topics missed. |
| **5–6 correct** | Need more study. Hit the corresponding sections in [quiz_5_00_study_guide.md](quiz_5_00_study_guide.md) before the real quiz. |
| **< 5 correct** | Good thing this was taken early. Start with the ✅ topics in the study guide (4, 8, 9) to build momentum, then tackle the ⚠ and ❌ ones. |

---

## Topic coverage map

Each practice question maps to a focus topic from [quiz_5_00_study_guide.md](quiz_5_00_study_guide.md):

| Question | Focus topic | Study guide section |
|---|---|---|
| Q1 | 1. MDP definitions | [§1.1](quiz_5_00_study_guide.md) |
| Q2 | 2. Dynamic programming for MDPs | [§1.2](quiz_5_00_study_guide.md) |
| Q3 | 2. Dynamic programming for MDPs | [§1.2](quiz_5_00_study_guide.md) |
| Q4 | 3. Exploration vs exploitation | [§1.3](quiz_5_00_study_guide.md) |
| Q5 | 5. DQN | [§1.5](quiz_5_00_study_guide.md) |
| Q6 | 6. Policy gradients derivation | [§1.6](quiz_5_00_study_guide.md) |
| Q7 | 7. Semi / few / self-supervised | [§2.1](quiz_5_00_study_guide.md) |
| Q8 | 8. Self-supervised tasks | [§2.2](quiz_5_00_study_guide.md) |
| Q9 | 9. GANs and VAEs | [§3.1](quiz_5_00_study_guide.md) |
| Q10 | 10. Word2vec | [§4.1](quiz_5_00_study_guide.md) |

Topic 4 (challenges of RL) is not tested in this round — it's a qualitative topic better tested via free-response self-test prompts in the study guide checklist.

---

*Last updated: 2026-04-10.*
