# The Three Paradigms of Machine Learning — A History of How Machines Learned to Learn

> Part of the [Neural Network Study Timeline](00_timeline_neural_sequence_models.md). See also: [Rosenblatt and the Perceptron](02_rosenblatt_and_the_perceptron.md), [Minsky — Death and Resurrection](06_minsky_perceptrons_death_and_resurrection.md), [Backprop: The Wiggle Ratio](09_backprop_the_wiggle_ratio.md), [Generative Models Taxonomy](24_generative_models_taxonomy.md), [Reinforcement Learning Overview](25_reinforcement_learning_overview.md).

---

## The Three Flavors — At a Glance

Before we tell the story, here's the map:

| | **Supervised** | **Unsupervised** | **Reinforcement** |
|---|---|---|---|
| **Signal** | Correct answer for every input | No labels at all | Delayed, sparse reward |
| **Question** | "What is this?" | "What structure is hiding here?" | "What should I do next?" |
| **Analogy** | Studying with an answer key | Finding patterns in a foreign language with no dictionary | Learning to ride a bike — nobody tells you the "right" muscle movements |
| **Math roots** | Regression (1805), classification (1936) | PCA (1901), clustering (1957) | Dynamic programming (1953), conditioning (1898) |
| **Philosophical ancestor** | Empiricism — knowledge from observation | Gestalt psychology — the mind organizes | Behaviorism — organisms learn from consequences |
| **When it dominated** | 1950s–present (always the workhorse) | 2006–2018 (the pre-training revolution) | 2013–present (games, then alignment) |
| **Modern role** | Fine-tuning, classification, regression | Foundation model pre-training (GPT, BERT) | Alignment (RLHF), robotics, game AI |

The real story is that these paradigms didn't develop in isolation — the same people kept crossing the boundaries, and breakthroughs in one paradigm triggered revolutions in another.

---

# Part I: Supervised Learning — 200 Years of Teaching Machines With Answer Keys

## The Prehistory: Mathematicians Who Didn't Know They Were Doing ML

### Legendre and Gauss — The Least Squares War (1805–1809)

The oldest ancestor of supervised learning is a bitter priority dispute between two mathematicians who never used the word "learning."

**Adrien-Marie Legendre** — a French mathematician known for his elegance and his legendary ugliness (one contemporary described him as having "the face of a gargoyle") — published the **method of least squares** in 1805 in a 10-page appendix to a book about comet trajectories. The idea: given noisy data points, find the line that minimizes the sum of squared errors.

```
The least squares idea, stripped bare:

  Given: data points (x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)
  Find:  line y = mx + b that minimizes Σ(yᵢ - mxᵢ - b)²

  This is literally what linear regression does today.
  Every neural network's loss function descends from this.
```

**Carl Friedrich Gauss** — widely considered the greatest mathematician in history — then claimed he'd been using the method since 1795, age 18, and just never bothered to publish it. Legendre was furious. The dispute was never resolved.

**Why it matters**: The idea that you can define "error" mathematically and minimize it — that's the seed of every supervised learning algorithm. Loss functions, gradient descent, backpropagation — all of it traces back to "minimize the squared difference between prediction and truth."

### Sir Francis Galton — Regression (1886)

Galton — Charles Darwin's cousin, a Victorian polymath and problematic eugenicist — studied the heights of parents and their children. He noticed that extremely tall parents tended to have children who were tall, but *less* extreme than their parents. He called this **"regression toward mediocrity"** (later softened to "regression to the mean").

The statistical technique he invented to quantify this relationship became **regression analysis** — the name literally means "going back toward the average." Every linear regression model carries the ghost of Galton measuring Victorian families.

### Karl Pearson — Correlation (1896–1901)

Galton's protégé **Karl Pearson** formalized correlation into the **Pearson correlation coefficient** and, in 1901, invented **Principal Component Analysis (PCA)** — which we'll meet again in the unsupervised learning story. Pearson was building the statistical toolkit that would eventually become the foundation of machine learning.

Pearson also founded the world's first university statistics department (UCL, 1911) and the journal *Biometrika*. He was a socialist, a Germanophile, a eugenicist, and one of the most influential scientists of the 20th century — a complicated man building tools that would outlive their original purposes.

### Ronald Fisher — Classification and the Iris Dataset (1936)

**Sir Ronald Aylmer Fisher** — statistician, geneticist, and another deeply problematic eugenicist — created what might be the most famous dataset in machine learning history: the **Iris dataset** (1936).

He measured four features (sepal length/width, petal length/width) of 150 iris flowers across three species. Then he developed **Linear Discriminant Analysis (LDA)** to classify them. This was the first formal **classification** method — drawing boundaries in feature space to separate categories.

```
Fisher's Iris dataset (1936):
  150 flowers → 3 species (setosa, versicolor, virginica)
  4 measurements each → plot in 4D space → draw separating planes

  This is supervised classification:
    Input:  flower measurements
    Label:  species name
    Goal:   learn the boundary

  Every intro ML course still uses this dataset 90 years later.
```

Fisher also developed **maximum likelihood estimation**, **analysis of variance (ANOVA)**, and the **F-distribution** (named after him). The statistical framework he built became the language supervised learning would eventually speak.

---

## The Machine Age: When Neurons Started Learning

### Rosenblatt's Perceptron (1957–1958) — Psychology Meets the Cold War

*(Detailed biography in [Note 02](02_rosenblatt_and_the_perceptron.md))*

**Frank Rosenblatt** wasn't a mathematician or an engineer — he was a **psychologist** at Cornell who wanted to understand how brains learn. In 1957, funded by the **U.S. Office of Naval Research** (Project PARA — Perceiving and Recognizing Automaton), he built the **Perceptron** — the first machine that could learn from labeled examples.

The Cold War context matters. Sputnik launched in October 1957. The Perceptron was conceived the same year. America was panicking about Soviet technological supremacy, and the Navy was throwing money at anything that might create machine intelligence.

The Perceptron's learning rule was shockingly simple:

```
For each training example (input x, correct label y):
  1. Make a prediction: ŷ = sign(w · x + b)
  2. If wrong: w ← w + η(y - ŷ) · x
  3. If right: do nothing

This is supervised learning in its purest form:
  "Here's the answer → adjust if wrong."
```

The *New York Times* ran a front-page story in 1958: **"New Navy Device Learns by Doing"** — calling it an "electronic brain" that could "walk, talk, see, write, reproduce itself and be conscious of its existence." Rosenblatt himself was more measured, but the hype machine was already running.

### The Minsky-Papert Kill (1969)

*(Full story in [Note 06](06_minsky_perceptrons_death_and_resurrection.md))*

**Marvin Minsky** and **Seymour Papert** — both from MIT, both champions of symbolic AI — published *Perceptrons* (1969), proving that single-layer perceptrons can't solve XOR or any non-linearly-separable problem. Their math was correct but their conclusion was misleading: they strongly implied that multi-layer networks would face the same problems. The field collapsed.

Rosenblatt died in a sailing accident in 1971, two years later, on his 43rd birthday. Funding dried up. The **first AI winter** descended.

**The supervised learning insight the winter obscured**: the problem wasn't with learning from labels — it was with single-layer architectures. The concept of "here's the answer, adjust your weights" was sound. The architecture wasn't deep enough.

---

## The Comeback: Backpropagation and the Deep Learning Explosion

### Paul Werbos — The Right Answer, 12 Years Too Early (1974)

**Paul John Werbos** — a Harvard PhD student in statistics — had a radical idea for his 1974 dissertation: apply the chain rule of calculus *backward* through the layers of a neural network to compute how each weight contributed to the error. He called it **backpropagation**.

Nobody cared. The AI winter was in full swing. Neural networks were considered a dead end. His thesis advisors told him it wasn't publishable in any major journal. He filed it, moved on to work at the Department of Energy, and watched the field ignore him for 12 years.

The cruel detail: **Rosenblatt coined the phrase "back-propagating errors" in his 1962 book** *Principles of Neurodynamics* — naming the solution 24 years before anyone made it work.

### Rumelhart, Hinton & Williams — The Nature Paper (1986)

In 1986, **David Rumelhart**, **Geoffrey Hinton**, and **Ronald Williams** published *"Learning representations by back-propagating errors"* in *Nature*. They demonstrated that multi-layer networks trained with backprop could solve XOR and much more.

```
Backprop solved the supervised learning crisis:

  Before 1986:  Can learn, but only linearly separable problems (perceptron)
  After 1986:   Can learn ANY continuous function (universal approximation)

  Single layer:   w ← w - η · (∂Loss/∂w)
  Multi-layer:    w ← w - η · (∂Loss/∂w) via chain rule through ALL layers

  (See Note 09 for the intuitive "wiggle ratio" explanation)
```

Supervised learning was back. But it needed two more ingredients to truly explode.

### Vladimir Vapnik — From Soviet Theory to SVMs (1962 → 1995)

**Vladimir Naumovich Vapnik** — born in 1936 in the Soviet Union — developed the foundations of **statistical learning theory** in the 1960s with his mentor **Alexei Chervonenkis**. Their work on **VC dimension** (1964) — a measure of a model's capacity — was published in Soviet journals that few Western researchers read.

After the Soviet Union collapsed, Vapnik emigrated to the United States and joined **Bell Labs** in 1991. There, in 1995, he published the **Support Vector Machine (SVM)** — a supervised classification method that found the optimal separating boundary (the "maximum margin hyperplane") between classes.

```
SVM insight:
  Don't just find ANY separating line — find the one
  with the MAXIMUM margin (distance to nearest points).

  This is provably the best generalization you can get
  for a given model complexity.

  The "kernel trick" lets SVMs work in infinite-dimensional
  space without actually computing in that space.
```

SVMs dominated machine learning from 1995 to about 2012. For nearly two decades, they were the gold standard for classification — elegant, mathematically grounded, and effective. Vapnik received the Paris Kanellakis Award, the Benjamin Franklin Medal, and became one of the most cited computer scientists alive.

**The irony**: SVMs proved that the statistical framework *behind* supervised learning mattered more than any specific architecture. Then deep learning came along and overwhelmed that framework with raw scale.

### LeCun's LeNet (1988–1998) — Reading Millions of Checks

**Yann LeCun** — a French researcher who had worked with Hinton — built **LeNet** at AT&T Bell Labs. It was a **convolutional neural network** (CNN) that read handwritten digits on bank checks. By the late 1990s, LeNet was processing 10–20% of all checks deposited in American banks.

This was one of the first commercial successes of supervised deep learning — a system trained on millions of labeled examples (digits + correct labels) that actually worked in production.

### AlexNet (2012) — The Cambrian Explosion

**Alex Krizhevsky**, **Ilya Sutskever**, and **Geoffrey Hinton** at the University of Toronto trained a deep CNN called **AlexNet** on **ImageNet** — a dataset of 1.2 million labeled images across 1,000 categories. They won the ImageNet competition by a staggering margin (15.3% error vs. 26.2% for the runner-up).

What changed wasn't the algorithm — backprop and CNNs had existed for decades. What changed was:

1. **Data** — ImageNet provided millions of labeled examples
2. **Compute** — Krizhevsky trained on two NVIDIA GTX 580 GPUs
3. **Scale** — 60 million parameters, 8 layers deep

AlexNet proved that supervised learning + deep networks + enough data + GPUs = superhuman performance. The deep learning revolution had begun.

**The 200-year arc**: Legendre's squared errors (1805) → Fisher's classification (1936) → Rosenblatt's learning rule (1957) → Werbos's backprop (1974) → Rumelhart/Hinton/Williams (1986) → AlexNet (2012). Supervised learning wasn't invented — it was **assembled over 200 years** by statisticians, psychologists, Soviet mathematicians, French researchers, and a team with two consumer gaming GPUs.

---

# Part II: Unsupervised Learning — Letting Machines Find Their Own Patterns

## The Philosophical Difference

Supervised learning gives a teacher. Unsupervised learning says: "Here's the data. Figure it out."

No labels. No answer key. No reward signal. Just structure waiting to be discovered.

For most of ML's history, unsupervised learning was considered the "weaker sibling" — interesting theoretically, but supervised learning was where the money was. Then something flipped: the modern AI revolution (GPT, BERT, diffusion models) is built on unsupervised pre-training. The "secondary" paradigm became the foundation.

---

## The Pioneers

### Karl Pearson — PCA (1901)

We already met Pearson in the supervised learning story. But his most lasting contribution to unsupervised learning came first: **Principal Component Analysis (PCA)**, published in 1901 in *Philosophical Magazine*.

The idea: given high-dimensional data with no labels, find the directions of maximum variance — the axes along which the data "spreads out" the most. Project onto those axes to reduce dimensionality while losing as little information as possible.

```
PCA in one picture:

  Original data: a cloud of points in 100 dimensions
  PCA: find the 2-3 directions that capture most of the spread
  Result: a 2D or 3D map that preserves the essential structure

  No labels needed. Just geometry.
```

Pearson developed this for biological data analysis. **Harold Hotelling** formalized it further in the 1930s, connecting it to eigenvalue decomposition of the covariance matrix. PCA remains one of the most widely used techniques in data science — still the first thing many people try when they have high-dimensional unlabeled data.

### Donald Hebb — "Neurons That Fire Together Wire Together" (1949)

**Donald Olding Hebb** — a Canadian psychologist at McGill University who had studied under Karl Lashley (the great behaviorist) — published *The Organization of Behavior* in 1949. His key insight:

> "When an axon of cell A is near enough to excite a cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased."

The popular summary — **"neurons that fire together wire together"** — was coined much later by neuroscientist Carla Shatz, not by Hebb himself.

**Why this belongs in the unsupervised story**: Hebbian learning is inherently unsupervised. There's no teacher, no correct answer. Neurons strengthen connections based purely on co-activation patterns — the brain finding its own structure in sensory data. This principle directly inspired:

- **Kohonen's Self-Organizing Maps** (1980s)
- **Competitive learning** networks
- **Boltzmann machines** (Hinton, 1985)
- Modern **contrastive learning** methods

Hebb never wrote an algorithm. He described a *principle*. It took decades for that principle to become code.

### Stuart Lloyd — K-Means, Unpublished for 25 Years (1957 → 1982)

**Stuart P. Lloyd** — a mathematician at **Bell Labs** — developed the **k-means clustering algorithm** in 1957 as part of his work on pulse-code modulation (PCM) for telephone signals. The idea: given data points with no labels, partition them into k groups by iteratively assigning points to the nearest center and recomputing centers.

```
K-means algorithm (1957):
  1. Pick k random center points
  2. Assign each data point to its nearest center
  3. Recompute each center as the mean of its assigned points
  4. Repeat steps 2-3 until nothing changes

  No labels. No teacher. Just "find k natural groups."
```

Here's the remarkable part: **Lloyd never published it**. It circulated as an internal Bell Labs technical note for **25 years**. The algorithm was so useful that it spread through the research community by word of mouth and photocopies. **James MacQueen** independently developed and named it "k-means" in a 1967 paper, giving credit to Lloyd's unpublished work.

Lloyd finally published his version in 1982 — a quarter century after he'd invented it. By then, k-means was already one of the most widely used algorithms in data analysis.

**The Bell Labs detail**: Lloyd worked in the same building where Claude Shannon had developed information theory, where the transistor had been invented, and where Vapnik would later develop SVMs. Bell Labs in its golden era was perhaps the most productive research laboratory in human history.

### Teuvo Kohonen — Self-Organizing Maps (1982)

**Teuvo Kohonen** — a Finnish professor at the Helsinki University of Technology — created **Self-Organizing Maps (SOMs)**, inspired directly by how the brain's cortex organizes itself topologically. The visual cortex, for example, maps nearby regions of the visual field to nearby neurons — and it does this without a teacher.

SOMs learned to create 2D maps of high-dimensional data, preserving topological relationships. Feed in 100-dimensional data points, and the SOM arranges them on a 2D grid so that similar data points end up near each other — like the brain mapping the body onto the somatosensory cortex.

Kohonen worked largely in isolation from the American ML community. His SOMs became hugely popular in Europe and Japan but were less known in the US until the 1990s.

---

## Hinton's Revolution — From Boltzmann Machines to Deep Belief Nets

### Geoffrey Hinton — The Man Who Wouldn't Give Up

*(Hinton appears throughout the timeline — see [Note 06](06_minsky_perceptrons_death_and_resurrection.md) and [Note 09](09_backprop_the_wiggle_ratio.md))*

**Geoffrey Everest Hinton** — great-great-grandson of logician **George Boole** (the man whose algebra made the McCulloch-Pitts neuron possible) — is the single most important figure in the unsupervised learning story. He kept working on neural networks throughout the AI winter when almost everyone else had abandoned them.

Born in Wimbledon, England, he studied experimental psychology at Cambridge (like Hebb, the connectionists kept coming from psychology, not math). His father was an entomologist, his family full of scientists. He has said he entered AI because he was "deeply suspicious of symbolic AI" — the idea that intelligence was manipulation of logical symbols felt wrong to him. Intelligence, he believed, came from **learning representations from data**.

### Boltzmann Machines (1985)

Hinton and **Terry Sejnowski** (a neuroscientist at Johns Hopkins) created the **Boltzmann machine** — a network where units are connected bidirectionally and the system is defined by an energy function borrowed from statistical physics (specifically, the **Boltzmann distribution** used to describe particle energies in thermodynamics).

```
Boltzmann Machine:
  - Every unit connects to every other unit (no layers)
  - Network has an "energy" for each configuration
  - Lower energy = more probable configuration
  - Learning = adjusting weights so that low-energy states
    correspond to the patterns in the training data

  Named after Ludwig Boltzmann (1844–1906), the Austrian
  physicist who connected microscopic particle behavior
  to macroscopic thermodynamic properties.
  (Boltzmann tragically died by suicide in 1906.)
```

Boltzmann machines were theoretically beautiful but practically nightmarish — training required sampling via Markov Chain Monte Carlo, which was agonizingly slow. But the key insight — that you could learn to model data distributions using energy-based networks — would eventually transform the field.

### Restricted Boltzmann Machines and Deep Belief Nets (2006)

In 2006, Hinton published a paper that many consider the starting gun of the deep learning revolution: **"A Fast Learning Algorithm for Deep Belief Nets"** (with Simon Osindero and Yee-Whye Teh).

The trick was **Restricted Boltzmann Machines (RBMs)** — Boltzmann machines where units are organized into layers and connections only exist *between* layers (not within). This restriction made training tractable via a method called **contrastive divergence**.

```
Deep Belief Net training (2006):
  Layer 1: Train an RBM on raw data (pixels)
  Layer 2: Train an RBM on Layer 1's hidden activations
  Layer 3: Train an RBM on Layer 2's hidden activations
  ...
  Each layer learns increasingly abstract features.

  Then optionally: fine-tune the whole stack with
  supervised backprop (labels).

  Unsupervised pre-training → supervised fine-tuning.
  Sound familiar? This is the template GPT follows.
```

**Why this was revolutionary**: for 20 years, deep networks had been impossible to train — gradients vanished, training got stuck. Hinton showed that *unsupervised pre-training* could initialize the weights in a good region of parameter space, making subsequent supervised training work. The "secondary" paradigm (unsupervised) was the key to unlocking the "primary" one (supervised).

### Autoencoders Beat PCA (2006)

In the same miraculous year, Hinton and **Ruslan Salakhutdinov** published *"Reducing the Dimensionality of Data with Neural Networks"* in *Science*. They showed that deep autoencoders — networks trained to compress data into a bottleneck and then reconstruct it (an unsupervised task: no labels needed) — could discover better low-dimensional representations than PCA.

```
Autoencoder:
  Input → [Encoder] → Bottleneck (compressed) → [Decoder] → Reconstructed Input

  Loss = difference between input and reconstruction
  No labels needed — the data IS the label.

  PCA: linear compression only
  Deep autoencoder: nonlinear compression → captures curves, manifolds
```

This was the moment unsupervised learning went from "interesting but secondary" to "potentially more powerful than supervised learning for learning representations."

---

## The Generative Explosion

### VAEs — The Reparameterization Trick (2013)

**Diederik Kingma** (a Dutch PhD student at the University of Amsterdam) and **Max Welling** (his advisor) published the **Variational Autoencoder (VAE)** in December 2013.

*(See [Note 24](24_generative_models_taxonomy.md) for the full generative models taxonomy)*

The problem they solved: you want to learn a probabilistic model of data — not just compress it, but generate *new* data that looks like the training data. The math requires sampling from a distribution, but sampling isn't differentiable — you can't backpropagate through random noise.

Kingma's **reparameterization trick** was elegant: instead of sampling z ~ N(μ, σ²), write z = μ + σ · ε where ε ~ N(0,1). Now the randomness is in ε (a fixed input), and the network only needs to learn μ and σ (deterministic functions of the input). Backprop works.

```
VAE = autoencoder where the bottleneck is a probability distribution

  Input → Encoder → (μ, σ) → z = μ + σ·ε → Decoder → Reconstructed Input
                                   ↑
                              random noise
                            (reparameterized)

  You can now:
  1. Sample z from the distribution → generate new data
  2. Interpolate between two z's → smooth transitions
  3. Do arithmetic in latent space → "man" - "king" + "queen" = "woman"
```

### GANs — The Bar Scene (2014)

**Ian Goodfellow** — a PhD student at the University of Montreal, studying under **Yoshua Bengio** — was at a bar with friends in 2014, arguing about generative models. Someone suggested an idea; Goodfellow said it wouldn't work. Walking home slightly drunk, he thought of a different approach. He went home, coded it up that night, and **it worked on the first try**.

The idea: **Generative Adversarial Networks (GANs)** — pit two networks against each other:

```
GAN = a counterfeiter vs. a detective

  Generator: takes random noise → produces fake images
  Discriminator: looks at real and fake images → says "real" or "fake"

  Generator wants to fool the Discriminator.
  Discriminator wants to catch the Generator.
  They get better together, in a minimax game.

  No labels needed for the images. No explicit density model.
  Just: "can you tell the difference?"
```

Yann LeCun called GANs **"the most interesting idea in the last 10 years in ML"** (2016). They generated increasingly photorealistic images, culminating in faces so real they were indistinguishable from photographs (StyleGAN, 2018).

**The GAN lore**: Goodfellow's original paper was rejected from NIPS (now NeurIPS) on first submission. A reviewer argued the method couldn't possibly work. Goodfellow added more experiments, it was accepted, and within a few years it was one of the most cited papers in ML history.

### Diffusion Models — From Physics to Art (2015 → 2020)

**Jascha Sohl-Dickstein** — a physicist-turned-ML-researcher at Stanford (later Google Brain) — published *"Deep Unsupervised Learning using Nonequilibrium Thermodynamics"* in 2015. The idea came from physics: thermodynamic processes gradually destroy structure (add noise), and this process can be reversed.

```
Diffusion model intuition:

  Forward process (easy): gradually add Gaussian noise to an image
    Real image → slightly noisy → more noisy → ... → pure noise

  Reverse process (learned): gradually remove noise
    Pure noise → slightly less noisy → ... → generated image

  Train a neural network to predict the noise at each step.
  At generation time, start with pure noise and denoise step by step.
```

This sat mostly dormant until **Jonathan Ho**, **Ajay Jain**, and **Pieter Abbeel** at UC Berkeley published **DDPM (Denoising Diffusion Probabilistic Models)** in 2020. Their paper showed that diffusion models could generate images as good as or better than GANs — and with much more stable training.

By 2022, diffusion models powered **DALL-E 2**, **Stable Diffusion**, and **Midjourney**, generating the AI art explosion. GANs had been dethroned by an idea from thermodynamics.

### The Philosophical Flip

The unsupervised learning story has a satisfying arc:

```
1901: PCA — find the spread (a statistics tool)
1949: Hebb — find co-activations (a neuroscience principle)
1957: K-means — find clusters (an engineering tool, unpublished!)
1982: SOMs — find topology (a brain-inspired map)
1985: Boltzmann machines — find energy minima (a physics import)
2006: Deep Belief Nets — find representations (the pre-training revolution)
2013: VAEs — find latent distributions (the generative revolution)
2014: GANs — find adversarial equilibria (game theory meets generation)
2020: Diffusion — find denoising paths (thermodynamics meets art)

The "secondary" paradigm became the foundation of modern AI.
GPT, BERT, Stable Diffusion — all unsupervised at their core.
```

---

# Part III: Reinforcement Learning — The Oldest Idea, The Longest Road

## From Animal Psychology to AlphaGo

Reinforcement learning is the oldest of the three paradigms in spirit — animals learning from consequences predates any formal mathematics. But it took the longest to become computationally useful.

*(Note 25 covers RL techniques and applications in detail. This section focuses on the historical lore — the people and the ideas.)*

---

## The Animal Psychologists

### Edward Thorndike — Cats in Puzzle Boxes (1898)

**Edward Lee Thorndike** — a 24-year-old graduate student at Columbia University — was running an experiment that his landlady wasn't happy about. He'd built wooden **puzzle boxes** in his apartment and was putting cats inside them to see if they could learn to escape.

The cat would scratch, claw, and flail randomly until it accidentally pressed a lever or pulled a loop that opened the door. Thorndike timed how long it took. Then he put the cat back in. Over dozens of trials, the cat escaped faster and faster — not through "insight" or "reasoning," but through gradually strengthening the behaviors that led to freedom and weakening those that didn't.

Thorndike formalized this as the **Law of Effect** (1898):

> "Responses that produce a satisfying effect in a particular situation become more likely to occur again in that situation, while responses that produce a discomforting effect become less likely to occur."

```
Thorndike's Law of Effect (1898):
  Action → satisfying consequence → action becomes more likely
  Action → annoying consequence  → action becomes less likely

  This IS reinforcement learning, stated 55 years before
  anyone built a computer to do it.
```

**The landlady detail**: Thorndike originally tried to run his experiments at Harvard, but the university wouldn't give him space. He moved the puzzle boxes to his apartment. His landlady objected to the noise (and presumably the cats). William James — the legendary psychologist — let Thorndike run experiments in James's *own basement*. American psychology's most important animal learning experiments happened in William James's cellar.

### Ivan Pavlov — Classical Conditioning (1903)

**Ivan Petrovich Pavlov** — a Russian physiologist who won the **Nobel Prize in Physiology (1904)** for his work on digestion — is famous for his dogs. Ring a bell before feeding a dog; eventually, the bell alone triggers salivation. **Classical conditioning**: an involuntary response gets associated with a new stimulus.

Pavlov was studying digestion, not learning. He noticed the dogs were salivating *before* food arrived — at the sight of the lab assistant, the sound of footsteps, eventually at any signal that food was coming. His lab assistant **Ivan Tolochinov** actually first reported the phenomenon, but Pavlov systematized it.

```
Pavlov vs Thorndike — two types of conditioning:

  Classical (Pavlov):     Stimulus → Response (involuntary)
                          Bell → Salivation (the dog doesn't "decide" to drool)

  Operant (Thorndike):   Action → Consequence → Action changes (voluntary)
                          Press lever → Food → Press lever more often

  Reinforcement learning descends from OPERANT conditioning.
  The agent CHOOSES actions and learns from consequences.
```

**B.F. Skinner** later formalized operant conditioning in the 1930s–50s, building elaborate "Skinner boxes" that automated Thorndike's puzzle box idea. Skinner's terminology — positive reinforcement, negative reinforcement, punishment — became the language of behaviorism and eventually the language of RL algorithms.

---

## The Mathematicians

### Richard Bellman — Dynamic Programming, and How to Name Things Under RAND (1953)

**Richard Ernest Bellman** (1920–1984) — born in Brooklyn to a family that ran a **small grocery store** on Bergen Street — would become one of the most important applied mathematicians of the 20th century.

After serving in World War II and earning his PhD from Princeton under Solomon Lefschetz, Bellman joined the **RAND Corporation** in 1952. RAND — "Research ANd Development" — was a Cold War think tank funded by the Air Force, tackling everything from nuclear strategy to operations research.

At RAND, Bellman developed **dynamic programming** — a method for solving complex optimization problems by breaking them into simpler subproblems. The core idea:

```
The Bellman Equation:
  V(s) = max_a [ R(s,a) + γ · V(s') ]

  "The value of being in a state = the best available
   (immediate reward + discounted future value)"

  This is recursive: the value depends on future values.
  Solving this recursion IS the core of RL.
```

**The naming story** is legendary. Bellman himself told it in his autobiography:

> "The 1950s were not good years for mathematical research. I felt I had to do something to shield [dynamic programming] ... I decided therefore to use the word 'programming.' I wanted to get across the idea that this was dynamic, this was multistage, this was time-varying ... I thought dynamic programming was a good name. It was something not even a Congressman could object to."

He was working under **Charles Wilson**, the Secretary of Defense, who Bellman described as having "a pathological fear and hatred of the word 'research.'" So Bellman chose words that sounded applied and practical rather than theoretical.

Bellman also coined the phrase **"curse of dimensionality"** — the observation that the number of states grows exponentially with the number of dimensions, making exhaustive solutions impossible for real-world problems. This curse is exactly what makes RL hard: a chess board has more possible positions than atoms in the universe.

### Arthur Samuel — Checkers, Self-Play, and Coining "Machine Learning" (1952–1962)

**Arthur Lee Samuel** (1901–1990) — an electrical engineer at **IBM** — started building a checkers-playing program in 1952 on the **IBM 701**, one of the first commercial scientific computers.

Samuel's program was remarkable for two reasons:

1. **Self-play**: the program played against itself to generate training data — no human opponent needed. This is the ancestor of AlphaGo Zero's self-play, 65 years later.

2. **The term "machine learning"**: Samuel's 1959 paper, *"Some Studies in Machine Learning Using the Game of Checkers,"* is where the phrase "machine learning" first appears in print.

```
Samuel's checkers program:
  - Evaluated board positions using a scoring function
  - Improved by playing against a copy of itself
  - Used alpha-beta pruning (search tree optimization)
  - Beat a former Connecticut state champion in 1962

  Demo on live TV (1956): Thomas Watson Sr. (IBM chairman)
  predicted a 15-point IBM stock rise from the demo.
  He was right.

  One of the first times AI was used to move a stock price.
```

Samuel was in his 50s when he built this — already a senior engineer at IBM. He worked on it as a passion project, not as his main assignment. He once said the program taught him more about checkers than years of playing had.

---

## The Modern Synthesis

### A. Harry Klopf — Reward-Seeking as Fundamental (1972)

**A. Harry Klopf** — a researcher at the **Air Force's Wright-Patterson Air Force Base** in Ohio — published a series of provocative papers in the 1970s arguing that the fundamental drive of intelligence isn't *prediction* (as most AI researchers believed) but **reward-seeking** — what he called the "hedonistic" hypothesis.

His 1972 paper proposed that neurons themselves are reward-seeking entities — not passive processors but active agents trying to maximize their own "satisfaction." This was a radical departure from the prevailing view of neural networks as error-minimizing machines.

**Why Klopf matters**: he directly inspired **Richard Sutton** to pursue temporal difference learning. Sutton has called Klopf's ideas the spark that lit his career.

### Richard Sutton & Andrew Barto — TD Learning at UMass (1980s)

**Richard Sutton** — a Canadian who started in psychology before moving to computer science — and **Andrew Barto** — a physicist-turned-computer-scientist — met at the **University of Massachusetts Amherst** in the late 1970s.

Sutton was Barto's PhD student. Together, they developed the framework that unified animal learning theory, dynamic programming, and neural networks into what we now call **modern reinforcement learning**.

Their key contribution was **Temporal Difference (TD) learning** (1988) — the insight that you don't have to wait until the end of an episode to learn. You can update your estimates at every time step based on the *difference* between consecutive predictions:

```
TD learning:
  V(s) ← V(s) + α · [r + γ·V(s') - V(s)]
                       └─────────────────┘
                         "TD error" — the surprise

  In English:
  "My new estimate = my old estimate + learning rate ×
   (what actually happened - what I expected)"

  This is learning from SURPRISES — when reality differs
  from expectation, that's information.
```

Their 1998 textbook, *Reinforcement Learning: An Introduction*, became the bible of the field. It remains the standard reference — a rare case where one book defines an entire subfield.

**The neuroscience connection**: In 1997, **Wolfram Schultz** and colleagues at the University of Cambridge discovered that **dopamine neurons** in the monkey brain fire in a pattern that looks *exactly* like the TD error signal. When a monkey receives an unexpected reward, dopamine spikes (positive TD error). When an expected reward doesn't arrive, dopamine drops (negative TD error). Biology had independently discovered TD learning. Sutton and Barto's algorithm was a mathematical description of what primate brains actually do.

### Chris Watkins — Q-Learning (1989)

**Christopher Watkins** — a British PhD student at Cambridge — published **Q-learning** in his 1989 thesis (and a key 1992 paper with Peter Dayan). The breakthrough: a **model-free** algorithm that learns the value of actions *directly*, without needing a model of the environment.

```
Q-learning update:
  Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') - Q(s,a)]

  "Learn the value of each action in each state —
   without knowing the rules of the game."

  Model-based RL: "I know the rules, I'll plan ahead"
  Model-free RL:  "I don't know the rules, I'll learn from experience"

  Q-learning is model-free. It's what DQN (2013) scaled up
  with deep neural networks.
```

### TD-Gammon (1992) — The Proof of Concept

**Gerald Tesauro** at IBM combined a neural network with TD learning to play **backgammon**. TD-Gammon learned entirely through self-play — 1.5 million games against itself — and reached expert-level performance.

The remarkable detail: professional backgammon players *adopted strategies they learned from TD-Gammon*. The machine didn't just match human play — it discovered better strategies that humans then copied. This happened 24 years before AlphaGo's "Move 37."

*(See [Note 25](25_reinforcement_learning_overview.md) for the full DQN → AlphaGo → RLHF pipeline)*

---

## The Game AI Pipeline (2013–2022)

The full story of DQN, AlphaGo, AlphaStar, OpenAI Five, and RLHF is told in [Note 25](25_reinforcement_learning_overview.md). Here's the biographical lore:

### DeepMind and Demis Hassabis

**Demis Hassabis** — born in London to a Greek-Cypriot father and a Chinese-Singaporean mother — was a chess prodigy (ranked second in the world for his age at 13), a video game designer (co-created *Theme Park* at age 17 while at Bullfrog Productions), and a neuroscience PhD from UCL before founding **DeepMind** in 2010.

His thesis — literally — was that the hippocampus enables imagination by recombining memories. He brought this neuroscience perspective to AI: intelligence requires memory, imagination, and planning, not just pattern matching.

DeepMind's trajectory: **DQN** (2013, Atari) → Google acquires for ~$500M (2014) → **AlphaGo** (2016, beats Lee Sedol) → **AlphaGo Zero** (2017, no human data) → **AlphaZero** (2017, chess/Go/shogi) → **AlphaFold** (2020, protein folding, Nobel Prize 2024).

### Move 37 — The Moment RL Entered Popular Culture

In Game 2 of the AlphaGo vs. Lee Sedol match (March 2016), the AI played **Move 37** — placing a stone on the upper right side of the board in a position that no human expert would consider. The commentators thought it was a mistake.

Fan Hui, the European Go champion who had previously lost to AlphaGo, was watching from the audience. He later said: *"It's not a human move. I've never seen a human play this move. So beautiful."*

The move turned out to be brilliant. It set up a strategic position that won the game. 200 million people watched. Lee Sedol won one game back (Game 4, move 78 — a "God move" of his own), but lost the match 4-1.

After the match, Lee Sedol said: *"I apologize for being so powerless."* He retired from professional Go in 2019, saying AI could never be defeated.

### The "Reward Is Enough" Thesis (2021)

**David Silver** — the lead researcher behind AlphaGo — co-authored *"Reward Is Enough"* in 2021, arguing that **all of intelligence** — perception, language, social behavior, knowledge — can arise from reward maximization in a sufficiently complex environment.

This is a philosophical claim as much as a technical one: you don't need separate systems for vision, language, and reasoning. You just need one reward-maximizing agent in a rich enough world. It's the strongest claim RL has ever made about its own importance.

---

# Part IV: The Convergence — How the Three Paradigms Merged

The story of the three paradigms isn't three separate stories. It's one story with three threads that keep crossing and eventually converge.

## Self-Supervised Learning — Blurring the Line (2018)

**Self-supervised learning** — the paradigm behind BERT (2018) and GPT (2018) — sits in the crack between supervised and unsupervised:

```
Self-supervised learning:
  "Here's a sentence with a word removed: 'The cat sat on the ___'"
  "Predict the missing word."

  Is this supervised? There ARE labels (the missing word).
  Is this unsupervised? The labels come FROM the data itself.

  It's both. Or neither. The boundary dissolved.

  BERT: mask 15% of words → predict them (MLM)
  GPT:  predict the next word → repeat forever
```

Yann LeCun calls self-supervised learning the **"dark matter of intelligence"** — the vast majority of what humans learn comes not from labeled examples (supervised) or from explicit rewards (RL) but from predicting what comes next in our sensory experience.

## RLHF — Reinforcement Learning Meets Language (2022)

The connection between RL and language models became explicit with **RLHF (Reinforcement Learning from Human Feedback)**, which turned GPT-3 (a raw text predictor) into ChatGPT (a helpful assistant).

```
The convergence in one pipeline:

  1. UNSUPERVISED pre-training
     GPT-3 predicts next tokens on internet text
     (no labels, no rewards — just patterns)

  2. SUPERVISED fine-tuning
     Human contractors write ideal responses
     (labeled examples: prompt → good response)

  3. REINFORCEMENT LEARNING alignment
     Humans rank responses → train a reward model →
     use PPO to optimize the language model
     (delayed reward signal, same algorithm as OpenAI Five/Dota 2)

  All three paradigms, in sequence, in one system.
```

The PPO algorithm that taught an AI to play Dota 2 (2019) is the same algorithm that made ChatGPT helpful (2022). The paradigms didn't just converge theoretically — the same *code* crossed the boundary.

## The Foundation Model Pattern

Modern AI has settled into a pattern that uses all three paradigms:

```
┌─────────────────────────────────────────────────┐
│           THE FOUNDATION MODEL RECIPE           │
│                                                 │
│  Step 1: UNSUPERVISED pre-training              │
│          Learn the structure of language/images  │
│          (billions of unlabeled examples)        │
│                                                 │
│  Step 2: SUPERVISED fine-tuning                  │
│          Learn to follow instructions            │
│          (thousands of labeled examples)         │
│                                                 │
│  Step 3: RL alignment (RLHF / RLAIF)            │
│          Learn to be helpful, harmless, honest   │
│          (human preference rankings)             │
│                                                 │
│  This is how GPT-4, Claude, Gemini are made.    │
│  Three paradigms. One model.                    │
└─────────────────────────────────────────────────┘
```

---

## The Grand Timeline

```
SUPERVISED                  UNSUPERVISED                REINFORCEMENT
──────────                  ────────────                ─────────────

1805: Least squares         1898: Thorndike's cats
      (Legendre/Gauss)            (Law of Effect)
                            1901: PCA (Pearson)
                                                        1903: Pavlov's dogs
1936: Fisher's Iris                                           (conditioning)
      (classification)
                            1949: Hebb's rule
                                  ("fire together")
                                                        1953: Bellman (dynamic
1957: Perceptron                                              programming)
      (Rosenblatt)          1957: K-means (Lloyd,       1952: Samuel's checkers
                                  unpublished!)               → "machine learning"
1960: ADALINE
      (gradient descent)

1969: Minsky kills it ──────── AI WINTER ──────────────────────────────
      (XOR limit)

1974: Werbos backprop
      (ignored)
                            1982: SOMs (Kohonen)        1972: Klopf (hedonistic)
                            1985: Boltzmann
1986: Backprop works!             machines (Hinton)     1988: TD learning
      (Rumelhart/Hinton)                                      (Sutton & Barto)
                                                        1989: Q-learning
1989: CNN (LeCun)                                             (Watkins)
                                                        1992: TD-Gammon
1995: SVM (Vapnik)                                            (Tesauro)

                            2006: Deep Belief Nets
                                  (Hinton) ─── "unsupervised pre-training
                                               enables supervised learning"
2012: AlexNet ─────────────── DEEP LEARNING REVOLUTION ────────────────
      (GPU + data + scale)
                            2013: VAE (Kingma)          2013: DQN (DeepMind)
                            2014: GAN (Goodfellow)      2016: AlphaGo (Move 37)
                                                        2017: AlphaZero

2018: ──── SELF-SUPERVISED (BERT, GPT) ──── blurs supervised/unsupervised

                            2020: Diffusion (DDPM)

2022: ──── RLHF (ChatGPT) ──── all three converge ────────────────────

2023-now: Foundation models = unsupervised + supervised + RL
```

---

## The One-Sentence Summary

**Supervised learning taught machines to match answers (200 years of statistics → perceptron → backprop → AlexNet), unsupervised learning taught them to find structure (PCA → Boltzmann machines → GANs → diffusion), and reinforcement learning taught them to seek rewards (Thorndike's cats → Bellman's equations → AlphaGo → RLHF) — and modern AI uses all three, in sequence, in every foundation model.**

---

## Cross-References

- [Timeline](00_timeline_neural_sequence_models.md) — master chronology where all these milestones appear
- [Rosenblatt and the Perceptron](02_rosenblatt_and_the_perceptron.md) — full biography of the supervised learning pioneer
- [Minsky — Death and Resurrection](06_minsky_perceptrons_death_and_resurrection.md) — how the AI winter happened
- [Backprop: The Wiggle Ratio](09_backprop_the_wiggle_ratio.md) — intuitive backpropagation explanation
- [Generative Models Taxonomy](24_generative_models_taxonomy.md) — VAEs, GANs, diffusion, Boltzmann machines in depth
- [Reinforcement Learning Overview](25_reinforcement_learning_overview.md) — RL techniques, the game pipeline (DQN → AlphaGo → RLHF), applications
