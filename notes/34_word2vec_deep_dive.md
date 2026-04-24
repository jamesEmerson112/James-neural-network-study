# Word2vec — Skip-gram, CBOW, and Negative Sampling

## What this note unpacks

- The **distributional hypothesis** — "a word is known by the company it keeps"
- **Skip-gram** and **CBOW** — the two Word2vec architectures
- The **softmax bottleneck** — why training with full softmax over the vocabulary is too slow
- **Negative sampling** — Word2vec's fix for the bottleneck, with a full derivation
- **Hierarchical softmax** — the alternative fix using a Huffman tree
- The **vector arithmetic** result (king − man + woman ≈ queen) and why it works
- History: Mikolov 2013 and what came before

---

## The distributional hypothesis

From the linguist **J. R. Firth (1957)**:

> *"You shall know a word by the company it keeps."*

**Claim:** words that appear in similar contexts tend to have similar meanings. "dog" and "cat" appear near words like "pet," "fur," "walk," "bowl." "car" and "truck" appear near words like "drive," "highway," "engine." Two words being interchangeable in many contexts suggests they refer to similar concepts.

Word2vec operationalizes this hypothesis: **learn vector representations such that words with similar contexts have similar vectors.** The "similarity" is measured by dot product (or equivalently, cosine similarity after normalization). Training pushes contextually-similar word vectors close together in $\mathbb{R}^d$.

**The big result:** once you have these vectors, semantic relationships show up as *geometric directions*. "The direction from 'man' to 'woman' is similar to the direction from 'king' to 'queen'" — because both pairs capture the same "male → female" semantic shift. We'll see why this happens below.

---

## Skip-gram vs CBOW — two architectures

Word2vec (Mikolov et al. 2013) comes in two flavors. Both learn word vectors, but they frame the prediction task differently.

### Skip-gram: predict context from center

Given the center word, predict which words appear in its context window.

```
  sentence:  "the quick brown fox jumps over the lazy dog"
  center:    "fox"
  window:    2 words on each side

  task:  given "fox", predict {"quick", "brown", "jumps", "over"}
```

**Formally:** for each center word $w_c$ in the training corpus, maximize the log-probability of its context words $w_{c+j}$ (for $-m \leq j \leq m$, $j \neq 0$):

$$
\max \sum_{c} \sum_{-m \leq j \leq m, \, j \neq 0} \log p(w_{c+j} \mid w_c)
$$

where $m$ is the window size (typically 5–10).

### CBOW: predict center from context

The reverse: given the surrounding context words, predict the center word.

```
  context:  {"quick", "brown", "jumps", "over"}
  task:     predict the center word "fox"
```

**Formally:**

$$
\max \sum_{c} \log p(w_c \mid w_{c-m}, \ldots, w_{c-1}, w_{c+1}, \ldots, w_{c+m})
$$

### Architecture diagrams

```
  ── SKIP-GRAM ──                       ── CBOW ──

    center word                         context words
       │                                 │  │  │  │
       ▼                                 ▼  ▼  ▼  ▼
   ┌────────┐                         ┌──────────────┐
   │embedding│                         │ embeddings + │
   │  lookup │                         │   average    │
   └────────┘                         └──────┬───────┘
       │                                      │
       ▼                                      ▼
   ┌────────┐                             ┌────────┐
   │  linear │                             │  linear │
   │  + softmax                            │  + softmax
   │  over V │                             │  over V │
   └────────┘                             └────────┘
       │                                      │
       ▼                                      ▼
   P(context | center)               P(center | context)
```

**Both architectures are shallow**: just a lookup table + one linear layer + softmax. No hidden layers. No nonlinearities. This is deliberate — Mikolov's 2013 contribution wasn't a new architecture, it was showing that even this simple setup produces useful embeddings when trained on enough data.

**Which to use?**
- **Skip-gram** is slower to train but works better for infrequent words (each center word generates $2m$ training examples, one per context position).
- **CBOW** is faster to train and smooths over the contexts (the average over context words acts as regularization), but infrequent words suffer more.
- In practice, Skip-gram with negative sampling is the most common choice.

---

## The softmax bottleneck

Both Skip-gram and CBOW end with a softmax over the entire vocabulary:

$$
p(w_O \mid w_I) = \frac{\exp(v_{w_O}^\top u_{w_I})}{\sum_{w \in V} \exp(v_w^\top u_{w_I})}
$$

where:
- $u_{w_I}$ is the "input" embedding of the input word $w_I$
- $v_{w_O}$ is the "output" embedding of the output word $w_O$
- $V$ is the vocabulary

Crucially, **each word has two embeddings**: $u$ (used when it's the input) and $v$ (used when it's the output). This is a Word2vec convention. After training, usually only $u$ is kept as the "word vector," though some systems use $(u + v) / 2$.

**The problem:** the denominator sums over the **entire vocabulary**. For a 1 million word vocabulary, computing this sum — and its gradient — requires **1 million dot products per training example**. Word2vec papers used corpora with billions of tokens, so this is completely infeasible.

```
  cost per training step ∝ |V| = vocabulary size

  for a 1M word vocabulary and 1B training tokens:
  total operations ≈ 10⁶ × 10⁹ = 10¹⁵ = petaFLOPs
```

This is the **softmax bottleneck**. Word2vec's main technical contribution is two ways to work around it: **negative sampling** and **hierarchical softmax**.

---

## Negative sampling — the main trick

**The idea:** replace the softmax over the full vocabulary with a **binary classification** task. For each (center, context) pair in the data, sample a handful of random "negative" words that are not in the context, and train a logistic classifier to say "yes, this is a real pair" for positives and "no, this is not" for negatives.

**Intuitively:** instead of asking "of all possible words, which is the right one?", ask "is this word the right one, yes or no?" Then do that for a few random words as negative examples. The training signal becomes $O(k)$ per step instead of $O(|V|)$, where $k$ is typically 5–20.

### Formal derivation

Start from the full softmax objective for Skip-gram:

$$
\log p(w_O \mid w_I) = v_{w_O}^\top u_{w_I} - \log \sum_{w \in V} \exp(v_w^\top u_{w_I})
$$

The expensive part is the $\log \sum_w$ term. Replace it with a binary classification loss that is much cheaper.

Define the **positive pair loss** using a sigmoid $\sigma(x) = 1 / (1 + e^{-x})$:

$$
\mathcal{L}_{\text{pos}} = \log \sigma(v_{w_O}^\top u_{w_I})
$$

This says: "the sigmoid output for the real (input, output) pair should be close to 1." Maximizing this pushes $v_{w_O}^\top u_{w_I}$ to be large and positive.

For each positive pair, sample $k$ **negative** words $w_{n_1}, \ldots, w_{n_k}$ from a distribution $P_{\text{neg}}(w)$ (typically the unigram distribution raised to the 3/4 power — more on this below). Each negative contributes:

$$
\mathcal{L}_{\text{neg}_i} = \log \sigma(-v_{w_{n_i}}^\top u_{w_I})
$$

Note the negative sign inside the sigmoid. This says: "the sigmoid output should be close to 0" — equivalently, "$\sigma(-v_{n_i}^\top u_{w_I})$ should be close to 1." Maximizing this pushes $v_{w_{n_i}}^\top u_{w_I}$ to be large and negative (i.e., the negative word vector should point *away* from the input word).

Total loss for one positive pair plus its negatives:

$$
\mathcal{L}_{\text{NS}} = \log \sigma(v_{w_O}^\top u_{w_I}) + \sum_{i=1}^{k} \log \sigma(-v_{w_{n_i}}^\top u_{w_I})
$$

Maximize this across all (center, context) pairs in the training corpus.

**Cost analysis:** each training step touches $k + 1$ word vectors (one positive, $k$ negatives) instead of $|V|$. For $k = 5$ and $|V| = 10^6$, that's a speedup of ~$10^5$ per step. Word2vec trains in hours instead of weeks.

### Why the 3/4 power?

Mikolov's paper recommends sampling negatives from:

$$
P_{\text{neg}}(w) \propto f(w)^{3/4}
$$

where $f(w)$ is the raw frequency of word $w$ in the corpus. The 3/4 exponent down-weights frequent words (like "the", "is") and up-weights rare words. This is an empirically-tuned hyperparameter — there's no deep theory behind 3/4, but it works better than raw unigram probabilities or uniform sampling.

**Intuition for why:** if you sampled negatives purely uniformly, rare words would be picked too often (wasting training on obvious non-matches). If you sampled from raw frequency, common stopwords would dominate. The 3/4 power is a compromise: rare words get more attention than pure frequency would allow, but not as much as uniform.

---

## Hierarchical softmax — the alternative trick

**The idea:** organize the vocabulary as a binary tree (specifically, a **Huffman tree** built from word frequencies). Every word becomes a leaf. To compute $p(w \mid \text{context})$, walk from the root to the leaf and make a binary decision ("go left or right?") at each internal node. The total cost is $O(\log |V|)$ instead of $O(|V|)$.

```
             root
            /    \
          n₁      n₂
         / \     / \
       n₃   n₄  w₁  n₅
      / \  / \      / \
    w₂ w₃ w₄ w₅   w₆ w₇

  word "w₄" = path (right, left, right) from root
  p(w₄ | context) = σ(score₁) × (1 - σ(score₂)) × σ(score₃)
```

Each internal node has its own learned vector. At each node, you compute the sigmoid of the dot product between the context vector and the node's vector — that's the probability of "going right." The probability of a full word is the product of these sigmoids along the path.

**Cost analysis:** each word's path has depth $\log_2 |V|$ nodes. For a 1 million word vocabulary, that's ~20 nodes instead of 1 million softmax terms. A ~50,000x speedup.

**Huffman coding detail:** the tree is built using Huffman coding, which assigns shorter paths to more frequent words. Frequent words (like "the") get paths of length ~3, while rare words get paths of length ~25. On average, the tree weighted by word frequency has depth $\sim H(V)$, the entropy of the unigram distribution.

**Negative sampling vs hierarchical softmax**:
- **Negative sampling** is simpler to implement and often empirically better for Skip-gram
- **Hierarchical softmax** gives exact probabilities (negative sampling is an approximation)
- Both are much faster than full softmax
- Most implementations default to negative sampling now, but the Word2vec paper supported both

---

## Why vector arithmetic works

The famous result:

$$
\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})
$$

Or equivalently: $\text{vec}(\text{king}) - \text{vec}(\text{queen}) \approx \text{vec}(\text{man}) - \text{vec}(\text{woman})$.

**Why does this happen?**

Word vectors are trained to predict context. "king" and "queen" appear in many identical contexts (royalty, throne, crown, palace), but also in a few contexts that differ by gender (masculine vs feminine pronouns, certain roles). The "king" and "queen" vectors end up similar along most axes (the shared royalty contexts) but different along a "gender axis" — the components that relate to masculine vs feminine contexts.

"man" and "woman" have a similar gender-axis difference, for the same reason.

So:
- $\text{vec}(\text{king}) - \text{vec}(\text{queen})$ ≈ the "male → female" direction (from royalty context)
- $\text{vec}(\text{man}) - \text{vec}(\text{woman})$ ≈ the "male → female" direction (from general context)
- These two differences are similar, so by rearranging: $\text{vec}(\text{king}) - \text{vec}(\text{man}) \approx \text{vec}(\text{queen}) - \text{vec}(\text{woman})$, i.e., $\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})$

**The deeper reason:** semantic relationships end up as consistent *directions* because the training objective pushes words with similar contexts into similar regions. Linear algebra (vector addition / subtraction) captures these relationships because the loss is linear in the dot product, and "similar contexts" compose additively.

**This doesn't always work.** The most famous success cases are gender (king/queen, man/woman), verb tense (walking/walked, eating/ate), and geography (Paris/France, Tokyo/Japan). But for many relationships, vector arithmetic fails — Word2vec doesn't encode every semantic relation as a direction, just the most statistically salient ones.

**Modern critique:** when BERT-style contextual embeddings came along, people realized static word vectors like Word2vec can't distinguish "bank" (financial) from "bank" (river). Contextual embeddings assign different vectors to the same word in different contexts. But the vector arithmetic property of Word2vec is still a beautiful and pedagogically important result.

---

## Training loop

```
algorithm: Skip-gram with negative sampling
  initialize all word vectors u_w, v_w ∈ R^d (small random values)
  build Huffman tree or negative sampling distribution P_neg

  for each (center, context) pair in the training corpus:
    # positive pair
    score ← v_context^T u_center
    loss_pos ← log σ(score)

    # k negative samples
    for i = 1, ..., k:
      n_i ← sample from P_neg
      score_n ← v_{n_i}^T u_center
      loss_neg_i ← log σ(-score_n)

    total_loss ← loss_pos + Σ_i loss_neg_i

    # gradient ascent on total_loss
    update u_center, v_context, v_{n_1}, ..., v_{n_k}

  return word embeddings {u_w} for all w in V
```

The updates are tiny: each training step only touches $k + 1 + 1 = k + 2$ vectors (input, positive output, $k$ negatives). That's why Word2vec can train on billion-token corpora in a few hours on a single machine.

---

## History/lore

- **1954 — Zellig Harris** publishes *Distributional Structure*, the first formal statement of the distributional hypothesis: meaning is usage, usage is context, so distribution is meaning.
- **1957 — J. R. Firth** coins the famous phrase "You shall know a word by the company it keeps" in *A Synopsis of Linguistic Theory 1930–1955*. Firth was a British linguist at the University of London; his students would later form the basis of the "London School" of linguistics.
- **1990s — LSA (Latent Semantic Analysis)** and **LDA (Latent Dirichlet Allocation)** are earlier statistical NLP techniques that learn word representations from context. They use matrix factorization and probabilistic modeling instead of neural networks.
- **2003 — Bengio et al.** publish *A Neural Probabilistic Language Model*, which introduces learned word embeddings as a byproduct of training a neural language model. This is the conceptual ancestor of Word2vec, but too slow to train at scale.
- **2008 — Collobert & Weston** publish *A Unified Architecture for Natural Language Processing*, showing neural word embeddings can be learned end-to-end for many NLP tasks. Pre-trained embeddings become a thing.
- **2013 — Tomas Mikolov** (at Google Brain, with Kai Chen, Greg Corrado, Jeff Dean) publishes two papers on arXiv in January and September: *Efficient Estimation of Word Representations in Vector Space* and *Distributed Representations of Words and Phrases and their Compositionality*. These introduce Skip-gram, CBOW, negative sampling, hierarchical softmax, and the famous vector arithmetic results. The second paper demonstrates "king − man + woman ≈ queen" for the first time.
- **2013 — Google releases Word2vec** as open-source code along with pre-trained 300-dimensional vectors for 3 million English words and phrases. This single release put high-quality word embeddings in every NLP practitioner's hands.
- **2014 — GloVe** (Pennington, Socher, Manning at Stanford) offers an alternative approach using word co-occurrence statistics. Shows that Word2vec and GloVe are doing essentially the same thing from different angles.
- **2016 — FastText** (Bojanowski et al. at Facebook) extends Word2vec with subword information (character n-grams), solving the out-of-vocabulary problem.
- **2018 — ELMo, BERT** introduce **contextual** embeddings that replace the same word's representation in different contexts. This is the beginning of the end for static embeddings like Word2vec — but Word2vec remains a pedagogical and historical touchstone, and the "meaning as vector geometry" insight carries forward.

**Mikolov's career**: Tomas Mikolov was a PhD student in Brno, Czech Republic, then joined Google Brain for the Word2vec work. He later moved to Facebook AI (where he worked on FastText) and then to the Czech Technical University. Word2vec is probably the most cited NLP paper of the 2010s.

---

## Takeaway

- **Distributional hypothesis** (Firth 1957): words with similar contexts have similar meanings.
- **Skip-gram**: given center word, predict context words. **CBOW**: given context words, predict center word.
- **Full softmax**: $p(w_O \mid w_I) = \exp(v_{w_O}^\top u_{w_I}) / \sum_{w \in V} \exp(v_w^\top u_{w_I})$ — denominator is over whole vocabulary, too slow.
- **Negative sampling**: replace softmax with binary classification on positive pair + $k$ random negatives. Loss: $\log \sigma(v_{w_O}^\top u_{w_I}) + \sum_i \log \sigma(-v_{n_i}^\top u_{w_I})$.
- **Hierarchical softmax**: organize vocab as Huffman tree, $\log |V|$ binary decisions instead of $|V|$ softmax terms.
- **Vector arithmetic**: $\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})$. Semantic relationships show up as consistent directions because similar-context words are trained to be similar vectors.
- **History**: Harris 1954 → Firth 1957 → Bengio 2003 → Mikolov 2013 (Word2vec) → GloVe 2014 → ELMo/BERT 2018 (contextual replaces static).
