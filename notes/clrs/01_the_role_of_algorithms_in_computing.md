# Chapter 1 — The Role of Algorithms in Computing

*Source: CLRS (Cormen, Leiserson, Rivest, Stein) — Introduction to Algorithms*

## What this chapter is really about

Before diving into sorting and graphs, CLRS asks: what IS an algorithm, why do we care, and what can't algorithms do? This chapter sets up the motivation for the entire book — and connects directly to why deep learning works (and where it cheats).

---

## What is an algorithm?

A well-defined computational procedure that takes an input and produces an output. That's it.

```
Input  ──►  [ Algorithm ]  ──►  Output

"A sequence of computational steps that transforms
 the input into the output."
```

More precisely: for every valid input, the algorithm must:
1. Terminate (eventually stop)
2. Produce the correct output

An algorithm is correct if, for every input instance, it halts with the correct output.

### DL connection

A trained neural network is NOT an algorithm by this strict definition — it doesn't guarantee correct output for every input. It's an approximation. But the *training procedure* (SGD, backprop) is an algorithm. The distinction matters: algorithms give guarantees, models give estimates.

---

## The sorting problem — the canonical example

CLRS uses sorting as the running example throughout the book because it's:
- Easy to state
- Hard to do efficiently
- Foundational to many other algorithms

**Formal definition:**

```
Input:   a sequence of n numbers ⟨a₁, a₂, ..., aₙ⟩
Output:  a permutation ⟨a'₁, a'₂, ..., a'ₙ⟩ such that a'₁ ≤ a'₂ ≤ ... ≤ a'ₙ
```

An instance of the sorting problem: $\langle 31, 41, 59, 26, 41, 58 \rangle$

Sorted output: $\langle 26, 31, 41, 41, 58, 59 \rangle$

### Why sorting matters beyond sorting

```
Binary search:          needs sorted data              O(log n) vs O(n)
Database indexing:      B-trees keep keys sorted       fast lookup
Closest pair:           sort by coordinate first       geometry
Median finding:         sort then pick middle          statistics
Removing duplicates:    sort then scan adjacent        data cleaning
```

### DL connection

Sorting shows up in deep learning more than you'd think:
- **Top-k predictions**: softmax outputs are sorted to find the highest probability classes
- **Non-max suppression** in object detection: sort bounding boxes by confidence, greedily remove overlapping ones
- **Beam search** in language models: sort candidate sequences by probability at each decoding step
- **Argsort** for ranking: used everywhere in recommendation systems, attention score ordering

---

## Why algorithm efficiency matters

CLRS makes the case that a faster algorithm beats faster hardware.

```
Computer A:  10 billion ops/sec, runs insertion sort:  c₁·n²
Computer B:  10 million ops/sec, runs merge sort:      c₂·n·log(n)

Sorting 10 million numbers:

Computer A (fast machine, slow algorithm):
  2·(10⁷)² / (10¹⁰) = 20,000 seconds ≈ 5.5 hours

Computer B (slow machine, fast algorithm):
  50·(10⁷)·log(10⁷) / (10⁷) ≈ 1,163 seconds ≈ 20 minutes

The slow computer with the better algorithm wins by 17×.
```

At $n = 10^7$, the difference between $O(n^2)$ and $O(n \log n)$ is the difference between "done in 20 minutes" and "wait 5 hours."

### DL connection

This is exactly why attention complexity matters:

```
Original Transformer attention:  O(n²) where n = sequence length
  512 tokens:    262,144 operations    ← fine
  4096 tokens:   16,777,216 operations ← getting heavy
  100k tokens:   10,000,000,000 ops    ← impossible

Linear attention variants:  O(n)
  100k tokens:   100,000 operations    ← feasible

Same trade-off as CLRS sorting example.
Algorithm choice > hardware choice.
```

---

## Growth of functions — the intuition

CLRS introduces the idea that we care about how algorithms scale as input grows, not how fast they are on small inputs.

```
n = 10          n = 1,000          n = 1,000,000
──────          ─────────          ─────────────
O(1)         1              1                  1
O(log n)     3              10                 20
O(n)         10             1,000              1,000,000
O(n log n)   30             10,000             20,000,000
O(n²)        100            1,000,000          1,000,000,000,000
O(2ⁿ)        1,024          10³⁰⁰              ← universe dies first
```

The key insight: constant factors and lower-order terms don't matter at scale. $100n$ is still better than $n^2$ once $n > 100$.

### DL connection

```
Forward pass through layers:     O(n) in number of layers (sequential)
Attention per layer:             O(n²) in sequence length
Total training:                  O(epochs × dataset × model_size)

When people say "transformers don't scale to long sequences"
they mean O(n²) attention. When they say "linear attention scales"
they mean O(n). Same language as CLRS, same concept.
```

---

## Hard problems — NP-completeness

CLRS introduces the idea that some problems have no known efficient (polynomial-time) algorithm.

### P vs NP

```
P problems:        easy to SOLVE in polynomial time
                   (sorting, shortest path, matrix multiply)

NP problems:       easy to VERIFY in polynomial time
                   (given a proposed answer, you can check it fast)

NP-complete:       the hardest problems in NP
                   (if you solve one efficiently, you solve them ALL)

NP-hard:           at least as hard as NP-complete
                   (might not even be verifiable in polynomial time)
```

### Verify vs Solve

"Verify" = someone hands you a proposed answer, you check if it's correct.

```
SUDOKU:
  Solve:    fill in the grid from scratch          (hard, backtracking)
  Verify:   check a completed grid is valid        (easy, scan rows/cols/boxes)

TRAVELING SALESMAN:
  Solve:    find the shortest route visiting all cities    (try all permutations)
  Verify:   given a route, add up distances, check total   (one pass)

SUBSET SUM:
  Solve:    find a subset of numbers that adds to target   (2ⁿ subsets to try)
  Verify:   given a subset, add them up, check = target    (one pass)
```

### Why "verify" is NOT just a derivative of "solve"

It *feels* like if you can check an answer, you should be able to find one. But they're fundamentally different operations.

```
Lock analogy:
  Solve:    build the key from scratch        (hard)
  Verify:   try a key someone hands you       (just turn it)

Turning a key doesn't teach you how to MAKE one.
```

Traveling Salesman makes this concrete:

```
Possible routes for n cities:   (n-1)!/2

  5 cities:     12 routes                      ← check all, easy
  10 cities:    181,440 routes                 ← still okay
  20 cities:    ~60 quadrillion                ← good luck
  50 cities:    more than atoms in the universe

Verify ONE route:               O(n)           ← just add up distances
Find the BEST route:            O(n!)          ← no known polynomial shortcut
```

The gap between "easy to check" and "hard to find" is the entire P vs NP question. If checking *were* just a derivative of solving, you could always convert a verifier into a solver — and that would mean P = NP.

### The $1,000,000 question

Does P = NP? Can every problem that's easy to verify also be solved efficiently?

- If **yes**: encryption breaks, optimization becomes trivial, the world changes
- If **no**: some problems are fundamentally harder to solve than to check
- Most people believe P ≠ NP but nobody has proven it
- Clay Mathematics Institute offers $1,000,000 for a proof either way

### DL connection

Training a neural network is NP-hard in the general case (finding the global optimum of a non-convex loss landscape). Deep learning "cheats" by:

1. **Not finding the global optimum** — SGD finds a local minimum, which is good enough
2. **Overparameterization** — more parameters than data points creates many good local minima
3. **Stochastic noise** — SGD's randomness helps escape bad local minima

This is why deep learning works in practice despite the underlying problem being theoretically intractable. CLRS teaches you what "intractable" formally means.

```
CLRS world:     "this problem is NP-hard, so we prove approximation bounds"
DL world:       "this problem is NP-hard, so we use SGD and hope for the best"
                 (and it works surprisingly well)
```

---

## The combinatorics you'll see everywhere

The "how many pairs" formula from this chapter:

$$\binom{n}{2} = \frac{n(n-1)}{2}$$

```
n items, pick 2:

3 people:   3×2/2 = 3 pairs
4 people:   4×3/2 = 6 pairs
50 people:  50×49/2 = 1,225 pairs
n tokens:   n(n-1)/2 ≈ O(n²) pairs
```

This is why:
- Self-attention is $O(n^2)$ — every token attends to every other token, that's $\binom{n}{2}$ pairs
- Fully connected layers have $n \times m$ weights — every input paired with every output
- Pairwise distance computation (used in contrastive learning, nearest neighbors) scales quadratically

The general formula for choosing $k$ items from $n$:

$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

- $\binom{n}{1} = n$ — choosing 1 from $n$ (trivial)
- $\binom{n}{2} = \frac{n(n-1)}{2}$ — pairs
- $\binom{n}{n} = 1$ — only one way to choose everything

---

## Algorithms as technology

CLRS ends Chapter 1 by arguing that algorithms are a *technology* — like hardware, networking, or compilers. Even with infinite hardware, a bad algorithm loses to a good one on large inputs.

The practical takeaway: before optimizing code, optimize the algorithm. Going from $O(n^2)$ to $O(n \log n)$ matters more than any hardware upgrade.

### DL connection — this principle in action

```
2017: Original Transformer          O(n²) attention, max ~512 tokens
2020: Efficient Transformers        O(n√n) or O(n log n) attention
2023: State space models (Mamba)    O(n) sequence processing
2024: Ring attention                O(n²) but distributed across GPUs

Each step: same insight as CLRS Chapter 1.
Better algorithms > more hardware.
```

---

## Quick-fire self-test

1. What makes a correct algorithm? *(Halts for every valid input and produces the correct output)*
2. Why is insertion sort O(n²) worse than merge sort O(n log n)? *(At large n, n² grows much faster — a slow computer with merge sort beats a fast computer with insertion sort)*
3. What does it mean for a problem to be in NP? *(Given a proposed solution, you can verify it's correct in polynomial time)*
4. What's the difference between NP-hard and NP-complete? *(NP-complete = NP-hard AND in NP. NP-hard might not even be verifiable quickly)*
5. Why is $\binom{n}{2} = n(n-1)/2$? *(First item pairs with n-1 others, second with n-2, etc., divide by 2 because each pair counted twice)*
6. How does self-attention relate to $\binom{n}{2}$? *(Every token attends to every other token — n(n-1)/2 pairs, hence O(n²))*
7. Why does deep learning work despite training being NP-hard? *(SGD doesn't find the global optimum — it finds a good-enough local minimum. Overparameterization helps create many good minima.)*
