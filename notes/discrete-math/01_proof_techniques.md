# Proof Techniques

*The 4 core methods for proving mathematical statements.*

---

## Why proofs matter for CS and ML

CLRS uses proofs on nearly every page — proving algorithm correctness, proving runtime bounds, proving that greedy choices are optimal. Without these 4 techniques, CLRS is unreadable. They also show up in ML theory: convergence proofs for SGD, proving that baselines don't add bias to policy gradients, proving the Bellman equation has a unique fixed point.

---

## The 4 techniques at a glance

| Technique | Strategy | When to reach for it |
|-----------|----------|---------------------|
| Direct proof | Assume $P$, chain steps to $Q$ | Straightforward "if $P$ then $Q$" |
| Contradiction | Assume the opposite, derive impossibility | Proving something doesn't exist or can't happen |
| Contrapositive | Prove "if not $Q$ then not $P$" instead | Forward direction is messy, reverse is clean |
| Induction | Base case + "if $n$ works, $n+1$ works" | Anything with "for all $n$" |

---

## 1. Direct proof

**Pattern**: assume $P$ is true, use definitions and algebra to arrive at $Q$.

**Example**: prove that if $n$ is even, then $n^2$ is even.

$$n \text{ is even} \implies n = 2k \text{ for some integer } k$$

$$n^2 = (2k)^2 = 4k^2 = 2(2k^2)$$

$2(2k^2)$ is divisible by 2, so $n^2$ is even. $\blacksquare$

**The move**: rewrite things using definitions until the conclusion falls out.

### Where this shows up

- **CLRS**: proving loop invariants maintain a property at each step
- **ML**: proving that the expected value of an unbiased estimator equals the true parameter — just expand the definition and simplify

---

## 2. Proof by contradiction

**Pattern**: assume the OPPOSITE of what you want to prove, show it leads to something impossible.

**Example**: prove that $\sqrt{2}$ is irrational.

Assume $\sqrt{2}$ IS rational. Then $\sqrt{2} = \frac{a}{b}$ where $a, b$ are integers with no common factors (reduced fraction).

$$2 = \frac{a^2}{b^2} \implies a^2 = 2b^2$$

So $a^2$ is even, which means $a$ is even. Write $a = 2c$:

$$a^2 = 4c^2 = 2b^2 \implies b^2 = 2c^2$$

So $b^2$ is even, which means $b$ is even.

But $a$ and $b$ are both even — contradicts our assumption that $\frac{a}{b}$ was reduced. $\blacksquare$

**The move**: "suppose for contradiction that..." then derive $X$ and $\neg X$.

### Where this shows up

- **CLRS**: proving lower bounds ("no comparison sort can do better than $O(n \log n)$" — assume one does, derive contradiction)
- **ML**: proving the impossibility theorem in fairness — assume calibration + equal FPR + equal FNR all hold, derive contradiction when base rates differ

---

## 3. Proof by contrapositive

**Pattern**: instead of "if $P$ then $Q$", prove the equivalent "if not $Q$ then not $P$".

These are logically identical:

$$P \implies Q \quad \equiv \quad \neg Q \implies \neg P$$

**Example**: prove that if $n^2$ is odd, then $n$ is odd.

Direct: hard to start from "$n^2$ is odd."

Contrapositive: prove "if $n$ is even, then $n^2$ is even."

$$n = 2k \implies n^2 = 4k^2 = 2(2k^2) \implies n^2 \text{ is even} \quad \blacksquare$$

**The move**: flip the direction when the forward path is awkward.

### Why this works — truth table

```
P       Q       P → Q       ¬Q → ¬P
──      ──      ─────       ────────
T       T       T           T
T       F       F           F
F       T       T           T
F       F       T           T
                ↑ same ↑    ↑ same ↑
```

The columns for $P \implies Q$ and $\neg Q \implies \neg P$ are identical. Proving one proves the other.

### Where this shows up

- **CLRS**: "if the algorithm outputs $X$, then the input had property $Y$" — easier to prove "if input lacks $Y$, algorithm can't output $X$"
- **ML**: "if the model converges, the learning rate satisfies condition $C$" — easier to prove "if learning rate violates $C$, model diverges"

---

## 4. Proof by induction

**Pattern**: two steps.

1. **Base case**: prove the statement for $n = 1$ (or $n = 0$)
2. **Inductive step**: assume it's true for $n$ (the "inductive hypothesis"), prove it for $n + 1$

This creates a domino chain: base case knocks over $n=1$, which knocks over $n=2$, which knocks over $n=3$, forever.

**Example**: prove that $1 + 2 + 3 + \ldots + n = \frac{n(n+1)}{2}$

**Base case** ($n = 1$):

$$1 = \frac{1(1+1)}{2} = \frac{2}{2} = 1 \quad \checkmark$$

**Inductive step**: assume true for $n$. Prove for $n + 1$:

$$1 + 2 + \ldots + n + (n+1) = \frac{n(n+1)}{2} + (n+1)$$

$$= \frac{n(n+1) + 2(n+1)}{2} = \frac{(n+1)(n+2)}{2} \quad \blacksquare$$

**The move**: use the assumption for $n$ to bootstrap the proof for $n+1$.

### The domino analogy

```
Base case proves n=1:     [1] falls

Inductive step:           if [n] falls → [n+1] falls

Chain reaction:           [1] → [2] → [3] → [4] → ... → [∞]
```

### Strong induction (variant)

Regular induction: assume true for $n$, prove $n+1$.
Strong induction: assume true for ALL values $\leq n$, prove $n+1$.

Same idea, but you get to use more dominoes:

```
Regular:    only [n] knocks over [n+1]
Strong:     [1], [2], ..., [n] ALL help knock over [n+1]
```

Strong induction is useful for recursive algorithms where $f(n)$ depends on $f(n/2)$, not just $f(n-1)$.

### Where this shows up

- **CLRS**: correctness of merge sort (strong induction — each half is sorted by inductive hypothesis), loop invariant proofs, recurrence relation solutions
- **ML**: proving that Value Iteration converges — show $\|V_{k+1} - V^*\| \leq \gamma \|V_k - V^*\|$, base case is $V_0$, inductive step shows error shrinks by $\gamma$ each iteration
- **Backpropagation**: the chain rule applied layer by layer is essentially an inductive argument — if gradients are correct for layers $1$ through $n$, they're correct for layer $n+1$

---

## Choosing the right technique

```
"Prove that if P then Q"
  └─ Can you chain P → ... → Q directly?
       YES → direct proof
       NO  → Is "not Q → not P" easier?
              YES → contrapositive
              NO  → Try contradiction

"Prove for all n ≥ 1"
  └─ Induction (almost always)

"Prove something is impossible / doesn't exist"
  └─ Contradiction (assume it exists, derive absurdity)
```

---

## Quick-fire self-test

1. Which technique assumes the opposite and derives absurdity? *(Contradiction)*
2. "If not $Q$ then not $P$" is the ______ of "if $P$ then $Q$"? *(Contrapositive)*
3. What are the two steps of induction? *(Base case + inductive step)*
4. Why does CLRS use induction so heavily? *(Algorithms are recursive/iterative — induction matches that structure)*
5. Why is the contrapositive logically equivalent to the original? *(Same truth table — both false only when $P$ is true and $Q$ is false)*
6. Strong induction differs from regular induction how? *(Assume true for all values $\leq n$, not just $n$)*
7. "No comparison sort beats $O(n \log n)$" — which proof technique? *(Contradiction — assume one does, derive impossibility)*
