# Turing — The Man Who Invented Computation

## Who Was He?

**Alan Turing** (1912–1954), born in Maida Vale, London, England.

A quiet, awkward, brilliant man:

- **Stammered** when he talked
- **Chained his coffee mug** to a radiator so nobody would steal it
- Was a serious **long-distance runner** — nearly Olympic-qualifying times
- Graduated from **King's College, Cambridge**, earned his doctorate at **Princeton** (1938)
- Colleagues described him as eccentric, deeply kind, and operating on a different plane

---

## The Turing Machine (1936) — What Computation IS

At age 23, Turing asked the most important question in the history of computer science:

> *"What does it mean for something to be computable?"*

His answer was the **Turing Machine** — a theoretical device so simple it seems like a toy:

```
    THE TURING MACHINE
    ═══════════════════════════════════════

    ... [ ] [ ] [1] [0] [1] [ ] [ ] ...     ← Infinite tape of symbols
                  ▲
                  │
            ┌─────────┐
            │  HEAD    │     ← Reads/writes one symbol at a time
            │  State: A│     ← Has a current "state"
            └─────────┘
                  │
            ┌─────────┐
            │  RULES  │     ← A table that says:
            │         │        "If in state A reading 1,
            │         │         write 0, move right,
            │         │         go to state B"
            └─────────┘

    That's it. A tape, a head, and rules.
```

### Why Does This Matter?

Turing proved that this absurdly simple device can compute **anything that is computable**. Period. This is the **Church-Turing thesis**:

- Every algorithm ever written
- Every program on every computer
- Every neural network running on every GPU
- Every app on your phone

They're all just **fancy Turing machines**. Nothing more powerful exists (that we know of). If a Turing machine can't compute it, nothing can.

### The Halting Problem

In the same paper, Turing also proved what computation **can't** do. He showed there is no general algorithm that can determine whether an arbitrary program will eventually stop or run forever. This was the **Halting Problem** — the first proof that some problems are fundamentally unsolvable by any computer.

---

## The Direct Lineage to Deep Learning

McCulloch and Pitts read Turing's 1936 paper. **Seven years later** (1943), they proved that networks of neurons are equivalent to Turing machines. That's the direct pipeline:

```
    1936  TURING MACHINE        →  "Here's what computation IS"
                                            │
                                            ▼
    1943  McCULLOCH-PITTS       →  "The brain does this too"
                                            │
                                            ▼
    1945  VON NEUMANN           →  "Let me BUILD one"
                                            │
                                            ▼
    1958  ROSENBLATT            →  "Let me make it LEARN"
          (Perceptron)                      │
                                            ▼
    1986  RUMELHART et al.      →  "Let me make DEEP ones learn"
          (Backpropagation)                 │
                                            ▼
    2012+ DEEP LEARNING ERA     →  "Let me scale it up"
```

Without Turing's formal definition of computation, McCulloch and Pitts would have had nothing to prove their neurons were equivalent **to**. He built the measuring stick.

---

## Bletchley Park — Winning World War II

### The Enigma Problem

The Nazis encrypted all their military communications with the **Enigma machine** — an electromechanical device with approximately **158 million million million** (158 × 10¹⁸) possible settings. They changed the settings **every single day**.

The German military believed it was unbreakable. Mathematically, they had reason to — brute-force searching that space was impossible.

### The Solution

Turing was recruited to **Bletchley Park**, Britain's secret codebreaking center, in 1939 — the day after war was declared. He led **Hut 8**, responsible for cracking German naval Enigma.

His approach was pure computational thinking:

```
    TURING'S INSIGHT
    ═══════════════════════════════════════

    You don't need to check ALL possible settings.

    If you KNOW something about the plaintext
    (e.g., weather reports always start with "WETTER"
     or messages end with "HEIL HITLER"),

    you can use those "cribs" to eliminate
    vast swaths of the search space.
```

He designed the **Bombe** — an electromechanical machine (building on earlier Polish work by Marian Rejewski) that could test thousands of Enigma settings rapidly, guided by these logical constraints.

### The Result

The Allies could read Nazi messages in near real-time:

- They knew where the **U-boats** were (saving the Atlantic convoys)
- They knew **troop movements** before they happened
- They knew **battle plans** in advance

Historians estimate Turing's work **shortened the war by 2–4 years** and saved **millions of lives**.

It was classified until the 1970s. For decades, nobody outside intelligence circles knew what he'd done.

---

## The Turing Test (1950)

In 1950, Turing published "Computing Machinery and Intelligence" — opening with the line:

> *"Can machines think?"*

He proposed the **Imitation Game** (now called the Turing Test): if a human judge can't tell whether they're talking to a human or a machine through a text conversation, the machine should be considered intelligent.

This paper also anticipated nearly every objection that would be raised against AI for the next 75 years — and offered counterarguments to each one.

```
    THE TURING TEST
    ═══════════════════════════════════════

    ┌──────────┐         ┌──────────┐
    │  HUMAN   │         │ MACHINE  │
    │  (or AI) │         │  (or AI) │
    └────┬─────┘         └────┬─────┘
         │                    │
         │    text only       │
         ▼                    ▼
    ┌──────────────────────────────┐
    │         JUDGE (human)        │
    │                              │
    │   "Which one is the human?"  │
    └──────────────────────────────┘

    If the judge can't reliably tell →
    the machine passes the test.
```

---

## Conviction and Chemical Castration

### What Happened

In **January 1952**, Turing's house in Wilmslow was broken into. He reported it to the police. During the investigation, he matter-of-factly mentioned he was in a sexual relationship with a man — **Arnold Murray**, age 19.

**Homosexuality was a criminal offense in Britain** under Section 11 of the Criminal Law Amendment Act 1885 — the same law used to convict Oscar Wilde in 1895.

Turing was charged with **"gross indecency."**

### The "Choice"

He was given two options:

```
    OPTION A:  Prison

    OPTION B:  "Chemical castration"
               → One year of injections of synthetic estrogen
                 (diethylstilbestrol / DES)
               → Probation instead of prison
```

He chose Option B so he could continue his work.

### The Effects

For a year, the man who helped save Britain was injected with hormones by Britain:

- **Rendered impotent**
- **Grew breast tissue** (gynecomastia)
- Experienced mood changes and depression
- His **security clearance was revoked** — he was permanently barred from all government cryptographic work

### The Relationship

The relationship with Murray was brief — only a few weeks. At trial, Murray's own lawyer turned against Turing, arguing that if Murray "had not met Turing he would not have indulged in that practice." Murray received a conditional discharge. They never saw each other again. Turing traveled to Norway that summer, seeking more tolerant societies. Murray lived until 1990.

---

## His Death

On **June 7, 1954**, Turing's housekeeper found him dead in bed at his home in Wilmslow. He was **41 years old**.

Cause of death: **cyanide poisoning**. A half-eaten apple was found next to him.

The coroner ruled it suicide. But there's genuine ambiguity:

- Turing kept chemistry equipment at home and regularly worked with cyanide
- His mother always maintained it was **accidental inhalation**
- The apple was **never tested** for cyanide
- Friends said he seemed in good spirits in his final days
- He had made plans for the coming week

We will never know for certain.

*(A persistent but unconfirmed legend says the Apple Inc. logo — an apple with a bite taken out — was inspired by Turing. Apple's designer Rob Janoff has said it wasn't intentional, but "it's a wonderful urban legend.")*

---

## The Pardon — 59 Years Too Late

| Year | Event |
|------|-------|
| 1954 | Turing dies at 41 |
| 1970s | Bletchley Park work finally declassified |
| 2009 | Prime Minister Gordon Brown issues formal public apology |
| 2013 | Queen Elizabeth II grants **posthumous royal pardon** |
| 2014 | *The Imitation Game* film (Benedict Cumberbatch) |
| 2017 | **"Turing's Law"** retroactively pardons ~49,000 men convicted under same laws |
| 2021 | Turing appears on the **British £50 note** |

---

## The Timeline of Cruelty

```
    1936 ─── Invents the concept of computation itself
                │
    1939 ─── Recruited to crack Enigma — the day after war begins
                │
    1940-45 ── Saves millions of lives. Shortens the war by years.
                │
    1945 ─── Work is classified TOP SECRET. Nobody is told.
                │
    1950 ─── Proposes the Turing Test. Founds artificial intelligence.
                │
    1952 ─── Convicted of being gay. Given the "choice":
                prison or chemical castration.
                │
    1952-53 ── Injected with synthetic estrogen for one year.
                Security clearance revoked. Barred from the work
                that defined his life.
                │
    1954 ─── Dead at 41.
                │
    2013 ─── Pardoned by the Queen. 59 years late.
```

---

## Why He Matters to This Course

Every time you train a neural network, you are using a system that traces its theoretical foundation directly back to Turing:

1. **Turing Machine (1936)** — defined computation
2. **McCulloch-Pitts (1943)** — proved neurons compute like Turing machines
3. **Von Neumann (1945)** — built the hardware to run it
4. **Everything since** — is refinement and scale

Turing didn't just contribute to the history of computing. He **is** the history of computing. The entire field is a footnote to his 1936 paper.
