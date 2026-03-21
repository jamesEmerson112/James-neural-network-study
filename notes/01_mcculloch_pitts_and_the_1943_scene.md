# McCulloch-Pitts Neuron & The 1943 Scene

## The People

### Warren McCulloch (1898–1969)
- Neurophysiologist/philosopher at University of Illinois, Chicago
- Obsessed with one question: *how does the mind work, physically?*
- Not rich — a working academic

### Walter Pitts (1923–1969)
- A tragic genius. **Homeless teenager** who ran away from an abusive home in Detroit
- Wandered into a library, read Russell & Whitehead's *Principia Mathematica*, found errors in it, and wrote to Russell
- Showed up at UChicago and started hanging around academics — McCulloch took him in
- **No formal degree. Ever.**
- Co-authored the foundational neural network paper at age **20**

---

## The Paper (1943)

**"A Logical Calculus of the Ideas Immanent in Nervous Activity"**

### Core Argument
Neurons in the brain can be modeled as simple logical units — and any computable function can be built from networks of these units.

### The Logic
1. Neurons fire or they don't (binary — on/off)
2. A neuron fires when it gets enough input from other neurons
3. This looks exactly like a **logic gate** (AND, OR, NOT)
4. Logic gates can compute anything (Turing completeness)
5. Therefore: **the brain is a computing machine**

### Why It Matters
- First paper to bridge **neuroscience, mathematical logic, and computation** into one framework
- Proved neural networks are **Turing complete** — as powerful as any computer that could ever exist
- Direct origin of all modern neural networks

---

## The Turing Machine Connection (1936)

Alan Turing described the Turing Machine 7 years earlier — the theoretical foundation for ALL computation.

```
                         HEAD
                          |
                          v
                       [READ/WRITE]
                          |
    ... | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | ...
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              INFINITE TAPE (memory)

    RULES (State Table):
    ┌──────────┬──────┬───────┬──────┬────────────┐
    │  State   │ Read │ Write │ Move │ Next State  │
    ├──────────┼──────┼───────┼──────┼────────────┤
    │  START   │  0   │   1   │  →   │  STATE_A   │
    │  START   │  1   │   0   │  ←   │  STATE_B   │
    │  STATE_A │  0   │   0   │  →   │  HALT      │
    │  STATE_B │  1   │   1   │  ←   │  START     │
    └──────────┴──────┴───────┴──────┴────────────┘
```

A tape, a head that reads/writes, and rules. Anything that can be computed, can be computed by this machine (Church-Turing thesis).

### Equivalence

```
    TURING MACHINE              McCULLOCH-PITTS NETWORK
    ┌───────────┐               ┌───────────────────┐
    │  Tape +   │               │  Network of       │
    │  Head +   │  equivalent   │  binary neurons    │
    │  Rules    │ ============= │  with AND/OR/NOT   │
    │           │   power       │  connections       │
    └───────────┘               └───────────────────┘
```

---

## The 1943 Snapshot — Who Was Alive and Doing What

### Princeton / Institute for Advanced Study
- **Albert Einstein** (64) — working on unified field theory
- **John von Neumann** (40) — splitting time between IAS and the bomb; arguably the smartest human alive
- **Kurt Gödel** (37) — Einstein's daily walking buddy; had already proved mathematics is incomplete at age 25 (see section below)
- **Robert Oppenheimer** (39) — just left for Los Alamos

### Los Alamos (Manhattan Project)
- **Oppenheimer** (39) — leading the project
- **von Neumann** (40) — designing the implosion lens
- **Richard Feynman** (25) — doing grunt computation, would later win the Nobel Prize
- **Turing** — consulted, but mainly at Bletchley Park

### Chicago / MIT (Cybernetics Circle)
- **McCulloch** (45) & **Pitts** (20) — writing their neuron paper
- **Norbert Wiener** (49) — developing cybernetics at MIT
- **Claude Shannon** (27) — information theory at Bell Labs; had invented it at age 21

### Bletchley Park, England
- **Alan Turing** (31) — breaking Nazi codes with early computers

### Context
This was WWII. The war created massive government interest in computation, signals, and control systems. It was arguably the most concentrated explosion of genius in human history, and **war was the catalyst**.

Von Neumann later read McCulloch-Pitts' paper (alerted to it by Wiener) and it directly influenced his design of modern computer architecture (the von Neumann architecture). His EDVAC report — the blueprint for the stored-program computer — had only **one citation**: the McCulloch-Pitts paper.

---

## Kurt Gödel — The Man Who Broke Mathematics

**Kurt Gödel** (1906–1978) keeps appearing in this story because he was at the center of the Princeton intellectual universe — and because his work set the theoretical limits that frame everything from Turing machines to neural networks.

### What He Proved

At age **25** (1931), Gödel published his **Incompleteness Theorems**, which shattered the foundations of mathematics:

> **First Theorem:** In any consistent mathematical system powerful enough to describe basic arithmetic, there are true statements that **cannot be proved** within that system.

> **Second Theorem:** Such a system **cannot prove its own consistency**.

In the early 1900s, mathematicians (led by **David Hilbert**) believed they could build a complete, consistent set of axioms for ALL of mathematics — a perfect logical foundation. Gödel proved this is **impossible**. Mathematics will always contain truths it cannot prove.

Einstein called him *"the greatest logician since Aristotle."* He was not joking.

### The Connection to Turing, Computing, and Neural Networks

Gödel's work directly influenced the foundations of computer science:

```
    Gödel (1931)                    Turing (1936)                   McCulloch-Pitts (1943)
    ┌──────────────────┐            ┌──────────────────┐            ┌──────────────────┐
    │ Math cannot       │            │ Computers cannot  │            │ Neural networks   │
    │ prove all truths  │  ──────→   │ solve all problems│  ──────→   │ are Turing        │
    │ about itself      │ inspired   │ (Halting Problem) │ equivalent │ complete — same    │
    │                   │            │                   │            │ power, same limits │
    └──────────────────┘            └──────────────────┘            └──────────────────┘
```

- **Turing (1936)** — the Halting Problem (you can't write a program that determines whether all programs will halt) is essentially Gödel's theorem applied to computation
- **Von Neumann** — immediately recognized Gödel's work as revolutionary and helped promote it
- **McCulloch-Pitts (1943)** — proved neural networks are Turing complete, meaning they inherit both Turing's power AND Gödel's limits

### The Friendship with Einstein

Both were at the **Institute for Advanced Study** in Princeton. Starting in the 1940s, they walked together to and from the institute **nearly every day** for years, discussing math, physics, philosophy, and life. Their conversations were a mystery to other Institute members.

Einstein said in his later years: *"My own work no longer means much. I come to the Institute mainly to have the privilege of walking home with Gödel."*

They shared a deep belief that mathematical and physical truth is **objective** — something discovered, not invented.

### The Tragic End

Gödel suffered from severe paranoia — specifically, an obsessive fear of **being poisoned**. He would only eat food tasted first by his wife Adele. In 1977, Adele was hospitalized for six months. Gödel **refused to eat**. He died on January 14, 1978, weighing **65 pounds**. The death certificate read: "malnutrition and inanition caused by personality disturbance."

The greatest logician since Aristotle, starved to death because his proof system for food safety had a single point of failure.

### The Princeton Density of Genius

In the late 1940s, these people were all at Princeton IAS — having lunch, walking, arguing:
- **Einstein** (70) — unified field theory
- **Gödel** (43) — their daily walks
- **Von Neumann** (46) — computing + bomb
- **Oppenheimer** (45) — directing IAS, under FBI surveillance

This concentration of genius at a single institution may never be matched.

---

## The Tragedies

- **Walter Pitts** — Wiener (age 48) took Pitts (age 20) under his wing in 1943, becoming a father figure to the kid who'd run away from an abusive home at 15. Wiener recruited McCulloch to MIT in 1951, reuniting the group. But in 1952, Wiener's wife Margaret — who despised McCulloch and thought his circle was "too free" (wild parties, skinny-dipping at McCulloch's farm) — told Wiener that the "boys" had **seduced** their daughter Barbara (~age 25, an adult) when she'd stayed at McCulloch's house in Chicago. Oliver Selfridge (one of the accused) called this "absolutely false." Barbara herself was later interviewed by biographers and agreed to let the story be published. The biographers' conclusion (*Dark Hero of the Information Age*, Conway & Siegelman, 2005): it was a stratagem by Margaret to destroy a group she hated. Though some historians note we only have the accused parties' version. Either way — Wiener (age 57) cut off all contact with everyone in the group without discussion. Pitts (age 29) was shattered — a father figure had abandoned him for the second time. He started drinking, withdrew from everyone. In 1959 (age 36) he burned his unpublished doctoral dissertation and all research notes; refused to sign the paperwork when offered his PhD. Died of bleeding esophageal varices (cirrhosis) in a boarding house in Cambridge, May 14, 1969, age 46. No degree. No recognition in his lifetime. McCulloch died four months later.
- **Alan Turing** — chemically castrated by the British government for being gay in 1952. Died June 7, 1954, age 41, from cyanide poisoning. An inquest ruled suicide, but it remains debated — his mother believed it was accidental (careless storage of lab chemicals), and his biographer Hodges theorized Turing made it look accidental to protect her. Posthumously pardoned in 2013; became the face of the £50 note in 2021.

---

## The Lineage to Deep Learning

```
1943  McCulloch-Pitts    →  Neurons are logic gates
1958  Rosenblatt         →  What if we LEARN the weights? (Perceptron)
1986  Rumelhart/Hinton/Williams →  What if we stack layers and backpropagate? (MLPs)
2012+ Deep Learning      →  What if we go REALLY deep?
```

Every neural network traces back to this 1943 paper.
