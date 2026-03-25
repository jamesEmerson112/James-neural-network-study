# The Foundations, Pioneers & AI Winter (1847-1985)

> [Back to Timeline Hub](00_timeline.md) | [Next: The Resurrection](00b_timeline_1986-2017_resurrection_and_revolution.md)

## The Foundations (1931-1948)

| Year | What | Who | Key Idea |
|------|------|-----|----------|
| 1931 | **Incompleteness theorems** | Kurt Gödel | Math cannot prove all truths about itself — there are limits to formal systems |
| 1847 | **Boolean algebra** | George Boole ("Laws of Thought") | All logical reasoning = AND/OR/NOT. Provably correct but too rigid for how humans think |
| 1936 | **Turing machine** | Alan Turing | Defined "what is computable" — abstract machine that can compute anything computable. The goalpost for everything after |
| 1943 | **McCulloch-Pitts neuron** | McCulloch & Pitts (18-yr-old logician) | Fused neuroscience + Boolean logic + Turing: networks of binary neurons are Turing-complete. Brain = computer. But can't learn — weights set by hand |
| 1948 | **Information theory** | Claude Shannon (Bell Labs) | "A Mathematical Theory of Communication" — defined entropy, bits, and the math of information. Shannon, a 32-year-old who juggled on a unicycle through Bell Labs hallways, gave us the foundation for KL divergence, cross-entropy loss, and everything that makes Boltzmann machines and VAEs work. Without Shannon, there's no principled way to measure "how different two distributions are" |

## The Pioneers (1949-1962)

| Year | What | Who | Key Idea |
|------|------|-----|----------|
| 1949 | **Hebbian learning** | Donald Hebb | "Neurons that fire together wire together" — first learning principle, but no algorithm |
| 1950 | **Turing Test** | Turing | Flipped the question: not "can machines compute?" but "can machines think?" Kicked off AI as a field |
| 1950 | **Nash equilibrium** | John Nash (Princeton) | Proved every finite game has a point where no player gains by changing strategy alone — 27-page PhD thesis, age 22. Nash and Turing overlapped at Princeton; both brilliant, both later persecuted (Nash for schizophrenia, Turing for homosexuality). GANs (2014) are a Nash equilibrium problem: the generator and discriminator play a minimax game until neither can improve. Nobel Prize 1994 |
| 1951 | **SNARC** | Minsky & Edmonds | First physical neural net machine — 40 neurons, solved mazes. Trained on reward (reinforcement), not error |
| 1956 | **"Artificial Intelligence" coined** | Dartmouth Workshop | McCarthy, Minsky, Shannon, Rochester — the field gets a name |
| 1957 | **Perceptron** | Rosenblatt | First *learnable* neuron — weights adjust from data. Error measured AFTER threshold (binary: right/wrong). `w ← w + η(y - ŷ) · x` — the seed of all training |
| 1959 | **"Machine learning" coined** | Arthur Samuel (IBM) | Checkers program that learned from self-play (built since 1952). The 1959 paper is where the term "machine learning" first appears |
| 1959 | **Visual cortex mapping** | Hubel & Wiesel | Discovered receptive fields in cat visual cortex that detect edges/orientations (simple/complex cell taxonomy formalized 1962). Nobel Prize 1981. Directly inspired CNNs decades later |
| 1960 | **ADALINE** | Widrow & Hoff (Stanford) | ADAptive LInear NEuron — error measured BEFORE threshold (continuous: "how far off?"). Introduced LMS (Least Mean Squares) — first use of gradient descent in a neural network. Essentially an adaptive noise filter; lasting use in telephone echo cancellation. Ted Hoff later co-invented the Intel 4004 microprocessor |
| 1960 | **Backprop precursor** | Henry Kelley | First derivation of backpropagation-like gradient computation, but in control theory (optimal flight paths) — not applied to neural networks |
| 1960 | **MINOS I** | Stanford Research Institute (Rosen & Brain) | Neural network machine for pattern classification — classifying symbols on army maps. MINOS II (1963) scaled to 6,600 adjustable weights |
| 1962 | **Samuel's Checkers** | Arthur Samuel (IBM) | Checkers program beat a former Connecticut champion — one of the first demos of ML winning a real game. Started in 1950s on IBM 701; demoed on live TV in 1956. Thomas Watson Sr. predicted a 15-point IBM stock rise from the demo — and it happened. One of the first times AI was used to sell stock |
| 1962 | **MADALINE** | Widrow & Hoff | Many ADALINEs — one of the first multi-layer neural networks. Used MADALINE Rule I: each ADALINE trained independently with LMS, but hidden layers kept fixed (no way to train them yet). The missing piece was backprop — 24 years away |
| 1962 | ***Principles of Neurodynamics*** | Rosenblatt | 616-page treatise — the theoretical framework for perceptrons. **Coined the term "back-propagating errors"** but didn't know how to implement it. Named the solution 24 years before Rumelhart/Hinton/Williams made it work |

## The AI Winter (1969-1985)

| Year | What | Who | Key Idea |
|------|------|-----|----------|
| 1965 | **GMDH** | Ivakhnenko & Lapa | Group Method of Data Handling — sometimes called the first deep learning. Trained networks layer by layer. By 1971 had 8-layer networks. Largely forgotten in the West |
| 1966 | **ELIZA** | Weizenbaum (MIT) | First chatbot — pattern-matched text to simulate a therapist. No learning at all, but fooled people. Early proof that "seeming intelligent" ≠ "being intelligent" |
| 1969 | **Apollo 11 Moon landing** | NASA | Humanity lands on the Moon — proof that massive government R&D investment works. But funding pivots to space; AI loses priority. The same year neural nets get killed below |
| 1969 | **Perceptron limitations** | Minsky & Papert | Proved single-layer perceptrons can't learn XOR. Triggered ~15 years of reduced funding. McCulloch & Pitts both died the same year |
| 1971 | Rosenblatt dies | — | Drowned in sailing accident on his 43rd birthday. Never saw backprop vindicate his ideas |
| 1974 | **Backprop for neural nets** | Paul Werbos (Harvard) | PhD thesis: first to explicitly apply the chain rule backward through layers to train neural networks. The right answer, 12 years too early — almost nobody read it |

---

> See also: [Visual Map of this era](00d_timeline_pre1962_visual_map.md) | [Back to Timeline Hub](00_timeline.md)
