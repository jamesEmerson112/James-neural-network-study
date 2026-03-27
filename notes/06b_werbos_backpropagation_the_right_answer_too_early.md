# Paul Werbos & Backpropagation — The Right Answer, Twelve Years Too Early

> **The man who solved the biggest problem in neural networks published during the deepest part of the AI winter, when nobody was paying attention to neural networks.**

---

## Part 1: The Person

### Paul John Werbos (Born September 4, 1947)

Born in the suburbs of Philadelphia, Pennsylvania. His father had German engineering heritage; his mother, Margaret M. Smith, was Irish Catholic. By age eight, Paul had adopted atheism after questioning religious teachings against scientific authorities. At nine, he transferred from Catholic school to Chestnut Hill Academy, where he began teaching himself algebra in secret from one of his mother's old textbooks.

He was, by any measure, a prodigy — but the quiet, self-directed kind. He finished a full calculus course by the end of **sixth grade**. In seventh grade, his school sent him to the **University of Pennsylvania** to take the junior honors calculus course (complex variables and such). While still in high school, he transferred to **Lawrenceville School** in New Jersey specifically to access Princeton resources. There, he took the bus from his high school to Princeton University to sit in **Alonzo Church's graduate seminar in logic** — and received graduate course credit before he graduated from high school.

Think about that: the same Alonzo Church who supervised Alan Turing's PhD at Princeton in the 1930s, whose lambda calculus is the theoretical foundation of functional programming, was teaching a high school kid in the early 1960s.

### Education: Four Degrees Across Two Continents

Werbos entered **Harvard University** in 1964 as an undergraduate, but not in math or computer science — he chose **economics**. He was interested in distributed optimization and decision-making systems with social implications. He wanted to understand how complex systems like nations and economies work.

He went on to collect four degrees from **Harvard** and the **London School of Economics**:

1. **Economics** (Harvard)
2. **International political systems**, emphasizing European economic institutions (London School of Economics)
3. **Applied mathematics**, with a major in quantum physics and a minor in decision and control (Harvard)
4. **PhD in applied mathematics** — the interdisciplinary thesis that contained backpropagation (Harvard, 1974)

This was not a man who stayed in one lane. He was a polymath who read Freud, studied quantum physics, built econometric models, and was equally comfortable with political theory and differential equations.

### The Personality

Werbos was not the typical academic. He lived in **Roxbury**, one of Boston's poorest neighborhoods, during graduate school — subsisting on minimal food while facing computing resource constraints that would later shape his emphasis on efficient algorithms. The financial hardship was real. He wasn't at Harvard because of family wealth; he was there because he could solve problems nobody else could.

He had an intellectual confidence that bordered on audacity. When he believed he had the right answer, he would walk up to the biggest name in the room and say so — as we'll see with his approach to Marvin Minsky. But he also had the mathematical rigor to back it up. He studied John von Neumann's theoretical work and drew from an absurdly wide range of influences: control theory, Freudian psychology, dynamic programming, political science.

---

## Part 2: The Inspiration & Motivation

### The Problem: How Do You Train a Multi-Layer Network?

By the late 1960s, neural networks were in crisis. Minsky and Papert's *Perceptrons* (1969) had proved that single-layer perceptrons couldn't learn XOR or any non-linearly separable function. Everyone knew, in theory, that adding hidden layers could fix this. **But nobody knew how to train them.**

The problem was called **credit assignment**: if a network has multiple layers and it makes a mistake, which weights in which layers are responsible? How do you distribute blame backward through the network? A single-layer perceptron can just compare output to target and adjust — but with hidden layers, the hidden neurons have no "target" to compare against.

This was the roadblock. This was why Minsky could credibly claim neural nets were a dead end. Not because multi-layer networks were impossible in principle, but because nobody had a practical algorithm to train them.

### How Werbos Got There: From Minsky to Freud to the Chain Rule

The intellectual path that led Werbos to backpropagation is one of the strangest in the history of computer science.

**Step 1: Minsky's inspiration.** Werbos read Minsky's chapter in *Computers and Thought*, where Minsky discussed reinforcement learning as a path to building human-like intelligence. This fired Werbos up. He believed the approach could work — you could build intelligent systems that learned from experience. He went to meet Minsky in person.

**Step 2: Minsky's rejection.** When Werbos met Minsky, the encounter was deflating. Minsky told him: *"Nah, that idea never worked. I couldn't figure out how to do it."* The attitude was essentially: if Marvin Minsky couldn't solve it, it must be impossible. Werbos disagreed. He believed he could.

**Step 3: Freud's "psychic energy."** This is the truly bizarre part. Werbos was reading Sigmund Freud's *Project for a Scientific Psychology* (1895), in which Freud — who trained as a neurologist before inventing psychoanalysis — described how "psychic energy" flows through neural pathways. Freud argued that selective learning could only happen if the presynaptic neuron was as influenced as the postsynaptic neuron during excitation — that signals needed to flow backward through the system.

Werbos realized that Freud's "backward flow of psychic energy" could be formalized mathematically as the **chain rule applied in reverse** through a computational graph. Instead of mystical energy flowing backward through neurons, you could compute **derivatives flowing backward through layers** — telling each weight exactly how much it contributed to the error.

**Step 4: The chain law for ordered derivatives.** Werbos formalized this into what he called "the chain law for ordered derivatives" — a rigorous general theorem that later became known as the **reverse mode of automatic differentiation**, the **adjoint method**, or simply **backpropagation**. The core insight: you can compute the gradient of a loss function with respect to every parameter in a network by making one forward pass and one backward pass. The backward pass chains partial derivatives from output to input using the chain rule.

### The Intellectual Climate: A Field Frozen Solid

It's 1972-1974. The timing could not have been worse.

- Minsky and Papert's *Perceptrons* (1969) had devastated neural network research
- Government funding had evaporated — DARPA and other agencies redirected money to symbolic AI
- The leading AI labs (MIT, Stanford, Carnegie Mellon) were all-in on rule-based, symbolic approaches
- Rosenblatt, the perceptron's champion, had drowned in 1971 — gone at 43
- McCulloch and Pitts, the founders of computational neuroscience, had both died in 1969
- The very phrase "neural network" had become toxic in grant applications

Werbos was solving a problem that the establishment had decided didn't need solving — because they'd decided the whole approach was wrong.

---

## Part 3: The Creation Story

### The Thesis: "Beyond Regression" (Harvard, 1974)

The full title was **"Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences"**, submitted to the Committee on Applied Mathematics at Harvard University in August 1974.

His advisor was **Karl Wolfgang Deutsch** (1912-1992) — not a mathematician, not a computer scientist, but a **political scientist**. Deutsch was a Czech-born Harvard professor famous for his quantitative approach to political science. His 1953 book *Nationalism and Social Communication* reconceptualized nations as outcomes of communication networks rather than primordial ethnic entities. He used cybernetics — the science of feedback and control — to study human relationships and governance.

### What the Thesis Was Actually About

Here's the crucial detail that explains why nobody in AI read it: **the thesis was not about neural networks.** It was about prediction in the behavioral sciences.

The work began as an attempt to apply classical statistical and econometric techniques to the **Deutsch-Solow model of nationalism** — Deutsch's own framework for predicting political mobilization and national assimilation. Werbos's thesis committee agreed the work would qualify for a PhD if he could demonstrate that using the reverse differentiation method would enable more sophisticated time-series prediction methods, leading to the first successful implementation of Karl Deutsch's model of nationalism and social communications.

Backpropagation was a **tool** buried inside a political science thesis. The algorithm that would eventually train every neural network on Earth was presented as a means of doing better regression analysis for predicting nationalism.

### The Faculty's Reaction

When Werbos presented his 1972 thesis proposal to the Harvard faculty — explaining backpropagation and how it fit into a larger framework for intelligent systems — **they didn't believe it at first**. The faculty asked him to focus his dissertation on a smaller, more comprehensible portion of his broader framework. They wanted the prediction angle, not the grand vision of artificial intelligence.

So backpropagation got compressed into one part of a thesis that was framed around behavioral science prediction. The revolutionary algorithm was there, fully derived, mathematically rigorous — but buried in a context that no AI researcher would naturally encounter.

### Why Nobody Read It

Multiple factors conspired:

1. **Wrong field** — The thesis was in applied mathematics / behavioral science, not computer science or AI. It sat in a Harvard library, not in the proceedings of an AI conference.

2. **Wrong era** — Neural networks were dead. Nobody was looking for ways to train multi-layer networks because the establishment had declared them a dead end.

3. **No journal publication** — The thesis was not immediately published as a journal paper in a venue that AI or machine learning researchers would read. It was an unpublished doctoral dissertation.

4. **Wrong framing** — The title says "Beyond Regression" and "Behavioral Sciences." Nothing signals "this is the solution to the credit assignment problem in neural networks."

5. **The advisor gap** — Karl Deutsch was brilliant but was a political scientist. He had no connections to the AI or neural network research communities. If Werbos had been advised by someone in that world, the work might have been circulated through very different channels.

### The Minsky Encounter

At some point, Werbos approached Minsky directly. In his own words (from a later interview):

> *"Marvin, you've got this great book, but the thing is, the problem can be solved. Here's how to solve it. Why don't we become co-authors so that you don't be embarrassed when it comes out?"*

Minsky declined. The collaboration never happened.

Think about the audacity — and the tragedy — of this moment. A graduate student walks up to the man who had just killed neural network research and says: "I have the fix." And Minsky, the most powerful figure in AI, the man who had declared the problem unsolvable, says no.

Werbos later explained Minsky's mindset: *"If I can't do it, it must be impossible."* Minsky had already committed to symbolic AI. He had built his career and his lab on the premise that neural networks were a dead end. Acknowledging Werbos's solution would have meant admitting his own book was misleading.

---

## Part 4: The Aftermath — Life After the Invention

### Immediate Career: Not Academia

After his PhD, Werbos did not become a famous professor riding the wave of his discovery. There was no wave.

```
1973-1975  MIT postdoctoral researcher
1975-1978  University of Maryland, Assistant Professor
1978-1979  U.S. Census Bureau analyst
1979-1989  Energy Information Administration (Dept. of Energy)
              — lead analyst for long-term energy models
1988-2015  National Science Foundation, Program Director
              — led Adaptive and Intelligent Systems program
2015-now   Retired from NSF; Adjunct Professor, Missouri S&T
```

Look at that trajectory. The man who invented backpropagation spent the decade after his PhD working at the Census Bureau and the Department of Energy, building econometric models for electricity and natural gas forecasting. Not because the work wasn't important — he was applying his own algorithms to real-world prediction problems — but because the academic AI world had no interest in neural networks.

### The 1982 Paper: Trying Again

Werbos didn't give up on getting the word out. In 1982, he published **"Applications of Advances in Nonlinear Sensitivity Analysis"** in the proceedings of the 10th IFIP Conference (presented in New York, August-September 1981). This paper made the neural network application of backpropagation more explicit than the 1974 thesis had. According to Juergen Schmidhuber's analysis, this was "the first neural network-specific application of efficient backpropagation."

But again, the timing was off. It was 1982. The AI winter was still deep. The paper appeared in a conference proceedings volume on system modeling and optimization — not exactly the front page of AI research.

### 1986: Somebody Else Makes It Famous

Then came 1986. **David Rumelhart, Geoffrey Hinton, and Ronald Williams** published **"Learning representations by back-propagating errors"** in *Nature* (Volume 323, pages 533-536, October 9, 1986). The paper demonstrated experimentally that backpropagation could train multi-layer networks to develop useful internal representations in hidden layers.

**Did they know about Werbos?** The evidence says **no**. Their 1985/1986 papers do not cite Werbos's 1974 thesis or his 1982 paper. Rumelhart himself later stated he had no idea that Werbos had done work on backpropagation. The prior work was so obscure that the rediscovery was genuinely independent.

This is one of the great parallel inventions in the history of science. The same fundamental idea — apply the chain rule backward through a computational graph to compute gradients — was discovered or formulated by at least four independent groups across three decades:

- **Henry Kelley (1960)** — continuous-time gradient computation for optimal control
- **Seppo Linnainmaa (1970)** — reverse mode automatic differentiation (general algorithm, not applied to neural networks)
- **Paul Werbos (1974)** — first explicit application to neural networks
- **Rumelhart, Hinton & Williams (1986)** — experimental demonstration and popularization

### Did Werbos Get Credit?

Eventually, yes — but slowly and incompletely.

**Awards and recognition:**
- **1995** — IEEE Neural Network Pioneer Award "for the discovery of backpropagation and other basic neural network learning frameworks such as Adaptive Dynamic Programming"
- **2011** — Donald O. Hebb Award from the International Neural Network Society (INNS)
- **2022** — IEEE Frank Rosenblatt Award — the organization's top award for computational intelligence, named after the man whose work Werbos had vindicated

He was also elected a Fellow of IEEE and INNS, and served as one of the three original two-year Presidents of the International Neural Network Society in the early 1990s.

**Was he bitter?** The record suggests something more nuanced than bitterness. Werbos has described the real history as *"like a soap opera"* that *"you wouldn't believe."* In interviews, he comes across as someone who knows he got there first, who has documented his priority claim carefully (publishing *The Roots of Backpropagation* through Wiley in 1994 specifically to make his original thesis accessible), but who also understands why it happened the way it did.

He didn't become a billionaire from backpropagation. He didn't become a celebrity scientist like Hinton. He spent 27 years as a government bureaucrat at the NSF — a very good government bureaucrat who wielded his program directorship to shape the field, but not the kind of career that makes headlines.

### The NSF Years: Quiet Power

Here's where Werbos's story takes an ironic turn. From his desk at the National Science Foundation (1988-2015), he became one of the most influential funders of neural network research in America. He ran the Adaptive and Intelligent Systems program and co-directed the Learning and Intelligent Systems initiative, allocating approximately **$20 million annually** for AI and machine learning research.

In 2008, through a program called **COPN** (Cognitive Optimization and Prediction), Werbos funded **Andrew Ng** and **Yann LeCun** to test neural networks on crucial benchmark challenges. People inside NSF fought him on it. Someone threatened him with a lawsuit, arguing the work was "not sufficiently innovative" because it used "old algorithms on problems people have studied before." Werbos pushed back, found a procedural exception, and funded the grants anyway.

After Ng and LeCun demonstrated to Google that neural networks could outperform classical methods on these benchmarks, Google launched its neural network initiatives — and the modern deep learning revolution began.

The man who invented backpropagation in 1974 was also the man who funded the researchers who proved it could change the world in 2008. That's not a footnote; that's a full narrative arc.

### Did He Benefit Financially?

There is no indication that Werbos earned significant money from backpropagation itself. There were no patents (it was an academic thesis), no startup, no acquisition. His financial life was the life of a government scientist — comfortable but not wealthy. He married Ludmilla Dolmatova and lives in Virginia with their three children (Alex, Lissa, and Maia).

Compare this to Hinton, who went on to become a Google VP, shared the 2018 Turing Award, and became arguably the most famous scientist in AI. Or to Andrew Ng and Yann LeCun, whom Werbos funded, who went on to lead AI at Google Brain/Coursera and Facebook/Meta respectively.

### Current Status

As of the most recent available information (2023), Werbos is alive at age 78 (born 1947), retired from NSF since 2015, and holds an adjunct professorship at the Kummer Institute Center for AI and Autonomous Systems at Missouri University of Science and Technology. He has been exploring connections between quantum mechanics, reinforcement learning, and consciousness — remaining the same kind of boundary-crossing thinker he was at Harvard 50 years ago.

---

## Part 5: The Connections — A Tangled Web of Independent Discovery

### The Precursors Werbos May or May Not Have Known About

**Henry J. Kelley (1926-1988)** published "Gradient Theory of Optimal Flight Paths" in the ARS Journal in 1960. Kelley was a control theorist at Grumman Aircraft, later a professor at Virginia Tech. His paper derived gradient computations for optimizing flight trajectories — essentially computing how small changes in control inputs propagate through a dynamic system. The math is recognizably similar to backpropagation, but it was framed entirely in the language of optimal control theory, not neural networks.

**Arthur Bryson and Yu-Chi Ho** extended Kelley's work in their 1969 textbook *Applied Optimal Control*, systematically showing reverse-gradient techniques for optimization. This was the control theory tradition that ran parallel to — but mostly separate from — the neural network tradition.

**Seppo Linnainmaa** (born 1945), a Finnish mathematics student at the University of Helsinki, published the modern reverse mode of automatic differentiation in his 1970 master's thesis. His FORTRAN implementation (chapters 6-7 of the thesis) described efficient error backpropagation in arbitrary, discrete, possibly sparsely connected, network-like structures. He developed it to analyze the cumulative effect of rounding errors in long chains of computations — a pure numerical analysis problem, with no reference to neural networks.

Linnainmaa's 1970 thesis was written in Finnish and published in a Finnish journal. It was essentially unknown outside Finland until Juergen Schmidhuber began championing it decades later. As of 2020, all modern deep learning frameworks (TensorFlow, PyTorch) are fundamentally implementations of Linnainmaa's 1970 algorithm.

**Did Werbos know about these prior works?** Werbos himself has acknowledged the longer lineage of backpropagation's mathematical foundations, particularly from the control theory tradition. Yann LeCun noted in 1988 that "back-propagation had been used in the field of optimal control long before its application to connectionist systems." The specific connection between Werbos's 1974 work and Linnainmaa's 1970 work is less clear — given that Linnainmaa's thesis was in Finnish and in numerical analysis, not AI, it seems unlikely Werbos knew about it at the time.

### Also in the Mix: Shun'ichi Amari (1967-68)

It's worth noting that **Shun'ichi Amari**, a Japanese researcher, proposed training deep multilayer perceptrons using stochastic gradient descent as early as 1967-68. With his student Saito, he created a five-layer network that learned internal representations to classify non-linearly separable patterns — one of the earliest deep learning achievements. This work was published in Japanese and similarly overlooked by the English-speaking AI community.

### Rumelhart, Hinton & Williams: The Team That Made It Famous

**David Rumelhart** (1942-2011) was a cognitive psychologist at UC San Diego. He developed backpropagation independently in spring 1982 — eight years after Werbos, but without knowledge of his work. **Geoffrey Hinton** (born 1947, the same year as Werbos) was a British-Canadian cognitive scientist who had been stubbornly working on neural networks throughout the AI winter. **Ronald Williams** was a computer scientist at Northeastern University.

Their 1986 Nature paper didn't just describe the algorithm — it demonstrated, with compelling experiments, that backpropagation-trained networks could discover useful hidden representations on their own. This was the proof of concept that the field needed. It wasn't enough to have the math; you needed to show the skeptics that it actually worked on interesting problems.

The Rumelhart-Hinton-Williams paper succeeded where Werbos's thesis failed for several reasons:
- **Right venue** — *Nature*, the most prestigious scientific journal in the world
- **Right time** — 1986, when computing power had caught up enough to run meaningful experiments
- **Right framing** — demonstrated results, not just theory
- **Right network** — Hinton in particular had been building connections with the PDP (Parallel Distributed Processing) group for years
- **Right champions** — Hinton would go on to spend four decades evangelizing neural networks, eventually winning the 2018 Turing Award

### The AI Winter Context

Werbos's thesis (1974) landed at the absolute nadir of neural network research. The timeline tells the story:

- **1969** — Minsky & Papert publish *Perceptrons*; funding collapses
- **1971** — Rosenblatt drowns
- **1974** — **Werbos publishes the solution** (nobody notices)
- **1974** — The Lighthill Report in the UK further devastates AI funding
- **1975-1985** — Deep winter; only a handful of researchers persist (Hinton, Grossberg, Fukushima, Kohonen)
- **1986** — Rumelhart, Hinton & Williams publish in Nature; the thaw begins

The irony is exquisite and brutal. The solution to the problem that "killed" neural networks existed within five years of the kill shot. It sat in a Harvard library for twelve years while the field languished. When the field finally revived, it was because someone else independently rediscovered the same idea.

---

## Part 6: The Irony and the Drama

### A Man Out of Time

Paul Werbos is the kind of figure that makes you question whether scientific progress is meritocratic. He was smarter than almost everyone in the room. He had the right answer. He published it. And it didn't matter — because he published it in the wrong place, at the wrong time, with the wrong framing, under the wrong advisor, in a field that had collectively decided not to listen.

He offered to collaborate with Minsky. Minsky said no. He presented to the Harvard faculty. They told him to scale down his ambitions. He published a thesis that contained the mathematical key to training deep networks, and it sat unread while the entire AI establishment spent a decade building expert systems that would themselves become obsolete.

And then, when the vindication came, it came with someone else's name on it.

### What Werbos Built Instead

But Werbos's story isn't really a tragedy. He went on to develop **backpropagation through time** (BPTT) for recurrent neural networks, extending his original algorithm to handle temporal sequences — the theoretical foundation for training the LSTMs and RNNs that would dominate NLP before Transformers. He pioneered **Adaptive Dynamic Programming** (ADP), a framework combining neural networks with dynamic programming that is now recognized as a precursor to modern **reinforcement learning**.

From his position at NSF, he shaped the direction of an entire field of research for 27 years. He funded the people who would build the deep learning revolution. When he says there have been three revolutions in neural networks, he was present at or responsible for enabling all three.

He published *The Roots of Backpropagation* (Wiley, 1994) — reprinting his original 1974 thesis alongside newer papers, with a guide to the field — essentially saying to history: "I was here first, and here's the proof." The thesis has accumulated over 4,800 citations.

### The Recognition Gap

Geoffrey Hinton shared the 2018 Turing Award (with Yann LeCun and Yoshua Bengio) for conceptual and engineering breakthroughs in deep learning. The award specifically credited work on backpropagation. Paul Werbos was not included.

The IEEE has given Werbos its Neural Network Pioneer Award (1995) and Frank Rosenblatt Award (2022). These are significant honors within the computational intelligence community. But they don't carry the cultural weight of a Turing Award or a Nobel Prize.

Werbos got recognition from the specialists. Hinton got recognition from the world.

---

## Connect the Dots

- **Backward to Minsky & Papert (1969)**: Werbos solved the exact problem that *Perceptrons* used to kill neural networks — how to train multi-layer networks. See: `06_minsky_perceptrons_death_and_resurrection.md`

- **Backward to Rosenblatt (1962)**: Rosenblatt coined the term "back-propagating errors" in *Principles of Neurodynamics* but didn't know how to implement it. Werbos provided the implementation 12 years later. See: `02_rosenblatt_and_the_perceptron.md`

- **Backward to Kelley (1960) and control theory**: The mathematical idea of computing gradients backward through a system originated in optimal control theory for rocket trajectories. Werbos bridged this to neural networks.

- **Parallel to Linnainmaa (1970)**: The Finnish master's student who independently derived reverse-mode automatic differentiation for numerical error analysis — the same algorithm, different context, four years before Werbos. Basically never cited until the 2010s.

- **Forward to Rumelhart, Hinton & Williams (1986)**: Independent rediscovery and experimental validation that launched the neural network renaissance. See: `09_backprop_the_wiggle_ratio.md` and `10_mlp_backprop_and_the_birth_of_rnns.md`

- **Forward to backpropagation through time (BPTT)**: Werbos's own extension of backprop to recurrent networks — the reason LSTMs can be trained. See: `11_lstm_the_memory_machine.md`

- **Forward to the deep learning revolution (2009-)**: Werbos at NSF funded Andrew Ng and Yann LeCun, whose benchmark results convinced Google to invest in neural networks, triggering the modern AI industry.

- **Forward to modern autodiff**: Every call to `loss.backward()` in PyTorch is running a descendant of the algorithm Werbos described in 1974 (and Linnainmaa described in 1970). TensorFlow, JAX, PyTorch — all reverse-mode automatic differentiation.

- **The credit assignment problem in science itself**: Werbos's story is a case study in how scientific credit gets assigned. The first to discover, the first to publish, and the first to make people care are often three different people. Werbos was the first to apply it to neural nets. Rumelhart/Hinton/Williams were the first to make people care.

---

## Key Sources

- [Paul Werbos - Wikipedia](https://en.wikipedia.org/wiki/Paul_Werbos)
- [Paul Werbos Biography - National Space Society](https://nss.org/paul-werbos-biography/)
- [Grokipedia: Paul Werbos](https://grokipedia.com/page/paul_werbos)
- [How Marvin Minsky Inspired Artificial Neural Networks - Mind Matters](https://mindmatters.ai/2021/06/how-marvin-minsky-inspired-artificial-neural-networks/)
- [Neural Networks and Quantum Physics are the Future: Paul Werbos - IEEE Young Professionals](https://yp.ieee.org/blog/2023/04/10/neural-networks-and-quantum-physics-are-the-future-paul-werbos/)
- [Who Invented Backpropagation? - Schmidhuber/IDSIA](https://people.idsia.ch/~juergen/who-invented-backpropagation.html)
- [Backpropagation is Older Than You Think - Learning from Examples](https://www.learningfromexamples.com/p/backpropagation-is-older-than-you)
- [Paul Werbos - Regulating AI](https://regulatingai.org/expert/paul-werbos/)
- [Lifeboat Foundation: Dr. Paul John Werbos](https://lifeboat.com/ex/bios.paul.john.werbos)
- [The Roots of Backpropagation (Wiley, 1994)](https://www.wiley.com/en-us/The+Roots+of+Backpropagation:+From+Ordered+Derivatives+to+Neural+Networks+and+Political+Forecasting+-p-9780471598978)
- [Rumelhart, Hinton & Williams (1986) - Nature](https://www.nature.com/articles/323533a0)
- [Werbos 1974 thesis PDF (via Gwern)](https://gwern.net/doc/ai/nn/1974-werbos.pdf)
- [Werbos 1982: Applications of Advances in Nonlinear Sensitivity Analysis](https://link.springer.com/chapter/10.1007/BFb0006203)
- [Karl Deutsch - Wikipedia](https://en.wikipedia.org/wiki/Karl_Deutsch)
- [Seppo Linnainmaa - Wikipedia](https://en.wikipedia.org/wiki/Seppo_Linnainmaa)
