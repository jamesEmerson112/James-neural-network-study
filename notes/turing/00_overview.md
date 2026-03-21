# Alan Turing (1912-1954)

> Part of the [Neural Network Study Timeline](../00_timeline_neural_sequence_models.md). See also: [McCulloch-Pitts](../01_mcculloch_pitts_and_the_1943_scene.md), [Backprop](../09_backprop_the_wiggle_ratio.md).

## The Man Who Invented Computation

Born in Maida Vale, London. Graduated King's College Cambridge, PhD from Princeton (1938). Eccentric, kind, brilliant — a serious long-distance runner who chained his coffee mug to a radiator.

---

## Core Contributions

| Year | Contribution | Significance |
|------|-------------|--------------|
| 1936 | **Turing Machine** | Defined what computation IS. Proved the Church-Turing thesis — anything computable can be computed by this simple device. Also proved the Halting Problem: some questions are fundamentally unanswerable. |
| 1939-45 | **Enigma codebreaking** | Designed the Bombe at Bletchley Park. Cracked Nazi naval Enigma. Shortened WWII by an estimated 2-4 years, saving millions of lives. Classified until the 1970s. |
| 1950 | **"Computing Machinery and Intelligence"** | Proposed the Turing Test (Imitation Game). Founded the field of artificial intelligence. Anticipated and rebutted every major objection to AI for the next 75 years. |
| 1952 | **Morphogenesis paper** | Showed how reaction-diffusion equations produce biological patterns (spots, stripes) from uniform initial conditions. The mathematical ancestor of modern diffusion models. |

---

## Connection to Deep Learning

```
1936  Turing Machine         →  defined computation
        │
1943  McCulloch-Pitts        →  proved neurons compute like Turing machines
        │
1945  Von Neumann            →  built the hardware
        │
2020  Diffusion models       →  Turing's noise-to-pattern idea, computed by
                                 neural networks, on von Neumann machines
```

He built both halves of the bridge: defined computation (1936) and discovered that diffusion creates structure from noise (1952). Modern AI connected them.

---

## His Persecution and Death

Convicted of "gross indecency" in 1952 for being gay. Given the choice: prison or chemical castration. Chose the injections so he could keep working. Security clearance revoked. Found dead of cyanide poisoning on June 7, 1954, age 41. Pardoned by the Queen in 2013 — 59 years too late. Now on the British 50-pound note.

---

## Notes in This Folder

| File | Topic |
|------|-------|
| `01_turing_the_man_who_invented_computation.md` | Full biography and historical context |
| `02_turing_morphogenesis_his_final_beautiful_idea.md` | Reaction-diffusion, pattern formation |
| `03_turing_machine_technical_deep_dive.md` | Formal definitions, complexity theory, Halting Problem |
| `04_turing_unfinished_works.md` | Incomplete research at time of death |
| `05_turing_to_diffusion_models_the_lineage.md` | 90-year lineage from Turing to DALL-E |
| `06_the_halting_problem.md` | Complete deep dive: proof, history, Godel/Cantor connections, AI implications |
| `turing_machine_simulator.py` | Working Python implementation |
