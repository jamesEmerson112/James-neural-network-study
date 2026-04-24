# Calibration and the Fairness Impossibility Theorem

## Calibration — "when the model says 80%, it should be right 80% of the time"

A classifier is **well-calibrated** if its predicted probabilities match reality.

$$\forall p \in [0,1], \quad P(\hat{Y} = Y \mid \hat{P} = p) = p$$

- $\hat{P} = p$ — take all observations where the model predicted probability $p$
- $\hat{Y} = Y$ — the predicted label matches the true label
- $P(\ldots) = p$ — the fraction that actually are positive should equal $p$

**Concrete example:** the model scores 100 people with $\hat{P} = 0.8$. If the model is well-calibrated, ~80 of those 100 really are positive. If only 50 are positive, the model is **overconfident**. If 95 are positive, the model is **underconfident**.

---

## The confusion matrix — FPR, FNR, and the four outcomes

Every binary classifier puts each prediction into one of four buckets:

```
                          Actual Positive (Y=1)     Actual Negative (Y=0)
                         ┌────────────────────────┬────────────────────────┐
  Model says Positive    │  True Positive (TP)     │  False Positive (FP)   │
  (predicted Y=1)        │  "correctly caught"     │  "falsely accused"     │
                         ├────────────────────────┼────────────────────────┤
  Model says Negative    │  False Negative (FN)    │  True Negative (TN)    │
  (predicted Y=0)        │  "slipped through"      │  "correctly cleared"   │
                         └────────────────────────┴────────────────────────┘
```

**FPR — False Positive Rate:** of everyone who is actually *negative*, what fraction did the model wrongly flag?

$$\text{FPR} = \frac{FP}{FP + TN} = \frac{\text{falsely accused}}{\text{all actual negatives}}$$

Think of it as: "I'm innocent — what's my chance of being wrongly flagged?" A high FPR means the model is trigger-happy, flagging people who shouldn't be flagged. In a criminal justice context, this is an innocent person being told they're high risk.

**FNR — False Negative Rate:** of everyone who is actually *positive*, what fraction did the model miss?

$$\text{FNR} = \frac{FN}{FN + TP} = \frac{\text{slipped through}}{\text{all actual positives}}$$

Think of it as: "I'm actually dangerous — what's the chance the model misses me?" A high FNR means the model is too lenient, letting real positives walk. In a medical context, this is a sick patient being told they're healthy.

**Quick mnemonic:**
- FPR = how badly the model treats **innocent** people (false alarms)
- FNR = how badly the model treats **guilty/sick** people (missed catches)
- Both range from 0% (perfect) to 100% (worst)
- They trade off: making the model more aggressive (flag more people) → lowers FNR but raises FPR

**Concrete example:** A cancer screening model tests 1000 people. 100 have cancer, 900 don't.

```
                          Has cancer (100)     No cancer (900)
Model says "cancer"          80 (TP)              90 (FP)
Model says "no cancer"       20 (FN)             810 (TN)
```

- $\text{FPR} = 90/900 = 10\%$ — 10% of healthy people get a scary false alarm
- $\text{FNR} = 20/100 = 20\%$ — 20% of actual cancer patients are told they're fine (dangerous!)

---

## Three fairness properties (for a risk score $S$ applied to two groups A, B)

### 1. Calibration

Among all people scored $S = s$, the actual positive rate is the same regardless of group:

$$P(Y = 1 \mid S = s, \text{Group} = A) = P(Y = 1 \mid S = s, \text{Group} = B)$$

"A score of 7 means the same thing no matter who you are."

### 2. Balance for the positive class (equal FNR)

Among people who actually are positive ($Y = 1$), the average risk score is the same across groups. Equivalently: **false negative rates are equal**.

### 3. Balance for the negative class (equal FPR)

Among people who actually are negative ($Y = 0$), the average risk score is the same across groups. Equivalently: **false positive rates are equal**.

---

## The Impossibility Theorem

> **Chouldechova (2017), Kleinberg-Mullainathan-Raghavan (2016)**

**You cannot satisfy all three simultaneously — unless the base rates are equal across groups.**

If $P(Y = 1 \mid \text{Group} = A) \neq P(Y = 1 \mid \text{Group} = B)$, then any classifier that is calibrated **must** violate either equal FPR or equal FNR (or both).

This is a **mathematical fact**, not an algorithm limitation. No amount of engineering can fix it.

### Why it's impossible — intuition

Calibration locks the meaning of each score to a fixed positive rate. But if Group A has a higher base rate than Group B, then:
- Group A will have more true positives at every score threshold → different FNR
- Group B will have more false positives relative to its smaller positive pool → different FPR

You can't redistribute the errors evenly across groups without breaking the calibration guarantee.

### Worked example — seeing the impossibility with numbers

We'll walk through this slowly. Imagine a judge uses a model to decide who gets extra supervision after release. The model outputs a risk score, and anyone labeled "High Risk" gets supervised.

#### Step 1: Set the scene — two groups with different base rates

```
Group A: 1000 people, 50% base rate (500 will actually reoffend, 500 won't)
Group B: 1000 people, 20% base rate (200 will actually reoffend, 800 won't)
```

The base rates are different — Group A reoffends at a higher rate than Group B. This is just a fact about the populations. The model didn't cause this; it's the ground truth the model has to work with.

#### Step 2: What calibration demands

The model assigns some people to "High Risk." Being calibrated means: **of all people the model calls High Risk, exactly 60% actually reoffend.** This must hold *within each group separately*. A High Risk label can't mean "60% for Group A but 40% for Group B" — that would be uncalibrated (and arguably discriminatory in itself, since the same label would mean different things for different people).

So:
- High Risk in Group A → 60% of them actually reoffend
- High Risk in Group B → 60% of them actually reoffend

#### Step 3: Why you can't label the same number of people High Risk in both groups

Naive attempt: label 400 people as High Risk in each group.

Calibration says 60% of 400 = **240 true positives** in the High Risk bucket for each group.

```
                          Group A                    Group B
                     Reoffend  Won't             Reoffend  Won't
High Risk (400)        240      160                240      160
Low Risk  (600)        260      340                ???      ???
                       ---      ---                ---      ---
Column totals:         500      500                200      800
```

Group A works fine: 500 - 240 = 260 reoffenders left for Low Risk. No problem.

Group B is broken: it only has 200 total reoffenders, but we already put 240 in the High Risk bucket. That's **more reoffenders than exist in the entire group**. The Low Risk bucket would need -40 reoffenders, which is nonsensical.

The core issue: Group B doesn't have enough actual positives to fill up a large High Risk bucket while maintaining 60% precision. The lower base rate constrains how many people the model can call High Risk.

#### Step 4: A calibrated model that actually works

The model must label **fewer** people as High Risk in Group B to stay calibrated. Let's say:
- Group A: 400 labeled High Risk
- Group B: 200 labeled High Risk

Now fill in the confusion matrices. Calibration requires 60% of each High Risk group to be true positives:

```
Group A (1000 people, base rate 50%):

                     Reoffend    Won't
                     (Y=1)       (Y=0)
High Risk (400)       240         160       ← 240/400 = 60% ✓ calibrated
Low Risk  (600)       260         340
                      ---         ---
                      500         500
```

```
Group B (1000 people, base rate 20%):

                     Reoffend    Won't
                     (Y=1)       (Y=0)
High Risk (200)       120          80       ← 120/200 = 60% ✓ calibrated
Low Risk  (800)        80         720
                      ---         ---
                      200         800
```

Both groups are perfectly calibrated. A "High Risk" label means 60% reoffend, period, regardless of group. This is fair in the calibration sense.

#### Step 5: Now compute the error rates — here's where it breaks

**False Positive Rate (FPR)** = "of all the people who WON'T reoffend, what fraction did the model incorrectly flag as High Risk?"

$$\text{FPR} = \frac{\text{False Positives}}{\text{Total Actual Negatives}}$$

```
Group A:  FPR = 160 / 500 = 32%
Group B:  FPR =  80 / 800 = 10%
```

**Group A's innocent people get flagged at 3x the rate of Group B's.** If you're in Group A and you're NOT going to reoffend, you have a 32% chance of being wrongly supervised. In Group B, only 10%. This is what ProPublica measured in the COMPAS case.

**False Negative Rate (FNR)** = "of all the people who WILL reoffend, what fraction did the model miss?"

$$\text{FNR} = \frac{\text{False Negatives}}{\text{Total Actual Positives}}$$

```
Group A:  FNR = 260 / 500 = 52%
Group B:  FNR =  80 / 200 = 40%
```

**Group A's actual reoffenders slip through more often** — 52% missed vs. 40%. So the model also protects Group B's community better than Group A's.

#### Step 6: Can we fix this? Let's try forcing equal FPR.

We want FPR = 10% in both groups. For Group A, that means:

$$\text{FP}_A = 0.10 \times 500 = 50 \text{ false positives (down from 160)}$$

If the High Risk bucket in Group A now has only 50 false positives, and we still want 60% calibration:

$$0.60 = \frac{\text{TP}}{\text{TP} + 50} \implies \text{TP} = 75$$

So Group A's High Risk bucket: 75 true positives + 50 false positives = 125 people.

```
Group A (forced equal FPR):

                     Reoffend    Won't
High Risk (125)        75          50       ← 75/125 = 60% ✓ still calibrated
Low Risk  (875)       425         450
                      ---         ---
                      500         500
```

```
FPR = 50/500 = 10% ✓ matches Group B!
FNR = 425/500 = 85% ← we now miss 85% of reoffenders in Group A
```

We equalized FPR, and calibration still holds — but **FNR exploded to 85%.** We're now catching only 15% of Group A's reoffenders while catching 60% of Group B's. That's a massive FNR gap, arguably even more unfair than before.

The math is unforgiving: **every knob you turn to fix one error rate breaks another**, as long as the base rates differ.

#### Step 7: The impossibility, summarized

```
Attempt 1 (calibrated):     Cal ✓    FPR 32% vs 10% ✗    FNR 52% vs 40% ✗
Attempt 2 (force equal FPR): Cal ✓    FPR 10% vs 10% ✓    FNR 85% vs 40% ✗✗✗
```

There is no Attempt 3 that gets all three checkmarks. This isn't a modeling failure — it's a **mathematical constraint**. The unequal base rates (50% vs 20%) create a tension that no classifier can resolve. You must choose which type of fairness matters more, and that's a human values decision, not an engineering one.

---

## The COMPAS case — the canonical example

COMPAS is a recidivism risk tool used in U.S. courts. ProPublica investigated it in 2016.

| Stakeholder | Fairness criterion they used | Satisfied? |
|---|---|---|
| **Northpointe** (built COMPAS) | Calibration — "a score of 7 means the same recidivism risk for any race" | Yes |
| **ProPublica** (investigated COMPAS) | Equal FPR — "Black defendants had ~2x the false positive rate" | No |

**Both sides were mathematically correct.** Because base rates differed across racial groups, COMPAS couldn't satisfy calibration and equal error rates simultaneously. The impossibility theorem proves this.

---

## FAT = Fairness, Accountability, and Transparency

The impossibility theorem is a central result in the FAT literature. Its takeaway:

- **Fairness** is not a single metric — there are multiple incompatible definitions
- The choice between calibration vs. equal error rates is a **values decision**, not a technical one
- **Accountability** requires making that choice explicit and defensible
- **Transparency** means the tradeoff is documented, not buried inside the algorithm

---

## Quick-fire self-test

1. What does it mean for a classifier to be well-calibrated? *(When it says p%, approximately p% of those cases are actually positive.)*
2. State the three fairness properties. *(Calibration, balance for positive class / equal FNR, balance for negative class / equal FPR.)*
3. When can you satisfy all three? *(Only when base rates are equal across groups.)*
4. In the COMPAS debate, what did ProPublica measure vs. Northpointe? *(ProPublica: FPR disparity. Northpointe: calibration.)*
5. Why is the impossibility result important for FAT? *(It forces an explicit values choice — you can't dodge it with better algorithms.)*
