# ADALINE as a Noise Filter — The Signal Processing Origin of Neural Networks

## Core Insight

ADALINE and LMS are fundamentally **adaptive filters**, not "brain models." The math is identical whether you're:

- Removing echo from a phone line
- Predicting tomorrow's weather
- Classifying inputs into categories

It's all: **"given noisy input, find the signal."** Adjust weights to minimize the difference between what you expected and what you got.

## Why This Matters

- Widrow came from **electrical engineering**, not AI — he was building a better filter, not modeling the brain
- The "neuron" framing was almost incidental; the math came from signal processing and statistics
- LMS is still used today in echo cancellation on telephone lines — its most lasting practical application
- Ted Hoff (co-author of LMS) went on to co-invent the Intel 4004 microprocessor — the concepts of signal processing and computation are deeply connected

## The Reframe

Early "neural networks" were really just **adaptive linear filters with a marketing rebrand**.

The deep learning revolution came when people:
1. **Stacked** them (multiple layers)
2. **Added nonlinearities** (activation functions)
3. **Ran backprop** through the whole thing (Rumelhart, Hinton, Williams — 1986)

## Connection to This Assignment

The same gradient-based optimization principle flows through every model in Assignment 3:
- **LSTM** — gates control what signal to keep/forget, trained by gradient descent
- **Seq2Seq** — encoder filters source signal, decoder reconstructs it in target language
- **Transformer** — attention is a learned, dynamic filter over all positions
