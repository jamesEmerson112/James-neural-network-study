# Chapter 1 — Functions and Parameters

Book: *The Little Learner: A Straight Line to Deep Learning* (Friedman & Christiansen)

## Scheme Basics

- `(lambda (x) (+ x 1))` = anonymous function, same as Python `lambda x: x + 1`
- Syntax: `(operator operand1 operand2 ...)` — prefix notation, everything in parens

## Key Laws

- **Law of Parameters**: Parameters are placeholders — they have no value until arguments are supplied.

## Reduction / Evaluation

- **Redex** (reducible expression): an expression that can still be simplified
- **Normal form**: fully reduced, nothing left to evaluate
- Book notation: **dashed line** = still reducible, **solid line** = final form
- This is **beta reduction** from lambda calculus (Church, 1930s): substitute argument into function body

Example:
$(λ(w\;b).\; w \cdot 8 + b)(3, 1) \rightarrow 3 \cdot 8 + 1 \rightarrow 25$

## Bridge to ML (why this matters later)

- Lambda parameters → trainable weights and biases (θ, w, b)
- Function composition → stacking neural network layers
- Beta reduction → forward pass evaluation
- A function that takes a function and returns a function → automatic differentiation
