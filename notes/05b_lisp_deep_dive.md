# LISP Deep Dive — The Language That Changed Everything

> *Last updated: 2026-03-16*
> *See also: [04_john_mccarthy.md](04_john_mccarthy.md) and [05_lisp_why_it_matters.md](05_lisp_why_it_matters.md)*

## Programming Before LISP (The 1950s)

### How You Wrote Code

Paper → punch cards → wait hours/days:

1. Write program on coding sheets by hand
2. Keypunch operator punches onto cards (one line = one card)
3. Submit deck to computer operator
4. Wait **hours or DAYS**
5. Get printed output — probably a compilation error
6. Fix one card, resubmit entire deck

No screen. No keyboard. No interactive anything.

### Languages That Existed

| Language | Year | Purpose | Paradigm |
|----------|------|---------|----------|
| Assembly | 1949 | Direct hardware control | Move bytes between registers |
| FORTRAN | 1957 | Scientific/numerical computation | Imperative — DO loops, arrays, GOTO |
| IPL | 1956 | List processing (AI) | Assembly-like syntax, clunky |
| **LISP** | **1958** | **Symbolic reasoning / AI** | **Functional — recursion, lists, lambda calculus** |

## LISP's 9 Revolutionary Principles

### 1. Recursion
FORTRAN literally couldn't do it (no stack, fixed return addresses). LISP made it natural:
```lisp
(defun factorial (n)
  (if (zerop n) 1
      (* n (factorial (- n 1)))))
```

### 2. Homoiconicity (code = data)
The mind-bender. Code and data are the SAME thing (lists):
```lisp
'(+ 1 2 3)          ;; DATA — a list
(+ 1 2 3)           ;; CODE — evaluates to 6
(eval '(+ 1 2 3))   ;; turn data INTO code at runtime!
```
Programs that generate, examine, and modify other programs. Impossible in FORTRAN.

### 3. First-Class Functions
Functions are values you pass around:
```lisp
(mapcar #'sqrt '(1 4 9 16))  ;; applies sqrt to each → (1 2 3 4)
```
FORTRAN: functions were subroutines. Period.

### 4. Garbage Collection
Automatic memory management. FORTRAN/Assembly: manage every byte yourself.

### 5. Dynamic Typing
A variable can hold anything:
```lisp
(setq x 5)        ;; integer
(setq x "hello")  ;; now a string
(setq x '(a b))   ;; now a list
```

### 6. The REPL (Read-Eval-Print Loop)
Type code, see result instantly:
```lisp
CL-USER> (+ 2 2)
4
```
FORTRAN: submit punch cards, wait hours.

### 7. Lambda Functions
Anonymous inline functions, from Alonzo Church's lambda calculus.

### 8. Lists as Fundamental Data
Not fixed-size arrays. Flexible, nested, recursive data structures.

### 9. Macros
Write code that writes code at compile time — extend the language itself:
```lisp
(defmacro unless (test &body body)
  `(if (not ,test) (progn ,@body)))
;; You just invented a new language construct
```

## Side-by-Side: FORTRAN vs LISP

**Task: Sum of squares of a list of numbers**

FORTRAN (1957):
```fortran
      PROGRAM SUMSQ
      DIMENSION X(100)
      INTEGER N, I, SUM
      SUM = 0
      DO 10 I = 1, N
         SUM = SUM + X(I) * X(I)
   10 CONTINUE
      PRINT *, SUM
      END
```

LISP (1958):
```lisp
(reduce #'+ (mapcar (lambda (x) (* x x)) '(1 2 3 4 5)))
;; Returns 55
```

**Task: Check if list contains an odd number**

FORTRAN:
```fortran
      HASODD = .FALSE.
      DO 10 I = 1, N
         IF (MOD(X(I), 2) .NE. 0) THEN
            HASODD = .TRUE.
            GO TO 20
         END IF
   10 CONTINUE
   20 CONTINUE
```

LISP:
```lisp
(some #'oddp '(2 4 6 7 8))  ;; Returns T
```

## The Eval Story

McCarthy published `eval` as **theoretical mathematical notation** — how LISP *would* work, not meant to be implemented. He told student Steve Russell: *"This is for reading, not for computing."*

**Russell implemented it anyway.** Hand-compiled the math into IBM 704 machine code. It worked. McCarthy was surprised.

Russell created the first LISP interpreter — a language that evaluates itself. A **metacircular evaluator**. Foundation of every modern interpreter.

## The Philosophical Split

```
FORTRAN: Von Neumann architecture
         "Tell the machine what STEPS to take"
         Imperative — move data, loop, branch, store
         Designed for: number crunching

LISP:    Lambda calculus
         "Describe what things ARE"
         Functional — transform, map, reduce, recurse
         Designed for: symbolic reasoning
```

## Why LISP Didn't Kill FORTRAN

**Speed.** FORTRAN compiled to optimized machine code. LISP was interpreted (eval at runtime). For number crunching, FORTRAN was orders of magnitude faster.

```
FORTRAN: compiled → machine code → hardware speed
LISP:    interpreted → eval reads each expression → much slower
```

| Task | FORTRAN wins | LISP wins |
|------|-------------|-----------|
| Matrix multiplication | Blazing fast, contiguous memory | Slow — lists aren't contiguous |
| Numerical simulation | Built for this | Massive GC overhead |
| Recursion | Impossible until 1990 | Natural and fundamental |
| Code that writes code | Impossible | Macros, eval |
| Symbolic reasoning | Can't represent symbols | Built for it |
| Interactive dev | Punch cards, wait hours | REPL — instant feedback |

**Neither killed the other. They served different masters.**

## The Two Lineages (Still Alive Today)

```
FORTRAN lineage (number crunching):
  FORTRAN → C → C++ → CUDA → modern GPU computing
  "Make the machine go FAST"
  Users: physicists, engineers, HPC, game engines

LISP lineage (symbolic reasoning):
  LISP → Scheme → ML → Haskell → Clojure
  "Make the programmer think CLEARLY"
  Users: AI researchers, language designers, academics

Hybrid lineage (borrowed from both):
  Python, JavaScript, Ruby, Java
  Got GC, dynamic typing, first-class functions from LISP
  Got imperative control flow, practical speed from FORTRAN/C
```

## Where Rust Fits

Rust borrows from **both lineages** and adds something new:

```
FORTRAN lineage: Speed, compiled, no garbage collector
LISP lineage:    Type system, pattern matching, closures, macros
NEW (Rust only):  Ownership & borrow checker — compile-time memory safety
```

| Feature | FORTRAN | LISP | C/C++ | Rust |
|---------|---------|------|-------|------|
| Speed | Fast | Slow | Fast | Fast |
| Memory mgmt | Manual | GC | Manual (crashes) | **Ownership** (compile-time) |
| Recursion | No (until 1990) | Yes | Yes | Yes |
| First-class functions | No | Yes | Sort of | Yes |
| Pattern matching | No | Yes | No | Yes |
| Macros | No | Yes | Crude preprocessor | Yes (hygienic) |
| Type safety | Weak | Dynamic | Weak (segfaults) | **Strongest** |

Rust solved the impossible tradeoff:
```
C:     Fast but unsafe    → "segmentation fault"
LISP:  Safe but slow      → garbage collector pauses
Rust:  Fast AND safe      → compiler proves memory correctness
```

### Rust's Family Tree

```
1957  FORTRAN    → speed, compiled, arrays, numerical
1958  LISP       → safety, recursion, functions-as-values, macros
1972  C          → FORTRAN speed + systems programming
1979  ML         → LISP ideas + static types + pattern matching
1983  C++        → C + object-oriented
2010  Rust       → C++ speed + ML types + LISP macros + NEW ownership model
```

## The Deep Learning Connection

Modern deep learning runs on **both lineages simultaneously**:

- **The math** (matrix multiply, backprop) needs FORTRAN-speed → GPUs via CUDA (C/FORTRAN lineage)
- **The interface** (PyTorch, TensorFlow) uses Python → inherited GC, dynamic typing, closures from LISP

When you write `loss.backward()` in PyTorch, you're using **LISP's ideas** (dynamic, expressive, interactive) to control **FORTRAN's muscle** (compiled, optimized, fast).

Rust is increasingly showing up in ML infrastructure:
- Hugging Face tokenizers (Rust for speed)
- candle / burn (Rust ML frameworks)
- ruff (Python linter in Rust, 100x faster)

Pattern: Python (LISP lineage) for researcher interface, Rust (FORTRAN+LISP hybrid) for performance internals.

**Neither lineage won. They merged.**
