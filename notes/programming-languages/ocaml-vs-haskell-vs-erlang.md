# OCaml vs Haskell vs Erlang — Functional Programming Languages Compared

Three influential functional programming languages, each with a different philosophy. This note assumes **zero FP experience** and explains concepts from scratch.

> Connected to the neural network study via the Lambda Calculus → LISP → ML lineage. See [John McCarthy](../04_john_mccarthy.md), [Why LISP Matters](../05_lisp_why_it_matters.md), and [LISP Deep Dive](../05b_lisp_deep_dive.md). Back to [master timeline](../00_timeline.md).

---

## What Is Functional Programming?

Functional programming (FP) is a paradigm where you build programs by composing **pure functions** — functions that take inputs and return outputs without modifying anything outside themselves.

Core ideas:

| Concept | What it means | Why it matters |
|---|---|---|
| **Immutability** | Once a value is created, it never changes | No surprise mutations — easier to reason about code |
| **First-class functions** | Functions are values: pass them around, store them, return them | Enables composition — build complex behavior from small pieces |
| **Pattern matching** | Destructure data directly in function definitions | Cleaner than chains of if/else, catches missing cases at compile time |
| **Pure functions** | Same input → always same output, no side effects | Testable, predictable, parallelizable |
| **Recursion over loops** | Repeat by calling the function itself, not with for/while | Natural fit when data is immutable (no loop counter to mutate) |

**Key mental shift**: In imperative programming (Python, Java, C), you tell the computer *what to do step by step*. In FP, you describe *what things are* — transformations on data.

---

## The Intellectual Roots of FP

Functional programming didn't appear out of nowhere — it has roots older than computers themselves.

**1932 — Lambda Calculus (Alonzo Church, Princeton)**
Before electronic computers existed, mathematician Alonzo Church invented the **lambda calculus** — a formal system for expressing computation using nothing but functions. Church first published the lambda calculus in 1932; his famous 1936 paper applied it to the Entscheidungsproblem (decidability), proving that some mathematical questions have no algorithmic answer. Meanwhile, Alan Turing independently reached the same conclusion with his theoretical tape machine. These two approaches turned out to be equivalent (the Church-Turing thesis), but they spawned very different programming traditions: Turing's machine model led to imperative languages (C, Java), Church's function model led to functional ones.

**1958 — Lisp (John McCarthy, MIT)**
The first functional programming language. McCarthy wanted a language for AI research — one that could manipulate symbolic expressions (not just numbers) and treat code as data. Lisp introduced garbage collection, recursion as a primary control flow, and the idea that functions are first-class values. Nearly every FP concept traces back to Lisp.

**1973 — ML (Robin Milner, University of Edinburgh)**
Milner was building **LCF**, a system for proving mathematical theorems by computer. He needed a language where programs could be *proved correct* — so he invented ML (Meta Language) with a powerful type system that catches errors at compile time. ML introduced **type inference** (the compiler deduces types without you writing them) and **pattern matching**. This is the direct ancestor of OCaml.

**1986-1990 — The branching point.** Three separate communities, solving three unrelated problems, independently decide that functions are the answer:
- French computer scientists at INRIA evolve ML → **Caml** (1987) → **OCaml** (practical, pragmatic)
- Academic researchers consolidate lazy FP languages → **Haskell** (pure, principled)
- Swedish telecom engineers at Ericsson need fault tolerance → **Erlang** (concurrent, resilient)

### Family Tree

```
Lambda Calculus (1932, Alonzo Church)
│
└── Lisp (1958, John McCarthy, MIT)
    │
    └── ML (1973, Robin Milner, Edinburgh)          ← born from theorem proving
        │
        ├── Standard ML (1983)
        ├── Caml (1987, Ascánder Suárez, INRIA France)
        │   └── OCaml (1996, Xavier Leroy)          ← "let's make ML practical"
        │
        └── Miranda (1985, David Turner)            ← lazy, proprietary
            └── Haskell (1990, academic committee)  ← "one open standard for lazy FP"

Prolog (1972, Alain Colmerauer, Marseille)
│
└── Erlang (1986, Joe Armstrong, Ericsson)          ← started as Prolog extensions for telecom
```

### Timeline — What Was Happening in Parallel

```
Year   OCaml lineage          Haskell lineage         Erlang lineage
─────  ────────────────────   ─────────────────────   ─────────────────────────
1973   ML born (Milner)
1985                          Miranda (Turner)
1986                                                  Armstrong starts Erlang at Ericsson
1987   Caml (INRIA)           FPCA conference: "we
                              need ONE lazy language"
1989                          GHC prototype (Hammond)
1990                          Haskell 1.0 released
1992                          Wadler's monads papers   Erlang goes into production
                                                        at Ericsson (phone switches)
1996   OCaml 1.0 (Leroy)
1998                                                  Ericsson BANS Erlang internally
                                                       → open-sourced (Dec 1998)
2003                          Haskell 98 standard
2003   Jane Street adopts                             Erlang gains traction outside
        OCaml for trading                              telecom
2010                          Haskell Platform         Erlang at WhatsApp
2014                                                  WhatsApp (~35 eng, 450M users)
                                                       acquired by Facebook for $19B
2020   OCaml 5 (multicore)   GHC 9.x series          Elixir matures on BEAM VM
```

---

## Quick Comparison

| | **OCaml** | **Haskell** | **Erlang** |
|---|---|---|---|
| **Born** | 1996 (INRIA, France) | 1990 (academic committee) | 1986 (Ericsson, Sweden) |
| **Philosophy** | Pragmatic FP — lets you be impure when needed | Pure FP — side effects must be explicit (monads) | Concurrency-first — built for fault-tolerant distributed systems |
| **Type system** | Static, strong, inferred | Static, strong, inferred (more powerful than OCaml's) | Dynamic — no compile-time type checking |
| **Evaluation** | Strict (eager) — evaluates arguments immediately | Lazy — only evaluates when the value is actually needed | Strict (eager) |
| **Concurrency** | Library-level (recently added multicore support) | Library-level (STM, async) | **Built into the language** — lightweight processes, actor model |
| **Purity** | Impure (side effects anywhere) | Pure (side effects tracked by the type system) | Impure (but message-passing encourages isolation) |
| **Compilation** | Native code (fast) | Native code (GHC) | Bytecode on BEAM VM |
| **Famous users** | Jane Street (finance), Meta (Hack/Flow), Docker | Facebook (spam filter), Standard Chartered, GitHub (Semantic) | WhatsApp, Ericsson, Discord (originally), RabbitMQ |

---

## OCaml — The Pragmatic One

**ML family** (Meta Language). OCaml stands for "Objective Caml" — it supports OOP too, but nobody really uses that part.

### History & Lore

The story starts in **Edinburgh, 1973**. Robin Milner — one of the most important computer scientists of the 20th century (later won the Turing Award) — was building a system called **LCF** (Logic for Computable Functions) to help mathematicians prove theorems using computers. He needed a programming language where you could trust that programs were correct. Existing languages were too sloppy — C let you corrupt memory, Lisp had no type checking. So Milner invented **ML** (Meta Language) with an iron-clad type system that could catch errors at compile time, *and* a type inference algorithm (Algorithm W) so you didn't have to tediously annotate every variable.

ML crossed the English Channel to **INRIA** (France's national computer science lab). In 1987, **Ascánder Suárez**, working under Gérard Huet's Formel project, implemented **Caml** (Categorical Abstract Machine Language) — a French dialect of ML. Then came **Caml Light**, a leaner version by Xavier Leroy and Damien Doligez.

In **1996**, **Xavier Leroy** (also INRIA) created **OCaml** — Objective Caml. The "Objective" refers to an object system he added, though it's rarely used in practice. The real contribution was a blazing-fast native code compiler and a practical, batteries-included standard library. Leroy wanted ML to stop being an academic curiosity and become a real tool for real software.

The twist came in **2003**: **Yaron Minsky**, fresh from his Cornell PhD (2002), joined **Jane Street** (a quantitative trading firm in NYC) and fell in love with OCaml. He convinced the firm to bet their entire trading infrastructure on it — billions of dollars flowing through OCaml code daily. Jane Street became the world's largest industrial OCaml user, hired many core OCaml developers, open-sourced major libraries (Core, Async, ppx), and basically bankrolled the language's modern ecosystem. The lesson: a language born from theorem proving ended up running Wall Street.

### What makes it special
- **Type inference** — you rarely write type annotations; the compiler figures them out
- **Strict evaluation** — behaves predictably, like Python or Java (no lazy surprises)
- **Escape hatches** — mutable references, exceptions, and imperative loops are available when FP is awkward
- **Fast** — compiles to native code, performance comparable to C++ for many workloads

### Code example

```ocaml
(* Factorial — recursive, pattern matching *)
let rec factorial n =
  match n with
  | 0 -> 1                    (* base case: 0! = 1 *)
  | n -> n * factorial (n - 1) (* recursive case *)

(* Filter even numbers from a list *)
let evens lst =
  List.filter (fun x -> x mod 2 = 0) lst
  (* fun x -> ... is a lambda / anonymous function *)

(* Using it *)
let () =
  let result = factorial 5 in
  Printf.printf "5! = %d\n" result;   (* 120 *)
  let nums = evens [1; 2; 3; 4; 5; 6] in
  List.iter (Printf.printf "%d ") nums (* 2 4 6 *)
```

### Key insight
OCaml is "FP with training wheels off" — it won't force you to be purely functional, but its type system catches a huge class of bugs at compile time. Jane Street runs billions of dollars through OCaml code because of this safety/pragmatism balance.

---

## Haskell — The Pure One

**The academic flagship of functional programming.** If OCaml is pragmatic FP, Haskell is principled FP — it takes purity seriously.

### History & Lore

By the late 1980s, the FP research community had a problem: there were too many lazy functional languages. Miranda, Clean, Gofer, Lazy ML, Orwell — every research group had their own, and they were all slightly different. Researchers couldn't share code or compare results because everyone was on a different language island.

The breaking point came at the **FPCA conference in Portland, Oregon, September 1987**. A group of researchers — frustrated by the fragmentation — decided: *"Let's design one single, open-standard, lazy, purely functional language."* A committee formed, led by luminaries including:

- **Simon Peyton Jones** (University of Glasgow, later Microsoft Research) — would become the long-time lead of GHC, Haskell's compiler, and the language's tireless champion for decades
- **Philip Wadler** (University of Glasgow) — wrote the landmark paper on **monads** (1992) that solved Haskell's biggest practical problem: how do you do I/O in a language that forbids side effects?
- **Paul Hudak** (Yale) — one of the original designers, later wrote "A Gentle Introduction to Haskell"

They named it after **Haskell Brooks Curry** (1900–1982), an American mathematician and logician whose work in combinatory logic was foundational to FP. The technique called **currying** — transforming a function that takes multiple arguments into a chain of functions each taking one argument: `f(x, y)` becomes `f(x)(y)` — actually traces back to Gottlob Frege (1893) and Moses Schönfinkel (1924). Curry developed the idea further in the 1930s, and the name "currying" was coined by Christopher Strachey in 1967. Curry never knew a language would be named after him.

**Haskell 1.0** arrived in **1990**. The **Glasgow Haskell Compiler (GHC)** began as a prototype by Kevin Hammond in **1989**, with a first beta release in April 1991 and the first full release in 1992. It became the de facto implementation and is still the standard today — one of the longest-running open source compiler projects in history, shepherded by Simon Peyton Jones for over 30 years.

The **monads breakthrough** deserves its own mention. Pure languages had a fundamental problem: if functions can't have side effects, how do you print to the screen? Read a file? Get the current time? Philip Wadler presented "The essence of functional programming" at POPL in January 1992, followed by "Monads for functional programming" at the Marktoberdorf summer school later that year (published 1995). These papers showed how to *model* effects as values in the type system — the IO monad. This was arguably the most influential programming languages idea of the 1990s and its concepts rippled into Scala, Rust, Swift, and even JavaScript (though JS Promises, despite looking monadic, actually violate the monad laws because they automatically flatten nested Promises — `Promise<Promise<T>>` collapses to `Promise<T>`).

Haskell never became a mainstream industry language, but it has been a **factory for ideas that go mainstream**: type classes → Rust traits, monads → async/await, algebraic data types → TypeScript unions, pattern matching → now in Python 3.10+. The joke in the community is: *"Haskell: avoid success at all costs"* — which Simon Peyton Jones says was intentionally ambiguous. Does it mean "avoid success, at all costs" (stay niche) or "avoid success-at-all-costs" (don't compromise purity for popularity)?

### What makes it special
- **Purity** — functions cannot have side effects. Printing to the screen, reading a file, getting the current time — all require explicit marking via the **IO monad**
- **Lazy evaluation** — nothing is computed until its result is actually needed. You can define an infinite list and only compute the first 10 elements
- **Type classes** — like interfaces but more powerful. Define behavior (e.g., "things that can be compared") and the compiler fills in the implementation
- **Monads** — a pattern for sequencing operations that have effects. Sounds scary, but it's just a design pattern for chaining computations

### Code example

```haskell
-- Factorial — pattern matching on the argument directly
factorial :: Int -> Int        -- type signature (optional but good practice)
factorial 0 = 1                -- base case
factorial n = n * factorial (n - 1)  -- recursive case

-- Filter even numbers from a list
evens :: [Int] -> [Int]
evens lst = filter even lst    -- 'even' is a built-in function

-- Lazy evaluation: infinite list, take only what you need
naturals :: [Int]
naturals = [1..]               -- infinite list: 1, 2, 3, 4, ...

firstTenEvens :: [Int]
firstTenEvens = take 10 (filter even naturals)
-- Result: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
-- The infinite list is fine because Haskell only computes what 'take 10' demands

-- Side effects must go in the IO monad
main :: IO ()
main = do
  putStrLn ("5! = " ++ show (factorial 5))  -- 'do' notation sequences IO actions
  print firstTenEvens
```

### The monad intuition (no math required)

Think of a monad as a **context wrapper**. `IO Int` means "an Int that requires doing something in the real world to produce." The type system forces you to acknowledge this — you can't accidentally mix pure computation with side effects.

```
Pure function:    Int -> Int           (always safe, always predictable)
Effectful action: Int -> IO Int        (does something — the type tells you)
```

### Key insight
Haskell's purity seems restrictive but it gives you guarantees no other language can: if a function's type says `Int -> Int`, it truly has no side effects. This makes refactoring, testing, and reasoning about code dramatically easier — at the cost of a steeper learning curve.

---

## Erlang — The Concurrent One

**Built for telecom systems that must never go down.** Erlang doesn't care about type theory elegance — it cares about millions of simultaneous connections and zero downtime.

### History & Lore

While academics in Edinburgh, Glasgow, and Portland were debating type theory and purity, **Joe Armstrong** at **Ericsson's Computer Science Laboratory** in Stockholm had a very different problem: telephone switches.

In **1986**, Ericsson was the world's largest telecom equipment maker. Their switches handled millions of simultaneous phone calls, and the requirements were brutal:
- **99.999% uptime** ("five nines" — about 5 minutes of downtime per year) was the telecom industry standard
- Software updates **without taking the system offline** (you can't tell a country "no phone calls for 10 minutes while we deploy")
- Handling **millions of concurrent connections**, each independent

Armstrong's team tried existing languages — C was too crash-prone, Prolog was too slow. So Armstrong started extending **Prolog** (a logic programming language) with concurrency and fault-tolerance features. Erlang literally began as Prolog macros. This is why Erlang's syntax looks a bit like Prolog — the pattern matching, the period-terminated clauses, the `fun/end` blocks. It kept Prolog's best ideas and ditched the rest.

The name "**Erlang**" has a delicious double meaning:
1. **Ericsson Language** — the corporate origin
2. **Agner Krarup Erlang** (1878–1929) — a Danish mathematician who founded **telecom queuing theory** (the math behind "how many phone lines does a city need?"). The **erlang** is still a standard unit for measuring telephone traffic. Naming the language after him was a wink to telecom insiders.

Erlang went into production inside Ericsson in the early 1990s, powering the **AXD 301 ATM switch** — deployed by British Telecom, this system reportedly achieved **99.9999999% uptime** ("nine nines"), far exceeding the five-nines requirement. But then came the twist.

In **1998, Ericsson's management banned Erlang**. The reasoning: they wanted to use "industry standard" languages like C++ and Java. They didn't want to depend on a niche language they'd have to maintain. Armstrong and his colleagues were devastated — and in December 1998, during or shortly after the ban, they **open-sourced Erlang**. Armstrong himself left Ericsson.

The ban backfired spectacularly. The open-source community adopted Erlang for exactly the properties Ericsson management didn't appreciate. Ericsson quietly started using it again. Armstrong returned.

The ultimate validation came with **WhatsApp**. In **2014**, WhatsApp had only about **35 engineers** (roughly 55 total employees) serving **450 million users** — and their backend was Erlang. The "let it crash" philosophy, the lightweight processes, the hot code swapping — all the features Armstrong built for phone switches turned out to be exactly what you need for a messaging app at planetary scale. Facebook acquired WhatsApp for **$19 billion**, making it the most expensive Erlang application in history.

Joe Armstrong passed away in **April 2019**, but Erlang lives on through the **BEAM VM** — the virtual machine that also runs **Elixir** (a modern language created by José Valim in 2011 that brought Ruby-like syntax to Erlang's runtime). Armstrong was supportive of Elixir, seeing it as a way to bring his ideas to a new generation.

### What makes it special
- **Lightweight processes** — spawn millions of them (not OS threads, much lighter). Each has its own memory, communicates only by message passing
- **Actor model** — processes are independent actors that send messages to each other. No shared memory = no locks = no deadlocks
- **"Let it crash"** — instead of defensive error handling everywhere, let processes crash and have supervisors restart them. Sounds crazy, works brilliantly
- **Hot code swapping** — update running code without stopping the system
- **OTP (Open Telecom Platform)** — battle-tested libraries for building fault-tolerant systems (supervisors, gen_servers, etc.)

### Code example

```erlang
%% Factorial — pattern matching with guards
factorial(0) -> 1;                          %% base case
factorial(N) when N > 0 -> N * factorial(N - 1).  %% recursive case

%% Filter even numbers
evens(List) ->
    lists:filter(fun(X) -> X rem 2 =:= 0 end, List).
    %% fun(X) -> ... end is a lambda

%% Concurrency: spawn a process, send it a message
start() ->
    %% Spawn a new lightweight process running the 'loop' function
    Pid = spawn(fun loop/0),

    %% Send it a message (the ! operator sends messages)
    Pid ! {greet, "Hello from another process"},

    ok.

loop() ->
    receive                              %% wait for a message
        {greet, Msg} ->
            io:format("Got message: ~s~n", [Msg]),
            loop()                       %% tail-recursive: loop forever
    end.
```

### The "let it crash" philosophy

In most languages:
```
try {
    riskyOperation();
} catch (Exception e) {
    // 50 lines of recovery code you hope is correct
}
```

In Erlang:
```
Just let it crash. The supervisor notices, restarts the process
with clean state, and the rest of the system is unaffected.
```

This works because processes are **isolated** — one crashing process can't corrupt another's memory. WhatsApp served hundreds of millions of users with ~35 engineers partly because Erlang handles failure so gracefully.

### Key insight
Erlang teaches you that **fault tolerance matters more than fault prevention**. Instead of trying to handle every possible error, build systems that recover automatically. This philosophy has influenced Elixir, Akka (Scala/Java), and distributed systems design broadly.

---

## Side-by-Side: The Same Task in Three Languages

**Task**: Sum a list of numbers using recursion.

```ocaml
(* OCaml — strict, pattern matching on the list *)
let rec sum = function
  | []     -> 0              (* empty list → 0 *)
  | x :: rest -> x + sum rest  (* head :: tail pattern *)
```

```haskell
-- Haskell — same idea, lazy, type signature optional
sum :: [Int] -> Int
sum []     = 0
sum (x:xs) = x + sum xs
```

```erlang
%% Erlang — no static types, semicolons separate clauses
sum([])     -> 0;
sum([X|Rest]) -> X + sum(Rest).
```

Notice how similar the structure is — all three use **pattern matching** to destructure the list into head and tail. This is a hallmark of functional programming.

---

## When to Pick Which

| You want to... | Pick | Why |
|---|---|---|
| Learn FP fundamentals with a gentle on-ramp | **OCaml** | Strict evaluation is intuitive, escape hatches available, fast compiler feedback |
| Go deep on type theory and pure FP | **Haskell** | The best language for understanding monads, type classes, and mathematical FP |
| Build distributed / fault-tolerant systems | **Erlang** (or **Elixir**) | Nothing else handles concurrency and failure recovery this naturally |
| Get a job in quantitative finance | **OCaml** | Jane Street (largest OCaml shop) actively hires and teaches it |
| Explore a modern take on Erlang | **Elixir** | Same BEAM VM and concurrency model, but with nicer syntax and better tooling |
| Write Haskell but with more pragmatism | **OCaml** or **F#** | Same ML-family DNA, less ceremony around purity |

---

## What Each Language Teaches You

- **OCaml** → Strong static types catch bugs before runtime. Type inference means safety doesn't require verbosity.
- **Haskell** → Separating pure computation from effects makes programs easier to reason about. Laziness changes how you think about data.
- **Erlang** → Concurrency and fault tolerance are system-level concerns, not afterthoughts. The actor model is a fundamentally different way to think about state.

Even if you never use these languages in production, studying them changes how you write Python, Java, or JavaScript — you start using immutable data, composing small functions, and thinking about where side effects belong.

---

## Questions

- How do monads in Haskell compare to effect systems in other languages (e.g., algebraic effects)?
- What is Elixir's relationship to Erlang, and when would you pick one over the other?
- How does OCaml's module system (functors) compare to Haskell's type classes?
- What does "lazy evaluation" actually look like in memory — how does Haskell represent unevaluated expressions (thunks)?
