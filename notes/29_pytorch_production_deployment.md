# PyTorch Production Deployment

Eager mode vs TorchScript, JIT compiler optimizations, and the path from research prototype to production serving.

Context: Vincent Quenneville-Belair's talk "Scaling Deep Learning from Experiment to Production" — he led torchaudio and the optimizer module in PyTorch at Meta.

---

## Two modes of running a model

```
EXPERIMENT (eager mode)                 PRODUCTION (TorchScript mode)
════════════════════════                ═══════════════════════════════

Python interpreter runs                Model gets COMPILED into a
your code line by line.                standalone graph — no Python needed.

  model = MyNet()                        scripted = torch.jit.script(model)
  x = preprocess(img)                   scripted.save("model.pt")
  out = model(x)          ──convert──►
  print(out)                             # In production (C++ server):
  breakpoint()                           model = torch.jit.load("model.pt")
                                         out = model(input)

✓ Flexible                              ✓ Fast (no Python overhead)
✓ Debug with print/breakpoint           ✓ Portable (runs in C++, mobile)
✓ Dynamic control flow                  ✓ Optimizable (JIT can fuse ops)
✗ Slow (Python overhead per op)         ✗ Restricted Python subset
✗ Needs Python runtime                  ✗ Harder to debug
✗ Can't deploy to mobile/C++
```

---

## The Python bottleneck

In eager mode, every operation goes through Python:

```
Python: "hey C++, multiply these tensors"
  → C++ does the multiply, returns to Python
Python: "now add this bias"
  → C++ does the add, returns to Python
Python: "now apply ReLU"
  → C++ does ReLU, returns to Python
...

Each round-trip = overhead.
Python is the bottleneck, not the math.
```

TorchScript eliminates the round-trips entirely. The model graph runs in C++ from start to finish.

---

## Conversion methods

Two ways to convert a PyTorch model to TorchScript:

| Method | How it works | Handles if/else? | Trade-off |
|--------|-------------|-----------------|-----------|
| `torch.jit.trace(model, example_input)` | Runs the model once with example input, records every op | No — misses branches not taken during tracing | Easy to use, but can silently produce wrong results if model has data-dependent control flow |
| `torch.jit.script(model)` | Reads the Python source code, compiles it to TorchScript IR | Yes — analyzes all branches | Handles control flow, but only supports a restricted Python subset |

```python
# Tracing — easy but fragile
traced = torch.jit.trace(model, torch.randn(1, 3, 224, 224))

# Scripting — robust but restrictive
scripted = torch.jit.script(model)

# Either way, save for production
traced.save("model.pt")
```

---

## JIT compiler optimizations

Once the model is in TorchScript's intermediate representation (IR), the JIT compiler applies optimizations before running it. From Vincent's talk:

### Algebraic rewriting

Classic compiler transforms applied to tensor operations:

```python
# Before (what you wrote)
x = input * 1.0        # pointless multiply
y = x + 0              # pointless add
z = relu(relu(y))      # redundant activation

# After (JIT rewrites)
z = relu(input)         # same result, 3 ops eliminated
```

Also includes constant folding (precompute `3*4` → `12`), common subexpression elimination (compute shared sub-expressions once), dead code elimination (remove ops whose results are never used), and loop unrolling.

### Out-of-order execution

Re-ordering operations to reduce memory pressure and improve cache locality. If operation B doesn't depend on operation A, the JIT can reorder them to minimize peak memory usage. Same concept as CPU out-of-order execution — the dependency graph determines what can be rearranged.

### Kernel fusion

The biggest performance win for GPU workloads. Combines multiple operations into a single GPU kernel to avoid per-op memory round-trips:

```
WITHOUT FUSION:                      WITH FUSION:

  matmul                               matmul + bias + relu
    ↓ write to GPU memory              = ONE kernel launch
  bias_add                             = ONE memory round-trip
    ↓ read, compute, write back
  relu                                 For small ops, the memory
    ↓ read, compute, write back        round-trip dominates compute.
                                       Fusing them → 10-50x faster.
  = 3 kernel launches
  = 3 memory round-trips
```

Each GPU kernel launch has fixed overhead. Reading/writing GPU global memory takes ~100x longer than the actual arithmetic. Fusion eliminates intermediate reads and writes.

### Target-dependent code generation

Compiling the operation graph for specific hardware — generating different GPU kernels for an A100 vs a V100 vs a CPU. Related tools:

- **TVM** (Apache) — ML compiler framework, generates optimized kernels for any hardware target
- **Halide** — domain-specific language for image processing pipelines (MIT/Google)
- **Glow** (Meta) — graph-lowering compiler for neural networks
- **XLA** (Google) — Accelerated Linear Algebra compiler, used by JAX and TensorFlow

### Runtime: no GIL

Python's Global Interpreter Lock (GIL) means only one thread can execute Python bytecode at a time. TorchScript runs entirely in C++, so there's no GIL — true multi-threaded parallelism is possible. "Fork and wait" means spawning parallel threads for independent subgraphs and joining them when results are needed.

---

## `torch.compile()` — the modern successor

PyTorch 2.0 (2023) introduced `torch.compile()`, which largely supersedes TorchScript for most use cases:

```python
# One line — no model changes needed
compiled_model = torch.compile(model)
```

Unlike TorchScript, `torch.compile()` works with arbitrary Python code (no restricted subset), uses the TorchDynamo frontend to capture the computation graph at runtime, and the Inductor backend to generate optimized Triton (GPU) or C++ (CPU) kernels.

| | TorchScript | torch.compile() |
|---|-------------|----------------|
| Python support | Restricted subset | Full Python |
| Conversion effort | Manual trace/script | One-line decorator |
| Graph capture | Static (ahead of time) | Dynamic (at runtime) |
| Optimization backend | TorchScript JIT | Inductor (Triton/C++) |
| Export to C++ | Yes (`.save()`) | Via `torch.export()` |
| Status | Maintenance mode | Active development |

The core idea Vincent taught remains the same regardless of which tool is used: **research in Python (flexible, debuggable), deploy without Python (fast, portable).**

---

## Takeaway

- PyTorch eager mode executes ops through Python one at a time — flexible but slow due to Python overhead
- TorchScript compiles the model into a standalone C++ graph — no Python needed at runtime
- The JIT applies algebraic rewriting, kernel fusion (biggest GPU win), out-of-order execution, and hardware-specific code generation
- `torch.compile()` (PyTorch 2.0+) supersedes TorchScript with better Python support and the Inductor backend
- The fundamental principle: prototype in Python, deploy without Python

---

*See also:* [researchers/vincent_quenneville_belair.md](researchers/vincent_quenneville_belair.md)
