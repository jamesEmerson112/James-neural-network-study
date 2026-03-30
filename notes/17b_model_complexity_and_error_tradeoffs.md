# Model Complexity & the Three Errors

> **As a neural network grows in complexity, three types of error pull in different directions. Understanding them explains why architecture choices matter.**

---

## Part 1: The Three Errors

Every neural network's total error can be decomposed into three independent sources:

```
Total Error = Modeling Error + Estimation Error + Optimization Error
```

### Modeling Error (Approximation Error / Bias)

**What it is:** The gap between the true function and the best function your model *could possibly* learn — even with infinite data and perfect training.

**Analogy:** You're trying to fit a curve through data points. If your model is a straight line (linear regression), the modeling error is the gap between that best-fit line and the true curvy relationship. No amount of data or training will make a straight line curve.

```
True function: ~~~∿~~~     Your model class: ——————

                ∿
               ∿ ∿         The straight line can never
  ————————————∿———∿————    capture the wiggles.
             ∿     ∿       That gap = modeling error.
            ∿       ∿
```

**What controls it:** Model capacity — number of layers, neurons, type of architecture. A deeper, wider network has a richer hypothesis space and can represent more functions.

### Estimation Error (Variance)

**What it is:** The gap between the best your model *could* learn (with infinite data) and what it *actually* learns from your finite training set.

**Analogy:** You have a very flexible model (say, a polynomial of degree 100) but only 50 data points. The model has enough capacity to memorize all 50 points perfectly — including their noise. If you drew a different 50 points from the same distribution, you'd get a completely different fit. That instability = estimation error.

```
Dataset A:  ○  ○  ○  ○  ○       Dataset B:  ○  ○  ○  ○  ○
Fit A:      ∿∿∿∿∿∿∿∿∿∿∿∿       Fit B:      ∿∿∿∿∿∿∿∿∿∿∿∿
              (wiggly)                        (different wiggly)

Same model, different data → wildly different results = high estimation error
```

**What controls it:** The ratio of model parameters to training data. More parameters with same data → higher estimation error. More data with same model → lower estimation error.

### Optimization Error

**What it is:** The gap between the best your model could learn from this data and what your training algorithm (SGD, Adam, etc.) actually finds.

**Analogy:** The loss landscape is a mountain range. The global minimum is the deepest valley. But SGD might get stuck in a local minimum or saddle point, or stop early, or overshoot. The gap between where SGD lands and the true best solution = optimization error.

```
Loss landscape:

    ╲    ╱╲      ╱╲
     ╲  ╱  ╲    ╱  ╲
      ╲╱    ╲  ╱    ╲
       ↑     ╲╱      ╲╱
    SGD lands   ↑
    here     Global
             minimum

    The gap between these two = optimization error
```

**What controls it:** Model architecture (skip connections help), optimizer choice, learning rate schedule, initialization. More parameters can actually help — overparameterized models have smoother loss landscapes with more paths to good solutions.

---

## Part 2: As Complexity Grows

This is the exam question: **what happens to each error as you make the network more complex (deeper, wider, more parameters)?**

| Error Type | Trend | Why | Intuition |
|-----------|-------|-----|-----------|
| **Modeling error** | **Decreases** | More capacity = richer hypothesis space = can represent more functions | A 100-layer network can approximate almost anything; a single neuron can't |
| **Estimation error** | **Increases** | More parameters to estimate from the same finite data = more room to overfit | A 1M-parameter model memorizes 10K examples easily; a 100-parameter model can't |
| **Optimization error** | **Decreases** | More parameters = smoother loss landscape = more paths to good minima | Overparameterized networks are actually *easier* to optimize (the lottery ticket hypothesis) |

```
                ↑ Error
                |
Estimation      |            ╱╱╱
Error           |          ╱╱
                |        ╱╱
                |      ╱╱
                |    ╱╱
                |  ╱╱
                |╱╱
                |
                |╲╲
                |  ╲╲
Modeling        |    ╲╲
Error           |      ╲╲
                |        ╲╲
                |          ╲╲╲
                +————————————————→ Model Complexity
                simple          complex
```

**The tradeoff:** You can't minimize all three simultaneously. Making the network more complex reduces modeling error and optimization error, but increases estimation error. The art of architecture design is finding the sweet spot — or using regularization (dropout, weight decay, data augmentation) to push the estimation error curve down.

---

## Part 3: Applied to Vision Architectures

These three errors explain *why* different CNN architectures exist and *why* architecture choices matter.

### CNN Inductive Biases Reduce Modeling Error Efficiently

A CNN bakes in assumptions about images:
- **Local connectivity** — pixels near each other are related (spatial locality)
- **Translation equivariance** — a cat in the top-left should be detected the same way as a cat in the bottom-right (weight sharing)
- **Hierarchical features** — edges → textures → parts → objects (pooling + depth)

These biases mean a CNN can represent image functions with **far fewer parameters** than a fully-connected network — achieving low modeling error without blowing up estimation error.

```
Fully connected on 224×224 image:  224 × 224 × 3 × 1000 = ~150M params (first layer alone!)
CNN with 3×3 filters:              3 × 3 × 3 × 64 = 1,728 params (first layer)

Same modeling power for images, vastly different estimation error.
```

### Why ViTs Need More Data (Estimation Error)

From your [Note 16](16_do_vision_transformers_see_like_cnns.md): Vision Transformers need ~300M images to match ResNets trained on ~1.2M images.

**Why?** ViTs have fewer inductive biases — no built-in locality, no translation equivariance. They must *learn* these properties from data. This means:
- Same modeling error (both can represent the same functions eventually)
- Much higher estimation error (ViTs need more data to learn what CNNs get for free from their architecture)

| Architecture | Inductive Bias | Params for ImageNet | Data Needed | Why |
|-------------|---------------|--------------------|----|-----|
| ResNet-50 | Strong (local, hierarchical) | 25M | 1.2M images | Biases match image structure |
| ViT-Large | Weak (global attention) | 307M | 300M images | Must learn locality from scratch |

### Why ResNets Train Easier (Optimization Error)

Plain deep CNNs (e.g., VGG-19 with 19 layers) suffer from vanishing/exploding gradients — deeper doesn't always mean better because **optimization error increases** with depth.

ResNets (He et al., 2015) added **skip connections**:

```
Plain network:          ResNet:

x → [Conv] → [Conv] → y     x → [Conv] → [Conv] → (+) → y
                                                     ↑
                                                     x (skip)

If the layers learn nothing useful, a plain network outputs garbage.
A ResNet outputs x + garbage ≈ x. The skip connection provides a safe fallback.
```

Skip connections create **shortcuts in the loss landscape** — gradients can flow directly through the skip path, making deep networks as easy to optimize as shallow ones. This dramatically reduces optimization error, allowing ResNets to go 152 layers deep where VGG-19 already struggled.

### The Architecture Progression as Error Management

| Architecture | Year | Key Innovation | Which Error It Attacked |
|-------------|------|---------------|------------------------|
| LeNet-5 | 1989 | Convolutions + pooling | Modeling (proved CNNs work for digits) |
| AlexNet | 2012 | Depth + GPU + dropout | Estimation (dropout as regularization) |
| VGG | 2014 | Uniform 3×3 filters, deeper | Modeling (more depth = more capacity) |
| GoogLeNet | 2014 | Inception modules (multi-scale) | Modeling (capture features at multiple scales) |
| ResNet | 2015 | Skip connections | Optimization (train 152 layers without degradation) |
| DenseNet | 2017 | Dense skip connections | Optimization + Estimation (feature reuse = fewer params) |
| EfficientNet | 2019 | Compound scaling (depth × width × resolution) | All three (balanced scaling avoids waste) |
| ViT | 2020 | Pure attention, no convolution | Modeling (global context from layer 1) — but pays in estimation error |

---

## Part 4: The Bias-Variance Tradeoff

The three errors map onto the classic bias-variance tradeoff:

- **Bias** ≈ Modeling error — systematic under-representation of the true function
- **Variance** ≈ Estimation error — sensitivity to the specific training set
- **Irreducible error** — noise in the data itself (can't be reduced by any model)

```
            ↑ Error
            |
            |  ╲                    ╱
            |    ╲  Total Error   ╱
            |      ╲            ╱
            |        ╲        ╱
            |          ╲    ╱
            |            ╲╱  ← Sweet spot
            |           ╱ ╲
            |     Bias ╱    ╲ Variance
            |        ╱       ╲
            |      ╱           ╲
            +————————————────────→ Complexity
            underfitting      overfitting
```

### The Double Descent Surprise

Classical theory says: after the sweet spot, test error only goes up (overfitting).

Modern deep learning discovered **double descent** (Belkin et al., 2019): if you keep making the model bigger *past* the point where it perfectly fits the training data (the interpolation threshold), test error starts going **down** again.

```
            ↑ Test Error
            |
            |     ╱╲
            |    ╱  ╲
            |   ╱    ╲         Classical theory
            |  ╱      ╲        stops here
            | ╱        ╲
            |╱     ╱╲   ╲
            |     ╱  ╲   ╲
            |    ╱    ╲   ╲
            |   ╱      ╲   ╲╲╲╲  ← keeps going down!
            |  ╱        ╲
            + ╱──────────╲──────→ Model size
            under-    interpolation   over-
            parameterized  threshold  parameterized
```

This explains why modern networks (GPT with 175B parameters, ViT-Large with 307M) can be massively overparameterized yet still generalize well — they're operating in the second descent regime. The classical bias-variance U-curve is real but incomplete.

---

## Part 5: Practical Implications

### How to reduce each error in practice:

| Error | Reduction Strategy | Example |
|-------|-------------------|---------|
| **Modeling** | Increase capacity, use better architecture | ResNet-152 instead of ResNet-18 |
| **Estimation** | More data, regularization, data augmentation | Dropout, weight decay, random crops/flips |
| **Optimization** | Better optimizer, skip connections, learning rate scheduling | Adam instead of SGD, warm-up + cosine decay |

### The modern recipe (why it works):

```
1. Make the model very large        → modeling error ≈ 0
2. Use massive data + augmentation  → estimation error stays manageable
3. Use skip connections + Adam      → optimization error ≈ 0
4. Apply regularization (dropout, weight decay) → keeps estimation error in check

Result: all three errors are small simultaneously
```

This is exactly what AlexNet (2012) → ResNet (2015) → EfficientNet (2019) → ViT (2020) each improved upon, step by step.

---

## Part 6: Quantization — Trading Precision for Memory

Once a model is trained, the weights just sit there doing multiplications. Do they really need 32 bits of precision? Turns out: mostly no.

### Data types in neural networks

| Data Type | Bits | Bytes | Use case |
|-----------|------|-------|----------|
| **float32** | 32 | 4 | Standard training (the default) |
| **float16 / bfloat16** | 16 | 2 | Mixed-precision training, saves 2x memory |
| **int8** | 8 | 1 | Inference — effectively lossless |
| **int4** | 4 | 0.5 | Inference on consumer hardware |

### Real benchmarks: Llama 2 7B (WikiText-2 perplexity, lower = better)

| Precision | Perplexity | Loss vs FP16 |
|-----------|-----------|--------------|
| FP16 | 5.47 | — |
| INT8 | 5.48 | +0.01 (basically nothing) |
| INT4 (AWQ) | 5.60 | +0.13 (tiny) |
| INT3 | 6.29 | +0.82 (now it hurts) |

### The killer comparison: Big model quantized vs small model full precision

| Model | MMLU | HellaSwag | ARC | TruthfulQA |
|-------|------|-----------|-----|------------|
| **Llama 2 70B at INT4** (~40GB) | 69.5% | 85.2% | 64.5% | 54.0% |
| **Llama 2 13B at FP16** (~26GB) | 55.7% | 80.7% | 59.0% | 37.4% |

The 70B quantized to INT4 destroys the 13B at full precision on every benchmark — using only ~50% more memory. **Bigger model + fewer bits > smaller model + more bits.**

### Why it works: Bigger models survive quantization better

| Model Size | INT4 quality loss |
|-----------|------------------|
| 1-3B | 2-5% (noticeable) |
| 7-8B | 0.5-2% (minor) |
| 13B | <1% (minimal) |
| 70B+ | <0.5% (basically free) |

Larger models have more redundant parameters — they tolerate compression the way a 4K photo survives downscaling to 1080p, but a 480p photo looks terrible at 240p.

### Why training and inference differ

```
Training:   Need float16+ because gradients are tiny numbers (0.00001)
            and small rounding errors accumulate over millions of steps.

Inference:  Weights are fixed, just doing multiplications.
            Whether a weight is 0.2471893 or 0.247 barely matters.
```

That's why training stays at float16/bfloat16 but inference can go to int4.

### The practical memory calculation

For our CNN example (2,157,577 activation values):
```
float32:  2,157,577 × 4 bytes  = 8.2 MB
float16:  2,157,577 × 2 bytes  = 4.1 MB
int8:     2,157,577 × 1 byte   = 2.1 MB
int4:     2,157,577 × 0.5 byte = 1.0 MB
```

For a real model — Llama 7B (7 billion parameters):
```
float32:  7B × 4 bytes = 28 GB  — needs a server GPU (A100)
float16:  7B × 2 bytes = 14 GB  — fits on RTX 4090
int8:     7B × 1 byte  =  7 GB  — fits on RTX 3080
int4:     7B × 0.5     =  3.5 GB — fits on RTX 3060 or a Mac M1
```

---

## Connect the Dots

- **CNN architectures and tasks** — the full zoo of vision architectures, each managing these tradeoffs differently. See: [CNN & Vision Tasks](17_cnn_vision_tasks.md)

- **ViT vs CNN representations** — empirical evidence for why ViTs have higher estimation error (need more data to learn what CNNs get from inductive biases). See: [Do Vision Transformers See Like CNNs?](16_do_vision_transformers_see_like_cnns.md)

- **Vanishing gradients** — the core optimization error problem that skip connections solved. See: [Vanishing Gradient and Tanh](13_vanishing_gradient_and_tanh.md)

- **Backprop** — the algorithm that powers all optimization (minimizing training loss). Optimization error is the gap between what backprop finds and the true minimum. See: [Backprop: The Wiggle Ratio](09_backprop_the_wiggle_ratio.md)

- **Training stages** — practical view of overfitting (estimation error) during training. See: [Training Stages](James_notes_deepLearning_team_project/07_training-stages.md)

- **Generative models** — VAEs and GANs face the same three errors. Mode collapse in GANs = optimization error. Blurriness in VAEs = modeling error of the Gaussian assumption. See: [Generative Models Taxonomy](24_generative_models_taxonomy.md)
