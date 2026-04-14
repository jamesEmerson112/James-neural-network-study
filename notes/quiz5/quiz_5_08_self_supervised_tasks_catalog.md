# 08. Self-Supervised Tasks — Catalog with Inputs, Outputs, and Losses

## What this note unpacks

Quiz topic 8 asks: "**What are the types of self-supervised tasks, and what are their inputs, outputs, and losses?**" This note is the catalog. For each canonical SSL task, we write down:

- **Input**: what goes in
- **Output / target**: what the model tries to predict
- **Loss**: the explicit loss function
- **Example**: a concrete instance

The six task families covered: **masked language modeling**, **autoregressive prediction**, **contrastive learning**, **self-distillation**, **masked image modeling**, **denoising autoencoders**, plus classical **pretext tasks** (rotation, jigsaw, colorization).

See [quiz_5_07_learning_paradigms_comparison.md](quiz_5_07_learning_paradigms_comparison.md) for the big-picture placement of self-supervised learning among other paradigms.

---

## 1. Masked language modeling (MLM) — BERT

**The idea:** take a sentence, randomly mask ~15% of the tokens, and train the model to predict the original tokens at the masked positions.

```
  input:   "The [MASK] sat on the [MASK]."
  target:  "cat" at position 1, "mat" at position 5
```

**Input**: a sequence of tokens with ~15% replaced by a special `[MASK]` token (following BERT: of the 15%, 80% replaced with `[MASK]`, 10% replaced with a random token, 10% left unchanged — this is the "BERT masking trick" that prevents the model from only learning to handle `[MASK]` tokens).

**Output**: a probability distribution over the vocabulary at each masked position.

**Loss**: cross-entropy on the masked positions only. If $\mathcal{M}$ is the set of masked positions and $y_i$ is the original token at position $i$:

$$
\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log p_\theta(y_i \mid x_{\setminus \mathcal{M}})
$$

Reading: average negative log-probability that the model assigns to the correct token at each masked position, conditioned on the rest of the sentence.

**Why it works:** to predict masked tokens accurately, the model must understand syntax, semantics, and world knowledge. A sentence like "The [MASK] that the engineer designed __ ran on electricity" requires multi-sentence reasoning to fill in "machine" or "train." Solving MLM forces the model to learn representations that generalize.

**Who uses it:** BERT (2018), RoBERTa, ALBERT, DeBERTa, and most encoder-only language models. See [../20_bert.md](../20_bert.md).

---

## 2. Autoregressive prediction (next-token) — GPT

**The idea:** given a sequence of tokens, predict the next one. Repeat at every position.

```
  input:   "The cat sat on the"
  target:  "mat"
```

**Input**: a sequence of tokens $x_1, x_2, \ldots, x_{t-1}$.

**Output**: probability distribution over the vocabulary for the next token $x_t$.

**Loss**: cross-entropy on the next token, applied at every position (i.e., the loss at position $t$ uses tokens $1 \ldots t-1$ as context):

$$
\mathcal{L}_{\text{AR}} = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

This is exactly the **log-likelihood of the sequence** factored via the chain rule. Maximizing this loss is equivalent to maximum likelihood estimation of the sequence distribution.

**Masked attention:** autoregressive models use **causal masking** in self-attention so that position $t$ can only attend to positions $1, \ldots, t-1$. This ensures the model doesn't cheat by looking at the answer.

**Why it works:** predicting the next token well requires modeling the entire distribution over plausible continuations, which in turn requires deep knowledge of syntax, semantics, logical structure, world facts, and even mathematical reasoning. GPT-3 and later LLMs showed that scaling this simple objective produces surprisingly general intelligence.

**Who uses it:** GPT-1/2/3/4, LLaMA, Claude, Gemini, PaLM, Mistral — every decoder-only language model.

---

## 3. Contrastive learning — SimCLR, MoCo

**The idea:** take two augmented views of the same image (or data point). Pull them together in embedding space, push apart from unrelated images.

```
  input:   image x
           │
           ├── augmentation a₁ → view x_a  (e.g., crop + color jitter)
           └── augmentation a₂ → view x_b  (e.g., different crop + flip)

  target:  embed(x_a) should be close to embed(x_b),
           far from embed(other images)
```

**Input**: a batch of $N$ images. For each image, generate two augmented views. The batch now has $2N$ augmented examples forming $N$ "positive pairs" (both views of the same image).

**Output**: an embedding $z_i \in \mathbb{R}^d$ for each view. Typically from a backbone encoder (ResNet, ViT) followed by a small projection head (MLP).

**Loss**: the **NT-Xent** loss (Normalized Temperature-scaled Cross-Entropy), introduced by SimCLR (Chen et al. 2020). For a positive pair $(i, j)$:

$$
\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

where:
- $\text{sim}(u, v) = u^\top v / (\|u\| \|v\|)$ is cosine similarity
- $\tau$ is the temperature (typically 0.1–0.5)
- The denominator sums over **all** other augmented views in the batch (positives and negatives)
- The loss is symmetric: you compute $\mathcal{L}_{i,j}$ and $\mathcal{L}_{j,i}$ and average

**Reading**: "maximize the softmax-like probability that the positive view $j$ is the correct match for $i$, among all other views in the batch." This is exactly a softmax cross-entropy with the positive pair as the target class.

**Why it works:** two augmented views of the same image share semantic content (it's still a cat after cropping and color jitter), but differ in low-level pixel statistics. Pulling them together forces the model to learn **invariant semantic features** rather than superficial pixel patterns.

**Who uses it:** SimCLR (2020), MoCo (2020), MoCo v2, SwAV, BYOL (no negatives variant), and CLIP (cross-modal contrastive).

**Critical detail:** large batch sizes matter. SimCLR used batch size 4096 because you need lots of negatives to make the contrastive loss informative. MoCo avoids this by maintaining a "momentum queue" of past embeddings as negatives, which is more memory-efficient.

---

## 4. Self-distillation — DINO

**The idea:** two networks (student and teacher) see different augmentations of the same image. The student is trained to match the teacher's output distribution. The teacher is an exponential moving average of the student — no labels anywhere.

```
  image x
    ├── aug₁ → teacher → distribution p_teacher
    └── aug₂ → student → distribution p_student

  target: KL(p_teacher || p_student) minimized
  teacher parameters: exponential moving average of student parameters
```

**Input**: augmented views of images (like SimCLR).

**Output**: a probability distribution (softmax over a fixed-size output) from both student and teacher.

**Loss**: cross-entropy between the student distribution and the teacher distribution (teacher is the target, no gradients flow through teacher):

$$
\mathcal{L}_{\text{DINO}} = -\sum_k p_{\text{teacher}}(k) \log p_{\text{student}}(k)
$$

with **centering + sharpening** to prevent collapse (the teacher's output is centered by subtracting a running mean; the student's output uses a lower temperature than the teacher's).

**Why it works:** even though there are no labels, the consistency constraint plus asymmetric temperature forces the student to learn features that are **stable under augmentation**. The surprising result from DINO: these features spontaneously learn to segment objects and attend to semantic parts — you can visualize the attention maps and they look like object segmentation without ever being trained to do that.

**Who uses it:** DINO (2021), DINOv2 (2023), iBOT.

---

## 5. Masked image modeling (MIM) — MAE

**The idea:** mask a large fraction of image patches (typically 75%), reconstruct the pixel values at the masked positions.

```
  input:  image split into 16x16 patches, 75% of patches replaced with [MASK]
  target: reconstruct the RGB values at the masked patches
```

**Input**: an image split into patches (e.g., 14×14 = 196 patches for a 224×224 image with 16×16 patch size). A random 75% of patches are masked (replaced with a learned `[MASK]` token).

**Output**: predicted pixel values (or patch embeddings) at the masked positions.

**Loss**: mean squared error on the masked patches only:

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \| \hat{x}_i - x_i \|^2
$$

where $\hat{x}_i$ is the model's prediction for masked patch $i$ and $x_i$ is the true pixel values.

**Clever architectural trick (MAE 2022):** the encoder only sees the 25% of unmasked patches, not the full image. The decoder is a separate smaller transformer that receives the encoder's representations plus mask tokens and reconstructs the missing pixels. This makes pretraining 3–4× faster because the encoder handles far fewer tokens.

**Why it works:** reconstructing 75% of an image from 25% of the pixels requires learning holistic image understanding. Simple interpolation doesn't work at this mask ratio — the model needs to understand objects, textures, and context.

**Who uses it:** MAE (Masked Autoencoders, He et al. 2022 at Facebook AI), BEiT, SimMIM. MAE is the vision analog of BERT.

---

## 6. Denoising autoencoder

**The idea:** corrupt the input with noise, train the model to reconstruct the clean version.

```
  input:  clean image x
          │
          ▼
        add noise (Gaussian, dropout, salt-pepper, masking, etc.)
          │
          ▼
        corrupted x̃
          │
          ▼
        encoder + decoder
          │
          ▼
        reconstruction x̂

  target: x̂ should equal the original clean x
```

**Input**: a corrupted version $\tilde{x}$ of a clean input $x$.

**Output**: a reconstruction $\hat{x}$.

**Loss**: mean squared error (for continuous data) or cross-entropy (for discrete data):

$$
\mathcal{L}_{\text{DAE}} = \| \hat{x} - x \|^2
$$

**Why it works:** to reconstruct clean data from corrupted input, the model must learn the underlying data distribution — specifically, it must learn to project corrupted inputs back onto the data manifold. This was the conceptual breakthrough of Vincent et al. (2008): forcing the model to denoise is equivalent to learning the data's structure.

**Who uses it:** the original denoising autoencoder (Vincent, Larochelle, Bengio 2008), stacked denoising autoencoders (2010), and — critically — **DDPM** (2020), which can be viewed as a denoising autoencoder trained at every noise level simultaneously. See [../ddpm/06_reverse_process.md](../ddpm/06_reverse_process.md).

**Lineage:** denoising autoencoder → BERT (masking is a form of noise) → MAE (masking applied to images) → DDPM (continuous noise schedule, multiple noise levels). All of them are "corrupt and reconstruct."

---

## 7. Classical pretext tasks (mostly superseded)

Before contrastive and masked methods took over, a family of hand-designed self-supervised tasks dominated. These are called **pretext tasks** because they're designed as a pretext for learning useful features, without being inherently useful themselves.

### Rotation prediction

**Task:** rotate the image by 0°, 90°, 180°, or 270°. Predict which rotation was applied.

- **Input**: rotated image
- **Output**: one of 4 classes (0, 90, 180, 270)
- **Loss**: cross-entropy (4-way classification)

**Why it works:** to predict the rotation, the model must recognize the canonical orientation of objects — which requires understanding what objects are. Proposed by Gidaris et al. (2018).

### Jigsaw puzzle

**Task:** split the image into $3 \times 3 = 9$ patches, shuffle them, predict the permutation.

- **Input**: shuffled patches
- **Output**: the permutation (classification over a fixed set of permutations, e.g., 100 possibilities)
- **Loss**: cross-entropy

Proposed by Noroozi & Favaro (2016).

### Colorization

**Task:** convert an image to grayscale, train the model to predict the color channels.

- **Input**: grayscale image
- **Output**: the two chrominance channels (ab in Lab color space)
- **Loss**: cross-entropy over quantized color bins, or regression loss

Proposed by Zhang, Isola, Efros (2016).

**Why these are mostly gone:** contrastive learning (SimCLR/MoCo 2020) and masked image modeling (MAE 2022) outperform these pretext tasks substantially. The classical tasks taught the model *something*, but not as much as contrastive and masked methods do. Pretext tasks are mostly historical, but they still come up on quizzes as examples of "creative SSL task design."

---

## Master comparison table

| Task | Input | Output | Loss | Canonical paper |
|---|---|---|---|---|
| **MLM** | Sentence with 15% tokens masked | Original tokens at masked positions | Cross-entropy on masked positions | BERT (2018) |
| **Autoregressive** | Token sequence $x_{<t}$ | Next token $x_t$ | Cross-entropy on next token | GPT (2018) |
| **Contrastive** | Two augmentations of same image + others | Embeddings that pull positive pair together | NT-Xent (softmax cross-entropy on similarity) | SimCLR (2020) |
| **Self-distillation** | Two augmentations, student + teacher | Student matches teacher distribution | Cross-entropy between student and teacher | DINO (2021) |
| **Masked image modeling** | Image with 75% patches masked | Pixel values at masked patches | MSE on masked patches | MAE (2022) |
| **Denoising autoencoder** | Input + noise | Clean input | MSE (or CE for discrete) | Vincent et al. (2008) |
| **Rotation prediction** | Rotated image | Rotation class (0/90/180/270) | Cross-entropy (4-way) | RotNet (2018) |
| **Jigsaw** | Shuffled patches | Permutation | Cross-entropy (over permutations) | Noroozi & Favaro (2016) |
| **Colorization** | Grayscale image | Color channels | Cross-entropy over color bins | Zhang et al. (2016) |

---

## The unifying principle

**Every self-supervised task is "predict something about the data from something else about the data."** The machinery (input, target, loss, backprop) is borrowed from supervised learning, but the target is generated from the data itself, not from a human annotator.

Four common patterns:
1. **Mask and predict** — hide part of the input, predict the hidden part. BERT, MAE, denoising autoencoders, DDPM.
2. **Augment and match** — produce two related views of the same data, learn representations that make them match. SimCLR, MoCo, DINO.
3. **Ordering / transformation prediction** — apply a known transformation (rotation, shuffling), predict the transformation. Rotation prediction, jigsaw.
4. **Next-step prediction** — given history, predict the future. GPT, video prediction, time-series forecasting.

Modern SSL is dominated by patterns 1 and 2. Pattern 4 dominates language modeling specifically.

---

## Takeaway

- **Every SSL task has three parts**: input (some view of the data), output (some other view or prediction target), and loss (cross-entropy, MSE, or contrastive NT-Xent).
- **MLM**: mask tokens, predict them, cross-entropy. BERT.
- **Autoregressive**: predict next token, cross-entropy. GPT.
- **Contrastive NT-Xent**: pull augmented pairs together in embedding space, push others apart. SimCLR, MoCo.
- **Masked image modeling**: mask 75% of patches, reconstruct pixels with MSE. MAE.
- **Denoising autoencoder**: add noise, reconstruct clean. The ancestor of all mask-and-predict methods and of DDPM.
- **Pretext tasks**: rotation, jigsaw, colorization — classical SSL, mostly superseded.

Next note: [quiz_5_09_gan_and_vae.md](quiz_5_09_gan_and_vae.md) — generative models: GAN and VAE training, loss functions, and the reparameterization trick.
