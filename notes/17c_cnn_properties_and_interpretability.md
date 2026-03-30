# CNN Properties & Interpretability

*Last updated: 2026-03-30*

---

## Rotation vs Translation Equivariance

CNNs get **translation equivariance** for free from weight sharing — the same filter slides everywhere, so a cat detected in the top-left triggers the same feature map response as a cat in the bottom-right.

```
Translation equivariance (built-in):

Input A:                    Input B (shifted right):
┌──────────┐                ┌──────────┐
│ 🐱       │                │       🐱 │
│          │                │          │
└──────────┘                └──────────┘
Feature map A:              Feature map B:
[peak here]                 [peak shifted right]

The peak MOVES with the object → equivariant.
```

But CNNs are **NOT rotation equivariant or invariant**. A rotated input produces a completely different feature map — standard filters don't "know" that a 45° edge is the same edge rotated:

```
Rotation (NOT built-in):

Input A:                    Input B (rotated 90°):
┌──────────┐                ┌──────────┐
│ 🐱       │                │ 🐱↻      │
│          │                │          │
└──────────┘                └──────────┘
Feature map A:              Feature map B:
[peak here]                 [DIFFERENT activation pattern]

The network might not even recognize it's the same object.
```

**How standard CNNs cope:** Data augmentation — train on random rotations so the network *learns* to handle them. This works but wastes capacity and isn't guaranteed.

**The principled fix — Group-Equivariant CNNs (G-CNNs):**

Cohen & Welling (2016) introduced G-CNNs that bake rotation equivariance into the architecture the same way standard CNNs bake in translation equivariance. Instead of sliding filters across positions only, G-CNN filters also rotate through a set of orientations.

```
Standard CNN filter:  slides across (x, y) positions         → translation equivariant
G-CNN filter:         slides across (x, y) AND rotates       → translation + rotation equivariant

Group theory connection:
- Translation equivariance uses the translation group
- Rotation equivariance uses rotation groups (e.g., 90° rotations = cyclic group C4)
- G-CNNs generalize to arbitrary symmetry groups
```

| Property | Standard CNN | G-CNN |
|---|---|---|
| Translation equivariant | Yes (weight sharing) | Yes |
| Rotation equivariant | No | Yes (group convolutions) |
| How rotation is handled | Data augmentation (learned) | Architecture (guaranteed) |
| Parameter cost | Lower | ~Same (filters are shared across rotations) |

---

## Guided Backpropagation

**What it does:** Produces a saliency map showing which input pixels the network "looked at" for its prediction.

**How it works:** Standard backpropagation passes gradients through ReLU as-is (zeroing only where the forward activation was negative). Guided backpropagation adds an extra gate — it also zeros out **negative gradients** during the backward pass:

```
Standard backprop through ReLU:
  Forward:  gate = (input > 0)         → zero where input was negative
  Backward: pass gradient × gate       → only masks by forward activation

Guided backprop through ReLU:
  Forward:  gate = (input > 0)         → same
  Backward: pass gradient × gate × (gradient > 0)
                                       → ALSO zero negative gradients

The extra (gradient > 0) mask means:
only positive influence flows backward → cleaner saliency maps
```

**Why zero negative gradients?** Negative gradients mean "this pixel made the prediction LESS confident." Guided backprop only keeps pixels that positively support the prediction, producing sharper, more interpretable maps.

**What you get:**

```
Input image:              Guided backprop saliency:
┌──────────────┐          ┌──────────────┐
│   🐱         │          │   ████       │  Bright pixels = "the network
│   on a       │    →     │              │   looked here to decide 'cat'"
│   couch      │          │              │
└──────────────┘          └──────────────┘

The saliency map highlights edges and textures of the cat,
roughly resembling a segmentation mask (though not precise).
```

**Guided Grad-CAM** combines guided backpropagation with Grad-CAM (class activation maps) for the best of both:

```
Guided backprop:  high-resolution but noisy (pixel-level detail)
Grad-CAM:         class-discriminative but coarse (spatial heatmap from final conv layer)

Guided Grad-CAM = element-wise multiply of both
                → high-resolution AND class-discriminative
```

---

## Transposed Convolution (Deconvolution)

Regular convolution shrinks spatial dimensions (encoder). Transposed convolution **grows** them back (decoder). Used in segmentation networks (U-Net, FCN) and generative models (GANs).

**The operation:** Each input value scales the entire kernel, placed at the corresponding output position. Overlapping regions are summed.

### Worked example from quiz

```
Input X = | 0  1 |     Kernel W = | 1  2 |
          | 2  3 |                | 3  4 |

Output is 3×3 (each 2×2 input position places a 2×2 kernel → output = (2-1) + 2 = 3)
```

**Step by step — each input element scales the full kernel:**

```
x(0,0) = 0:                    x(0,1) = 1:
0 × | 1  2 | = | 0  0 |       1 × | 1  2 | = | 1  2 |
    | 3  4 |   | 0  0 |           | 3  4 |   | 3  4 |
placed at (0,0)                placed at (0,1)

x(1,0) = 2:                    x(1,1) = 3:
2 × | 1  2 | = | 2  4 |       3 × | 1  2 | = | 3  6 |
    | 3  4 |   | 6  8 |           | 3  4 |   | 9 12 |
placed at (1,0)                placed at (1,1)
```

**Sum all contributions at each output position:**

```
Output position (0,0): 0                              = 0
Output position (0,1): 0 + 1                          = 1
Output position (0,2): 2                              = 2
Output position (1,0): 0 + 2                          = 2
Output position (1,1): 0 + 3 + 4 + 3                 = 10
Output position (1,2): 4 + 6                          = 10
Output position (2,0): 6                              = 6
Output position (2,1): 8 + 9                          = 17
Output position (2,2): 12                             = 12

Final output = |  0   1   2 |
               |  2  10  10 |
               |  6  17  12 |
```

**Key insight:** Transposed convolution is NOT the inverse of convolution. It's a learnable upsampling — the network learns how to "fill in" spatial detail during training.

```
Encoder (regular conv):     224×224 → 112×112 → 56×56 → 28×28
Decoder (transposed conv):  28×28 → 56×56 → 112×112 → 224×224

In segmentation: encoder extracts "what", decoder reconstructs "where"
```

| Upsampling method | Learnable? | Quality | Used in |
|---|---|---|---|
| Nearest neighbor | No | Blocky | Simple resizing |
| Bilinear interpolation | No | Smooth but generic | Feature pyramids |
| Transposed convolution | Yes | Best (learned) | U-Net, FCN, GANs |

**Checkerboard artifacts:** Transposed convolutions can produce grid-like artifacts when stride doesn't evenly divide kernel size (uneven overlap). Fix: use bilinear upsampling + regular conv instead, or choose kernel/stride carefully.

---

## Adversarial Examples

Small, carefully crafted perturbations to input images that fool CNNs while being imperceptible to humans. They expose a fundamental gap between how CNNs and humans process visual information.

```
Original image:              Adversarial image:
┌──────────────┐             ┌──────────────┐
│   🐱         │  + tiny     │   🐱         │   Looks identical to humans
│   cat        │  noise →    │   cat        │   but CNN says "toaster"
│   (99.2%)    │             │   (99.7%)    │   with high confidence
└──────────────┘             └──────────────┘
```

### FGSM — Fast Gradient Sign Method (Goodfellow et al., 2014)

The simplest attack. One gradient step in the direction that maximizes loss:

```
x_adv = x + ε × sign(∇_x L(θ, x, y))

Where:
  x         = original image
  ε         = perturbation budget (e.g., 8/255 — invisible to humans)
  ∇_x L     = gradient of loss with respect to input pixels
  sign()    = just take +1 or -1 (each pixel nudged by exactly ε)
  y         = true label
  θ         = model parameters (fixed, we're attacking the INPUT not the weights)

Key insight: we use the SAME gradient computation as training,
but instead of updating weights to reduce loss,
we update the INPUT to INCREASE loss.
```

**Why sign()?** Taking just the sign means every pixel gets perturbed by the same amount (±ε), distributing the perturbation evenly. This is more effective than using raw gradient magnitudes, which would concentrate changes on a few pixels.

### PGD — Projected Gradient Descent (Madry et al., 2017)

FGSM but iterative — take multiple smaller steps and project back into the allowed perturbation ball after each step:

```
PGD (k steps):
  x_0 = x + random noise (within ε-ball)       ← random start
  for i = 1 to k:
      x_i = x_{i-1} + α × sign(∇_x L(θ, x_{i-1}, y))    ← small FGSM step
      x_i = clip(x_i, x - ε, x + ε)            ← project back into ε-ball
      x_i = clip(x_i, 0, 1)                     ← keep valid pixel range

Where:
  α = step size (smaller than ε, e.g., 2/255)
  k = number of steps (e.g., 7-20)
  ε = total perturbation budget

FGSM = PGD with k=1 and no random start
```

**PGD is the "gold standard" attack** for evaluating robustness — if a model survives PGD, it's considered reasonably robust. Adversarial training (training on PGD-generated examples) is the strongest known defense.

### Why adversarial examples exist

```
Human vision:     relies on shapes, textures, spatial relationships
CNN "vision":     relies on statistical patterns in pixel values — many imperceptible to humans

CNNs use features that are predictive but brittle:
high-frequency texture patterns that humans don't notice but models depend on heavily.

A small perturbation can flip these features without changing anything a human would see.
```

| Method | Steps | Strength | Speed | Use case |
|---|---|---|---|---|
| **FGSM** | 1 | Weak (but fast) | Very fast | Quick robustness check, adversarial training |
| **PGD** | k (7-20) | Strong | Slower | Gold standard evaluation, robust training |

---

## Questions

- How do adversarial training and standard training interact — does adversarial robustness hurt clean accuracy?
- What's the relationship between G-CNN group equivariance and ViT's learned rotation handling?
