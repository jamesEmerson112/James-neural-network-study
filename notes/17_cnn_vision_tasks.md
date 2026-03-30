# CNN & Vision Tasks — From Classification to Structured Representations

*Last updated: 2026-03-15*

## What is a CNN?

A regular neural network takes a flat list of numbers as input. An image is a **grid of pixels** — a CNN is designed to work with that grid structure directly.

### The core operation: Convolution

A small filter (e.g., 3x3) **slides across the image** and detects a pattern at each position:

```
Image (5x5)              Filter (3x3)           Output
┌───┬───┬───┬───┬───┐    ┌───┬───┬───┐
│   │   │   │   │   │    │ 1 │ 0 │-1 │    Each position: multiply
│   │   │   │   │   │  × │ 1 │ 0 │-1 │  → overlapping values, sum
│   │   │   │   │   │    │ 1 │ 0 │-1 │    = one number in output
│   │   │   │   │   │    └───┴───┴───┘
│   │   │   │   │   │
└───┴───┴───┴───┴───┘

This particular filter detects vertical edges.
```

### What stacking layers does

```
Early layers:  detect edges, corners, simple textures
      │
Middle layers: detect parts (eyes, wheels, handles)
      │
Deep layers:   detect whole objects (faces, cars, dogs)
```

The network **learns these filters automatically** during training — you don't hand-design them.

### CNN building blocks

| Component | What it does |
|---|---|
| **Conv layer** | Slides filters across input, detects patterns |
| **Pooling** | Shrinks the spatial size (e.g., 224×224 → 112×112). Keeps important info, reduces computation |
| **Activation (ReLU)** | Adds non-linearity — lets the network learn complex patterns |
| **Fully connected** | At the end, flattens everything into a decision (e.g., "this is a cat") |

```
Input Image → [Conv → ReLU → Pool] × N → Flatten → Fully Connected → "cat"
```

### Receptive field

The **receptive field** of a neuron is the region of the original input that can influence that neuron's output.

```
Layer 1 (3×3 conv):   each neuron sees a 3×3 patch of the input
Layer 2 (3×3 conv):   each neuron sees a 3×3 patch of layer 1's output
                       → which means it sees a 5×5 patch of the original input
Layer 3 (3×3 conv):   each neuron sees 7×7 of the original input

Receptive field GROWS with depth:
  Layer 1: 3×3     Layer 2: 5×5     Layer 3: 7×7    ...
```

**Key properties:**

| Property | Why |
|---|---|
| Grows with depth | Each layer adds context from a wider spatial area |
| Pooling increases it faster | A 2×2 pool effectively doubles the receptive field of subsequent layers |
| FC layers see everything | Fully connected layers have a receptive field covering the entire input |
| Center pixels contribute more | A center pixel falls within more overlapping filter positions than an edge pixel — it influences more neurons |

```
Why center > edge:

3×3 filter sliding over 5×5 input:

  Corner pixel (0,0):  appears in 1 filter position
  Edge pixel (0,2):    appears in 2 filter positions
  Center pixel (2,2):  appears in 9 filter positions (all of them)

  → center pixels have MORE influence on the output
  → this is called the "effective" receptive field — technically the RF is
    a fixed size, but the center contributes disproportionately
```

**Why it matters:** A network needs a receptive field large enough to "see" the objects it's trying to classify. If the receptive field is too small, the network only sees local textures, never the whole object. This is why deeper networks and dilated convolutions (DeepLab) exist — to grow the receptive field without losing resolution.

### Worked example: Counting parameters (no bias)

Given: 224×224×3 input → Conv (3×3, stride 1, same padding) → 224×224×32 → Pool → 112×112×32 → FC → 1×1×9

**Conv layer — 864 parameters**
```
One 3×3 filter must cover all input channels:  3 × 3 × 3 = 27 weights per filter
We need 32 filters (one per output channel):   27 × 32 = 864

Key insight: these 27 weights are SHARED across all 224×224 positions.
The filter slides everywhere, reusing the same weights — that's why CNNs are efficient.
```

**Pooling layer — 0 parameters**
```
Pooling just picks the max (or average) from each window. No learnable weights.
112×112×32 output, but nothing to train.
```

**Fully connected layer — 3,612,672 parameters**
```
Flatten the pooling output:     112 × 112 × 32 = 401,408 input neurons
Each input connects to each output:  401,408 × 9 = 3,612,672

NO weight sharing — every connection has its own unique weight.
That's why FC layers are so expensive.
```

**Total: 864 + 0 + 3,612,672 = 3,613,536 parameters**

```
Conv layer:  ██ 0.02%                    (864)
FC layer:    ████████████████████ 99.98%  (3,612,672)

The FC layer dominates — this is why modern architectures (GoogLeNet, ResNet)
replaced large FC layers with global average pooling to cut parameters.
```

### Worked example: Backward pass — computing gradients

Setup: 2×2 kernel (W) slides over 3×3 input (X), no padding, stride=1, producing 2×2 output (H).

```
X = | x₁₁  x₁₂  x₁₃ |     W = | w₁₁  w₁₂ |     H = | h₁₁  h₁₂ |     → Loss
    | x₂₁  x₂₂  x₂₃ |         | w₂₁  w₂₂ |         | h₂₁  h₂₂ |
    | x₃₁  x₃₂  x₃₃ |

Forward: slide kernel over X, multiply and sum at each of 4 positions → H
Backward: given dH (blame at output), find dW and dX (blame for weights and inputs)
```

**dW₂₂ — "how much blame does kernel weight w₂₂ get?"**

At each of the 4 positions, w₂₂ (bottom-right of kernel) always multiplies the bottom-right value of the input patch. So dW₂₂ = dH × the input values that w₂₂ touched:

```
Given: dH = | 1.2  -0.5 |     X = | 9  4  6 |
            | 1.1   0.7 |         | 6  7  3 |
                                   | 4  5  5 |

Position 1: w₂₂ touched x₂₂ = 7  →  dh₁₁ × x₂₂ = 1.2 × 7 = 8.4
Position 2: w₂₂ touched x₂₃ = 3  →  dh₁₂ × x₂₃ = -0.5 × 3 = -1.5
Position 3: w₂₂ touched x₃₂ = 5  →  dh₂₁ × x₃₂ = 1.1 × 5 = 5.5
Position 4: w₂₂ touched x₃₃ = 5  →  dh₂₂ × x₃₃ = 0.7 × 5 = 3.5

dW₂₂ = 8.4 + (-1.5) + 5.5 + 3.5 = 15.9
```

For dW: multiply dH by **input values** (X). The kernel weight always sits in the same position (bottom-right), so the input values it touches shift predictably: x₂₂, x₂₃, x₃₂, x₃₃.

**dX₂₂ — "how much blame does input value x₂₂ get?"**

x₂₂ is in the center of X, so all 4 kernel positions touch it. But as the kernel slides, x₂₂ falls on a DIFFERENT kernel weight each time:

```
Given: dH = | 2    0.5 |     W = | 5  7 |
            | 1.4  1.2 |         | 3  2 |

Position 1 (top-left patch):     x₂₂ is at bottom-right → sits under w₂₂ = 2
Position 2 (top-right patch):    x₂₂ is at bottom-left  → sits under w₂₁ = 3
Position 3 (bottom-left patch):  x₂₂ is at top-right    → sits under w₁₂ = 7
Position 4 (bottom-right patch): x₂₂ is at top-left     → sits under w₁₁ = 5

dX₂₂ = dh₁₁ × w₂₂ + dh₁₂ × w₂₁ + dh₂₁ × w₁₂ + dh₂₂ × w₁₁
     = (2)(2)      + (0.5)(3)    + (1.4)(7)    + (1.2)(5)
     = 4           + 1.5         + 9.8          + 6.0
     = 21.3
```

For dX: multiply dH by **kernel weights** (W). Because x₂₂ shifts position inside the kernel window as the kernel slides, the kernel weights appear in reversed order (w₂₂, w₂₁, w₁₂, w₁₁).

**Why dW and dX look different:**

```
dW: the KERNEL WEIGHT stays fixed (always bottom-right)
    → the INPUT values it touches shift naturally (x₂₂, x₂₃, x₃₂, x₃₃)
    → no reversal

dX: the INPUT VALUE stays fixed (always at x₂₂)
    → which KERNEL WEIGHT touches it changes as the kernel moves over it
    → the kernel weights appear reversed (w₂₂, w₂₁, w₁₂, w₁₁)
    → this "flip" is not a rule — it's just geometry
       (kernel moves right → x₂₂ shifts left INSIDE the window)
```

**⚠️ STILL GROKKING THIS** — the reversal in dX feels unintuitive. The key is to just trace "which kernel weight lands on x₂₂?" at each position rather than trying to memorize a flip rule.

**What are dW and dX used for?**
```
dW → update the kernel weights:   w_new = w_old - learning_rate × dW    (learning)
dX → pass blame to previous layer: becomes dH for the layer before       (backprop continues)
```

---

## How All Tasks Share a CNN Backbone

All vision tasks **use a CNN as the backbone** for feature extraction, then add a different "head" on top:

```
                          ┌→ Classification head → "cat"
                          │
Input → CNN backbone ─────┼→ Detection head → boxes + labels
  (feature extraction)    │
                          └→ Segmentation head → pixel-level masks
```

---

## What Detection and Segmentation Actually Look Like

### Object Detection — "Draw a box around each object"

```
┌─────────────────────────┐
│                         │
│   ┌─────┐    ┌──────┐  │
│   │ cat │    │ dog  │  │
│   └─────┘    └──────┘  │
│         ┌────┐          │
│         │ball│          │
│         └────┘          │
└─────────────────────────┘

Output: [(cat, box), (dog, box), (ball, box)]
Each box = (x, y, width, height) + class label + confidence score
```

**Two-stage**: propose "interesting regions" first, then classify each one. More accurate, slower.
**Single-stage**: predict all boxes in one pass. Faster, simpler.

### Segmentation — "Color every pixel by what it belongs to"

```
Classification:  "There's a cat"
Detection:       "There's a cat HERE [box]"
Segmentation:    "THESE EXACT PIXELS are cat"

┌─────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░░ │  ░ = background
│ ░░░██████░░░▓▓▓▓▓▓▓░░░ │  █ = cat (every pixel)
│ ░░░██████░░░▓▓▓▓▓▓▓░░░ │  ▓ = dog (every pixel)
│ ░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────────────┘
```

---

## The Vision Task Hierarchy

Each level builds on the one before — you can't do the next without solving the previous.

```
Level 1: Classification        "What is in the image?"
    │
    ▼
Level 2: Object Detection      "What + Where?" (bounding boxes)
    │
    ▼
Level 3: Segmentation          "What + Where at pixel level?"
    │
    ▼
Level 4: Structured Representations   "What + Where + How things relate?"
```

---

## Level 1 — Classification

- Input: image → Output: single label ("cat", "car")
- The foundation — every other task depends on this
- Classic architectures: LeNet, AlexNet, VGG, ResNet, Inception

---

## Level 2 — Object Detection

Localizes multiple objects with bounding boxes + class labels.

### Two-Stage Detectors (8 key models)

Propose regions first, then classify each region. More accurate, slower.

| # | Model | Year | Key Idea |
|---|---|---|---|
| 1 | **R-CNN** | 2014 | Selective Search → CNN features → SVM classifier. First to apply CNNs to detection. |
| 2 | **SPPNet** | 2014 | Spatial Pyramid Pooling — run CNN once on full image, not per region. Big speedup over R-CNN. |
| 3 | **Fast R-CNN** | 2015 | RoI Pooling — single CNN pass + end-to-end training (no separate SVM). |
| 4 | **Faster R-CNN** | 2015 | Region Proposal Network (RPN) replaces Selective Search. Fully neural, end-to-end. |
| 5 | **FPN** | 2017 | Feature Pyramid Network — multi-scale feature maps for detecting small + large objects. |
| 6 | **Mask R-CNN** | 2017 | Adds segmentation branch to Faster R-CNN. Detection + instance segmentation together. |
| 7 | **Cascade R-CNN** | 2018 | Multiple detection heads at increasing IoU thresholds. Progressive refinement. |
| 8 | **HTC** | 2019 | Hybrid Task Cascade — interleaves detection and segmentation across cascade stages. |

### Single-Stage Detectors (8 key models)

Predict all boxes at once in one pass. Faster, simpler.

| # | Model | Year | Key Idea |
|---|---|---|---|
| 1 | **YOLO (v1)** | 2016 | Divide image into grid, predict boxes + classes in one pass. Real-time speed. |
| 2 | **SSD** | 2016 | Multi-scale feature maps with anchor boxes at each scale. Faster than Faster R-CNN. |
| 3 | **RetinaNet** | 2017 | Focal Loss — solves class imbalance problem. Single-stage accuracy matches two-stage. |
| 4 | **YOLOv3** | 2018 | Multi-scale predictions + Darknet-53 backbone. Practical balance of speed & accuracy. |
| 5 | **CornerNet** | 2018 | Anchor-free — detects objects as pairs of top-left and bottom-right corners. |
| 6 | **FCOS** | 2019 | Fully Convolutional One-Stage — anchor-free, per-pixel prediction. Simpler design. |
| 7 | **CenterNet** | 2019 | Objects as center points — keypoint detection approach, no anchors or NMS needed. |
| 8 | **DETR** | 2020 | Transformer-based detection. Set prediction with bipartite matching. No anchors, no NMS. |

---

## Level 3 — Segmentation

Pixel-level understanding of the scene.

### Types
- **Semantic segmentation** — label every pixel by class (all "car" pixels, all "road" pixels)
- **Instance segmentation** — distinguish individual objects (car #1 vs car #2)
- **Panoptic segmentation** — combines both (every pixel labeled + individual instances)

### Segmentation Networks (8 key models)

| # | Model | Year | Key Idea |
|---|---|---|---|
| 1 | **FCN** | 2015 | First fully convolutional approach — replaces FC layers with conv for dense prediction. |
| 2 | **SegNet** | 2015 | Encoder-decoder with pooling indices — unpooling for upsampling instead of deconv. |
| 3 | **U-Net** | 2015 | Encoder-decoder + skip connections. Designed for medical imaging, works with small datasets. |
| 4 | **DeepLab v2** | 2017 | Atrous/dilated convolutions + CRF post-processing. Larger receptive field without losing resolution. |
| 5 | **PSPNet** | 2017 | Pyramid Pooling Module — captures global context at multiple scales. |
| 6 | **DeepLab v3+** | 2018 | Atrous Spatial Pyramid Pooling (ASPP) + encoder-decoder structure. |
| 7 | **PANet** | 2018 | Path Aggregation Network — bottom-up path augmentation for better feature flow. |
| 8 | **HRNet** | 2019 | Maintains high-resolution representations throughout — no downsampling then upsampling. |

---

## Level 4 — Structured Representations

Goes beyond detecting/segmenting objects to understanding **relationships and organization** between them.

### Key forms

| Form | What it captures | Example |
|---|---|---|
| **Scene Graphs** | Object-relationship-object triples | "person *riding* bike", "cup *on* table" |
| **Feature Pyramids (FPN)** | Multi-scale hierarchical features | Small, medium, and large objects in one model |
| **Part-whole hierarchies** | How parts compose into objects | wheels → car, eyes → face → person |

### Scene Graphs — the NLP-Vision bridge

From Xu et al., "Scene Graph Generation by Iterative Message Passing" (2017):

- **Insight**: an object in a scene relates to surrounding objects the same way a word relates to surrounding words (skip-gram / Word2Vec analogy)
- **Method**: detect objects first, then use message passing between them to predict relationships
- **Output**: a graph where nodes = objects, edges = relationships

```
  [person] ──riding──▶ [bike]
     │                    │
  wearing              parked_on
     │                    │
     ▼                    ▼
  [helmet]             [sidewalk]
```

This is where vision systems go from "I see a list of objects" to "I understand the scene."

---

## Why This Hierarchy Matters

- Each level adds more **structure** to the output
- Detection → list of boxes. Segmentation → pixel maps. Scene graphs → relational understanding.
- Modern architectures often solve multiple levels at once (e.g., Mask R-CNN does detection + instance segmentation)
- Structured representations enable higher-level reasoning: visual question answering, image captioning, robotic scene understanding

---

## Embeddings, Encoders, and Decoders — The Big Picture

### Embeddings = a lookup table (dictionary)

An embedding is NOT an encoder or decoder. It's simpler — it comes *before* both.

```
Regular dictionary:   "cat" → "a small furry animal"     (human-readable)
Embedding table:      "cat" → [0.12, -0.34, 0.56, ...]  (machine-readable vector)
```

**Cosine similarity** is how the machine measures meaning — by comparing angles between vectors:
```
"cat"  ↔ "kitten"    = 0.92   (very close)
"cat"  ↔ "dog"       = 0.76   (related — both pets)
"cat"  ↔ "airplane"  = 0.15   (unrelated)
```

### How embeddings, encoders, and decoders fit together

```
Token IDs           Embeddings              Encoder / Decoder
(integers)          (lookup table)          (complex processing)

"cat" → 1037  →  [0.12, -0.34, ...]  →  [attention, feed-forward, ...]

   ①                   ②                          ③
 Symbol          →   Vector              →   Understanding
 (discrete)          (just a lookup)          (actual computation)
```

| Step | What it is | Complexity |
|---|---|---|
| **Embedding** | Lookup table. Token ID in → vector out. No context. | Dead simple |
| **Encoder** | Reads embeddings, applies attention across all positions → contextual representations | Complex |
| **Decoder** | Takes representations, generates output token by token | Complex |

### Static vs Contextualized

| | Standalone Embeddings | Embeddings + Encoder/Decoder |
|---|---|---|
| **Lookup** | "cat" → always the same vector | "cat" → same initial vector |
| **After** | That's it. Use cosine similarity. | Encoder/decoder transforms it based on context |
| **"cat food" vs "cat burglar"** | Same vector for "cat" | Different representations after processing |
| **Examples** | Word2Vec, GloVe, BigGraph | BERT, GPT, Claude |

### How models map to this framework

```
EMBEDDINGS ONLY          ENCODER               DECODER            ENCODER-DECODER
────────────────         ───────               ───────            ───────────────
Word2Vec, GloVe          BERT, RoBERTa         GPT, Claude        T5, BART
BigGraph (graphs)        ELMo (LSTM-based)     LLaMA, nanochat    Original Transformer
Node2Vec                 CNN backbones                            U-Net (vision)
                         (ResNet, VGG)
```

### Graph Embeddings — PyTorch-BigGraph

Same idea as Word2Vec, but for knowledge graphs instead of text.

| Embedding Type | Input | Learns from | Scale |
|---|---|---|---|
| **Word2Vec/GloVe** | Words | Word co-occurrence in text | ~3-6M words, 300 dims |
| **BERT token embeddings** | Tokens | Language modeling | ~30K tokens, 768 dims |
| **BigGraph (Wikidata)** | Graph entities | Graph structure & relationships | 78M entities, 200 dims |

---

## Questions

- How do graph neural networks (GNNs) fit into scene graph generation?
- What's the connection between structured representations and vision-language models (e.g., CLIP)?
