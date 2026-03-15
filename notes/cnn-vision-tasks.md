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
