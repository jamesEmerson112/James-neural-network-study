# Do Vision Transformers See Like Convolutional Neural Networks?

> **Raghu, Unterthiner, Kornblith, Zhang, Dosovitskiy (Google Brain, 2021)**
> Paper: https://arxiv.org/abs/2108.08810

---

## Assignment Questions

### Q1: Short Review (max 350 words)

*What is the main contribution? Key insights? Strengths and weaknesses?*

TODO

### Q2: Personal Takeaway (max 350 words)

*Perceived novelty, future directions, anything noteworthy?*

My takeaway is that despite achieving similar accuracy, ViTs and CNNs take fundamentally different paths to get there. ViTs rely heavily on large-scale data to learn properties that CNNs get for free from their architecture (like local feature detection). This makes CNNs more practical and data-efficient in resource-constrained settings. However, when sufficient data is available, ViTs develop unique properties — like preserved spatial localization and stronger intermediate representations — that CNNs cannot achieve due to their architectural constraints. The paper suggests neither architecture is strictly superior; the choice depends on the data regime and downstream task.

### Q3: Compare and contrast learned features of ViTs and CNNs (max 350 words)

*For differences, explain in terms of network architecture and training.*

Both ViTs and CNNs ultimately learn to detect low-level features (edges, textures) in early layers and task-relevant semantic features in later layers, converging to similar final performance on classification. However, the internal representations and how they arrive at these features differ significantly.

**Representation structure:** CKA analysis reveals that ViTs have a remarkably uniform representation across layers — early and late layers share high similarity. CNNs show a clear hierarchical stage structure, where early layers (edges) are very different from late layers (objects), with sharp transitions between stages.

**Local vs global processing:** The most fundamental difference stems from architecture. CNN convolutions have fixed small kernels (e.g. 3x3), forcing early layers to be strictly local; global context only emerges after stacking many layers. ViT's self-attention has no distance constraint, so even the first layer contains attention heads that attend both locally and globally. This means ViTs aggregate global information from the very start, while CNNs must build it up gradually.

**Skip connections:** In ViTs, skip connections carry most of the information, with each layer making small incremental adjustments. Removing a single skip connection fractures the model's representation and drops accuracy by 4%. In ResNets, skip connections are less dominant — each convolutional layer substantially transforms the representation.

**Role of training data:** ViTs must *learn* to attend locally from data, whereas CNNs get local processing for free from their architecture. When trained only on ImageNet (~1.2M images), ViT early layers never develop local attention heads, leading to worse performance. At scale (JFT-300M), ViTs learn a mix of local and global attention that produces features CNNs structurally cannot — the highest ViT layers have no CNN equivalent in cross-model CKA analysis. Furthermore, lower-layer representations converge with relatively little data, but higher-layer representations require massive datasets, especially for larger ViT models.

**Unique ViT properties:** ViTs preserve spatial localization through their final layer (aided by the CLS token), while CNNs smear spatial information through global average pooling. This makes ViT representations potentially more suitable for tasks requiring precise spatial information like object detection.

### Q4: Spatial localization — what is it and why might ViTs be better for object detection? (max 350 words)

**Spatial localization** refers to a model's ability to preserve information about *where* in the image a feature originates — not just *what* is present, but its precise position.

The paper measures this by comparing each token's final-layer representation against every spatial patch in the input image using CKA similarity. ViT tokens show peak similarity at their corresponding input location — the model knows where each token came from, even at the final layer. ResNet features show broad, diffuse similarity, meaning precise positional information has been lost.

**Why this happens:** ViT uses a CLS token for classification, acting as a separate aggregation channel. Spatial tokens never need to carry global class information — they stay spatially grounded. ResNet uses Global Average Pooling, forcing every feature to carry global info, destroying spatial precision. When the authors train ViT with GAP instead of CLS, localization degrades significantly, confirming that the CLS token is key. ViT's strong skip connections also faithfully propagate spatial information across layers.

**Why ViTs may be better for object detection:** Detection requires both classifying objects AND predicting precise bounding box locations. CNNs lose spatial precision by their final layers and must rely on feature pyramids to recover it. ViTs maintain spatial precision throughout while having global context from layer 1 via self-attention — both ingredients a detector needs, without architectural workarounds.

---

## Reading Notes

### ViT vs CNN Comparison Table

#### Architecture & Design

| Aspect | ViT | CNN (ResNet) |
|--------|-----|--------------|
| Basic unit | Self-attention over image patches (tokens) | Stacked convolutional filters with fixed kernels |
| Input processing | Chop image into patches (e.g. 16x16), treat as tokens | Slide filters across raw pixels |
| Inductive bias | Minimal — must learn spatial structure from data | Strong — locality and translation equivariance built in |
| Classification mechanism | CLS token (one special token aggregates global info) | Global Average Pooling (average all spatial features) |
| Skip connections | Carry most of the information; layers make small tweaks | Less influential; each layer substantially transforms the representation |

#### Internal Representations (CKA Analysis)

| Aspect | ViT | CNN (ResNet) |
|--------|-----|--------------|
| Overall structure | Uniform — layers look similar to each other throughout | Hierarchical — clear stages with sharp transitions |
| Low↔high layer similarity | High — features shared across depth | Low — early (edges) very different from late (objects) |
| Depth efficiency | Bottom half of ResNet maps to just bottom quarter of ViT | Needs ~2x layers to reach same representational level |
| Highest layers | Learn novel representations with no CNN equivalent | — |
| Without skip connections | Model fractures into two halves; 4% accuracy drop | Less dramatic effect |

#### Local vs Global Processing

| Aspect | ViT | CNN (ResNet) |
|--------|-----|--------------|
| Early layers | Both local AND global from layer 1 | Forced local only (small receptive field) |
| Late layers | Fully global (all attention heads) | Finally global after many stacked layers |
| How local→global happens | Self-attention — no distance constraint, immediate | Must stack many conv layers to gradually grow receptive field |
| Effective receptive field | Large from early layers, expands fast | Tiny in early layers, grows gradually |
| Early layer features | Mix of CNN-like local features + novel global features | Local only (edges, textures) |

#### Spatial Localization & Object Detection

| Aspect | ViT | CNN (ResNet) |
|--------|-----|--------------|
| Spatial info at final layer | Preserved — each token knows where it came from | Smeared out — loses precise spatial info |
| CLS vs GAP effect | CLS token keeps spatial tokens localized; GAP blurs it | Uses GAP — all features forced to carry global info |
| Object detection potential | Strong — spatial info preserved + global context from layer 1 | Weaker — needs feature pyramids to recover spatial info |

#### Data & Scale

| Aspect | ViT | CNN (ResNet) |
|--------|-----|--------------|
| Data requirements | ~300M images to learn local attention; on ImageNet alone, never learns it | Works well on ImageNet (~1.2M images) |
| Lower layer convergence | With just 3% of data, already similar to full-data model | Less sensitive to data scale overall |
| Higher layer convergence | Needs massive data; bigger models need even more | Less sensitive to scale (priors compensate) |
| Transfer learning | Larger ViTs + more data → stronger intermediate representations | Weaker intermediate representations than large ViTs on JFT |

### Paper Strengths

1. **Comprehensive methodology** — CKA heatmaps, attention distance analysis, effective receptive fields, linear probes, interventional experiments. Each finding cross-validated by multiple methods.
2. **Interventional experiments** — Removing skip connections (Fig 8), swapping CLS for GAP (Fig 10), varying dataset size (Fig 12). Moves beyond correlation toward causation.
3. **Practical implications** — Spatial localization → object detection, data scale → transfer learning. Actionable insights.
4. **First systematic comparison** — Before this paper (NeurIPS 2021), nobody had rigorously compared how ViTs and CNNs process information internally.

### Paper Weaknesses

1. **Data fairness is murky** — Both train on JFT-300M, but ViT benefits more from large data. Comparing at respective sweet spots would be more revealing.
2. **CKA limitations** — Collapses a layer's entire representation into one scalar. May miss fine-grained differences.
3. **No actual detection experiments** — Claims ViTs are promising for detection based on localization analysis, but never runs a detection benchmark.
4. **Only classification models** — Would findings hold for self-supervised or segmentation models?
5. **Limited CNN architectures** — Only ResNets. What about EfficientNets, ConvNeXts?
6. **Google-internal dataset** — JFT-300M isn't public. Hard to reproduce core experiments.

### Key Figures

| Figure | What it shows |
|--------|--------------|
| Fig 1 | CKA heatmaps within models — ViT uniform, ResNet hierarchical |
| Fig 2 | Cross-model CKA — bottom half of ResNet ≈ bottom quarter of ViT |
| Fig 3 | Attention head mean distances — early layers mix local+global, late layers all global |
| Fig 4 | On ImageNet only, ViT never learns local attention |
| Fig 5 | Local ViT heads ≈ ResNet early layers; global heads learn something different |
| Fig 6 | Effective receptive fields — ViT expands fast, ResNet gradual |
| Fig 7 | Skip connections carry most info in ViT |
| Fig 8 | Removing skip connection fractures ViT representation |
| Fig 9 | ViT preserves spatial localization at final layer; ResNet doesn't |
| Fig 10 | GAP degrades ViT's spatial localization |
| Fig 11 | Linear probes — individual ViT spatial tokens poor at classification |
| Fig 12 | Higher layers need more data; lower layers converge fast |
| Fig 13 | ViTs on JFT produce stronger intermediate representations than ResNets |
