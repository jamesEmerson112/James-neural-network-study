# The Resurrection & Deep Learning Revolution (1986-2017)

> [Previous: Foundations, Pioneers & AI Winter](00a_timeline_1847-1985_foundations_pioneers_winter.md) | [Back to Timeline Hub](00_timeline.md) | [Next: The LLM Era](00c_timeline_2018-now_llm_era.md)

## The Resurrection (1986-2008)

| Year | What | Who | Key Idea |
|------|------|-----|----------|
| 1985 | **Boltzmann machine** | Hinton & Sejnowski | Energy-based generative model — uses the Boltzmann distribution to define probabilities over states. Training via MCMC sampling. Led to RBMs and deep belief nets (2006), which helped trigger the deep learning revolution |
| 1986 | **MLP + Backprop** | Rumelhart, Hinton, Williams | "Learning representations by back-propagating errors" in *Nature* 323. Multi-layer networks trained with backpropagation. Solved XOR. Neural nets revived |
| 1986 | **PDP Book** | Rumelhart, McClelland (eds.) | *Parallel Distributed Processing* (MIT Press, 2 vols). Chapter 8 = full backprop treatment. Final section introduces training recurrent nets via unrolling. 30,000+ citations. Also introduced the autoencoder concept (see below) |
| 1986 | **Autoencoder** | Rumelhart & Hinton | Described in the PDP book: compress input through a bottleneck layer, then reconstruct it. The network learns an efficient internal representation without labels — unsupervised feature learning. The bottleneck forces the network to discover what matters. Direct ancestor of VAEs: Kingma & Welling added probabilistic sampling to this exact architecture 27 years later |
| 1986 | **Jordan Network** | Michael Jordan (UCSD) | "Serial Order" — first RNN variant. Output fed back to context units. Network remembers what it *said* |
| 1986 | **RNN** | Rumelhart, Hinton, Williams | Recurrent connections described in PDP Ch. 8 — share weights across time, feed hidden state back. Networks process sequences with memory |
| 1989 | **CNN** | Yann LeCun | Convolutional neural networks — local filters for image recognition |
| 1989 | **Q-learning** | Chris Watkins | Model-free RL algorithm — learn action values directly from experience without knowing the environment's rules. Foundation for DQN decades later |
| 1990 | **Elman Network** | Jeffrey Elman (UCSD) | "Finding Structure in Time" — hidden state (not output) fed back to context units. Became the standard "vanilla RNN" architecture |
| 1990 | **BPTT formalized** | Paul Werbos | "Backpropagation Through Time: What It Does and How to Do It" — unroll the RNN, run standard backprop on the unrolled graph |
| 1991 | Vanishing gradient proved | Hochreiter | Diploma thesis *"Untersuchungen zu dynamischen neuronalen Netzen"* (in German, TU Munich, advisor: Schmidhuber). Proved gradients decay exponentially through time steps |
| 1992 | **TD-Gammon** | Gerald Tesauro (IBM) | Neural net learns backgammon via self-play + temporal difference learning. Expert-level play; changed how humans played the game. Proved neural nets + RL + self-play works |
| 1994 | Vanishing gradient confirmed | Bengio, Simard, Frasconi | "Learning Long-Term Dependencies with Gradient Descent is Difficult" — confirmed Hochreiter's result in English, widely read |
| 1997 | **LSTM** | Hochreiter & Schmidhuber | "Long Short-Term Memory" in *Neural Computation* 9(8). Cell state ("constant error carousel") + input/output gates solve vanishing gradients. Original had NO forget gate **(Assignment Phase 1)** |
| 2000 | **Forget gate added to LSTM** | Gers, Schmidhuber, Cummins | "Learning to Forget: Continual Prediction with LSTM" — without forget gate, cell state grows forever. This completed the modern LSTM architecture used today |
| 2006 | **Deep belief nets** | Geoffrey Hinton (U of Toronto) | "A Fast Learning Algorithm for Deep Belief Nets" — stack Restricted Boltzmann Machines (RBMs) layer by layer, each trained greedily on the previous layer's hidden activations. Solved the "deep networks won't train" problem by pre-training unsupervised first, then fine-tuning with backprop. Hinton, working with just two grad students in Toronto, reopened the door Minsky & Papert slammed in 1969. This paper triggered the deep learning revolution — six years before AlexNet |
| 2006 | **Deep autoencoder beats PCA** | Hinton & Salakhutdinov | "Reducing the Dimensionality of Data with Neural Networks" in *Science* — deep autoencoders with pre-trained layers outperform PCA at dimensionality reduction. Used the same RBM pre-training trick as deep belief nets. Published in the same year — 2006 was Hinton's annus mirabilis. Proved that deep nonlinear compression learns structure that 105 years of linear PCA (Pearson, 1901) couldn't find |
| 2008 | **Denoising autoencoder** | Vincent, Larochelle, Bengio (Montreal) | Corrupt the input with noise, then train the network to reconstruct the clean version. Forces the model to learn robust features instead of just memorizing. The core idea — "add noise, learn to remove it" — is exactly what diffusion models (DDPM, 2020) do at massive scale. A 12-year bridge from autoencoders to Stable Diffusion |

## The Deep Learning Revolution (2012-2017)

| Year | What | Who | Key Idea |
|------|------|-----|----------|
| 2011 | **NADE** | Larochelle & Murray | Neural Autoregressive Density Estimator — model p(x) as a product of conditionals, each parameterized by a neural net. Exact log-likelihood, no approximation needed. Predecessor to PixelRNN: same "generate one piece at a time" idea, applied to images five years later |
| 2012 | **AlexNet** | Krizhevsky, Sutskever, Hinton | CNN wins ImageNet by a landslide — deep learning revolution begins. GPUs + data + scale |
| 2013 | **DQN** | Mnih et al. (DeepMind) | Deep Q-Network — CNN + Q-learning plays 49 Atari games from raw pixels. Experience replay + target networks. Put DeepMind on the map; Google acquired them for ~$500M |
| 2013 | **VAE** | Kingma & Welling | Variational Autoencoder — built on the autoencoder architecture (1986) — encoder compresses to latent space, decoder reconstructs. Approximate density via ELBO. The reparameterization trick (z = μ + σ·ε) made backprop through sampling possible. Smooth latent space but blurry outputs |
| 2014 | **GANs** | Goodfellow | Generator vs discriminator — adversarial training for image generation — a minimax game (Nash equilibrium, 1950). Conceived during a bar argument in Montreal; worked on the first try |
| 2014 | **Seq2Seq** | Sutskever, Vinyals, Le | Encoder → fixed vector → Decoder. First serious neural machine translation |
| 2014 | **Attention** | Bahdanau, Cho, Bengio | Decoder "looks back" at all encoder states instead of one fixed vector **(Assignment Phase 2)** |
| 2014 | GRU | Cho et al. | Simplified LSTM — 2 gates instead of 3, no separate cell state |
| 2015 | Attention variants | Luong et al. | Different scoring: dot product, cosine similarity, etc. |
| 2015 | **Normalizing flows** | Rezende & Mohamed (DeepMind) | Start with simple noise, apply a chain of invertible transformations to warp it into complex data. Exact density via the change-of-variables formula. A third generative path alongside VAEs (blurry but stable) and GANs (sharp but unstable): no blurriness, no adversarial training, but architecturally constrained to invertible functions |
| 2015 | **DCGAN** | Radford, Metz, Chintala | "Deep Convolutional GANs" — the recipe that made GANs actually work for images: use convolutions instead of fully connected layers, batch normalization, and specific activations (ReLU in generator, LeakyReLU in discriminator). Before DCGAN, GANs were an unstable novelty; after, they generated coherent faces. Alec Radford was 24 — he went on to co-author GPT-1 three years later |
| 2016 | **AlphaGo** | Silver et al. (DeepMind) | Beat Lee Sedol 4-1 at Go — watched by 200M people. Supervised learning + RL self-play + Monte Carlo Tree Search. "Move 37" discovered strategies 3,000 years of human play hadn't found |
| 2016 | **PixelRNN / PixelCNN** | van den Oord et al. (DeepMind) | Autoregressive image generation — predict one pixel at a time. Tractable density but painfully slow |
| 2017 | **AlphaZero** | Silver et al. (DeepMind) | Learned chess, shogi, and Go from scratch via pure self-play — no human data. Beat Stockfish (best chess engine) after 4 hours of training |
| 2017 | **Transformer** | Vaswani et al. (Google) | "Attention Is All You Need" — drop recurrence. Self-attention over all positions in parallel **(Assignment Phases 3 & 4)** |
| 2017 | **WGAN** | Arjovsky, Chintala, Bottou | Wasserstein GAN — replaced the discriminator's binary "real/fake" with a continuous score measuring distribution distance (Wasserstein/Earth Mover's distance). Solved GAN training's two worst problems: mode collapse (generator only producing one type of output) and vanishing gradients. The math was unusually rigorous — Arjovsky was a math PhD student at NYU's Courant Institute, and it shows |

---

> [Back to Timeline Hub](00_timeline.md) | [Study Guide](00e_timeline_study_guide.md)
