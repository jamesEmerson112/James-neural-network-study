# Study Guide — Deep Dives & Detailed Notes

> Part of the [Timeline Hub](00_timeline.md).

All notes in this repository, organized by narrative arc.

### Foundations & Historical Figures

- [Kurt Gödel — Incompleteness & the Limits of Formal Systems](godel/00_overview.md)
- [Alan Turing — Computation, Codebreaking & Morphogenesis](turing/00_overview.md)
- [John von Neumann — The Bridge Between Brains and Machines](von_neumann/00_overview.md)
- [John Nash — The Man Who Solved the Game](nash/00_overview.md) — Nash equilibrium, game theory, and why GANs are so hard to train
- [McCulloch-Pitts & the 1943 Scene](01_mcculloch_pitts_and_the_1943_scene.md) — the neuron that fused Boolean logic, neuroscience, and Turing

### Pioneers & AI Winter

- [Rosenblatt and the Perceptron](02_rosenblatt_and_the_perceptron.md) — the first learnable neuron
- [ADALINE as Noise Filter](03_adaline_as_noise_filter.md) — continuous error and the birth of gradient descent
- [John McCarthy](04_john_mccarthy.md) — the man who named AI and invented LISP
- [Why LISP Matters](05_lisp_why_it_matters.md) — code as data, garbage collection, and AI's first language
- [LISP Deep Dive](05b_lisp_deep_dive.md) — technical deep dive into LISP internals
- [Minsky, Perceptrons — Death and Resurrection](06_minsky_perceptrons_death_and_resurrection.md) — how one book killed neural nets for 15 years

### Backprop, RNNs & LSTMs

- [Backprop: The Wiggle Ratio](09_backprop_the_wiggle_ratio.md) — chain rule through layers, intuitively
- [MLP, Backprop, and the Birth of RNNs](10_mlp_backprop_and_the_birth_of_rnns.md) — multi-layer networks learn
- [Vanishing Gradient and tanh](13_vanishing_gradient_and_tanh.md) — why deep nets forget
- [LSTM vs Seq2Seq vs Transformer](07_lstm_vs_seq2seq_vs_transformer.md) — three architectures compared
- [Phase 1: Naive LSTM](08_phase1_naive_lstm.md) — building LSTM from scratch
- [LSTM: The Memory Machine](11_lstm_the_memory_machine.md) — gates, cell state, and constant error carousel
- [NN LSTM Inputs and Outputs](15_nn_lstm_inputs_and_outputs.md) — practical tensor shapes and data flow
- [Softmax: Turning Scores into Probabilities](22_softmax.md) — logits, temperature, and why CrossEntropyLoss includes it

### Seq2Seq, Attention & Transformers

- [From Phrase Tables to Seq2Seq](14_from_phrase_tables_to_seq2seq.md) — machine translation's evolution
- [Assignment 3 Battle Plan](12_assignment3_battle_plan.md) — hands-on implementation strategy
- [Transformers](19_transformers.md) — encoder/decoder, self-attention, positional encoding
- [NLP Models Overview](18_nlp_models_overview.md) — ELMo → BERT → GPT word representation evolution
- [BERT](20_bert.md) — masked language modeling, encoder-only Transformer

### Vision

- [CNN Vision Tasks](17_cnn_vision_tasks.md) — convolutional neural networks for image understanding
- [Do Vision Transformers See Like CNNs?](16_do_vision_transformers_see_like_cnns.md) — ViT vs CNN comparison

### Attention Mechanics

- [Scaled Dot-Product Attention](23_scaled_dot_product_attention.md) — why √d, dot product vs cosine similarity, the Goldilocks argument

### Generative Models

- [Generative Models Taxonomy](24_generative_models_taxonomy.md) — the full tree: VAEs, GANs, Boltzmann machines, autoregressive models, diffusion models
- [From Turing to Diffusion Models](turing/05_turing_to_diffusion_models_the_lineage.md) — the full lineage from Turing's morphogenesis to DALL-E

### Learning Paradigms & Reinforcement Learning

- [The Three Paradigms of Learning](26_three_paradigms_of_learning.md) — how supervised, unsupervised, and reinforcement learning emerged, evolved, and converged — from Legendre (1805) to RLHF (2022)
- [Reinforcement Learning Overview](25_reinforcement_learning_overview.md) — trial-and-error learning, the game pipeline (checkers → AlphaGo → RLHF), why RL is hard, real-world applications

### LLM Inference & Benchmarks

- [LLM Inference Stack](llm-inference-stack/) — how LLMs actually run in production
- [LLM Benchmarks](21_llm_benchmarks.md) — benchmark descriptions and what they measure

### Applied ML & Forecasting

- [SPADE, GEANN, and Goldilocks](30_spade_geann_goldilocks.md) — Vincent's Amazon forecasting trilogy: cold-start via graphs, peak decomposition via attention, training selection via bandits

### Programming Languages

- [OCaml vs Haskell vs Erlang](programming-languages/ocaml-vs-haskell-vs-erlang.md) — three functional programming traditions compared. Connected to the LISP lineage: [McCarthy](04_john_mccarthy.md) → [Why LISP Matters](05_lisp_why_it_matters.md) → Lambda Calculus → ML → OCaml/Haskell
