# The LLM Era (2018-Now)

> [Previous: Resurrection & Revolution](00b_timeline_1986-2017_resurrection_and_revolution.md) | [Back to Timeline Hub](00_timeline.md)

| Year | What | Who | Key Idea |
|------|------|-----|----------|
| 2018 | **BERT** | Devlin et al. (Google) | Encoder-only transformer, pretrained with masked language modeling. `TransformerTranslator` is this style |
| 2018 | **GPT-1** | Radford et al. (OpenAI) | Decoder-only transformer, autoregressive pretraining on BookCorpus |
| 2018 | **StyleGAN** | Karras, Laine, Aila (NVIDIA) | Generated photorealistic human faces so convincing they spawned thispersondoesnotexist.com — the moment the public realized AI could create people who never existed. Used progressive growing (start at 4×4 resolution, gradually increase) and style mixing at different layers. Cultural shock and deepfake anxiety followed |
| 2019 | **GPT-2** | OpenAI | 1.5B params. Showed scaling up decoder-only transformers yields strong zero-shot performance |
| 2019 | **T5** | Google | "Text-to-Text Transfer Transformer" — frames every NLP task as text-in, text-out |
| 2019 | **AlphaStar** | Vinyals et al. (DeepMind) | Grandmaster-level StarCraft II — real-time, imperfect information, multi-unit coordination. Top 0.2% of human players |
| 2019 | **OpenAI Five** | OpenAI | Beat world champions at Dota 2 (5v5). Trained with PPO — the same algorithm later used for RLHF in ChatGPT |
| 2019 | **Score matching** | Song & Ermon (Stanford) | "Generative Modeling by Estimating Gradients of the Data Distribution" — instead of learning p(x) directly, learn the *gradient* of log p(x) (the "score function") and follow it to generate data. Unified denoising autoencoders (2008), Boltzmann machines (1985), and noise-based generation into one framework. The theoretical foundation that made DDPM (2020) possible |
| 2020 | **GPT-3** | OpenAI | 175B params. In-context learning — few-shot via prompting, no fine-tuning needed |
| 2020 | **AlphaFold 2** | Jumper et al. (DeepMind) | Solved protein folding — predicted 3D protein structures with atomic accuracy. RL + attention. Nobel Prize in Chemistry 2024 |
| 2020 | **Diffusion models (DDPM)** | Ho, Jain, Abbeel (UC Berkeley) | Denoising Diffusion — gradually add noise, learn to reverse it. Stable training, sharp outputs. Eventually dethroned GANs for image generation. The same "corrupt then reconstruct" idea as denoising autoencoders (2008), scaled to thousands of noise steps |
| 2020 | **Vision Transformer (ViT)** | Dosovitskiy et al. (Google) | Applied transformers to images by splitting into patches. Transformers escape NLP |
| 2021 | **DALL-E** | OpenAI | Transformer generates images from text descriptions |
| 2022 | **ChatGPT / InstructGPT** | OpenAI | RLHF (reinforcement learning from human feedback) aligns GPT-3.5 to follow instructions |
| 2022 | **Stable Diffusion** | Rombach et al. | Latent diffusion — run denoising in compressed latent space, not pixel space. Open-sourced, triggered the AI art explosion |
| 2022 | **Chinchilla** | Hoffmann et al. (DeepMind) | Scaling laws — models were over-parameterized & under-trained. Train longer on more data |
| 2023 | **GPT-4** | OpenAI | Multimodal (text + vision), massive leap in reasoning |
| 2023 | **LLaMA** | Meta | Open-weight LLMs. Kicked off the open-source LLM explosion |
| 2023 | **Mixture of Experts (MoE)** | Mistral, Google | Only activate a subset of parameters per token — scale without scaling compute linearly |
| 2024 | **State-space models (Mamba)** | Gu & Dao | Alternative to attention — linear-time sequence modeling, challenging transformer dominance for long sequences |
| 2024-25 | **Reasoning models** | OpenAI (o1/o3), Anthropic (Claude), DeepSeek (R1) | Chain-of-thought at inference time. Models "think" before answering |

---

> [Back to Timeline Hub](00_timeline.md) | [Study Guide](00e_timeline_study_guide.md)
