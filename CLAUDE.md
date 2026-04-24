# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a personal study repository for learning neural network concepts, with a current focus on NLP models (BERT, ELMo, Transformers, etc.).

## Repository Structure

This is an early-stage learning repository. Code and notebooks will be organized by topic as study progresses.

## Key Concepts Being Studied

- Neural network fundamentals
- NLP model architectures (ELMo, BERT, Transformers)
- Contextualized word embeddings vs static embeddings

## Guidelines

- Prioritize clear, educational code with explanatory comments over production-grade abstractions
- When creating examples, include inline explanations of *why* each step matters, not just *what* it does
- Use Python with PyTorch or Hugging Face Transformers as the default stack unless otherwise specified

## Context History

### 2026-04-19 / 2026-04-20 (late night session)
- [feat] Expanded `notes/quiz5/quiz_5_11_calibration_and_fairness_impossibility.md` — added confusion matrix primer (FPR/FNR definitions), 7-step worked numerical example proving calibration + equal FPR + equal FNR can't coexist when base rates differ, plus forced-equal-FPR scenario showing FNR explosion to 85%
- [feat] Created 4 RL study note files in `notes/rl/` from UW CSE 579 PDFs: `01_markov_decision_processes.md`, `02_policy_iteration.md`, `03_temporal_difference_and_q_learning.md`, `04_policy_gradients.md`
- [research] Looked up Nirbhay Modhe (GT PhD, Dhruv Batra's MLP Lab, post-Zoox)
- [decision] Identified core career motivation: "healing people" — breaking, diagnosing, fixing. Explored biomedical engineering paths in SF (Neuralink, BCI startups, AbbVie Pleasanton, Penumbra Alameda, Mind Company SF)
- [feat] Submitted Neuralink application — drafted 4 "exceptional ability" bullets (PNA sole developer, memory optimization, BioBERT clinical ML, multimodal agent hackathon)
- [decision] Summer 2026 plan: DSA foundation gaps (DP, graphs, recursion) + math gaps (integral/sum equivalence, chain rule for log, gradient-integral swap). Resources: MIT 6.006, Skiena, targeted CLRS chapters
- Various concept explanations: backprop vs Value Iteration parallel, DQN algorithm breakdown, policy gradient likelihood ratio trick line-by-line, Rust vs Python for DSA comparison
