# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Personal study repository for neural networks, NLP, and ML history. It contains two kinds of content: Markdown study notes (`notes/`) and a collection of research papers as PDFs. There is no build system, linter, or test suite — the only runnable code is the occasional standalone Python script (e.g. `notes/turing/turing_machine_simulator.py`), run directly with `python`.

## Repository Structure

- `notes/` — the main content. General neural-network/NLP notes live at the root with numbered prefixes (`00_timeline.md` … `30_spade_geann_goldilocks.md`); deeper dives live in topic subfolders (`ddpm/`, `rl/`, `turing/`, `von_neumann/`, `godel/`, `nash/`, `exponential-and-logarithm/`, `llm-inference-stack/`, `clrs/`, etc.).
- `TODO/` — papers not yet read.
- `human_cognitive_ai/`, `general_ai_and_society/`, `CUDA class/` — papers that have been read, organized by topic.
- `README.md` — the reading index: a "To Read" table mirroring `TODO/` and a "Read" list mirroring the topic folders. When a paper moves from `TODO/` into a topic folder, update README.md to match.

## Note Conventions

- One file per topic; create separate files rather than appending unrelated sections to an existing one.
- Root notes use numbered prefixes in rough chronological/curriculum order. Insertions between existing numbers get a letter suffix (`00a_`, `06b_`, `17b_`). Topic subfolders start with a `00_overview.md`.
- Math uses GFM LaTeX syntax (`$...$` inline, `$$...$$` display), never Unicode approximations.
- Notes weave in historical context — who created the idea, why, what inspired them, and what happened after. The `history-lore-researcher` agent (`.claude/agents/`) exists for researching this.
- ASCII diagrams are used liberally for visual intuition (timelines, architecture sketches, geometric comparisons).

## Guidelines

- Prioritize clear, educational code with comments explaining *why* each step matters, not just *what* it does.
- Use Python with PyTorch or Hugging Face Transformers as the default stack unless otherwise specified.
