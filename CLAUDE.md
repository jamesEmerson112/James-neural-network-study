# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Personal study repository for neural networks, NLP, AI, and ML history. Most of it is Markdown study notes (`notes/`) and research-paper PDFs; there are also self-contained interactive HTML lessons and one real web app (the route lab below). There is no repo-wide build system or linter. `AGENTS.md` holds overlapping guidance for other agents — keep the two consistent when conventions change.

## Commands

Run from the repository root unless noted.

- Turing simulator (the only Python program): `python notes/turing/turing_machine_simulator.py` runs every demo.
- HTML lessons: open the file directly, or serve with `python -m http.server 8000` (needed for pages that load ES modules, e.g. the Three.js lab).
- Route lab — `cd "AI Fall 2026/CS 6601 Artificial Intelligence/metropolitan-route-lab"` (quote the path, it has spaces):
  - `npm ci` then `npm run dev`
  - `npm test` (Vitest; ~15 s because `beforeAll` preprocesses the whole graph)
  - Single test: `npx vitest run -t "matches Dijkstra"`
  - `npm run build` (`tsc -b && vite build`), `npm run benchmark`
  - `npm run data:refresh` re-fetches the OSM snapshot from public Overpass servers — only run it deliberately (see its `DATA_LICENSE.md`).

## Repository Structure

- `notes/` — the main content. Curriculum notes live at the root with numbered prefixes (`00_timeline.md` … `35_calibration_and_fairness.md`); deeper series live in topic subfolders (`ddpm/`, `rl/`, `turing/`, `godel/`, `exponential-and-logarithm/`, `llm-inference-stack/`, `NVIDIA_2026/`, `researchers/`, etc.).
- `AI Fall 2026/CS 6601 Artificial Intelligence/` — course material: numbered single-file HTML labs (`03_` … `09_`), reference PDFs under `bidirectional search/resources/`, and the `metropolitan-route-lab/` app.
- `DSA visualizations/` — standalone single-file HTML visualizations of graph algorithms.
- `TODO/` — papers not yet read. `human_cognitive_ai/`, `general_ai_and_society/`, `CUDA class/` — read papers, organized by topic.
- `README.md` — the reading index: a "To Read" table mirroring `TODO/` and a "Read" list mirroring the topic folders. When a paper is added or moves from `TODO/` into a topic folder, update README.md to match.
- `notes/job_insights/` and `notes/mira/` are private career notes listed in `.gitignore`; the existing files were committed before that rule, so edits to them are still tracked, but new files there are ignored.

## Paper Files

PDFs are named after the paper's title in Title Case. A title colon becomes ` - ` (e.g. `AI-Driven Clinical Decision Support - Enhancing Disease Diagnosis Exploiting Patients Similarity.pdf`), since the files must also work on Windows.

## Route Lab Architecture (`metropolitan-route-lab/`)

React + TypeScript + Vite + MapLibre GL app comparing exact shortest-path algorithms on a real OpenStreetMap graph of downtown San Francisco.

- `src/data/san-francisco.graph.json` is a checked-in, immutable OSM snapshot so results are reproducible; `runtimeGraph.ts` turns it into adjacency lists.
- `src/algorithms/search.ts` holds all the algorithms (the `AlgorithmId` union in `types.ts`: Dijkstra, bidirectional Dijkstra, A*, ALT, reach, reach+shortcuts, combo). `preprocessGraph` builds the landmark/reach/shortcut indexes; `runSearch` answers one query.
- `src/workers/search.worker.ts` runs preprocessing and queries off the main thread; the message protocol is the `WorkerRequest`/`WorkerResponse` types in `types.ts`. Preprocessed indexes are cached in IndexedDB (`data/indexCache.ts`).
- Dijkstra is the correctness oracle: `search.test.ts` checks that every optimized method returns Dijkstra's distance. A new algorithm must be added to that comparison.
- "Shortcuts" are exact degree-two corridor contractions that still unpack to the original road-node path.
- `dist/` is committed and linked from `06_metropolitan_route_search.html` (relative `base: "./"`), so rebuild and commit `dist/` after changing source.

## Note Conventions

- One file per topic; create separate files rather than appending unrelated sections to an existing one.
- Root notes use two-digit prefixes in rough chronological/curriculum order with lowercase snake_case names. Insertions between existing numbers get a letter suffix (`00a_`, `06b_`, `17b_`). A new topic subfolder starts with a `00_overview.md` (some older ones, like `ddpm/` and `rl/`, start at `01_`).
- Math uses GFM LaTeX syntax (`$...$` inline, `$$...$$` display), never Unicode approximations.
- Notes weave in historical context: who created the idea, why, what inspired them, and what happened after. The `history-lore-researcher` agent (`.claude/agents/`) exists for researching this.
- ASCII diagrams are used liberally for visual intuition (timelines, architecture sketches, geometric comparisons).
- HTML lessons are single self-contained files (inline CSS/JS; any library from a pinned CDN URL) so they open without a build step.

## Guidelines

- Prioritize clear, educational code with comments explaining *why* each step matters, not just *what* it does.
- Use Python with PyTorch or Hugging Face Transformers as the default stack unless otherwise specified.
- Commit messages: a short summary starting with a verb (`Add`, `Update`, `Create`) that names the topic, e.g. `Add DDPM reverse-process notes`, not `more notes`.
- Commit and push to `origin/main` automatically after finishing each change; don't ask first. Commit only the files the change touched, and pull first if `main` is behind.
