# Repository Guidelines

## Project Structure & Module Organization

This repository is a personal study collection rather than a packaged application. `notes/` contains the main Markdown material: numbered files at its root follow the broad curriculum, while its folders hold topic-specific series. The folders on the LLM learning path carry a two-digit rank in study order, from `01_exponential-and-logarithm/` through `09_ddpm/`; a new folder on that path takes the rank of its place, with a letter suffix such as `04b_` when it falls between two ranks. Folders off the path, such as `turing/` and `godel/`, stay unnumbered. Use `00_overview.md` to introduce a new series. `TODO/` stores unread papers; completed papers belong in `human_cognitive_ai/`, `general_ai_and_society/`, or `CUDA class/`. Keep the reading lists in `README.md` synchronized when moving papers. Standalone browser demonstrations live in `DSA visualizations/`; the only current Python program is `notes/turing/turing_machine_simulator.py`. Private CS 6601 study material goes in the gitignored `AI Fall 2026/CS 6601 Artificial Intelligence/private/` folder, which exists only in the main checkout, not in worktrees.

## Development and Validation Commands

There is no build system or dependency manifest. Run checks from the repository root:

- `python notes/turing/turing_machine_simulator.py` runs all simulator examples.
- `python -m py_compile notes/turing/turing_machine_simulator.py` checks Python syntax.
- `python -m http.server 8000` serves the repository for reviewing HTML visualizations at `http://localhost:8000/DSA%20visualizations/`.
- `git diff --check` detects whitespace errors before committing.

## Coding Style & Naming Conventions

Write focused Markdown files with descriptive headings and short, educational explanations. Use GFM math (`$...$` and `$$...$$`) instead of Unicode approximations, and include historical context or ASCII diagrams when they improve understanding. Root note names use two-digit ordering and lowercase snake case, for example `19_transformers.md`; insertions use suffixes such as `17b_model_complexity_and_error_tradeoffs.md`. For Python, use four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and comments that explain why an operation matters.

## Testing Guidelines

No automated test framework or coverage target is configured. For code changes, run the affected script and its syntax check. For notes, preview Markdown and verify headings, local links, tables, formulas, and image paths. For HTML changes, exercise controls and inspect the page at common desktop and mobile widths.

## Commit & Pull Request Guidelines

History favors short summaries beginning with verbs such as `Add`, `Update`, or `Create`. Make them specific (`Add DDPM reverse-process notes`) rather than generic (`more notes`). Keep each commit focused. Commit and push to `origin/main` automatically after finishing each change, without asking first; commit only the files the change touched, and pull first if `main` is behind. Pull requests should explain the purpose, list important paths changed, cite sources for substantive research additions, and link relevant issues. Include screenshots for visual HTML changes and call out any README index updates or moved binary files.
