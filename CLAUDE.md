# CLAUDE.md

Guidance for Claude when working with Damian on the mitarget project (empirical-Bayes target encoding and mutual-information estimation for large, under-sampled categorical features).

## Working with Damian

- **Language.** Reply in the language Damian writes in (Spanish or English). Code, commit messages, PRs, docs in the repos and the manuscript are always in English.
- **Who he is.** Physicist / statistician, author of the 2019 Bayesian MI estimator this project builds on. He is comfortable with full derivations: give the math (moments, likelihoods, limits) rather than hand-waving, and connect results to the literature (empirical Bayes, Robbins/Good-Turing, Valiant & Valiant, NPMLE).
- **Answer first, then the reasoning.** When he asks "what next?" or "is it worth it?", give a short numbered list in the order you would do it, with one recommendation. He picks by number ("let's go with 1 and 3").
- **Discussion vs. edits.** Conceptual questions ("let's discuss", "how should I interpret...") get an explanation, not a PR. Edit files only when he asks, or offer the edit in one line.
- **Be honest about novelty and evidence.** Name prior work that already contains a result. Say which claims come from a run and which are interpretation not backed by a run. Report negative or null results plainly (e.g. "ties the baseline within noise").
- **Quantify.** Report gaps in nats and as a share of the baseline's gain over the prior-only predictor; use paired tests (Nadeau-Bengio corrected, level-clustered SE) before claiming one method beats another.
- **Scope of changes.** Before changing a method, say what it implies for earlier results and what would need rerunning. Prefer adding a *named variant* (e.g. `baseline_unseen`, `baseline_robbins`) over redefining the baseline; the theory is about the plain beta baseline.
- **Settled decisions (do not reopen unless he does).** NPMLE is the "richer prior" in the paper; K-mixtures and CRP are dropped. The likelihood-region interval is the main MI interval; the moment interval is a conservative fallback in the appendix. Recommendation on m: "fit m, don't fix it".
- **Workflow.** One PR per coherent change, on the assigned branch; Damian reviews and merges himself. Paper changes that need new numbers come as a pair: scripts/tables PR in `mitarget` + text PR in `mitarget-paper`, cross-linked.
- **Results.** Each study writes its outputs and a short `README.md` (what was run, command, key numbers, caveats) to its own folder in the project files, e.g. `/mnt/project-files/<study>/`.

## Notation (paper convention)

- `p_x`: popularity (test weight) of level x; `q_x`: P(Y=1 | X=x); `n_x`: training count of level x.
- `m_j = E[q^j]`: prior moments; `F_n` (and `F_{n,k}`): number of levels with n rows (k positives); `\nlev = |X|` number of levels.
- `h`: (binary) entropy; pseudo-count `\hat s`; `pi^lev_n`: level-weighted law of q among levels of size n; `pi*_n`: test-weighted prior of stratum n.
- Older notes (`near_oracle.tex`) used `q_x` for popularity: always translate to the paper notation.

## This repository: `mitarget` (package + experiment scripts)

- Layout: package in `src/mitarget/` (baseline, competitors = GLMM/NPMLE/CatBoost, freq_prior, mi_interval, exchangeability, synthetic/); package benchmarks in `benchmarks/`; paper experiments in `experiments/`; figure/table generators for the manuscript in `experiments/paper/` (verbatim theory checks in `experiments/paper/theory/`, excluded from ruff).
- Methods are selected by name in `fit_counts(n, method)`; new estimators go in as new names with tests, never by changing an existing method's behaviour.
- Checks before every PR: `pip install -e ".[dev]"`, `ruff check .` (E/F/B, line length 110) and `pytest -q`. CI runs Python 3.10 and 3.12. If GitHub Actions fails at startup it is usually billing, not the code: say so and report the local results.
- Data is never committed (`/data/` is git-ignored); scripts take paths on the command line. Large raw data lives in `/mnt/project-files/datasets/`.
- Sweeps: run with `OMP_NUM_THREADS=1` and several worker processes (about 5x faster). Long runs: estimate the time first and say it.
- Tables for the paper are generated here and copied into `mitarget-paper/tables/`; if a table is edited by hand in the paper, update the generator too.
- Double-blind supplement: `scripts/anonymous_supplement.sh -n sparsete -r RESULTS` renames the package and adds result CSVs; tests must pass after the rename.
- Network notes: GroupLens, OpenML and Kaggle downloads are blocked here; GitHub raw, PyPI and Hugging Face work.
