# RSP: development-tuned representation sensitivity for medical QA selection

Research code for **An Empirical Study of Development-Tuned Representation Sensitivity for Medical QA Data Selection**, Yuexin Wu, Dayou Yu, and Vasile Rus, GenAI4Health 2026. [Paper (arXiv PDF)](https://arxiv.org/pdf/2609.09684).

Representation-Sensitivity Prioritization first restricts an existing labeled bank to its top `min(4B,N)` predictive-entropy examples. It reranks that pool using answer-flip frequency and non-flipping contraction of the top-two answer margin under final-block hidden-state perturbations. Candidate scoring reads questions/options without pool labels or auxiliary text; labeled development outcomes tune the scale coefficient. The final retained examples include auxiliary text as context, while only final answer tokens are supervised.

The study evaluates five medical QA benchmarks with `google/medgemma-4b-it`, budget B=128, and seeds 13/23/37. It reports descriptive, dataset-dependent improvements; paired intervals versus the pool-matched random comparator include zero. This is exploratory evidence, not established statistical superiority or clinical deployment readiness.

The release includes exact scoring/training/evaluation code, paper configuration, selected aggregate tables, and synthetic tests. No benchmark question text, model weights, selection UID lists, or original git history is redistributed. See [reproducibility](docs/REPRODUCIBILITY.md), [artifact mapping](docs/ARTIFACTS.md), and [license status](LICENSE_STATUS.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tests/synthetic_smoke.py
python -m unittest discover -p 'test_original_rsp*.py'
```

Use Python 3.11. Requirements describe necessary packages; exact historical versions and model commit were not retained. Obtaining MedGemma requires your own authorized access and compliance with the publisher's terms. No models are downloaded by the synthetic smoke test.

## Scientific limits

- RSP uses 78 development-evaluated fits and 15 tested winners; comparator tuning budgets are unequal. The retained-set budget does not equalize scoring, tuning, annotation, or total compute cost.
- The expanded lambda grid was designed after earlier analyses used the same test cohorts. Results remain exploratory even when the final coefficient selection uses development outcomes.
- Historical model revisions were unpinned. An archived no-update reference is not a fully matched causal before/after control.
- Benchmarks differ in their relationship to backbone post-training sources. MedExpQA has acquisition-pool overlap with development/test questions. Removing four test rows and two development rows does not undo overlapping training or create a fully deduplicated new experiment; repeated within-test questions remain.
- No matched answer-only control identifies a causal benefit of auxiliary-text conditioning. Adapted baselines are not exact reproductions of their source methods.

No additional experiments, model downloads, inference, or training were performed for this release. Citation metadata contains no invented DOI.
