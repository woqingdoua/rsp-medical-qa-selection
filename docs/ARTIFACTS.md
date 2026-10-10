# Artifact mapping

- `original_rsp_raw_utils.py`, `score_original_rsp_raw.py`: flip/contraction score and acquisition field allowlist.
- `train_original_rsp_raw.py`: exact final-answer-only LoRA branch training.
- `evaluate_original_rsp_raw.py`, `evaluate_original_rsp_selected.py`: raw and selected branch evaluation.
- `make_original_rsp_expanded_dev_lambda_manifest.py`, `select_original_rsp_expanded_dev_lambda.py`: expanded development selection.
- `score_original_rsp_extra_baselines.py`, `score_latest_*.py`, `train_latest_temp_direction.py`, `rationale_baselines.py`: adapted comparator implementations.
- `make_entropy_prefiltered_random_manifest.py`: pool-matched random comparator.
- `evaluate_raw_checkpoint_no_update.py`, `make_raw_checkpoint_no_update_manifest.py`, `aggregate_raw_checkpoint_no_update.py`: archived no-update reference pipeline.
- `results/updated_summary_results.csv`, `updated_per_seed_results.csv`: final leakage-filtered aggregate/per-seed metrics.
- `results/three_dataset_method_summary.csv`: restricted benchmark interpretation subset.
- `results/medexpqa_dev_lambda*.csv`: development overlap sensitivity summaries.
- `docs/paper_config.json`: final paper hyperparameters and full lambda union.
- `docs/source_code_manifest.json`: explicit copied-code list, original hashes and portability edits.

Camera-ready input: GenAI4Health-Paper-97-Camera-Ready-Source.zip, Library version 3, materialized and inspected locally. Anonymous prediction arrays, overlap rows, selection identifiers, manuscripts and third-party styles are excluded from the release.

- `analysis/recompute_existing_artifacts.py`: exact camera-ready paired bootstrap and score-scale computation from retained predictions.
- `analysis/recompute_submission_sensitivities.py`: exact final overlap/alignment audit, post-hoc evaluation filter, development-lambda sensitivity and five-/three-dataset aggregates. Item-level outputs must be reviewed before redistribution.
