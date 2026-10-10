# Reproducing the protocol

Acquire AfriMed-QA, MedExpQA, MedExQA, PubMedQA and MedMCQA from their original distributors with the required permissions. The data preparation scripts name the dataset sources and schemas. Do not assume a common redistribution license. Place authorized data outside tracked paths, using JSONL records with `uid`, `question`, `options` (letter-to-text map), `answer_letter`, `rationale` and optional `domain`. `pool.jsonl`, `dev.jsonl`, `test.jsonl` and `metadata.json` are expected per dataset. Synthetic examples are schema demonstrations only.

`prepare_additional_medical.py` prepares the five-dataset bank; `prepare_original_rsp_raw_data.py` is an earlier MedMCQA/MedPrompt protocol and must not be substituted for the paper split. Filesystem defaults have been made relative; source hashes record those edits. Existing full-precision aggregate evidence is retained, but exact historical splits require controlled source manifests.

The exact paper settings are in `paper_config.json`: final-block output perturbations, absolute Gaussian sigma 0.02, K=8 draws, entropy multiplier 4, B=128; LoRA rank 16, scale 32, dropout 0.05; two epochs, learning rate 1e-4, cosine schedule, warmup 0.03, fused AdamW, batch 1/accumulation 8, maximum length 1024. No answer-only warm-up and no requested quantization. Historical resolved precision and package versions were not retained.

Commands below describe reproduction for an authorized GPU environment. They were not run during release preparation.

```bash
python score_original_rsp_raw.py --phase clean --dataset medexpqa \
  --data_dir data/additional_medical/medexpqa --output_dir runs/demo/clean \
  --cache_dir model_cache --budget 128 --noise_layer -1 --noise_sigma 0.02
python score_original_rsp_raw.py --phase rsp --dataset medexpqa \
  --data_dir data/additional_medical/medexpqa --clean_dir runs/demo/clean \
  --output_dir runs/demo/rsp --cache_dir model_cache --budget 128 \
  --prefilter_mult 4 --mc_passes 8 --noise_layer -1 --noise_sigma 0.02 --seed 13
python train_original_rsp_raw.py --help
python evaluate_original_rsp_selected.py --help
```

Score defaults include legacy settings; always supply paper budget/noise settings. The original lambda derivation module's default grid is historical. The final grid is explicitly recorded in `paper_config.json`; use the expanded development manifest and its selection script rather than reporting a default-grid run as the final paper. Development selection maximizes mean accuracy, then mean Macro-F1, then prefers smaller lambda; signatures deduplicate identical ordered selections. Only winners proceed to test evaluation. Preserve seed pairing and normalized-question overlap filtering.

Scripts for downstream training/evaluation accept dataset, data paths, score file, output paths and seed. Inspect `--help` and the generated manifests. Test scoring uses restricted next-token softmax over legal answer letters. Training places auxiliary text in context and masks prompt tokens from the supervised loss. The final answer-only target is distinct from rationale-token supervision in earlier project variants.

Historical raw-checkpoint run: `raw_checkpoint_no_update_five_20260920`, no adapter/training, seed 13, max length 1024. `results/archived_no_update_summary.json` preserves the camera-ready aggregate reference and its provenance caveats. No missing control was recreated.

The manuscript's paired bootstrap requires retained item-level predictions with verified order and gold-vector alignment; aggregates alone cannot reproduce it. Such arrays are deliberately excluded from this conservative public release. Full bitwise historical replication is not promised because model revision, environment and some input identifiers are unavailable.

## Final development branch and tested winner example

After deriving `rsp_lambda_448` from the saved score components (the MedExpQA winner, selected using the development-only aggregate rule), the exact downstream branch interface is:

```bash
python train_original_rsp_raw.py --dataset medexpqa \
  --data_dir data/additional_medical/medexpqa --score_file runs/demo/derived_scores.jsonl \
  --output_dir runs/demo/dev_branch --method rsp_lambda_448 --budget 128 \
  --cache_dir model_cache --seed 13 --max_length 1024 --train_batch_size 1 \
  --gradient_accumulation_steps 8 --epochs 2 --learning_rate 1e-4 \
  --lora_rank 16 --lora_alpha 32 --eval_splits dev
python evaluate_original_rsp_selected.py --dataset medexpqa \
  --data_dir data/additional_medical/medexpqa \
  --adapter_path runs/demo/dev_branch/adapter --training_result runs/demo/dev_branch/result.json \
  --output_dir runs/demo/test_winner --method rsp_lambda_448 --lambda_value 448 \
  --cache_dir model_cache --seed 13 --max_length 1024
```

Repeat branches with seeds 13/23/37 and the final dataset-specific lambda winners: AfriMed-QA .25, MedExpQA 448, MedExQA 24, PubMedQA 0, MedMCQA 56. These commands describe the method, not authorization to start a new experiment. Historical run-family paths are recorded below; rebuilding new scores does not reconstruct unpinned historical weights.

`analysis/recompute_existing_artifacts.py` and `analysis/recompute_submission_sensitivities.py` are the exact camera-ready deterministic analysis code. The latter verifies gold-vector/order alignment and recomputes overlap filtering before final metrics. It can output anonymous arrays and row-level overlap reports for local research; those outputs are not automatically authorized for public redistribution. This release includes the code and selected aggregate tables only.

Canonical artifact families: `original_rsp_raw_five_lambda_expanded_dev_20260902` (RSP and uncertainty/diversity baselines), `latest_five_baselines_20260903` (RSR/ASLEC/NAIT), `latest_five_baselines_20260903_temp_recalibrated` (TEMP), `entropy_prefiltered_random_20260904` (dedicated Random@4B). The raw-checkpoint archival reference comes from `raw_checkpoint_no_update_five_20260920`.
