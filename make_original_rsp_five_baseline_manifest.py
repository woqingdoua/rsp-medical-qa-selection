#!/usr/bin/env python
"""Create the frozen five-dataset direct-baseline extension manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import write_json, write_jsonl
from derive_original_rsp_lambda_scores import LAMBDAS, method_name
from original_rsp_raw_utils import sha256


DATASETS = ("afrimedqa", "medexpqa", "medexqa", "pubmedqa", "medmcqa")
PRIOR_DATASETS = DATASETS[:-1]
SEEDS = (13, 23, 37)
BUDGET = 128
DIRECT_METHODS = ("random", "entropy", "margin")
EXTRA_METHODS = (
    "bald_hidden_dropout",
    "batchbald_hidden_dropout",
    "noise_stability",
    "sift",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--prior_run_root", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)

    prior_audit = args.prior_run_root / "completion_audit.json"
    if not prior_audit.exists():
        raise FileNotFoundError(f"Missing completed four-dataset source audit: {prior_audit}")
    if json.loads(prior_audit.read_text(encoding="utf-8")).get("status") != "complete":
        raise ValueError("Prior four-dataset run is not marked complete")

    data_manifest = {}
    for dataset in DATASETS:
        data_dir = args.data_root / dataset
        metadata_path = data_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pool_rows = sum(
            1 for line in (data_dir / "pool.jsonl").open(encoding="utf-8") if line.strip()
        )
        if pool_rows < BUDGET:
            raise ValueError(f"{dataset} has only {pool_rows} pool rows")
        data_manifest[dataset] = {
            "metadata_file": str(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "source_version": metadata.get("source_version"),
            "split_sizes": metadata.get("split_sizes"),
            "uid_sha256": metadata.get("uid_sha256"),
            "files": {
                split: sha256(data_dir / f"{split}.jsonl")
                for split in ("pool", "dev", "test")
            },
        }

    extra_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            score_file = args.run_root / "extra_scores" / dataset / f"seed_{seed}" / "scores.jsonl"
            for method in EXTRA_METHODS:
                extra_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "budget": BUDGET,
                        "eval_splits": ["dev", "test"],
                        "data_dir": str(args.data_root / dataset),
                        "score_file": str(score_file),
                        "output_dir": str(
                            args.run_root
                            / "extra_branches"
                            / dataset
                            / f"seed_{seed}"
                            / method
                        ),
                    }
                )
    write_jsonl(args.run_root / "extra_training_manifest.jsonl", extra_rows)

    medmcqa_rows = []
    dataset = "medmcqa"
    for seed in SEEDS:
        score_file = args.run_root / "rsp_scores" / dataset / f"seed_{seed}" / "scores_tuned.jsonl"
        for method in DIRECT_METHODS:
            medmcqa_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "lambda": None,
                    "budget": BUDGET,
                    "eval_splits": ["dev", "test"],
                    "data_dir": str(args.data_root / dataset),
                    "score_file": str(score_file),
                    "output_dir": str(
                        args.run_root / "medmcqa_branches" / f"seed_{seed}" / method
                    ),
                }
            )
        for value in LAMBDAS:
            method = method_name(value)
            medmcqa_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "lambda": value,
                    "budget": BUDGET,
                    "eval_splits": ["dev"],
                    "data_dir": str(args.data_root / dataset),
                    "score_file": str(score_file),
                    "output_dir": str(
                        args.run_root / "medmcqa_branches" / f"seed_{seed}" / method
                    ),
                }
            )
    write_jsonl(args.run_root / "medmcqa_training_manifest.jsonl", medmcqa_rows)

    protocol = {
        "name": "Original RSP--Raw five-dataset direct-baseline extension",
        "status": "exploratory extension",
        "backbone": "google/medgemma-4b-it",
        "answer_only_warmup": False,
        "datasets": list(DATASETS),
        "data_manifest": data_manifest,
        "acquisition_training_seeds": list(SEEDS),
        "budget_items": BUDGET,
        "selector_visible_fields": ["uid", "question", "options", "domain"],
        "direct_methods": list(DIRECT_METHODS),
        "additional_baselines": list(EXTRA_METHODS),
        "prior_four_dataset_run": {
            "path": str(args.prior_run_root),
            "completion_audit_sha256": sha256(prior_audit),
            "reused_datasets": list(PRIOR_DATASETS),
            "reused_methods": [*DIRECT_METHODS, "rsp_tuned", "raw_backbone"],
        },
        "medmcqa_rsp": {
            "formula": "Flip + lambda * non-flip positive confidence-margin Drop",
            "lambda_grid": list(LAMBDAS),
            "selection_rule": (
                "maximize mean dev accuracy across seeds; break ties by mean dev "
                "macro-F1 and then smaller lambda; test evaluated only after freeze"
            ),
            "passes": 8,
            "layer": -1,
            "absolute_sigma": 0.02,
            "entropy_prefilter_multiplier": 4,
        },
        "baseline_definitions": {
            "bald_hidden_dropout": {
                "passes": 8,
                "dropout_probability": 0.1,
                "layer": -1,
                "adaptation": "hidden-state dropout because MedGemma architectural dropout is zero",
            },
            "batchbald_hidden_dropout": {
                "passes": 8,
                "preselection": 5000,
                "joint_label_samples": 1000,
            },
            "noise_stability": {
                "passes": 8,
                "relative_sigma": 0.03,
                "layer": -1,
                "preselection": 5000,
            },
            "sift": {
                "entropy_preselection": 5000,
                "cosine_diversity_weight": 0.5,
            },
        },
        "training": {
            "initialization": "fresh LoRA on raw backbone",
            "rationale": "input context",
            "loss": "final-answer tokens only",
            "epochs": 2.0,
            "answer_replay": False,
        },
        "expected_extra_training_rows": len(extra_rows),
        "expected_medmcqa_training_rows": len(medmcqa_rows),
        "project_root": str(args.project_root),
        "data_root": str(args.data_root),
        "run_root": str(args.run_root),
    }
    write_json(args.run_root / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
