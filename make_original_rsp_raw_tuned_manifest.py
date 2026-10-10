#!/usr/bin/env python
"""Create manifests for the four-dataset dev-tuned Original RSP experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import write_json, write_jsonl
from derive_original_rsp_lambda_scores import LAMBDAS, method_name
from original_rsp_raw_utils import sha256


DATASETS = ("afrimedqa", "medexpqa", "medexqa", "pubmedqa")
SEEDS = (13, 23, 37)
BASELINES = ("random", "entropy", "margin")
BUDGET = 128


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    args = parser.parse_args()

    args.run_root.mkdir(parents=True, exist_ok=True)
    data_manifest = {}
    for dataset in DATASETS:
        data_dir = args.data_root / dataset
        metadata_path = data_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pool_rows = sum(1 for line in (data_dir / "pool.jsonl").open(encoding="utf-8") if line.strip())
        if pool_rows < BUDGET:
            raise ValueError(f"{dataset} has only {pool_rows} pool rows")
        data_manifest[dataset] = {
            "metadata_file": str(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "source_version": metadata["source_version"],
            "split_sizes": metadata["split_sizes"],
            "uid_sha256": metadata["uid_sha256"],
            "files": {
                split: sha256(data_dir / f"{split}.jsonl")
                for split in ("pool", "dev", "test")
            },
        }

    rows = []
    for dataset in DATASETS:
        data_dir = args.data_root / dataset
        for seed in SEEDS:
            score_file = args.run_root / "scores" / dataset / f"seed_{seed}" / "scores_tuned.jsonl"
            for method in BASELINES:
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "lambda": None,
                        "budget": BUDGET,
                        "eval_splits": ["dev", "test"],
                        "data_dir": str(data_dir),
                        "score_file": str(score_file),
                        "output_dir": str(
                            args.run_root / "branches" / dataset / f"seed_{seed}" / method
                        ),
                    }
                )
            for value in LAMBDAS:
                method = method_name(value)
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "lambda": value,
                        "budget": BUDGET,
                        "eval_splits": ["dev"],
                        "data_dir": str(data_dir),
                        "score_file": str(score_file),
                        "output_dir": str(
                            args.run_root / "branches" / dataset / f"seed_{seed}" / method
                        ),
                    }
                )
    write_jsonl(args.run_root / "training_manifest.jsonl", rows)
    protocol = {
        "name": "Original RSP--Raw four-dataset tuned-lambda extension",
        "status": "exploratory extension",
        "backbone": "google/medgemma-4b-it",
        "answer_only_warmup": False,
        "datasets": list(DATASETS),
        "data_manifest": data_manifest,
        "acquisition_training_seeds": list(SEEDS),
        "budget_items": BUDGET,
        "selector_visible_fields": ["uid", "question", "options", "domain"],
        "baselines": list(BASELINES),
        "rsp": {
            "formula": "Flip + lambda * non-flip positive confidence-margin Drop",
            "lambda_grid": list(LAMBDAS),
            "selection_rule": (
                "Per dataset, maximize mean dev accuracy across seeds; break ties by mean "
                "dev macro-F1 and then smaller lambda. Only development outcomes are used for selection; historical test cohorts were examined earlier."
            ),
            "entropy_prefilter_multiplier": 4,
            "passes": 8,
            "layer": -1,
            "absolute_sigma": 0.02,
        },
        "training": {
            "initialization": "fresh LoRA on raw backbone",
            "rationale": "input context",
            "loss": "final-answer tokens only",
            "epochs": 2.0,
            "answer_replay": False,
        },
        "candidate_training_rows": len(rows),
        "expected_baseline_rows": len(DATASETS) * len(SEEDS) * len(BASELINES),
        "expected_rsp_candidate_rows": len(DATASETS) * len(SEEDS) * len(LAMBDAS),
        "project_root": str(args.project_root),
        "data_root": str(args.data_root),
        "run_root": str(args.run_root),
    }
    write_json(args.run_root / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
