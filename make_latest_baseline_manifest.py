#!/usr/bin/env python
"""Freeze manifests for the 2026-09-03 five-baseline extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import write_json, write_jsonl
from original_rsp_raw_utils import sha256


DATASETS = ("afrimedqa", "medexpqa", "medexqa", "pubmedqa", "medmcqa")
SEEDS = (13, 23, 37)
BUDGET = 128
STATIC_METHODS = ("complexity_ft", "rsr", "aslec_drop")


def training_row(dataset: str, seed: int, method: str, score_file: Path,
                 output_dir: Path, data_root: Path) -> dict:
    return {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "budget": BUDGET,
        "eval_splits": ["dev", "test"],
        "data_dir": str(data_root / dataset),
        "score_file": str(score_file),
        "output_dir": str(output_dir),
    }


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
        metadata_file = data_dir / "metadata.json"
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        data_manifest[dataset] = {
            "metadata_file": str(metadata_file),
            "metadata_sha256": sha256(metadata_file),
            "source_version": metadata.get("source_version"),
            "split_sizes": metadata.get("split_sizes"),
            "files": {split: sha256(data_dir / f"{split}.jsonl") for split in ("pool", "dev", "test")},
        }

    direct_rows = []
    temp_rows = []
    for dataset in DATASETS:
        static_score = args.run_root / "scores" / "static" / dataset / "scores.jsonl"
        nait_score = args.run_root / "scores" / "nait" / dataset / "scores.jsonl"
        for seed in SEEDS:
            for method in STATIC_METHODS:
                direct_rows.append(
                    training_row(
                        dataset,
                        seed,
                        method,
                        static_score,
                        args.run_root / "branches" / dataset / f"seed_{seed}" / method,
                        args.data_root,
                    )
                )
            direct_rows.append(
                training_row(
                    dataset,
                    seed,
                    "nait",
                    nait_score,
                    args.run_root / "branches" / dataset / f"seed_{seed}" / "nait",
                    args.data_root,
                )
            )
            temp_score = args.run_root / "scores" / "temp" / dataset / f"seed_{seed}" / "scores.jsonl"
            temp_rows.append(
                training_row(
                    dataset,
                    seed,
                    "temp",
                    temp_score,
                    args.run_root / "branches" / dataset / f"seed_{seed}" / "temp",
                    args.data_root,
                )
            )
    write_jsonl(args.run_root / "direct_training_manifest.jsonl", direct_rows)
    write_jsonl(args.run_root / "temp_training_manifest.jsonl", temp_rows)

    protocol = {
        "name": "Latest raw-MedGemma five-baseline extension",
        "created": "2026-09-03",
        "status": "submitted exploratory extension",
        "backbone": "google/medgemma-4b-it",
        "answer_only_warmup": False,
        "datasets": list(DATASETS),
        "data_manifest": data_manifest,
        "acquisition_training_seeds": list(SEEDS),
        "budget_items": BUDGET,
        "methods": ["temp", "rsr", "nait", "complexity_ft", "aslec_drop"],
        "visibility": {
            "complexity_ft": "question/options only",
            "nait": "question/options only; dev question/options are the target-capability reference",
            "rsr": "privileged rationale-visible comparator",
            "aslec_drop": "privileged rationale-visible comparator",
            "temp": "privileged rationale-visible comparator; 64 dev rationales define each auxiliary direction",
        },
        "adaptations": {
            "nait": "first-to-last prompt activation deltas from all decoder layers; oriented PCA on dev; summed pool projections",
            "temp": "existing repository PEFT adaptation, reset to raw MedGemma with no answer-only adapter",
            "aslec_drop": "deterministic newline/sentence step segmentation",
            "rsr": "token-rank clipping at 100 retained from the repository implementation",
        },
        "training": {
            "initialization": "fresh LoRA on raw backbone",
            "lora_rank": 16,
            "lora_alpha": 32,
            "rationale": "input context",
            "loss": "final-answer tokens only",
            "epochs": 2.0,
            "learning_rate": 1e-4,
            "max_length": 1024,
            "answer_replay": False,
        },
        "expected_direct_training_rows": len(direct_rows),
        "expected_temp_training_rows": len(temp_rows),
        "expected_total_training_rows": len(direct_rows) + len(temp_rows),
        "project_root": str(args.project_root),
        "data_root": str(args.data_root),
        "run_root": str(args.run_root),
    }
    write_json(args.run_root / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
