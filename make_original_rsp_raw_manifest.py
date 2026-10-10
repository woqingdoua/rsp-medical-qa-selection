#!/usr/bin/env python
"""Create the frozen Original RSP--Raw training manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import write_json, write_jsonl


DATASETS = ("medmcqa", "medprompt")
SEEDS = (13, 23, 37)
METHODS = ("rsp", "random", "entropy", "margin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            score_file = args.run_root / "scores" / dataset / f"seed_{seed}" / "scores.jsonl"
            for method in METHODS:
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "budget": 2000,
                        "data_dir": str(args.data_root / dataset),
                        "score_file": str(score_file),
                        "output_dir": str(
                            args.run_root / "branches" / dataset / f"seed_{seed}" / method
                        ),
                    }
                )
    write_jsonl(args.run_root / "training_manifest.jsonl", rows)
    protocol = {
        "name": "Original RSP--Raw",
        "intentional_change_from_paper": (
            "No answer-only warm-up on either dataset; score and train directly from google/medgemma-4b-it."
        ),
        "backbone": "google/medgemma-4b-it",
        "datasets": list(DATASETS),
        "split": "fixed 80/10/10",
        "split_seed": 20260821,
        "acquisition_training_seeds": list(SEEDS),
        "methods": list(METHODS),
        "budget_items": 2000,
        "rsp": {
            "definition": "Flip + 0.5 * non-flip positive confidence-margin Drop",
            "entropy_prefilter_multiplier": 4,
            "passes": 8,
            "medmcqa": {"layer": -1, "absolute_sigma": 0.02},
            "medprompt": {"layer": -2, "absolute_sigma": 0.05},
        },
        "training": {
            "initialization": "fresh LoRA on raw backbone",
            "rationale": "input context",
            "loss": "final-answer tokens only",
            "epochs": 2.0,
            "answer_replay": False,
        },
        "training_rows": len(rows),
        "project_root": str(args.project_root),
        "data_root": str(args.data_root),
        "run_root": str(args.run_root),
    }
    write_json(args.run_root / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()

