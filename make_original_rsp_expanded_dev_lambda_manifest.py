#!/usr/bin/env python
"""Build a non-redundant, score-geometry-derived expanded lambda grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_jsonl, write_json, write_jsonl
from derive_original_rsp_lambda_scores import method_name
from make_original_rsp_five_baseline_manifest import DATASETS, PRIOR_DATASETS, SEEDS
from original_rsp_raw_utils import sha256


GLOBAL_GRID = (
    0.0,
    0.25,
    16.0,
    24.0,
    32.0,
    40.0,
    48.0,
    56.0,
    64.0,
    80.0,
    100.0,
    128.0,
    192.0,
    256.0,
    320.0,
    384.0,
    448.0,
    512.0,
    576.0,
    768.0,
    1024.0,
)
REUSABLE = (0.0, 0.25)
BUDGET = 128


def source_score_dir(
    dataset: str, seed: int, prior_run_root: Path, five_run_root: Path
) -> Path:
    if dataset in PRIOR_DATASETS:
        return prior_run_root / "scores" / dataset / f"seed_{seed}"
    return five_run_root / "rsp_scores" / dataset / f"seed_{seed}"


def source_branch(
    dataset: str,
    seed: int,
    method: str,
    prior_run_root: Path,
    five_run_root: Path,
) -> Path:
    if dataset in PRIOR_DATASETS:
        return prior_run_root / "branches" / dataset / f"seed_{seed}" / method
    return five_run_root / "medmcqa_branches" / f"seed_{seed}" / method


def selected_signature(rows: list[dict], value: float) -> tuple[int, ...]:
    scores = [
        -1e30
        if row["rsp_flip"] is None
        else float(row["rsp_flip"]) + value * float(row["rsp_drop"])
        for row in rows
    ]
    chosen = sorted(range(len(rows)), key=lambda index: (-scores[index], index))[:BUDGET]
    return tuple(sorted(chosen))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--prior_run_root", type=Path, required=True)
    parser.add_argument("--five_run_root", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)

    source_rows: dict[tuple[str, int], list[dict]] = {}
    source_hashes = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            path = source_score_dir(
                dataset, seed, args.prior_run_root, args.five_run_root
            ) / "scores.jsonl"
            source_rows[(dataset, seed)] = read_jsonl(path)
            source_hashes[str(path)] = sha256(path)

    grids = {}
    geometry = {}
    for dataset in DATASETS:
        seen = {}
        representatives = []
        previous_signature = None
        records = []
        for value in GLOBAL_GRID:
            signature = tuple(
                selected_signature(source_rows[(dataset, seed)], value)
                for seed in SEEDS
            )
            if signature in seen:
                records.append(
                    {
                        "lambda": value,
                        "representative": seen[signature],
                        "new_selection_signature": False,
                    }
                )
                continue
            mean_jaccard = None
            if previous_signature is not None:
                jaccards = []
                for current, previous in zip(signature, previous_signature):
                    current_set, previous_set = set(current), set(previous)
                    jaccards.append(
                        len(current_set & previous_set) / len(current_set | previous_set)
                    )
                mean_jaccard = sum(jaccards) / len(jaccards)
            seen[signature] = value
            representatives.append(value)
            records.append(
                {
                    "lambda": value,
                    "representative": value,
                    "new_selection_signature": True,
                    "mean_jaccard_to_previous_representative": mean_jaccard,
                }
            )
            previous_signature = signature
        grids[dataset] = representatives
        geometry[dataset] = records

    training_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            rows = source_rows[(dataset, seed)]
            expanded = []
            for row in rows:
                item = dict(row)
                for value in grids[dataset]:
                    item[method_name(value)] = (
                        float(item["rsp_flip"]) + value * float(item["rsp_drop"])
                        if item["rsp_flip"] is not None
                        else -1e30
                    )
                expanded.append(item)
            score_dir = args.run_root / "scores" / dataset / f"seed_{seed}"
            score_path = score_dir / "scores_expanded.jsonl"
            write_jsonl(score_path, expanded)
            write_json(
                score_dir / "metadata.json",
                {
                    "dataset": dataset,
                    "seed": seed,
                    "formula": "rsp_flip + lambda * rsp_drop",
                    "lambda_grid": grids[dataset],
                    "source_score_sha256": source_hashes[
                        str(
                            source_score_dir(
                                dataset, seed, args.prior_run_root, args.five_run_root
                            )
                            / "scores.jsonl"
                        )
                    ],
                    "output_score_sha256": sha256(score_path),
                    "test_labels_read": False,
                },
            )
            for value in grids[dataset]:
                if value in REUSABLE:
                    continue
                method = method_name(value)
                training_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "lambda": value,
                        "method": method,
                        "budget": BUDGET,
                        "eval_splits": ["dev"],
                        "data_dir": str(args.data_root / dataset),
                        "score_file": str(score_path),
                        "output_dir": str(
                            args.run_root
                            / "branches"
                            / dataset
                            / f"seed_{seed}"
                            / method
                        ),
                    }
                )
    write_jsonl(args.run_root / "training_manifest.jsonl", training_rows)
    protocol = {
        "name": "Original RSP--Raw expanded dev-only lambda search",
        "status": "exploratory expanded hyperparameter search",
        "selection_data": "development only",
        "test_labels_read_during_grid_construction": False,
        "global_geometry_probe_grid": list(GLOBAL_GRID),
        "dataset_nonredundant_lambda_grids": grids,
        "deduplication": (
            "retain the first lambda only when the exact selected 128-item signature "
            "across all three acquisition seeds has not appeared before"
        ),
        "selection_metric": (
            "mean dev accuracy across three seeds; break ties by mean dev macro-F1 "
            "and then smaller lambda"
        ),
        "reused_lambda_adapters": list(REUSABLE),
        "new_training_rows": len(training_rows),
        "budget_items": BUDGET,
        "seeds": list(SEEDS),
        "datasets": list(DATASETS),
        "score_geometry": geometry,
        "prior_run_root": str(args.prior_run_root),
        "five_baseline_run_root": str(args.five_run_root),
        "data_root": str(args.data_root),
        "run_root": str(args.run_root),
        "source_score_hashes": source_hashes,
        "caution": (
            "Large lambdas are required because flip has discrete 1/K-scale changes "
            "while non-flip margin drops are numerically much smaller."
        ),
    }
    write_json(args.run_root / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
