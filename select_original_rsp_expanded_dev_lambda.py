#!/usr/bin/env python
"""Select a non-redundant expanded RSP lambda on development data only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import write_json, write_jsonl
from derive_original_rsp_lambda_scores import method_name
from make_original_rsp_expanded_dev_lambda_manifest import REUSABLE, source_branch
from make_original_rsp_five_baseline_manifest import DATASETS, SEEDS
from original_rsp_raw_utils import sha256


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_path(
    run_root: Path,
    dataset: str,
    seed: int,
    value: float,
    prior_run_root: Path,
    five_run_root: Path,
) -> Path:
    method = method_name(value)
    if value in REUSABLE:
        return source_branch(
            dataset, seed, method, prior_run_root, five_run_root
        ) / "result.json"
    return run_root / "branches" / dataset / f"seed_{seed}" / method / "result.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    args = parser.parse_args()
    protocol = load(args.run_root / "protocol.json")
    prior_root = Path(protocol["prior_run_root"])
    five_root = Path(protocol["five_baseline_run_root"])
    rows = []
    selections = {}
    sources = {}
    test_manifest = []
    for dataset in DATASETS:
        candidates = []
        for value in protocol["dataset_nonredundant_lambda_grids"][dataset]:
            method = method_name(float(value))
            results = []
            for seed in SEEDS:
                path = result_path(
                    args.run_root,
                    dataset,
                    seed,
                    float(value),
                    prior_root,
                    five_root,
                )
                result = load(path)
                if result.get("evaluation_scope") != ["dev"] or "test" in result:
                    raise ValueError(f"Candidate was not dev-only: {path}")
                if result.get("method") != method or result.get("seed") != seed:
                    raise ValueError(f"Candidate identity mismatch: {path}")
                results.append(result)
                sources[str(path)] = sha256(path)
            accuracy = np.asarray([item["dev"]["accuracy"] for item in results])
            macro_f1 = np.asarray([item["dev"]["macro_f1"] for item in results])
            row = {
                "dataset": dataset,
                "lambda": float(value),
                "method": method,
                "n_seeds": len(SEEDS),
                "dev_accuracy_mean": float(accuracy.mean()),
                "dev_accuracy_std": float(accuracy.std(ddof=1)),
                "dev_macro_f1_mean": float(macro_f1.mean()),
                "dev_macro_f1_std": float(macro_f1.std(ddof=1)),
            }
            rows.append(row)
            candidates.append(row)
        chosen = max(
            candidates,
            key=lambda row: (
                row["dev_accuracy_mean"],
                row["dev_macro_f1_mean"],
                -row["lambda"],
            ),
        )
        selections[dataset] = chosen
        for seed in SEEDS:
            train_result = result_path(
                args.run_root,
                dataset,
                seed,
                chosen["lambda"],
                prior_root,
                five_root,
            )
            branch = train_result.parent
            test_manifest.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "lambda": chosen["lambda"],
                    "method": chosen["method"],
                    "data_dir": str(Path(protocol["data_root"]) / dataset),
                    "adapter_path": str(branch / "adapter"),
                    "training_result": str(train_result),
                    "output_dir": str(
                        args.run_root / "selected_test" / dataset / f"seed_{seed}"
                    ),
                }
            )
    with (args.run_root / "lambda_dev_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_jsonl(args.run_root / "selected_test_manifest.jsonl", test_manifest)
    write_json(
        args.run_root / "selected_lambda.json",
        {
            "selection_data": "development only",
            "test_opened": False,
            "selection_metric": (
                "mean dev accuracy; mean dev macro-F1 then lower lambda tie-break"
            ),
            "selections": selections,
            "source_files": sources,
        },
    )
    print(json.dumps(selections, indent=2))


if __name__ == "__main__":
    main()
