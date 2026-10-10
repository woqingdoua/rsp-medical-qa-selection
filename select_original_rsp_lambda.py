#!/usr/bin/env python
"""Select one Original RSP lambda per dataset using development results only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import write_json, write_jsonl
from derive_original_rsp_lambda_scores import LAMBDAS, method_name
from make_original_rsp_raw_tuned_manifest import DATASETS, SEEDS
from original_rsp_raw_utils import sha256


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose_candidate(candidates: list[dict]) -> dict:
    """Apply the prespecified development-only ordering."""
    return max(
        candidates,
        key=lambda row: (
            row["dev_accuracy_mean"],
            row["dev_macro_f1_mean"],
            -float(row["lambda"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    args = parser.parse_args()

    protocol = load(args.run_root / "protocol.json")
    if protocol.get("answer_only_warmup") is not False:
        raise ValueError("Unexpected warm-up protocol")
    summary_rows = []
    source_files = {}
    selections = {}
    test_manifest = []
    for dataset in DATASETS:
        candidates = []
        for value in LAMBDAS:
            method = method_name(value)
            seed_results = []
            for seed in SEEDS:
                path = args.run_root / "branches" / dataset / f"seed_{seed}" / method / "result.json"
                result = load(path)
                if result.get("evaluation_scope") != ["dev"] or "test" in result:
                    raise ValueError(f"RSP tuning result accessed test: {path}")
                if result.get("method") != method or result.get("seed") != seed:
                    raise ValueError(f"Result identity mismatch: {path}")
                seed_results.append(result)
                source_files[str(path)] = sha256(path)
            accuracies = np.asarray([item["dev"]["accuracy"] for item in seed_results])
            macro_f1 = np.asarray([item["dev"]["macro_f1"] for item in seed_results])
            row = {
                "dataset": dataset,
                "lambda": value,
                "method": method,
                "n_seeds": len(seed_results),
                "dev_accuracy_mean": float(accuracies.mean()),
                "dev_accuracy_std": float(accuracies.std(ddof=1)),
                "dev_macro_f1_mean": float(macro_f1.mean()),
                "dev_macro_f1_std": float(macro_f1.std(ddof=1)),
            }
            summary_rows.append(row)
            candidates.append(row)
        chosen = choose_candidate(candidates)
        selections[dataset] = chosen
        for seed in SEEDS:
            branch = args.run_root / "branches" / dataset / f"seed_{seed}" / chosen["method"]
            test_manifest.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "lambda": chosen["lambda"],
                    "method": chosen["method"],
                    "data_dir": str(Path(protocol["data_root"]) / dataset),
                    "adapter_path": str(branch / "adapter"),
                    "training_result": str(branch / "result.json"),
                    "output_dir": str(
                        args.run_root / "selected_test" / dataset / f"seed_{seed}"
                    ),
                }
            )

    write_csv(args.run_root / "lambda_dev_summary.csv", summary_rows)
    write_jsonl(args.run_root / "selected_test_manifest.jsonl", test_manifest)
    write_json(
        args.run_root / "selected_lambda.json",
        {
            "selection_data": "development only",
            "test_opened": False,
            "selection_metric": "mean dev accuracy; mean dev macro-F1 then lower lambda tie-break",
            "selections": selections,
            "source_files": source_files,
        },
    )
    print(json.dumps(selections, indent=2))


if __name__ == "__main__":
    main()
