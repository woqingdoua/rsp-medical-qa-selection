#!/usr/bin/env python
"""Aggregate completed latest-protocol baseline branches."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    args = parser.parse_args()
    manifests = [
        args.run_root / "direct_training_manifest.jsonl",
        args.run_root / "temp_training_manifest.jsonl",
    ]
    expected = [row for path in manifests for row in read_jsonl(path)]
    individual = []
    missing = []
    for row in expected:
        result_path = Path(row["output_dir"]) / "result.json"
        if not result_path.exists():
            missing.append(str(result_path))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for split in ("dev", "test"):
            individual.append(
                {
                    "dataset": result["dataset"],
                    "method": result["method"],
                    "seed": result["seed"],
                    "budget": result["budget_items"],
                    "split": split,
                    "accuracy": result[split]["accuracy"],
                    "macro_f1": result[split]["macro_f1"],
                    "result_file": str(result_path),
                }
            )
    if missing:
        raise RuntimeError(f"Missing {len(missing)} result files; first: {missing[:5]}")

    with (args.run_root / "individual_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(individual[0]))
        writer.writeheader()
        writer.writerows(individual)

    groups = defaultdict(list)
    for row in individual:
        groups[(row["dataset"], row["method"], row["budget"], row["split"])].append(row)
    summary = []
    for key, rows in sorted(groups.items()):
        accuracy = np.asarray([float(row["accuracy"]) for row in rows])
        macro_f1 = np.asarray([float(row["macro_f1"]) for row in rows])
        summary.append(
            {
                "dataset": key[0],
                "method": key[1],
                "budget": key[2],
                "split": key[3],
                "runs": len(rows),
                "accuracy_mean": float(accuracy.mean()),
                "accuracy_std": float(accuracy.std(ddof=1)) if len(rows) > 1 else 0.0,
                "macro_f1_mean": float(macro_f1.mean()),
                "macro_f1_std": float(macro_f1.std(ddof=1)) if len(rows) > 1 else 0.0,
            }
        )
    with (args.run_root / "summary_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_json(
        args.run_root / "completion_audit.json",
        {
            "status": "complete",
            "expected_branches": len(expected),
            "completed_branches": len(expected),
            "methods": sorted({row["method"] for row in individual}),
            "datasets": sorted({row["dataset"] for row in individual}),
            "summary_file": str(args.run_root / "summary_results.csv"),
        },
    )
    print(json.dumps({"branches": len(expected), "summary_rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
