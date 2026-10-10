#!/usr/bin/env python
"""Merge expanded dev-only RSP selection with frozen five-dataset baselines."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import write_json
from make_original_rsp_five_baseline_manifest import DATASETS, SEEDS
from original_rsp_raw_utils import sha256


METRICS = ("accuracy", "macro_f1", "brier", "ece")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    args = parser.parse_args()
    protocol = load(args.run_root / "protocol.json")
    base_root = Path(protocol["five_baseline_run_root"])
    selected = load(args.run_root / "selected_lambda.json")
    if selected.get("test_opened") is not False:
        raise ValueError("Expanded lambda was not frozen before test")
    individual = [
        row
        for row in read_csv(base_root / "individual_results.csv")
        if row["method"] != "rsp_tuned"
    ]
    sources = {
        str(args.run_root / "protocol.json"): sha256(args.run_root / "protocol.json"),
        str(args.run_root / "selected_lambda.json"): sha256(
            args.run_root / "selected_lambda.json"
        ),
        str(base_root / "individual_results.csv"): sha256(
            base_root / "individual_results.csv"
        ),
    }
    for dataset in DATASETS:
        chosen = selected["selections"][dataset]
        for seed in SEEDS:
            test_path = (
                args.run_root / "selected_test" / dataset / f"seed_{seed}" / "result.json"
            )
            test = load(test_path)
            if test.get("method") != chosen["method"] or float(test["lambda"]) != float(
                chosen["lambda"]
            ):
                raise ValueError(f"Selected test mismatch: {test_path}")
            train_path = Path(test["training_result"])
            train = load(train_path)
            if train.get("evaluation_scope") != ["dev"] or "test" in train:
                raise ValueError(f"Selected branch was not dev-only: {train_path}")
            row = {
                "dataset": dataset,
                "method": "rsp_expanded_dev_tuned",
                "lambda": chosen["lambda"],
                "seed": seed,
            }
            for metric in METRICS:
                row[f"dev_{metric}"] = float(train["dev"][metric])
                row[f"test_{metric}"] = float(test["test"][metric])
            individual.append(row)
            sources[str(train_path)] = sha256(train_path)
            sources[str(test_path)] = sha256(test_path)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in individual:
        grouped[(row["dataset"], row["method"])].append(row)
    summary = []
    for (dataset, method), rows in sorted(grouped.items()):
        if len(rows) != len(SEEDS):
            raise ValueError(f"{dataset}/{method}: {len(rows)} seeds")
        payload = {
            "dataset": dataset,
            "method": method,
            "lambda": rows[0]["lambda"],
            "n_seeds": len(rows),
        }
        for split in ("dev", "test"):
            for metric in METRICS:
                values = np.asarray([float(row[f"{split}_{metric}"]) for row in rows])
                payload[f"{split}_{metric}_mean"] = float(values.mean())
                payload[f"{split}_{metric}_std"] = float(values.std(ddof=1))
        summary.append(payload)
    if len(individual) != 120 or len(summary) != 40:
        raise ValueError(f"Unexpected aggregate sizes: {len(individual)}, {len(summary)}")
    write_csv(args.run_root / "individual_results.csv", individual)
    write_csv(args.run_root / "summary_results.csv", summary)

    display = {
        "random": "Random",
        "entropy": "Entropy",
        "margin": "Margin",
        "bald_hidden_dropout": "BALD (hidden dropout)",
        "batchbald_hidden_dropout": "BatchBALD (hidden dropout)",
        "noise_stability": "Noise-Stability",
        "sift": "SIFT",
        "rsp_expanded_dev_tuned": "Original RSP (expanded dev-tuned lambda)",
    }
    lines = [
        "# Original RSP: expanded non-redundant dev-only lambda search",
        "",
        "Candidates were retained only when they changed the exact 128-item acquisition signature across the three seeds.",
        "Lambda was selected by development accuracy only; test was evaluated after the selection file was frozen.",
        "",
        "| Dataset | Selected lambda | Method | Test Accuracy | Test Macro-F1 |",
        "|---|---:|---|---:|---:|",
    ]
    for dataset in DATASETS:
        chosen_lambda = selected["selections"][dataset]["lambda"]
        for method, label in display.items():
            row = next(
                item
                for item in summary
                if item["dataset"] == dataset and item["method"] == method
            )
            value = f"{chosen_lambda:g}" if method == "rsp_expanded_dev_tuned" else ""
            lines.append(
                f"| {dataset} | {value} | {label} | "
                f"{row['test_accuracy_mean']:.4f} +/- {row['test_accuracy_std']:.4f} | "
                f"{row['test_macro_f1_mean']:.4f} +/- {row['test_macro_f1_std']:.4f} |"
            )
    report = "\n".join(lines) + "\n"
    (args.run_root / "ORIGINAL_RSP_EXPANDED_DEV_LAMBDA_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    write_json(
        args.run_root / "completion_audit.json",
        {
            "status": "complete",
            "selection_data": "development only",
            "test_opened_only_after_freeze": True,
            "new_training_rows": protocol["new_training_rows"],
            "selected_test_evaluations": 15,
            "individual_rows": len(individual),
            "summary_rows": len(summary),
            "source_files": sources,
        },
    )
    print(report)


if __name__ == "__main__":
    main()
