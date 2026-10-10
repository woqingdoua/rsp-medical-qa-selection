#!/usr/bin/env python
"""Audit and aggregate the four-dataset dev-tuned Original RSP extension."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import write_json
from derive_original_rsp_lambda_scores import LAMBDAS
from make_original_rsp_raw_tuned_manifest import BASELINES, DATASETS, SEEDS, BUDGET
from original_rsp_raw_utils import sha256


METRICS = ("accuracy", "macro_f1", "brier", "ece")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    selected = load(args.run_root / "selected_lambda.json")
    if selected.get("test_opened") is not False:
        raise ValueError("Lambda selection was not test-blind")
    source_files = {
        str(args.run_root / "protocol.json"): sha256(args.run_root / "protocol.json"),
        str(args.run_root / "selected_lambda.json"): sha256(args.run_root / "selected_lambda.json"),
    }
    individual = []
    for dataset in DATASETS:
        chosen = selected["selections"][dataset]
        selected_method = chosen["method"]
        selected_lambda = float(chosen["lambda"])
        if selected_lambda not in LAMBDAS:
            raise ValueError(f"Unexpected lambda for {dataset}: {selected_lambda}")
        for seed in SEEDS:
            score_meta = args.run_root / "scores" / dataset / f"seed_{seed}" / "score_metadata.json"
            tuned_meta = args.run_root / "scores" / dataset / f"seed_{seed}" / "scores_tuned_metadata.json"
            for path in (score_meta, tuned_meta):
                source_files[str(path)] = sha256(path)
            if load(score_meta).get("answer_labels_read_during_acquisition") is not False:
                raise ValueError(f"Label-visible score: {score_meta}")
            if load(tuned_meta).get("test_labels_read") is not False:
                raise ValueError(f"Test-visible lambda derivation: {tuned_meta}")

            for method in BASELINES:
                path = args.run_root / "branches" / dataset / f"seed_{seed}" / method / "result.json"
                result = load(path)
                if result.get("evaluation_scope") != ["dev", "test"]:
                    raise ValueError(f"Incomplete baseline result: {path}")
                if result.get("budget_items") != BUDGET:
                    raise ValueError(f"Budget mismatch: {path}")
                row = {
                    "dataset": dataset,
                    "method": method,
                    "lambda": "",
                    "seed": seed,
                }
                for split in ("dev", "test"):
                    for metric in METRICS:
                        row[f"{split}_{metric}"] = float(result[split][metric])
                individual.append(row)
                source_files[str(path)] = sha256(path)

            train_path = (
                args.run_root / "branches" / dataset / f"seed_{seed}" / selected_method / "result.json"
            )
            train_result = load(train_path)
            if train_result.get("evaluation_scope") != ["dev"] or "test" in train_result:
                raise ValueError(f"Selected RSP candidate was not dev-only: {train_path}")
            test_path = args.run_root / "selected_test" / dataset / f"seed_{seed}" / "result.json"
            test_result = load(test_path)
            if float(test_result["lambda"]) != selected_lambda:
                raise ValueError(f"Selected-test lambda mismatch: {test_path}")
            row = {
                "dataset": dataset,
                "method": "rsp_tuned",
                "lambda": selected_lambda,
                "seed": seed,
            }
            for metric in METRICS:
                row[f"dev_{metric}"] = float(train_result["dev"][metric])
                row[f"test_{metric}"] = float(test_result["test"][metric])
            individual.append(row)
            source_files[str(train_path)] = sha256(train_path)
            source_files[str(test_path)] = sha256(test_path)

    grouped = defaultdict(list)
    for row in individual:
        grouped[(row["dataset"], row["method"])].append(row)
    summary = []
    for (dataset, method), rows in sorted(grouped.items()):
        if len(rows) != len(SEEDS):
            raise ValueError(f"{dataset}/{method} has {len(rows)} rows")
        payload = {
            "dataset": dataset,
            "method": method,
            "lambda": rows[0]["lambda"],
            "n_seeds": len(rows),
        }
        for split in ("dev", "test"):
            for metric in METRICS:
                values = np.asarray([row[f"{split}_{metric}"] for row in rows])
                payload[f"{split}_{metric}_mean"] = float(values.mean())
                payload[f"{split}_{metric}_std"] = float(values.std(ddof=1))
        summary.append(payload)

    write_csv(args.run_root / "individual_results.csv", individual)
    write_csv(args.run_root / "summary_results.csv", summary)
    raw = {
        dataset: load(args.run_root / "raw_eval" / dataset / "result.json")
        for dataset in DATASETS
    }
    display = {
        "random": "Random",
        "entropy": "Entropy",
        "margin": "Margin",
        "rsp_tuned": "RSP (dev-tuned lambda)",
    }
    lines = [
        "# Original RSP--Raw: four-dataset dev-tuned lambda extension",
        "",
        f"Exact acquisition budget: {BUDGET} items. Lambda is selected per dataset on development data only.",
        "",
        "| Dataset | Selected lambda | Dev Acc/F1 |",
        "|---|---:|---:|",
    ]
    for dataset in DATASETS:
        item = selected["selections"][dataset]
        lines.append(
            f"| {dataset} | {item['lambda']:g} | "
            f"{item['dev_accuracy_mean']:.4f}/{item['dev_macro_f1_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Dataset | Method | Test Accuracy | Test Macro-F1 | Brier | ECE |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        raw_test = raw[dataset]["test"]
        lines.append(
            f"| {dataset} | Raw MedGemma | {raw_test['accuracy']:.4f} | "
            f"{raw_test['macro_f1']:.4f} | {raw_test['brier']:.4f} | {raw_test['ece']:.4f} |"
        )
        for row in [item for item in summary if item["dataset"] == dataset]:
            lines.append(
                f"| {dataset} | {display[row['method']]} | "
                f"{row['test_accuracy_mean']:.4f} +/- {row['test_accuracy_std']:.4f} | "
                f"{row['test_macro_f1_mean']:.4f} +/- {row['test_macro_f1_std']:.4f} | "
                f"{row['test_brier_mean']:.4f} +/- {row['test_brier_std']:.4f} | "
                f"{row['test_ece_mean']:.4f} +/- {row['test_ece_std']:.4f} |"
            )
    report = "\n".join(lines) + "\n"
    (args.run_root / "ORIGINAL_RSP_RAW_TUNED_REPORT.md").write_text(report, encoding="utf-8")
    write_json(
        args.run_root / "completion_audit.json",
        {
            "status": "complete",
            "protocol": protocol,
            "candidate_training_rows": protocol["candidate_training_rows"],
            "individual_final_rows": len(individual),
            "summary_rows": len(summary),
            "selected_test_rows": len(DATASETS) * len(SEEDS),
            "source_files": source_files,
        },
    )
    print(report)


if __name__ == "__main__":
    main()
