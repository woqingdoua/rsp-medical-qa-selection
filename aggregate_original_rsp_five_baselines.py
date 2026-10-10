#!/usr/bin/env python
"""Audit and aggregate the frozen five-dataset direct-baseline extension."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import write_json
from make_original_rsp_five_baseline_manifest import (
    BUDGET,
    DATASETS,
    DIRECT_METHODS,
    EXTRA_METHODS,
    PRIOR_DATASETS,
    SEEDS,
)
from original_rsp_raw_utils import sha256


METRICS = ("accuracy", "macro_f1", "brier", "ece")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_both_splits(
    individual: list[dict],
    sources: dict[str, str],
    path: Path,
    dataset: str,
    method: str,
    seed: int,
    lambda_value: str | float = "",
) -> None:
    result = load(path)
    if result.get("dataset") != dataset or result.get("seed") != seed:
        raise ValueError(f"Result identity mismatch: {path}")
    if result.get("method") != method:
        raise ValueError(f"Method mismatch: {path}")
    if result.get("budget_items") != BUDGET:
        raise ValueError(f"Budget mismatch: {path}")
    if result.get("evaluation_scope") != ["dev", "test"]:
        raise ValueError(f"Incomplete evaluation scope: {path}")
    row = {"dataset": dataset, "method": method, "lambda": lambda_value, "seed": seed}
    for split in ("dev", "test"):
        for metric in METRICS:
            row[f"{split}_{metric}"] = float(result[split][metric])
    individual.append(row)
    sources[str(path)] = sha256(path)


def add_selected_rsp(
    individual: list[dict],
    sources: dict[str, str],
    dataset: str,
    seed: int,
    method: str,
    lambda_value: float,
    train_path: Path,
    test_path: Path,
) -> None:
    train = load(train_path)
    test = load(test_path)
    if train.get("evaluation_scope") != ["dev"] or "test" in train:
        raise ValueError(f"RSP candidate was not development-only: {train_path}")
    if train.get("dataset") != dataset or train.get("method") != method or train.get("seed") != seed:
        raise ValueError(f"RSP train identity mismatch: {train_path}")
    if test.get("dataset") != dataset or test.get("method") != method or test.get("seed") != seed:
        raise ValueError(f"RSP test identity mismatch: {test_path}")
    if float(test["lambda"]) != float(lambda_value):
        raise ValueError(f"RSP lambda mismatch: {test_path}")
    row = {"dataset": dataset, "method": "rsp_tuned", "lambda": lambda_value, "seed": seed}
    for metric in METRICS:
        row[f"dev_{metric}"] = float(train["dev"][metric])
        row[f"test_{metric}"] = float(test["test"][metric])
    individual.append(row)
    sources[str(train_path)] = sha256(train_path)
    sources[str(test_path)] = sha256(test_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    args = parser.parse_args()
    protocol = load(args.run_root / "protocol.json")
    prior_root = Path(protocol["prior_four_dataset_run"]["path"])
    prior_selected = load(prior_root / "selected_lambda.json")
    med_selected = load(args.run_root / "medmcqa_selected_lambda.json")
    if prior_selected.get("test_opened") is not False or med_selected.get("test_opened") is not False:
        raise ValueError("A selected lambda was not frozen before test evaluation")

    sources = {
        str(args.run_root / "protocol.json"): sha256(args.run_root / "protocol.json"),
        str(prior_root / "selected_lambda.json"): sha256(prior_root / "selected_lambda.json"),
        str(args.run_root / "medmcqa_selected_lambda.json"): sha256(
            args.run_root / "medmcqa_selected_lambda.json"
        ),
    }
    individual: list[dict] = []

    for dataset in PRIOR_DATASETS:
        for seed in SEEDS:
            for method in DIRECT_METHODS:
                add_both_splits(
                    individual,
                    sources,
                    prior_root / "branches" / dataset / f"seed_{seed}" / method / "result.json",
                    dataset,
                    method,
                    seed,
                )
            chosen = prior_selected["selections"][dataset]
            add_selected_rsp(
                individual,
                sources,
                dataset,
                seed,
                chosen["method"],
                float(chosen["lambda"]),
                prior_root / "branches" / dataset / f"seed_{seed}" / chosen["method"] / "result.json",
                prior_root / "selected_test" / dataset / f"seed_{seed}" / "result.json",
            )

    dataset = "medmcqa"
    chosen = med_selected["selection"]
    for seed in SEEDS:
        for method in DIRECT_METHODS:
            add_both_splits(
                individual,
                sources,
                args.run_root / "medmcqa_branches" / f"seed_{seed}" / method / "result.json",
                dataset,
                method,
                seed,
            )
        add_selected_rsp(
            individual,
            sources,
            dataset,
            seed,
            chosen["method"],
            float(chosen["lambda"]),
            args.run_root / "medmcqa_branches" / f"seed_{seed}" / chosen["method"] / "result.json",
            args.run_root / "medmcqa_selected_test" / f"seed_{seed}" / "result.json",
        )

    for dataset in DATASETS:
        for seed in SEEDS:
            score_meta = args.run_root / "extra_scores" / dataset / f"seed_{seed}" / "score_metadata.json"
            metadata = load(score_meta)
            if metadata.get("answer_labels_used_during_acquisition") is not False:
                raise ValueError(f"Label-visible baseline score: {score_meta}")
            if metadata.get("rationales_used_during_acquisition") is not False:
                raise ValueError(f"Rationale-visible baseline score: {score_meta}")
            sources[str(score_meta)] = sha256(score_meta)
            for method in EXTRA_METHODS:
                add_both_splits(
                    individual,
                    sources,
                    args.run_root
                    / "extra_branches"
                    / dataset
                    / f"seed_{seed}"
                    / method
                    / "result.json",
                    dataset,
                    method,
                    seed,
                )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in individual:
        grouped[(row["dataset"], row["method"])].append(row)
    expected_methods = (*DIRECT_METHODS, *EXTRA_METHODS, "rsp_tuned")
    summary = []
    for dataset in DATASETS:
        for method in expected_methods:
            rows = grouped[(dataset, method)]
            if len(rows) != len(SEEDS):
                raise ValueError(f"{dataset}/{method}: expected {len(SEEDS)} seeds, got {len(rows)}")
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

    raw = {}
    for dataset in DATASETS:
        path = (
            prior_root / "raw_eval" / dataset / "result.json"
            if dataset in PRIOR_DATASETS
            else args.run_root / "raw_eval" / dataset / "result.json"
        )
        raw[dataset] = load(path)
        sources[str(path)] = sha256(path)

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
        "rsp_tuned": "Original RSP (dev-tuned lambda)",
    }
    lines = [
        "# Original RSP--Raw: five-dataset direct-baseline extension",
        "",
        f"All acquisition methods use an exact {BUDGET}-item budget, raw google/medgemma-4b-it, fresh LoRA training, and three seeds.",
        "Selectors see question/options/domain only; answers and rationales are hidden during acquisition.",
        "BALD and BatchBALD use the documented hidden-dropout adaptation because the backbone's architectural dropout is zero.",
        "",
        "| Dataset | Method | Test Accuracy | Test Macro-F1 | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        raw_test = raw[dataset]["test"]
        lines.append(
            f"| {dataset} | Raw MedGemma | {raw_test['accuracy']:.4f} | "
            f"{raw_test['macro_f1']:.4f} | {raw_test['brier']:.4f} | {raw_test['ece']:.4f} |"
        )
        for method in expected_methods:
            row = next(item for item in summary if item["dataset"] == dataset and item["method"] == method)
            lines.append(
                f"| {dataset} | {display[method]} | "
                f"{row['test_accuracy_mean']:.4f} +/- {row['test_accuracy_std']:.4f} | "
                f"{row['test_macro_f1_mean']:.4f} +/- {row['test_macro_f1_std']:.4f} | "
                f"{row['test_brier_mean']:.4f} +/- {row['test_brier_std']:.4f} | "
                f"{row['test_ece_mean']:.4f} +/- {row['test_ece_std']:.4f} |"
            )
    report = "\n".join(lines) + "\n"
    (args.run_root / "ORIGINAL_RSP_FIVE_BASELINES_REPORT.md").write_text(report, encoding="utf-8")
    write_json(
        args.run_root / "completion_audit.json",
        {
            "status": "complete",
            "protocol_sha256": sha256(args.run_root / "protocol.json"),
            "reused_four_dataset_run": str(prior_root),
            "new_extra_training_rows": len(DATASETS) * len(SEEDS) * len(EXTRA_METHODS),
            "new_medmcqa_direct_and_rsp_rows": len(SEEDS) * (len(DIRECT_METHODS) + 1),
            "individual_rows": len(individual),
            "summary_rows": len(summary),
            "source_files": sources,
        },
    )
    print(report)


if __name__ == "__main__":
    main()
