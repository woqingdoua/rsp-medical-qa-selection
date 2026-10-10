#!/usr/bin/env python
"""Audit and aggregate the five-dataset Entropy-prefiltered Random control."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import write_json
from make_entropy_prefiltered_random_manifest import BUDGET, DATASETS, METHOD, SEEDS
from original_rsp_raw_utils import sha256


REUSED_DATASETS = ("medexpqa", "medmcqa")
ALL_DATASETS = ("afrimedqa", "medexpqa", "medexqa", "pubmedqa", "medmcqa")
METRICS = ("accuracy", "macro_f1", "brier", "ece")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--existing_run_root", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.run_root / "protocol.json"
    manifest_path = args.run_root / "training_manifest.jsonl"
    protocol = load_json(protocol_path)
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    if len(manifest) != len(DATASETS) * len(SEEDS):
        raise ValueError(f"Expected 9 new branches, found {len(manifest)}")

    individual = []
    source_hashes = {
        str(protocol_path): sha256(protocol_path),
        str(manifest_path): sha256(manifest_path),
    }
    for row in manifest:
        result_path = Path(row["output_dir"]) / "result.json"
        metadata_path = Path(row["score_metadata"])
        result = load_json(result_path)
        metadata = load_json(metadata_path)
        dataset, seed = row["dataset"], int(row["seed"])
        if result.get("dataset") != dataset or int(result.get("seed")) != seed:
            raise ValueError(f"Result identity mismatch: {result_path}")
        if result.get("method") != METHOD or result.get("budget_items") != BUDGET:
            raise ValueError(f"Protocol mismatch: {result_path}")
        if result.get("evaluation_scope") != ["dev", "test"]:
            raise ValueError(f"Incomplete evaluation: {result_path}")
        if result.get("adapter_saved") is not False:
            raise ValueError(f"Unexpected retained adapter: {result_path}")
        if result.get("selected_uid_sha256") != metadata.get("selected_uid_sha256"):
            raise ValueError(f"Selected UID audit failed: {result_path}")
        if metadata.get("answer_labels_used_during_acquisition") is not False:
            raise ValueError(f"Label-visible selection: {metadata_path}")
        if metadata.get("rationales_used_during_acquisition") is not False:
            raise ValueError(f"Rationale-visible selection: {metadata_path}")
        record = {
            "dataset": dataset,
            "method": METHOD,
            "seed": seed,
            "source": "new_top4B_random_run",
        }
        for split in ("dev", "test"):
            for metric in METRICS:
                record[f"{split}_{metric}"] = float(result[split][metric])
        individual.append(record)
        source_hashes[str(result_path)] = sha256(result_path)
        source_hashes[str(metadata_path)] = sha256(metadata_path)

    existing_path = args.existing_run_root / "individual_results.csv"
    existing = load_csv(existing_path)
    source_hashes[str(existing_path)] = sha256(existing_path)
    for dataset in REUSED_DATASETS:
        rows = [
            row
            for row in existing
            if row["dataset"] == dataset and row["method"] == "random"
        ]
        if sorted(int(row["seed"]) for row in rows) != list(SEEDS):
            raise ValueError(f"Could not recover three full-pool random seeds for {dataset}")
        if int(protocol["budget_items"]) * int(protocol["entropy_prefilter_multiplier"]) < (
            434 if dataset == "medexpqa" else 512
        ):
            raise ValueError(f"Full pool does not coincide with top-4B for {dataset}")
        for row in rows:
            record = {
                "dataset": dataset,
                "method": METHOD,
                "seed": int(row["seed"]),
                "source": "reused_full_pool_random_exactly_equal_to_top4B_control",
            }
            for split in ("dev", "test"):
                for metric in METRICS:
                    record[f"{split}_{metric}"] = float(row[f"{split}_{metric}"])
            individual.append(record)

    grouped = defaultdict(list)
    for row in individual:
        grouped[row["dataset"]].append(row)
    summary = []
    for dataset in ALL_DATASETS:
        rows = grouped[dataset]
        if sorted(row["seed"] for row in rows) != list(SEEDS):
            raise ValueError(f"{dataset}: expected seeds {SEEDS}, found {[r['seed'] for r in rows]}")
        output = {"dataset": dataset, "method": METHOD, "n_seeds": len(rows)}
        for split in ("dev", "test"):
            for metric in METRICS:
                values = np.asarray([row[f"{split}_{metric}"] for row in rows], dtype=float)
                output[f"{split}_{metric}_mean"] = float(values.mean())
                output[f"{split}_{metric}_std"] = float(values.std(ddof=1))
        summary.append(output)

    seed_macro = []
    for seed in SEEDS:
        rows = [row for row in individual if row["seed"] == seed]
        if len(rows) != len(ALL_DATASETS):
            raise ValueError(f"Seed {seed} has {len(rows)} datasets")
        seed_macro.append(
            {
                "seed": seed,
                "test_accuracy": float(np.mean([row["test_accuracy"] for row in rows])),
                "test_macro_f1": float(np.mean([row["test_macro_f1"] for row in rows])),
            }
        )
    macro_accuracy = np.asarray([row["test_accuracy"] for row in seed_macro])
    macro_f1 = np.asarray([row["test_macro_f1"] for row in seed_macro])
    dataset_macro = {
        "method": METHOD,
        "aggregation": "unweighted dataset mean within seed, then mean and sample SD across seeds",
        "per_seed": seed_macro,
        "test_accuracy_mean": float(macro_accuracy.mean()),
        "test_accuracy_std": float(macro_accuracy.std(ddof=1)),
        "test_macro_f1_mean": float(macro_f1.mean()),
        "test_macro_f1_std": float(macro_f1.std(ddof=1)),
    }

    individual.sort(key=lambda row: (ALL_DATASETS.index(row["dataset"]), row["seed"]))
    write_csv(args.run_root / "individual_results.csv", individual)
    write_csv(args.run_root / "summary_results.csv", summary)
    write_json(args.run_root / "dataset_macro_summary.json", dataset_macro)
    lines = [
        "# Entropy-prefiltered Random: five-dataset result",
        "",
        "New training was performed only for AfriMed-QA, MedExQA, and PubMedQA. "
        "MedExpQA and MedMCQA reuse their prior full-pool Random branches because each "
        "entire pool fits inside top-4B=512.",
        "",
        "| Dataset | Test Accuracy | Test Macro-F1 |",
        "|---|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['test_accuracy_mean']:.4f} +/- "
            f"{row['test_accuracy_std']:.4f} | {row['test_macro_f1_mean']:.4f} +/- "
            f"{row['test_macro_f1_std']:.4f} |"
        )
    lines.extend(
        [
            f"| Dataset-macro avg. | {dataset_macro['test_accuracy_mean']:.4f} +/- "
            f"{dataset_macro['test_accuracy_std']:.4f} | "
            f"{dataset_macro['test_macro_f1_mean']:.4f} +/- "
            f"{dataset_macro['test_macro_f1_std']:.4f} |",
            "",
        ]
    )
    (args.run_root / "ENTROPY_PREFILTERED_RANDOM_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    write_json(
        args.run_root / "completion_audit.json",
        {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "new_datasets": list(DATASETS),
            "reused_datasets": list(REUSED_DATASETS),
            "new_scientific_runs": len(DATASETS) * len(SEEDS),
            "reused_runs": len(REUSED_DATASETS) * len(SEEDS),
            "missing_runs": 0,
            "failed_runs": 0,
            "adapter_weights_retained": False,
            "selection_audit": "all selected UID hashes matched frozen top-4B score metadata",
            "source_hashes": source_hashes,
            "dataset_macro": dataset_macro,
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
