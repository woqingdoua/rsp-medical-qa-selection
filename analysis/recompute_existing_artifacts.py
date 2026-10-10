#!/usr/bin/env python3
"""Recompute paired comparisons from retained prediction and score artifacts.

This script performs no model training or inference.  It expects a local input
tree with the following retained files copied from the canonical experiment
folders::

    rsp/<dataset>/seed_<seed>.npz
    random4b/<dataset>/seed_<seed>.npz
    margin/<dataset>/seed_<seed>.npz
    scores/<dataset>/seed_<seed>.jsonl

The paired bootstrap conditions on the five study datasets.  Within each
dataset and replicate it resamples test items once, resamples the three paired
training seeds, and applies both draws identically to all compared methods.
Macro-F1 uses the fixed answer-position label space represented by each saved
probability matrix (A--C, A--D, or A--E).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DATASETS = ("afrimedqa", "medexpqa", "medexqa", "pubmedqa", "medmcqa")
SEEDS = (13, 23, 37)
COMPARATORS = ("random4b", "margin")
SELECTED_LAMBDAS = {
    "afrimedqa": 0.25,
    "medexpqa": 448.0,
    "medexqa": 24.0,
    "pubmedqa": 0.0,
    "medmcqa": 56.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def macro_f1(pred: np.ndarray, gold: np.ndarray, width: int) -> float:
    values: list[float] = []
    for label in range(width):
        tp = int(np.sum((pred == label) & (gold == label)))
        fp = int(np.sum((pred == label) & (gold != label)))
        fn = int(np.sum((pred != label) & (gold == label)))
        denom = 2 * tp + fp + fn
        values.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return float(np.mean(values))


def metrics(probs: np.ndarray, gold: np.ndarray, indices: np.ndarray | None = None) -> tuple[float, float]:
    if indices is not None:
        probs = probs[indices]
        gold = gold[indices]
    pred = probs.argmax(axis=1)
    return float(np.mean(pred == gold)), macro_f1(pred, gold, probs.shape[1])


def load_predictions(input_root: Path) -> tuple[dict, list[dict]]:
    loaded: dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = {}
    manifest: list[dict] = []
    for method in ("rsp", *COMPARATORS):
        loaded[method] = {}
        for dataset in DATASETS:
            loaded[method][dataset] = {}
            reference_gold: np.ndarray | None = None
            for seed in SEEDS:
                path = input_root / method / dataset / f"seed_{seed}.npz"
                if not path.is_file():
                    raise FileNotFoundError(path)
                with np.load(path) as artifact:
                    probs = np.asarray(artifact["test_probs"], dtype=np.float64)
                    gold = np.asarray(artifact["test_gold"], dtype=np.int64)
                if probs.ndim != 2 or gold.shape != (len(probs),):
                    raise ValueError(f"Malformed arrays in {path}: {probs.shape}, {gold.shape}")
                if reference_gold is None:
                    reference_gold = gold
                elif not np.array_equal(reference_gold, gold):
                    raise ValueError(f"Gold labels differ across seeds in {method}/{dataset}")
                loaded[method][dataset][seed] = (probs, gold)
                manifest.append(
                    {
                        "kind": "predictions",
                        "method": method,
                        "dataset": dataset,
                        "seed": seed,
                        "sha256": sha256(path),
                        "test_items": len(gold),
                        "answer_positions": probs.shape[1],
                    }
                )

    for dataset in DATASETS:
        reference = loaded["rsp"][dataset][SEEDS[0]][1]
        width = loaded["rsp"][dataset][SEEDS[0]][0].shape[1]
        for method in COMPARATORS:
            for seed in SEEDS:
                probs, gold = loaded[method][dataset][seed]
                if not np.array_equal(reference, gold):
                    raise ValueError(f"Gold labels are not paired for {method}/{dataset}/seed_{seed}")
                if probs.shape[1] != width:
                    raise ValueError(f"Answer-position width differs for {method}/{dataset}/seed_{seed}")
    return loaded, manifest


def point_estimates(loaded: dict) -> tuple[list[dict], dict[str, dict[str, float]]]:
    rows: list[dict] = []
    aggregate: dict[str, dict[str, float]] = {}
    for comparator in COMPARATORS:
        dataset_deltas: list[tuple[float, float]] = []
        for dataset in DATASETS:
            rsp_values = np.asarray([metrics(*loaded["rsp"][dataset][seed]) for seed in SEEDS])
            cmp_values = np.asarray([metrics(*loaded[comparator][dataset][seed]) for seed in SEEDS])
            delta = rsp_values.mean(axis=0) - cmp_values.mean(axis=0)
            dataset_deltas.append((float(delta[0]), float(delta[1])))
            rows.append(
                {
                    "comparison": f"rsp_minus_{comparator}",
                    "dataset": dataset,
                    "accuracy_delta": float(delta[0]),
                    "macro_f1_delta": float(delta[1]),
                }
            )
        deltas = np.asarray(dataset_deltas)
        aggregate[comparator] = {
            "accuracy_delta": float(deltas[:, 0].mean()),
            "macro_f1_delta": float(deltas[:, 1].mean()),
            "positive_accuracy_datasets": int(np.sum(deltas[:, 0] > 0)),
            "positive_macro_f1_datasets": int(np.sum(deltas[:, 1] > 0)),
        }
    return rows, aggregate


def hierarchical_bootstrap(loaded: dict, replicates: int, random_seed: int) -> dict:
    rng = np.random.default_rng(random_seed)
    draws = {
        comparator: {
            "accuracy_delta": np.empty(replicates, dtype=np.float64),
            "macro_f1_delta": np.empty(replicates, dtype=np.float64),
        }
        for comparator in COMPARATORS
    }
    for replicate in range(replicates):
        per_comparator = {name: [] for name in COMPARATORS}
        for dataset in DATASETS:
            gold = loaded["rsp"][dataset][SEEDS[0]][1]
            item_indices = rng.integers(0, len(gold), size=len(gold))
            paired_seed_indices = rng.integers(0, len(SEEDS), size=len(SEEDS))
            for comparator in COMPARATORS:
                seed_deltas = []
                for seed_index in paired_seed_indices:
                    seed = SEEDS[int(seed_index)]
                    rsp_metric = metrics(*loaded["rsp"][dataset][seed], indices=item_indices)
                    cmp_metric = metrics(*loaded[comparator][dataset][seed], indices=item_indices)
                    seed_deltas.append(
                        (rsp_metric[0] - cmp_metric[0], rsp_metric[1] - cmp_metric[1])
                    )
                per_comparator[comparator].append(np.mean(seed_deltas, axis=0))
        for comparator in COMPARATORS:
            across_datasets = np.mean(per_comparator[comparator], axis=0)
            draws[comparator]["accuracy_delta"][replicate] = across_datasets[0]
            draws[comparator]["macro_f1_delta"][replicate] = across_datasets[1]

    summary: dict[str, dict] = {}
    for comparator in COMPARATORS:
        summary[comparator] = {}
        for metric_name in ("accuracy_delta", "macro_f1_delta"):
            values = draws[comparator][metric_name]
            summary[comparator][metric_name] = {
                "bootstrap_mean": float(values.mean()),
                "ci95_percentile": [
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ],
            }
    return summary


def score_scale_audit(input_root: Path, manifest: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for dataset in DATASETS:
        scale = SELECTED_LAMBDAS[dataset]
        for seed in SEEDS:
            path = input_root / "scores" / dataset / f"seed_{seed}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            clean_margins: list[float] = []
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("rsp_prefilter_eligible"):
                        clean_margins.append(-float(row["margin"]))
            values = np.asarray(clean_margins, dtype=np.float64)
            count = int(np.sum(scale * values > 1.0))
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "selected_lambda": scale,
                    "eligible_candidates": len(values),
                    "lambda_times_clean_margin_gt_one": count,
                    "fraction": float(count / len(values)),
                }
            )
            manifest.append(
                {
                    "kind": "candidate_scores",
                    "dataset": dataset,
                    "seed": seed,
                    "sha256": sha256(path),
                    "eligible_candidates": len(values),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded, manifest = load_predictions(args.input_root)
    per_dataset, aggregate = point_estimates(loaded)
    bootstrap = hierarchical_bootstrap(loaded, args.bootstrap_replicates, args.bootstrap_seed)
    score_scale = score_scale_audit(args.input_root, manifest)

    write_csv(args.output_dir / "paired_differences.csv", per_dataset)
    write_csv(args.output_dir / "score_scale_audit.csv", score_scale)
    payload = {
        "analysis": "deterministic recomputation from retained predictions and scores",
        "new_model_training_or_inference": False,
        "datasets": list(DATASETS),
        "training_seeds": list(SEEDS),
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "rng_seed": args.bootstrap_seed,
            "scope": "conditional on the five fixed datasets",
            "hierarchy": "resample test items once within dataset; resample paired training seeds; use both draws for all methods",
            "interval": "two-sided 95% percentile interval",
            "label_space": "fixed answer-option positions from each probability matrix",
            "pairing_check": "NPZ files omit UIDs; rely on the common frozen test order and verify length, answer width, and complete ordered gold arrays",
        },
        "point_estimates": aggregate,
        "bootstrap_results": bootstrap,
        "input_manifest": manifest,
    }
    with (args.output_dir / "paired_bootstrap_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
