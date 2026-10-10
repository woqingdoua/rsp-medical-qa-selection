#!/usr/bin/env python3
"""Build and verify the anonymous submission-sensitivity artifact bundle.

This script performs deterministic analysis only.  It does not load a model,
train an adapter, run inference, create selections, or tune a new method.  It
binds the retained MedExpQA split order to saved probability arrays, removes
test questions whose normalized question also occurs in the acquisition pool,
recomputes the prespecified development-lambda sensitivity, and reports the
five- and three-dataset descriptive comparisons used in the paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


DATASETS = ("afrimedqa", "medexpqa", "medexqa", "pubmedqa", "medmcqa")
SENSITIVITY_DATASETS = ("afrimedqa", "medexpqa", "medexqa")
SEEDS = (13, 23, 37)
METHODS = (
    "entropy",
    "random4b",
    "margin",
    "bald",
    "batchbald",
    "noise_stability",
    "sift",
    "nait",
    "rsp",
    "rsr",
    "aslec_drop",
    "temp",
)
METHOD_LABELS = {
    "entropy": "Entropy",
    "random4b": "Random@4B",
    "margin": "Margin",
    "bald": "BALD-inspired",
    "batchbald": "BatchBALD-inspired",
    "noise_stability": "Noise-Stability",
    "sift": "SIFT-inspired",
    "nait": "NAIT-inspired",
    "rsp": "RSP (dataset-tuned)",
    "rsr": "RSR-inspired",
    "aslec_drop": "ASLEC-DROP-inspired",
    "temp": "TEMP-inspired",
}
TEST_SIZES = {
    "afrimedqa": 892,
    "medexpqa": 125,
    "medexqa": 200,
    "pubmedqa": 200,
    "medmcqa": 4096,
}
LAMBDA_TOKENS = ((0.0, "0"), (0.25, "0p25"), (448.0, "448"), (512.0, "512"), (576.0, "576"))
COMPARATORS = ("random4b", "margin")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def normalized_question(value: object) -> str:
    """Match prepare_additional_medical.py::_normalised_question exactly."""
    return re.sub(r"\s+", " ", safe_text(value)).strip().lower()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def macro_f1(pred: np.ndarray, gold: np.ndarray, width: int) -> float:
    values = []
    for label in range(width):
        tp = int(np.sum((pred == label) & (gold == label)))
        fp = int(np.sum((pred == label) & (gold != label)))
        fn = int(np.sum((pred != label) & (gold == label)))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
    return float(np.mean(values))


def metrics(probs: np.ndarray, gold: np.ndarray, indices: np.ndarray | None = None) -> tuple[float, float]:
    if indices is not None:
        probs = probs[indices]
        gold = gold[indices]
    pred = probs.argmax(axis=1)
    return float(np.mean(pred == gold)), macro_f1(pred, gold, probs.shape[1])


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def canonical_rows(args: argparse.Namespace, manifest: list[dict]) -> list[dict]:
    rows: dict[tuple[str, str, int], dict] = {}

    original_map = {
        "entropy": "entropy",
        "margin": "margin",
        "bald_hidden_dropout": "bald",
        "batchbald_hidden_dropout": "batchbald",
        "noise_stability": "noise_stability",
        "sift": "sift",
        "rsp_expanded_dev_tuned": "rsp",
    }
    for row in read_csv(args.original_results):
        method = original_map.get(row["method"])
        if method is None:
            continue
        key = (method, row["dataset"], int(row["seed"]))
        rows[key] = {
            "method": method,
            "dataset": row["dataset"],
            "seed": int(row["seed"]),
            "accuracy": float(row["test_accuracy"]),
            "macro_f1": float(row["test_macro_f1"]),
        }

    for row in read_csv(args.random_results):
        key = ("random4b", row["dataset"], int(row["seed"]))
        rows[key] = {
            "method": "random4b",
            "dataset": row["dataset"],
            "seed": int(row["seed"]),
            "accuracy": float(row["test_accuracy"]),
            "macro_f1": float(row["test_macro_f1"]),
        }

    for row in read_csv(args.latest_results):
        if row["split"] != "test" or row["method"] not in {"nait", "rsr", "aslec_drop"}:
            continue
        method = row["method"]
        key = (method, row["dataset"], int(row["seed"]))
        rows[key] = {
            "method": method,
            "dataset": row["dataset"],
            "seed": int(row["seed"]),
            "accuracy": float(row["accuracy"]),
            "macro_f1": float(row["macro_f1"]),
        }

    for row in read_csv(args.temp_results):
        if row["split"] != "test" or row["method"] != "temp":
            continue
        key = ("temp", row["dataset"], int(row["seed"]))
        rows[key] = {
            "method": "temp",
            "dataset": row["dataset"],
            "seed": int(row["seed"]),
            "accuracy": float(row["accuracy"]),
            "macro_f1": float(row["macro_f1"]),
        }

    expected = {(method, dataset, seed) for method in METHODS for dataset in DATASETS for seed in SEEDS}
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        raise ValueError(f"Canonical result coverage mismatch; missing={missing}, extra={extra}")

    for label, path in (
        ("original_results", args.original_results),
        ("random_results", args.random_results),
        ("latest_results", args.latest_results),
        ("temp_results", args.temp_results),
    ):
        manifest.append({"logical_source": label, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return [rows[key] for key in sorted(rows, key=lambda x: (METHODS.index(x[0]), DATASETS.index(x[1]), x[2]))]


def split_audit(medexp_root: Path, bundle: Path, manifest: list[dict]) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    split_rows = {split: read_jsonl(medexp_root / "data" / f"{split}.jsonl") for split in ("pool", "dev", "test")}
    for split, rows in split_rows.items():
        path = medexp_root / "data" / f"{split}.jsonl"
        manifest.append(
            {
                "logical_source": f"medexpqa_{split}_jsonl",
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "rows": len(rows),
                "ordered_uid_sha256": sha256_text("\n".join(row["uid"] for row in rows)),
            }
        )

    key_sets = {
        split: {normalized_question(row["question"]) for row in rows}
        for split, rows in split_rows.items()
    }
    relations = (("pool_dev", "pool", "dev"), ("pool_test", "pool", "test"), ("dev_test", "dev", "test"))
    overlap_rows: list[dict] = []
    relation_summary: dict[str, dict] = {}
    for name, left, right in relations:
        keys = sorted(key_sets[left] & key_sets[right])
        left_count = 0
        right_count = 0
        for key in keys:
            for split in (left, right):
                matches = [
                    (index, row)
                    for index, row in enumerate(split_rows[split])
                    if normalized_question(row["question"]) == key
                ]
                if split == left:
                    left_count += len(matches)
                else:
                    right_count += len(matches)
                for index, row in matches:
                    overlap_rows.append(
                        {
                            "relation": name,
                            "question_key_sha256": sha256_text(key),
                            "split": split,
                            "row_index": index,
                            "uid_sha256": sha256_text(row["uid"]),
                            "rows_for_key_in_split": len(matches),
                        }
                    )
        relation_summary[name] = {
            "key_count": len(keys),
            f"{left}_row_count": left_count,
            f"{right}_row_count": right_count,
        }

    internal_summary = {}
    for split, rows in split_rows.items():
        counts = Counter(normalized_question(row["question"]) for row in rows)
        duplicate_counts = [count for count in counts.values() if count > 1]
        internal_summary[split] = {
            "duplicate_key_count": len(duplicate_counts),
            "implicated_row_count": int(sum(duplicate_counts)),
            "excess_row_count": int(sum(count - 1 for count in duplicate_counts)),
        }

    pool_keys = key_sets["pool"]
    test_keep = np.asarray(
        [normalized_question(row["question"]) not in pool_keys for row in split_rows["test"]],
        dtype=bool,
    )
    dev_keep = np.asarray(
        [normalized_question(row["question"]) not in pool_keys for row in split_rows["dev"]],
        dtype=bool,
    )
    if int(test_keep.sum()) != 121 or int(dev_keep.sum()) != 61:
        raise ValueError(f"Unexpected overlap-filtered sizes: test={test_keep.sum()}, dev={dev_keep.sum()}")

    masks = {
        "normalization": "lowercase(strip(collapse_whitespace(safe_text(question))))",
        "test_filter_rule": "remove every test row whose normalized question occurs in the acquisition pool",
        "dev_filter_rule": "remove every development row whose normalized question occurs in the acquisition pool",
        "test_rows_before": len(split_rows["test"]),
        "test_rows_after": int(test_keep.sum()),
        "test_removed_row_indices": np.flatnonzero(~test_keep).tolist(),
        "test_removed_uid_sha256": [sha256_text(split_rows["test"][i]["uid"]) for i in np.flatnonzero(~test_keep)],
        "dev_rows_before": len(split_rows["dev"]),
        "dev_rows_after": int(dev_keep.sum()),
        "dev_removed_row_indices": np.flatnonzero(~dev_keep).tolist(),
        "dev_removed_uid_sha256": [sha256_text(split_rows["dev"][i]["uid"]) for i in np.flatnonzero(~dev_keep)],
        "cross_split": relation_summary,
        "within_split_duplicates": internal_summary,
        "within_test_duplicates_not_removed": True,
    }
    write_csv(bundle / "results" / "medexpqa_overlap_rows.csv", overlap_rows)
    write_json(bundle / "results" / "medexpqa_overlap_summary.json", masks)
    return masks, test_keep, dev_keep, split_rows["pool"]


def medexp_predictions(
    medexp_root: Path,
    canonical: list[dict],
    test_keep: np.ndarray,
    pool_rows: list[dict],
    bundle: Path,
    manifest: list[dict],
) -> tuple[dict, list[dict]]:
    canonical_index = {(row["method"], row["dataset"], row["seed"]): row for row in canonical}
    test_rows = read_jsonl(medexp_root / "data" / "test.jsonl")
    expected_gold = np.asarray(["ABCDE".index(row["answer_letter"]) for row in test_rows], dtype=np.int64)
    item_hashes = np.asarray([sha256_text(row["uid"]) for row in test_rows])
    pool_dev_keys = {
        normalized_question(row["question"])
        for row in pool_rows
        if normalized_question(row["question"])
        in {normalized_question(row["question"]) for row in read_jsonl(medexp_root / "data" / "dev.jsonl")}
    }
    pool_test_keys = {
        normalized_question(row["question"])
        for row in pool_rows
        if normalized_question(row["question"])
        in {normalized_question(row["question"]) for row in test_rows}
    }
    affected_pool = {
        row["uid"]: (
            normalized_question(row["question"]) in pool_dev_keys,
            normalized_question(row["question"]) in pool_test_keys,
        )
        for row in pool_rows
        if normalized_question(row["question"]) in pool_dev_keys | pool_test_keys
    }

    arrays: dict[str, np.ndarray] = {
        "item_uid_sha256": item_hashes,
        "gold": expected_gold,
        "leakage_free_keep_mask": test_keep,
    }
    output: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    overlap_selection_rows = []
    for method in METHODS:
        for seed in SEEDS:
            branch = medexp_root / "methods" / method / f"seed_{seed}"
            prediction_path = branch / "predictions.npz"
            result_path = branch / "result.json"
            with np.load(prediction_path) as artifact:
                probs = np.asarray(artifact["test_probs"], dtype=np.float64)
                gold = np.asarray(artifact["test_gold"], dtype=np.int64)
            if probs.shape[0] != len(test_rows) or gold.shape != expected_gold.shape:
                raise ValueError(f"Prediction shape mismatch: {method}/{seed}")
            if not np.array_equal(gold, expected_gold):
                raise ValueError(f"Ordered gold vector mismatch: {method}/{seed}")
            full_accuracy, full_f1 = metrics(probs, gold)
            prior = canonical_index[(method, "medexpqa", seed)]
            if not np.allclose([full_accuracy, full_f1], [prior["accuracy"], prior["macro_f1"]], atol=1e-12):
                raise ValueError(f"Saved summary does not match prediction array: {method}/{seed}")
            arrays[f"probs__{method}__seed_{seed}"] = probs
            output[(method, seed)] = (probs, gold)

            result = json.loads(result_path.read_text(encoding="utf-8"))
            selected = result.get("selected_uid_order")
            if not isinstance(selected, list) or len(selected) != 128:
                raise ValueError(f"Missing 128-item selection: {method}/{seed}")
            dev_hits = [uid for uid in selected if affected_pool.get(uid, (False, False))[0]]
            test_hits = [uid for uid in selected if affected_pool.get(uid, (False, False))[1]]
            union_hits = sorted(set(dev_hits) | set(test_hits))
            overlap_selection_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "selected_items": len(selected),
                    "pool_dev_overlap_selected": len(dev_hits),
                    "pool_test_overlap_selected": len(test_hits),
                    "union_overlap_selected": len(union_hits),
                    "affected_selected_uid_sha256": ";".join(sha256_text(uid) for uid in union_hits),
                    "selected_order_sha256": sha256_text("\n".join(selected)),
                }
            )
            for logical, path in (("predictions", prediction_path), ("selection_result", result_path)):
                manifest.append(
                    {
                        "logical_source": f"medexpqa_{method}_seed_{seed}_{logical}",
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )

    np.savez_compressed(bundle / "anonymous_inputs" / "medexpqa_all_methods_predictions.npz", **arrays)
    write_csv(bundle / "results" / "medexpqa_selected_overlap_audit.csv", overlap_selection_rows)
    return output, overlap_selection_rows


def lambda_sensitivity(
    medexp_root: Path,
    dev_keep: np.ndarray,
    bundle: Path,
    manifest: list[dict],
) -> dict:
    dev_rows = read_jsonl(medexp_root / "data" / "dev.jsonl")
    expected_gold = np.asarray(["ABCDE".index(row["answer_letter"]) for row in dev_rows], dtype=np.int64)
    arrays: dict[str, np.ndarray] = {
        "item_uid_sha256": np.asarray([sha256_text(row["uid"]) for row in dev_rows]),
        "gold": expected_gold,
        "overlap_free_keep_mask": dev_keep,
    }
    rows = []
    for value, token in LAMBDA_TOKENS:
        before_values = []
        after_values = []
        for seed in SEEDS:
            branch = medexp_root / "dev_lambda" / f"lambda_{token}" / f"seed_{seed}"
            prediction_path = branch / "predictions.npz"
            result_path = branch / "result.json"
            with np.load(prediction_path) as artifact:
                probs = np.asarray(artifact["dev_probs"], dtype=np.float64)
                gold = np.asarray(artifact["dev_gold"], dtype=np.int64)
            if not np.array_equal(gold, expected_gold):
                raise ValueError(f"Development alignment mismatch: lambda={value}/seed={seed}")
            before = metrics(probs, gold)
            after = metrics(probs, gold, np.flatnonzero(dev_keep))
            before_values.append(before)
            after_values.append(after)
            arrays[f"probs__lambda_{token}__seed_{seed}"] = probs
            rows.append(
                {
                    "lambda": value,
                    "seed": seed,
                    "dev_n_before": len(gold),
                    "dev_n_after": int(dev_keep.sum()),
                    "accuracy_before": before[0],
                    "accuracy_after": after[0],
                    "macro_f1_before": before[1],
                    "macro_f1_after": after[1],
                }
            )
            for logical, path in (("predictions", prediction_path), ("result", result_path)):
                manifest.append(
                    {
                        "logical_source": f"medexpqa_lambda_{token}_seed_{seed}_{logical}",
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )

    summary_rows = []
    for value, _ in LAMBDA_TOKENS:
        selected = [row for row in rows if row["lambda"] == value]
        summary_rows.append(
            {
                "lambda": value,
                "accuracy_before_mean": mean_sd(row["accuracy_before"] for row in selected)[0],
                "accuracy_before_sd": mean_sd(row["accuracy_before"] for row in selected)[1],
                "macro_f1_before_mean": mean_sd(row["macro_f1_before"] for row in selected)[0],
                "macro_f1_before_sd": mean_sd(row["macro_f1_before"] for row in selected)[1],
                "accuracy_after_mean": mean_sd(row["accuracy_after"] for row in selected)[0],
                "accuracy_after_sd": mean_sd(row["accuracy_after"] for row in selected)[1],
                "macro_f1_after_mean": mean_sd(row["macro_f1_after"] for row in selected)[0],
                "macro_f1_after_sd": mean_sd(row["macro_f1_after"] for row in selected)[1],
            }
        )
    choose = lambda row, suffix: (row[f"accuracy_{suffix}_mean"], row[f"macro_f1_{suffix}_mean"], -row["lambda"])
    before_winner = max(summary_rows, key=lambda row: choose(row, "before"))["lambda"]
    after_winner = max(summary_rows, key=lambda row: choose(row, "after"))["lambda"]
    payload = {
        "selection_rule": "mean development accuracy; then mean Macro-F1; then smaller lambda",
        "candidate_lambdas": [value for value, _ in LAMBDA_TOKENS],
        "development_rows_before": len(dev_rows),
        "development_rows_after": int(dev_keep.sum()),
        "winner_before": before_winner,
        "winner_after": after_winner,
        "winner_changed": before_winner != after_winner,
        "new_training_or_inference": False,
    }
    np.savez_compressed(bundle / "anonymous_inputs" / "medexpqa_dev_lambda_predictions.npz", **arrays)
    write_csv(bundle / "results" / "medexpqa_dev_lambda_per_seed.csv", rows)
    write_csv(bundle / "results" / "medexpqa_dev_lambda_summary.csv", summary_rows)
    write_json(bundle / "results" / "medexpqa_dev_lambda_selection.json", payload)
    return payload


def update_results(
    canonical: list[dict],
    medexp: dict,
    test_keep: np.ndarray,
    bundle: Path,
) -> tuple[list[dict], list[dict]]:
    updated = []
    medexp_before_after = []
    indices = np.flatnonzero(test_keep)
    for row in canonical:
        item = dict(row)
        item["n_test"] = TEST_SIZES[item["dataset"]]
        if item["dataset"] == "medexpqa":
            probs, gold = medexp[(item["method"], item["seed"])]
            after_accuracy, after_f1 = metrics(probs, gold, indices)
            medexp_before_after.append(
                {
                    "method": item["method"],
                    "seed": item["seed"],
                    "n_before": len(gold),
                    "n_after": len(indices),
                    "accuracy_before": item["accuracy"],
                    "accuracy_after": after_accuracy,
                    "macro_f1_before": item["macro_f1"],
                    "macro_f1_after": after_f1,
                }
            )
            item["accuracy"] = after_accuracy
            item["macro_f1"] = after_f1
            item["n_test"] = len(indices)
        updated.append(item)

    summary = []
    for method in METHODS:
        for dataset in DATASETS:
            selected = [row for row in updated if row["method"] == method and row["dataset"] == dataset]
            acc_mean, acc_sd = mean_sd(row["accuracy"] for row in selected)
            f1_mean, f1_sd = mean_sd(row["macro_f1"] for row in selected)
            summary.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "n_test": selected[0]["n_test"],
                    "accuracy_mean": acc_mean,
                    "accuracy_sd": acc_sd,
                    "macro_f1_mean": f1_mean,
                    "macro_f1_sd": f1_sd,
                }
            )
        seed_means = []
        for seed in SEEDS:
            selected = [row for row in updated if row["method"] == method and row["seed"] == seed]
            seed_means.append(
                (float(np.mean([row["accuracy"] for row in selected])), float(np.mean([row["macro_f1"] for row in selected])))
            )
        seed_means_array = np.asarray(seed_means)
        summary.append(
            {
                "method": method,
                "dataset": "mean_across_datasets",
                "n_test": "",
                "accuracy_mean": float(seed_means_array[:, 0].mean()),
                "accuracy_sd": float(seed_means_array[:, 0].std(ddof=1)),
                "macro_f1_mean": float(seed_means_array[:, 1].mean()),
                "macro_f1_sd": float(seed_means_array[:, 1].std(ddof=1)),
            }
        )
    write_csv(bundle / "results" / "updated_per_seed_results.csv", updated)
    write_csv(bundle / "results" / "medexpqa_before_after_per_seed.csv", medexp_before_after)
    write_csv(bundle / "results" / "updated_summary_results.csv", summary)
    return updated, summary


def core_prediction_bundle(
    paired_root: Path,
    medexp: dict,
    canonical: list[dict],
    bundle: Path,
    manifest: list[dict],
) -> dict:
    canonical_index = {(row["method"], row["dataset"], row["seed"]): row for row in canonical}
    loaded: dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = {
        method: {dataset: {} for dataset in DATASETS} for method in ("rsp", *COMPARATORS)
    }
    arrays: dict[str, np.ndarray] = {}
    for method in ("rsp", *COMPARATORS):
        for dataset in DATASETS:
            for seed in SEEDS:
                if dataset == "medexpqa":
                    probs, gold = medexp[(method, seed)]
                    logical = f"medexpqa_{method}_seed_{seed}_predictions"
                else:
                    path = paired_root / method / dataset / f"seed_{seed}.npz"
                    with np.load(path) as artifact:
                        probs = np.asarray(artifact["test_probs"], dtype=np.float64)
                        gold = np.asarray(artifact["test_gold"], dtype=np.int64)
                    manifest.append(
                        {
                            "logical_source": f"paired_{method}_{dataset}_seed_{seed}",
                            "sha256": sha256_file(path),
                            "bytes": path.stat().st_size,
                        }
                    )
                    logical = None
                prior = canonical_index[(method, dataset, seed)]
                before = metrics(probs, gold)
                if not np.allclose(before, [prior["accuracy"], prior["macro_f1"]], atol=1e-12):
                    raise ValueError(f"Core paired prediction mismatch: {method}/{dataset}/{seed}")
                loaded[method][dataset][seed] = (probs, gold)
                arrays[f"probs__{method}__{dataset}__seed_{seed}"] = probs
                if f"gold__{dataset}" not in arrays:
                    arrays[f"gold__{dataset}"] = gold
                elif not np.array_equal(arrays[f"gold__{dataset}"], gold):
                    raise ValueError(f"Gold mismatch: {method}/{dataset}/{seed}")
                del logical
    np.savez_compressed(bundle / "anonymous_inputs" / "core_paired_predictions.npz", **arrays)
    return loaded


def comparison_point(loaded: dict, datasets: tuple[str, ...], medexp_keep: np.ndarray) -> dict:
    output = {}
    for comparator in COMPARATORS:
        dataset_rows = []
        for dataset in datasets:
            indices = np.flatnonzero(medexp_keep) if dataset == "medexpqa" else None
            rsp_values = np.asarray([metrics(*loaded["rsp"][dataset][seed], indices=indices) for seed in SEEDS])
            cmp_values = np.asarray([metrics(*loaded[comparator][dataset][seed], indices=indices) for seed in SEEDS])
            delta = rsp_values.mean(axis=0) - cmp_values.mean(axis=0)
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "accuracy_delta": float(delta[0]),
                    "macro_f1_delta": float(delta[1]),
                }
            )
        output[comparator] = {
            "accuracy_delta": float(np.mean([row["accuracy_delta"] for row in dataset_rows])),
            "macro_f1_delta": float(np.mean([row["macro_f1_delta"] for row in dataset_rows])),
            "positive_accuracy_datasets": int(sum(row["accuracy_delta"] > 0 for row in dataset_rows)),
            "positive_macro_f1_datasets": int(sum(row["macro_f1_delta"] > 0 for row in dataset_rows)),
            "dataset_deltas": dataset_rows,
        }
    return output


def comparison_bootstrap(
    loaded: dict,
    datasets: tuple[str, ...],
    medexp_keep: np.ndarray,
    replicates: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    draws = {
        comparator: {metric: np.empty(replicates) for metric in ("accuracy_delta", "macro_f1_delta")}
        for comparator in COMPARATORS
    }
    eligible = {
        dataset: (np.flatnonzero(medexp_keep) if dataset == "medexpqa" else np.arange(len(loaded["rsp"][dataset][SEEDS[0]][1])))
        for dataset in datasets
    }
    for replicate in range(replicates):
        per_comparator = {name: [] for name in COMPARATORS}
        for dataset in datasets:
            base = eligible[dataset]
            indices = base[rng.integers(0, len(base), size=len(base))]
            paired_seed_indices = rng.integers(0, len(SEEDS), size=len(SEEDS))
            for comparator in COMPARATORS:
                seed_deltas = []
                for seed_index in paired_seed_indices:
                    branch_seed = SEEDS[int(seed_index)]
                    rsp_metric = metrics(*loaded["rsp"][dataset][branch_seed], indices=indices)
                    cmp_metric = metrics(*loaded[comparator][dataset][branch_seed], indices=indices)
                    seed_deltas.append((rsp_metric[0] - cmp_metric[0], rsp_metric[1] - cmp_metric[1]))
                per_comparator[comparator].append(np.mean(seed_deltas, axis=0))
        for comparator in COMPARATORS:
            values = np.mean(per_comparator[comparator], axis=0)
            draws[comparator]["accuracy_delta"][replicate] = values[0]
            draws[comparator]["macro_f1_delta"][replicate] = values[1]
    output = {}
    for comparator in COMPARATORS:
        output[comparator] = {}
        for metric_name, values in draws[comparator].items():
            output[comparator][metric_name] = {
                "bootstrap_mean": float(values.mean()),
                "ci95_percentile": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            }
    return output


def sensitivity_summaries(
    updated: list[dict],
    loaded: dict,
    test_keep: np.ndarray,
    bundle: Path,
    replicates: int,
    bootstrap_seed: int,
) -> tuple[dict, dict]:
    main_point = comparison_point(loaded, DATASETS, test_keep)
    main_bootstrap = comparison_bootstrap(loaded, DATASETS, test_keep, replicates, bootstrap_seed)
    main_payload = {
        "analysis": "five-dataset mixed-overlap benchmark audit with leakage-filtered MedExpQA test",
        "datasets": list(DATASETS),
        "bootstrap": {
            "replicates": replicates,
            "rng_seed": bootstrap_seed,
            "interval": "two-sided 95% percentile",
            "hierarchy": "resample eligible test items within dataset and paired branch-seed identifiers; equal dataset weighting",
        },
        "point_estimates": main_point,
        "bootstrap_results": main_bootstrap,
        "new_training_or_inference": False,
    }
    write_json(bundle / "results" / "five_dataset_paired_summary.json", main_payload)

    sensitivity_rows = []
    for method in METHODS:
        seed_means = []
        for seed in SEEDS:
            rows = [
                row for row in updated
                if row["method"] == method and row["seed"] == seed and row["dataset"] in SENSITIVITY_DATASETS
            ]
            seed_means.append(
                {
                    "method": method,
                    "seed": seed,
                    "accuracy": float(np.mean([row["accuracy"] for row in rows])),
                    "macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
                }
            )
        acc_mean, acc_sd = mean_sd(row["accuracy"] for row in seed_means)
        f1_mean, f1_sd = mean_sd(row["macro_f1"] for row in seed_means)
        sensitivity_rows.append(
            {
                "method": method,
                "accuracy_mean": acc_mean,
                "accuracy_sd": acc_sd,
                "macro_f1_mean": f1_mean,
                "macro_f1_sd": f1_sd,
            }
        )
    write_csv(bundle / "results" / "three_dataset_method_summary.csv", sensitivity_rows)
    three_point = comparison_point(loaded, SENSITIVITY_DATASETS, test_keep)
    three_bootstrap = comparison_bootstrap(loaded, SENSITIVITY_DATASETS, test_keep, replicates, bootstrap_seed)
    three_payload = {
        "analysis": "descriptive sensitivity restricted to study tests not directly reconstructed from documented MedGemma post-training source training collections",
        "datasets": list(SENSITIVITY_DATASETS),
        "not_claimed_clean_or_contamination_free": True,
        "bootstrap": main_payload["bootstrap"],
        "point_estimates": three_point,
        "bootstrap_results": three_bootstrap,
        "new_training_or_inference": False,
    }
    write_json(bundle / "results" / "three_dataset_paired_summary.json", three_payload)
    return main_payload, three_payload


def finish_bundle(bundle: Path, args: argparse.Namespace, source_manifest: list[dict], masks: dict, lambda_result: dict) -> None:
    config = {
        "paper_base_commit": args.git_commit,
        "analysis_date": args.analysis_date,
        "backbone_repository_id": "google/medgemma-4b-it",
        "backbone_revision": "not retained",
        "budget_items": 128,
        "entropy_prefilter": "top min(4B,N), B=128",
        "rsp": {"K": 8, "sigma": 0.02, "layer": "final decoder-block output", "lambda_selection": lambda_result},
        "branch_seeds": list(SEEDS),
        "medexpqa_masks": masks,
        "test_status": "retrospective exploratory reuse; candidate lambda files are dev-only but earlier test results predate the expanded grid",
        "scientific_computation": "deterministic reaggregation of retained arrays only; no training or inference",
        "source_manifest": sorted(source_manifest, key=lambda row: row["logical_source"]),
    }
    write_json(bundle / "configuration_manifest.json", config)

    files = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "bundle_manifest.json":
            files.append({"path": str(path.relative_to(bundle)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_json(
        bundle / "bundle_manifest.json",
        {
            "anonymous": True,
            "contains_raw_question_text": False,
            "contains_model_weights": False,
            "contains_credentials_or_private_paths": False,
            "files": files,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medexp-root", type=Path, required=True)
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--original-results", type=Path, required=True)
    parser.add_argument("--random-results", type=Path, required=True)
    parser.add_argument("--latest-results", type=Path, required=True)
    parser.add_argument("--temp-results", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--analysis-date", required=True)
    args = parser.parse_args()

    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    (args.bundle_dir / "anonymous_inputs").mkdir(exist_ok=True)
    (args.bundle_dir / "results").mkdir(exist_ok=True)
    source_manifest: list[dict] = []
    canonical = canonical_rows(args, source_manifest)
    write_csv(args.bundle_dir / "anonymous_inputs" / "canonical_per_seed_before.csv", canonical)
    masks, test_keep, dev_keep, pool_rows = split_audit(args.medexp_root, args.bundle_dir, source_manifest)
    medexp, _ = medexp_predictions(
        args.medexp_root, canonical, test_keep, pool_rows, args.bundle_dir, source_manifest
    )
    lambda_result = lambda_sensitivity(args.medexp_root, dev_keep, args.bundle_dir, source_manifest)
    updated, _ = update_results(canonical, medexp, test_keep, args.bundle_dir)
    loaded = core_prediction_bundle(args.paired_root, medexp, canonical, args.bundle_dir, source_manifest)
    sensitivity_summaries(
        updated,
        loaded,
        test_keep,
        args.bundle_dir,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    finish_bundle(args.bundle_dir, args, source_manifest, masks, lambda_result)
    print(json.dumps({"status": "complete", "bundle": str(args.bundle_dir)}, indent=2))


if __name__ == "__main__":
    main()
