#!/usr/bin/env python3
"""Evaluate a raw MedGemma checkpoint on one fixed test split, without LoRA."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from common import LETTER_TO_INDEX, classification_metrics, load_model, load_tokenizer, read_jsonl, set_seed, write_json
from original_rsp_raw_utils import next_letter_probabilities


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_indices(value: str) -> list[int]:
    if not value.strip():
        return []
    values = [int(token) for token in value.split(",")]
    if min(values) < 0 or len(set(values)) != len(values):
        raise ValueError("excluded indices must be unique nonnegative integers")
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="google/medgemma-4b-it")
    parser.add_argument("--cache-dir", default="model_cache")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--exclude-test-indices", default="")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    source_test = args.data_dir / "test.jsonl"
    source_rows = read_jsonl(source_test)
    excluded = parse_indices(args.exclude_test_indices)
    if any(index >= len(source_rows) for index in excluded):
        raise ValueError("excluded test index outside source test split")
    excluded_set = set(excluded)
    rows = [row for index, row in enumerate(source_rows) if index not in excluded_set]
    if not rows:
        raise ValueError("test set is empty after exclusions")
    if any(not row.get("uid") for row in rows):
        raise ValueError("all test rows require nonempty UIDs")

    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    # Explicitly pass adapter_path=None and trainable=False: this condition must
    # remain a checkpoint-only evaluation rather than an answer-only baseline.
    model = load_model(args.model_name, args.cache_dir, adapter_path=None, trainable=False)
    probabilities = next_letter_probabilities(model, tokenizer, rows, args.batch_size, args.max_length)
    gold = np.asarray([LETTER_TO_INDEX[row["answer_letter"]] for row in rows], dtype=np.int64)
    metrics = classification_metrics(probabilities, gold)
    uids = [row["uid"] for row in rows]
    payload = {
        "protocol": "Five-dataset raw checkpoint / no-update baseline",
        "dataset": args.dataset,
        "evaluation_initialization": "raw_backbone_without_training_or_adapter",
        "backbone": args.model_name,
        "adapter_path": None,
        "trainable": False,
        "seed": args.seed,
        "prompt": "question/options only; Choose one option. Return only the final option letter.",
        "option_scoring": "restricted next-token softmax over legal answer letters",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "source_test_path": str(source_test),
        "source_test_sha256": sha256(source_test),
        "source_test_rows": len(source_rows),
        "excluded_source_test_indices": excluded,
        "test_rows": len(rows),
        "ordered_test_uid_sha256": hashlib.sha256("\n".join(uids).encode("utf-8")).hexdigest(),
        "test": metrics,
    }
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        test_probs=probabilities,
        test_gold=gold,
        test_uids=np.asarray(uids),
    )
    write_json(args.output_dir / "result.json", payload)


if __name__ == "__main__":
    main()
