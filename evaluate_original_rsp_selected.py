#!/usr/bin/env python
"""Evaluate a development-selected RSP adapter on the test split.
Historical test cohorts had been examined in earlier analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import LETTER_TO_INDEX, classification_metrics, load_model, load_tokenizer, read_jsonl, set_seed, write_json
from original_rsp_raw_utils import next_letter_probabilities, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--training_result", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--lambda_value", type=float, required=True)
    parser.add_argument("--model_name", default="google/medgemma-4b-it")
    parser.add_argument("--cache_dir", default="model_cache")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    if training.get("evaluation_scope") != ["dev"] or "test" in training:
        raise ValueError("Selected training branch was not dev-only")
    if training.get("method") != args.method or training.get("seed") != args.seed:
        raise ValueError("Selected training branch identity mismatch")

    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    model = load_model(
        args.model_name,
        args.cache_dir,
        str(args.adapter_path),
        trainable=False,
    )
    rows = read_jsonl(args.data_dir / "test.jsonl")
    probabilities = next_letter_probabilities(
        model, tokenizer, rows, args.batch_size, args.max_length
    )
    gold = np.asarray(
        [LETTER_TO_INDEX[row["answer_letter"]] for row in rows], dtype=np.int64
    )
    metrics = classification_metrics(probabilities, gold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz", test_probs=probabilities, test_gold=gold
    )
    result = {
        "protocol": "Original RSP--Raw four-dataset tuned-lambda extension",
        "dataset": args.dataset,
        "method": args.method,
        "lambda": args.lambda_value,
        "seed": args.seed,
        "selection_data": "development only",
        "test_evaluation_after_freeze": True,
        "training_result": str(args.training_result),
        "training_result_sha256": sha256(args.training_result),
        "adapter_path": str(args.adapter_path),
        "test": metrics,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
