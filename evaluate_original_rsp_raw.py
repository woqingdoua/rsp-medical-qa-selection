#!/usr/bin/env python
"""Evaluate the unadapted MedGemma reference using paper candidate scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import LETTER_TO_INDEX, classification_metrics, load_model, load_tokenizer, read_jsonl, set_seed, write_json
from original_rsp_raw_utils import next_letter_probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="google/medgemma-4b-it")
    parser.add_argument("--cache_dir", default="model_cache")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    model = load_model(args.model_name, args.cache_dir, None, trainable=False)
    payload = {
        "protocol": "Original RSP--Raw",
        "dataset": args.dataset,
        "evaluation_initialization": "raw_backbone_without_training",
        "seed": args.seed,
    }
    arrays = {}
    for split in ("dev", "test"):
        rows = read_jsonl(args.data_dir / f"{split}.jsonl")
        probabilities = next_letter_probabilities(
            model, tokenizer, rows, args.batch_size, args.max_length
        )
        gold = np.asarray([LETTER_TO_INDEX[row["answer_letter"]] for row in rows], dtype=np.int64)
        payload[split] = classification_metrics(probabilities, gold)
        arrays[f"{split}_probs"] = probabilities
        arrays[f"{split}_gold"] = gold
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "predictions.npz", **arrays)
    write_json(args.output_dir / "result.json", payload)


if __name__ == "__main__":
    main()

