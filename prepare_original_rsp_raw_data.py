#!/usr/bin/env python
"""Create the paper's fixed 80/10/10 splits for Original RSP--Raw."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset

from common import write_json, write_jsonl
from original_rsp_raw_utils import sha256
from prepare_data import (
    _load_local_json_rows,
    _normalise_medmcqa,
    _normalise_medprompt,
)


def uid_hash(rows: list[dict]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(str(row["uid"]) for row in rows).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("medmcqa", "medprompt"), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split_seed", type=int, default=20260821)
    parser.add_argument(
        "--medmcqa_path",
        default=(
            "data/cache/openlifescienceai___medmcqa/default/0.0.0/"
            "91c6572c454088bf71b679ad90aa8dffcd0d5868/medmcqa-train.arrow"
        ),
    )
    parser.add_argument(
        "--medprompt_path",
        default=(
            "data/cache/datasets--HPAI-BSC--"
            "Medprompt-MedQA-CoT/snapshots/d121a17c371c24523d1c9b60d07f67f16ebb64aa/"
            "medprompt_medqa_cot_llama31.json"
        ),
    )
    args = parser.parse_args()

    if args.dataset == "medmcqa":
        source = Path(args.medmcqa_path)
        rows = _normalise_medmcqa(Dataset.from_file(str(source)))
    else:
        source = Path(args.medprompt_path)
        rows = _normalise_medprompt(_load_local_json_rows(str(source)))
    if not source.is_file():
        raise FileNotFoundError(source)
    if len({row["uid"] for row in rows}) != len(rows):
        raise ValueError("Dataset contains duplicate UIDs")

    random.Random(args.split_seed).shuffle(rows)
    pool_end = int(0.8 * len(rows))
    dev_end = pool_end + int(0.1 * len(rows))
    pool, dev, test = rows[:pool_end], rows[pool_end:dev_end], rows[dev_end:]
    if len(pool) < 2000:
        raise ValueError(f"Pool has only {len(pool)} rows; paper budget is 2000")
    if set(row["uid"] for row in pool) & set(row["uid"] for row in dev):
        raise RuntimeError("pool/dev overlap")
    if set(row["uid"] for row in pool) & set(row["uid"] for row in test):
        raise RuntimeError("pool/test overlap")
    if set(row["uid"] for row in dev) & set(row["uid"] for row in test):
        raise RuntimeError("dev/test overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("pool", pool), ("dev", dev), ("test", test)):
        write_jsonl(args.output_dir / f"{name}.jsonl", split)
    metadata = {
        "protocol": "Original RSP--Raw",
        "dataset": args.dataset,
        "source_file": str(source),
        "source_sha256": sha256(source),
        "usable_rows": len(rows),
        "split_seed": args.split_seed,
        "split_policy": "fixed seeded shuffle; floor(80%), floor(10%), remainder test",
        "split_sizes": {"pool": len(pool), "dev": len(dev), "test": len(test)},
        "uid_sha256": {
            "pool": uid_hash(pool),
            "dev": uid_hash(dev),
            "test": uid_hash(test),
        },
        "paper_acquisition_budget_items": 2000,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

