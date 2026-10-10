#!/usr/bin/env python
"""Add prespecified lambda variants to one Original RSP score file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_jsonl, write_json, write_jsonl
from original_rsp_raw_utils import sha256


LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def lambda_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def method_name(value: float) -> str:
    return f"rsp_lambda_{lambda_slug(value)}"


def derive(rows: list[dict], lambdas: tuple[float, ...] = LAMBDAS) -> list[dict]:
    output = []
    for row in rows:
        item = dict(row)
        eligible = bool(item["rsp_prefilter_eligible"])
        flip = item["rsp_flip"]
        drop = item["rsp_drop"]
        if eligible and (flip is None or drop is None):
            raise ValueError(f"Missing RSP components for eligible row {item['uid']}")
        for value in lambdas:
            item[method_name(value)] = (
                float(flip) + float(value) * float(drop) if eligible else -1e30
            )
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_file", type=Path, required=True)
    parser.add_argument("--score_metadata", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--output_metadata", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.score_metadata.read_text(encoding="utf-8"))
    if metadata.get("score_definition") != (
        "Flip + drop_weight * non-flip positive top1/top2 probability-margin degradation"
    ):
        raise ValueError("Unexpected source score definition")
    if metadata.get("answer_labels_read_during_acquisition") is not False:
        raise ValueError("Source acquisition used answer labels")
    if metadata.get("rationales_read_during_acquisition") is not False:
        raise ValueError("Source acquisition used rationales")

    rows = derive(read_jsonl(args.score_file))
    write_jsonl(args.output_file, rows)
    write_json(
        args.output_metadata,
        {
            "protocol": "Original RSP--Raw tuned-lambda extension",
            "source_score_file": str(args.score_file),
            "source_score_sha256": sha256(args.score_file),
            "source_metadata_file": str(args.score_metadata),
            "source_metadata_sha256": sha256(args.score_metadata),
            "lambda_grid": list(LAMBDAS),
            "method_columns": [method_name(value) for value in LAMBDAS],
            "formula": "rsp_flip + lambda * rsp_drop",
            "test_labels_read": False,
            "output_score_sha256": sha256(args.output_file),
        },
    )
    print(json.dumps({"rows": len(rows), "lambdas": LAMBDAS}, indent=2))


if __name__ == "__main__":
    main()
