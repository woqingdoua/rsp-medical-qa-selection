#!/usr/bin/env python3
"""Verify and summarize the five frozen raw-checkpoint evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from common import classification_metrics, write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_root.resolve()
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    manifest_path = run / "evaluation_manifest.jsonl"
    if sha256(manifest_path) != protocol["evaluation_manifest_sha256"]:
        raise ValueError("frozen evaluation manifest hash mismatch")
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    rows, hashes = [], {}
    for entry in manifest:
        output = Path(entry["output_dir"])
        result_path, prediction_path = output / "result.json", output / "predictions.npz"
        if not result_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"missing evaluation artifacts for {entry['dataset']}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("evaluation_initialization") != "raw_backbone_without_training_or_adapter":
            raise ValueError(f"{entry['dataset']}: not a raw/no-update result")
        if result.get("source_test_sha256") != entry["source_test_sha256"] or result.get("test_rows") != entry["expected_test_rows"]:
            raise ValueError(f"{entry['dataset']}: frozen test manifest mismatch")
        with np.load(prediction_path) as arrays:
            probs, gold = arrays["test_probs"], arrays["test_gold"]
            if len(probs) != entry["expected_test_rows"] or len(gold) != len(probs):
                raise ValueError(f"{entry['dataset']}: prediction shape mismatch")
            recomputed = classification_metrics(probs, gold)
        for metric, value in recomputed.items():
            if not np.isclose(value, result["test"][metric], atol=1e-12, rtol=0):
                raise ValueError(f"{entry['dataset']}: {metric} does not reproduce")
        hashes[entry["dataset"]] = {"result_sha256": sha256(result_path), "predictions_sha256": sha256(prediction_path)}
        rows.append({"dataset": entry["dataset"], "test_items": len(gold), **result["test"]})
    output = run / "aggregation"
    output.mkdir()
    fields = list(rows[0])
    with (output / "raw_checkpoint_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    means = {metric: float(np.mean([row[metric] for row in rows])) for metric in ("accuracy", "macro_f1", "brier", "ece")}
    report = ["# Raw checkpoint / no-update baseline", "", "| Dataset | Test items | Accuracy | Macro-F1 | Brier | ECE |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        report.append(f"| {row['dataset']} | {row['test_items']} | {100*row['accuracy']:.2f} | {100*row['macro_f1']:.2f} | {row['brier']:.4f} | {row['ece']:.4f} |")
    report += ["", "The final row below is an unweighted descriptive dataset mean; this condition contains no adapter, no fine-tuning, and one fixed inference run.", "", f"- Dataset-macro accuracy: {100*means['accuracy']:.2f}%", f"- Dataset-macro Macro-F1: {100*means['macro_f1']:.2f}%", ""]
    (output / "RAW_CHECKPOINT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(output / "completion_audit.json", {"status": "complete", "protocol_sha256": sha256(run / "protocol.json"), "artifacts": hashes, "dataset_macro": means})


if __name__ == "__main__":
    main()
