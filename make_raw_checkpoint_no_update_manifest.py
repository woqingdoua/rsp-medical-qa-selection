#!/usr/bin/env python3
"""Freeze the five current raw-checkpoint test evaluations before submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DATASETS = ("afrimedqa", "medexpqa", "medexqa", "pubmedqa", "medmcqa")
MED_EXPQA_EXCLUDED_TEST_INDICES = (4, 25, 41, 84)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def write_json(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    project, run = args.project_root.resolve(), args.run_root.resolve()
    if run.exists():
        raise FileExistsError(f"Run root already exists: {run}")
    run.mkdir(parents=True)
    manifest, hashes = [], {}
    for dataset in DATASETS:
        data_dir = project / "data" / "additional_medical" / dataset
        test_path = data_dir / "test.jsonl"
        if not test_path.is_file():
            raise FileNotFoundError(test_path)
        original_count = count_jsonl(test_path)
        excluded = list(MED_EXPQA_EXCLUDED_TEST_INDICES) if dataset == "medexpqa" else []
        if any(index >= original_count for index in excluded):
            raise ValueError(f"{dataset}: invalid frozen test exclusion")
        hashes[str(test_path)] = sha256(test_path)
        manifest.append({
            "dataset": dataset,
            "data_dir": str(data_dir),
            "source_test_sha256": hashes[str(test_path)],
            "source_test_rows": original_count,
            "excluded_source_test_indices": excluded,
            "expected_test_rows": original_count - len(excluded),
            "output_dir": str(run / "evaluations" / dataset),
        })
    manifest_path = run / "evaluation_manifest.jsonl"
    with manifest_path.open("x", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    source_files = ("evaluate_raw_checkpoint_no_update.py", "make_raw_checkpoint_no_update_manifest.py", "aggregate_raw_checkpoint_no_update.py", "common.py", "original_rsp_raw_utils.py")
    for name in source_files:
        path = project / name
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[str(path)] = sha256(path)
    write_json(run / "protocol.json", {
        "name": "Five-dataset raw checkpoint / no-update baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backbone": "google/medgemma-4b-it",
        "adapter": "none",
        "training": "none",
        "datasets": list(DATASETS),
        "evaluation_seed": 13,
        "prompt": "question/options only; Choose one option. Return only the final option letter.",
        "option_scoring": "restricted next-token softmax over legal answer letters",
        "max_length": 1024,
        "medexpqa_current_test_filter": list(MED_EXPQA_EXCLUDED_TEST_INDICES),
        "evaluation_manifest_sha256": sha256(manifest_path),
        "input_sha256": hashes,
    })
    print(json.dumps({"run_root": str(run), "evaluations": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
