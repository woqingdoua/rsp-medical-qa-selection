#!/usr/bin/env python
"""Create exact top-4B entropy-prefiltered random selections.

This extension reuses the frozen question-only entropy and deterministic random
score columns from the Original RSP--Raw experiment.  It never reads answer
labels or rationale text while constructing the acquisition ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common import read_jsonl, write_json, write_jsonl
from original_rsp_raw_utils import sha256


DATASETS = ("afrimedqa", "medexqa", "pubmedqa")
SEEDS = (13, 23, 37)
BUDGET = 128
PREFILTER_MULTIPLIER = 4
METHOD = "entropy_prefiltered_random"
SENTINEL = -1e30


def uid_hash(uids: list[str]) -> str:
    return hashlib.sha256("\n".join(uids).encode()).hexdigest()


def git_commit(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "not-a-git-worktree"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--source_run_root", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    source_hashes = {}
    for dataset in DATASETS:
        data_dir = args.data_root / dataset
        pool_path = data_dir / "pool.jsonl"
        pool = read_jsonl(pool_path)
        if len(pool) <= PREFILTER_MULTIPLIER * BUDGET:
            raise ValueError(
                f"{dataset} has {len(pool)} rows; no missing within-prefilter control is needed"
            )
        source_hashes[str(pool_path)] = sha256(pool_path)
        for seed in SEEDS:
            source_path = (
                args.source_run_root / "scores" / dataset / f"seed_{seed}" / "scores.jsonl"
            )
            scores = read_jsonl(source_path)
            if len(scores) != len(pool):
                raise ValueError(f"Pool/score length mismatch for {dataset}/seed {seed}")
            for index, (row, score) in enumerate(zip(pool, scores)):
                if row["uid"] != score["uid"]:
                    raise ValueError(f"UID mismatch for {dataset}/seed {seed} at {index}")
                if "entropy" not in score or "random" not in score:
                    raise ValueError(f"Missing entropy/random field in {source_path}")

            prefilter_size = PREFILTER_MULTIPLIER * BUDGET
            entropy_order = sorted(
                range(len(scores)), key=lambda index: (-float(scores[index]["entropy"]), index)
            )
            eligible_order = entropy_order[:prefilter_size]
            eligible = set(eligible_order)
            derived = []
            for index, score in enumerate(scores):
                item = dict(score)
                item[METHOD] = float(score["random"]) if index in eligible else SENTINEL
                derived.append(item)
            selected_order = sorted(
                eligible, key=lambda index: (-float(scores[index]["random"]), index)
            )[:BUDGET]
            if len(selected_order) != BUDGET or not set(selected_order).issubset(eligible):
                raise RuntimeError(f"Invalid selection for {dataset}/seed {seed}")

            score_dir = args.run_root / "scores" / dataset / f"seed_{seed}"
            score_path = score_dir / "scores.jsonl"
            write_jsonl(score_path, derived)
            selected_uids = [pool[index]["uid"] for index in selected_order]
            eligible_uids = [pool[index]["uid"] for index in eligible_order]
            metadata_path = score_dir / "metadata.json"
            write_json(
                metadata_path,
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": METHOD,
                    "budget": BUDGET,
                    "pool_size": len(pool),
                    "prefilter_size": prefilter_size,
                    "prefilter_rule": "top 4B by descending predictive entropy; stable index tie-break",
                    "random_rule": "top B by existing deterministic random score within prefilter; stable index tie-break",
                    "selector_visible_fields": ["uid", "question", "options", "domain"],
                    "answer_labels_used_during_acquisition": False,
                    "rationales_used_during_acquisition": False,
                    "source_score_file": str(source_path),
                    "source_score_sha256": sha256(source_path),
                    "derived_score_file": str(score_path),
                    "derived_score_sha256": sha256(score_path),
                    "pool_file": str(pool_path),
                    "pool_sha256": sha256(pool_path),
                    "eligible_uid_sha256": uid_hash(eligible_uids),
                    "selected_uid_order": selected_uids,
                    "selected_uid_sha256": uid_hash(selected_uids),
                },
            )
            source_hashes[str(source_path)] = sha256(source_path)
            manifest.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": METHOD,
                    "budget": BUDGET,
                    "eval_splits": ["dev", "test"],
                    "data_dir": str(data_dir),
                    "score_file": str(score_path),
                    "score_metadata": str(metadata_path),
                    "output_dir": str(
                        args.run_root / "branches" / dataset / f"seed_{seed}" / METHOD
                    ),
                }
            )

    manifest_path = args.run_root / "training_manifest.jsonl"
    write_jsonl(manifest_path, manifest)
    protocol = {
        "name": "Entropy-prefiltered Random missing-dataset extension",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(args.project_root),
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "budget_items": BUDGET,
        "entropy_prefilter_multiplier": PREFILTER_MULTIPLIER,
        "method": METHOD,
        "definition": (
            "Restrict to the top 4B=512 examples under frozen question-only predictive "
            "entropy, then choose B=128 by the retained deterministic random score."
        ),
        "backbone": "google/medgemma-4b-it",
        "initialization": "fresh LoRA from raw backbone; no answer-only warm-up",
        "training": {
            "epochs": 2.0,
            "learning_rate": 1e-4,
            "lora_rank": 16,
            "lora_alpha": 32,
            "max_length": 1024,
            "loss": "final-answer tokens only; acquired rationale is input context",
            "adapter_persistence": "disabled after in-memory dev/test evaluation",
        },
        "scientific_experiment_count": len(manifest),
        "source_run_root": str(args.source_run_root),
        "data_root": str(args.data_root),
        "run_root": str(args.run_root),
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": sha256(manifest_path),
        "source_hashes": source_hashes,
        "generation_command": " ".join(__import__("sys").argv),
    }
    write_json(args.run_root / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
