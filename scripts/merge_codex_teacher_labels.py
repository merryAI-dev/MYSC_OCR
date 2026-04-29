#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from settlement_tool.teacher_distill import (  # noqa: E402
    calibrate_policy_threshold,
    evaluate_source_selection,
    merge_teacher_labels,
    summarize_label_coverage,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_merged_policy_outputs(*, seed_path: Path, label_path: Path | None, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = read_jsonl(seed_path)
    label_rows = read_jsonl(label_path) if label_path else []
    merged = merge_teacher_labels(seed_rows, label_rows)
    calibration = calibrate_policy_threshold(merged)
    source_selection = evaluate_source_selection(merged, threshold=float(calibration["best_threshold"]))
    source_selection_by_threshold = {
        str(threshold): evaluate_source_selection(merged, threshold=threshold)
        for threshold in (10.0, 12.0, 14.0, float(calibration["best_threshold"]))
    }
    label_coverage = summarize_label_coverage(seed_rows, merged)

    merged_path = output_dir / "codex_policy_labels_merged.jsonl"
    eval_path = output_dir / "policy_eval.json"
    write_jsonl(merged_path, merged)

    labeled_count = sum(1 for row in merged if row.get("teacher_label") in {"accept", "reject"})
    summary = {
        "seed_path": str(seed_path),
        "label_path": str(label_path) if label_path else "",
        "merged_path": str(merged_path),
        "labeled_count": labeled_count,
        "label_coverage": label_coverage,
        "calibration": calibration,
        "source_selection": source_selection,
        "source_selection_by_threshold": source_selection_by_threshold,
        "notes": [
            "Merged labels are keyed by source_id, candidate_masked, variant, and prompt_id.",
            "Use the merged file for local policy/reranker evaluation; keep raw candidate features local only.",
        ],
    }
    eval_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Codex/manual teacher labels into policy label seed and evaluate threshold baselines.")
    parser.add_argument("--seed-path", type=Path, required=True)
    parser.add_argument("--label-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_merged_policy_outputs(seed_path=args.seed_path, label_path=args.label_path, output_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
