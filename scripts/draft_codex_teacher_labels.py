#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from settlement_tool.teacher_distill import draft_masked_context_teacher_label  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_draft_labels(*, seed_path: Path, output_path: Path, teacher_id: str) -> dict[str, Any]:
    seed_rows = read_jsonl(seed_path)
    labels = [draft_masked_context_teacher_label(row, teacher_id=teacher_id) for row in seed_rows]
    write_jsonl(output_path, labels)

    counts: dict[str, int] = {}
    for row in labels:
        label = str(row.get("teacher_label", ""))
        counts[label] = counts.get(label, 0) + 1
    return {
        "seed_path": str(seed_path),
        "output_path": str(output_path),
        "teacher_id": teacher_id,
        "label_count": len(labels),
        "label_counts": counts,
        "notes": [
            "This is a masked-context draft label set for policy/reranker smoke tests.",
            "Do not treat this as independent gold because labels are derived from the same local features being evaluated.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reproducible masked-context Codex draft teacher labels from seed records.")
    parser.add_argument("--seed-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--teacher-id", default="codex_draft_v1_masked_context")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_draft_labels(seed_path=args.seed_path, output_path=args.output_path, teacher_id=args.teacher_id)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
