#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from settlement_tool.teacher_distill import sweep_source_reranker  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def build_policy_reranker_sweep_outputs(
    *,
    merged_path: Path,
    output_dir: Path,
    thresholds: list[float] | None = None,
    margins: list[float] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(merged_path)
    sweep = sweep_source_reranker(rows, thresholds=thresholds, margins=margins)
    summary = {
        "merged_path": str(merged_path),
        "output_path": str(output_dir / "reranker_eval.json"),
        **sweep,
        "notes": [
            "Evaluation groups labeled candidates by source_name/source_id.",
            "Selection uses teacher_policy_score threshold plus a minimum top-vs-runner-up margin.",
            "Examples only include masked candidates and source metadata; raw account candidates are not emitted.",
        ],
    }
    (output_dir / "reranker_eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep policy reranker threshold and margin against merged teacher labels.")
    parser.add_argument("--merged-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", default="10,12,14,18")
    parser.add_argument("--margins", default="0,1,2,3,4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_policy_reranker_sweep_outputs(
        merged_path=args.merged_path,
        output_dir=args.output_dir,
        thresholds=parse_float_list(args.thresholds),
        margins=parse_float_list(args.margins),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
