#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_KEYS = [
    "human_positive_count",
    "human_review_count",
    "correct_positive",
    "wrong_positive",
    "missed_positive",
    "review_false_positive",
    "review_deferred",
]
METRIC_KEYS = [
    "positive_precision",
    "safe_selection_precision",
    "positive_recall",
    "review_false_positive_rate",
]


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _comparison_row(label: str, path: Path) -> dict[str, Any]:
    report = _load_report(path)
    summary = report.get("summary") or {}
    metrics = report.get("metrics") or {}
    row: dict[str, Any] = {"label": label, "path": str(path)}
    for key in SUMMARY_KEYS:
        row[key] = summary.get(key, 0)
    for key in METRIC_KEYS:
        row[key] = metrics.get(key, 0.0)
    return row


def compare_human_eval_reports(*, reports: list[tuple[str, Path]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_comparison_row(label, path) for label, path in reports]
    comparison_csv = output_dir / "human_eval_comparison.csv"
    fieldnames = ["label", "path", *SUMMARY_KEYS, *METRIC_KEYS]
    with comparison_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "comparison_csv": str(comparison_csv),
        "report_count": len(rows),
        "labels": [row["label"] for row in rows],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _parse_report_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("report must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("report label cannot be empty")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare human-label eval JSON reports across rerankers.")
    parser.add_argument("--report", action="append", type=_parse_report_arg, required=True, help="LABEL=/path/to/human_label_eval.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = compare_human_eval_reports(reports=args.report, output_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
