#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    ("full_ocr", "members"),
    ("full_ocr", "high_accounts"),
    ("full_ocr", "matched_names"),
    ("policy_resolution", "keep_existing"),
    ("policy_resolution", "auto_fill_single_candidate"),
    ("policy_resolution", "multiple_candidates_review"),
    ("policy_resolution", "no_candidate"),
    ("policy_resolution", "policy_audit_rows"),
    ("policy_resolution", "policy_rejected_rows"),
    ("targeted_retry", "targets"),
    ("targeted_retry", "runs"),
    ("targeted_retry", "high_accounts"),
    ("policy_resolution_targeted", "filled"),
    ("policy_resolution_targeted", "keep_existing"),
    ("policy_resolution_targeted", "auto_fill_single_candidate"),
    ("policy_resolution_targeted", "auto_fill_targeted_deepseek"),
    ("policy_resolution_targeted", "multiple_or_review"),
    ("policy_resolution_targeted", "targeted_retry_no_candidate"),
    ("policy_resolution_targeted", "no_candidate"),
    ("policy_resolution_targeted", "policy_audit_rows"),
    ("policy_resolution_targeted", "policy_rejected_rows"),
    ("final_workbook", "updated"),
    ("final_workbook", "skipped"),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def stage_summary(root: Path, stage: str) -> dict[str, Any]:
    return read_json(root / stage / "summary.json")


def numeric_delta(left: Any, right: Any) -> Any:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left - right
    return ""


def build_comparison(*models: tuple[str, Path]) -> dict[str, Any]:
    if len(models) != 2:
        raise ValueError("Exactly two models are required for delta comparison.")

    labels = [label for label, _ in models]
    rows: list[dict[str, Any]] = []
    for stage, metric in DEFAULT_METRICS:
        values = [stage_summary(root, stage).get(metric, "") for _, root in models]
        rows.append(
            {
                "stage": stage,
                "metric": metric,
                labels[0]: values[0],
                labels[1]: values[1],
                "delta": numeric_delta(values[0], values[1]),
            }
        )

    return {
        "models": labels,
        "roots": {label: str(root) for label, root in models},
        "rows": rows,
    }


def write_csv(path: Path, comparison: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = comparison["models"]
    fieldnames = ["stage", "metric", labels[0], labels[1], "delta"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison["rows"])


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected label=/path/to/output_root")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("Model label cannot be empty")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize two practical settlement OCR eval output roots.")
    parser.add_argument("--model", action="append", required=True, type=parse_model_arg, help="label=/path/to/output_root")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = build_comparison(*args.model)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
