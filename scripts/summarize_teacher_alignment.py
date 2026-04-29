#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FINAL_UPDATES = Path("final_workbook") / "account_updates_deepseek_resolution.csv"
DELTA_FIELDS = [
    "candidate",
    "scope",
    "relation",
    "key",
    "name",
    "cell",
    "teacher_status",
    "candidate_status",
    "teacher_decision",
    "candidate_decision",
    "teacher_source",
    "candidate_source",
    "teacher_account_masked",
    "candidate_account_masked",
    "teacher_candidate_accounts_masked",
    "candidate_candidate_accounts_masked",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> str:
    cell = row.get("cell", "").strip()
    name = row.get("name", "").strip()
    return cell or name


def read_final_rows(root: Path) -> dict[str, dict[str, str]]:
    path = root / FINAL_UPDATES
    rows = read_csv_rows(path)
    return {row_key(row): row for row in rows if row_key(row)}


def is_updated(row: dict[str, str] | None) -> bool:
    return bool(row) and row.get("status") == "updated"


def is_keep_existing(row: dict[str, str] | None) -> bool:
    return bool(row) and row.get("decision", "").startswith("keep_existing")


def row_in_scope(row: dict[str, str] | None, *, include_keep_existing: bool) -> bool:
    return is_updated(row) and (include_keep_existing or not is_keep_existing(row))


def safe_value(row: dict[str, str] | None, field: str) -> str:
    if row is None:
        return ""
    if field == "account":
        return ""
    return row.get(field, "")


def masked_account(row: dict[str, str] | None) -> str:
    return safe_value(row, "account_masked")


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def build_delta_row(
    *,
    candidate_label: str,
    scope: str,
    relation: str,
    key: str,
    teacher_row: dict[str, str] | None,
    candidate_row: dict[str, str] | None,
) -> dict[str, str]:
    visible = teacher_row or candidate_row or {}
    return {
        "candidate": candidate_label,
        "scope": scope,
        "relation": relation,
        "key": key,
        "name": visible.get("name", ""),
        "cell": visible.get("cell", ""),
        "teacher_status": safe_value(teacher_row, "status"),
        "candidate_status": safe_value(candidate_row, "status"),
        "teacher_decision": safe_value(teacher_row, "decision"),
        "candidate_decision": safe_value(candidate_row, "decision"),
        "teacher_source": safe_value(teacher_row, "source"),
        "candidate_source": safe_value(candidate_row, "source"),
        "teacher_account_masked": masked_account(teacher_row),
        "candidate_account_masked": masked_account(candidate_row),
        "teacher_candidate_accounts_masked": safe_value(teacher_row, "candidate_accounts_masked"),
        "candidate_candidate_accounts_masked": safe_value(candidate_row, "candidate_accounts_masked"),
    }


def classify_scope(
    *,
    candidate_label: str,
    scope: str,
    teacher_rows: dict[str, dict[str, str]],
    candidate_rows: dict[str, dict[str, str]],
    include_keep_existing: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    keys = sorted(set(teacher_rows) | set(candidate_rows))
    teacher_updated: set[str] = set()
    candidate_updated: set[str] = set()
    aligned: set[str] = set()
    teacher_missed: set[str] = set()
    candidate_only: set[str] = set()
    disagreements: set[str] = set()
    delta_rows: list[dict[str, str]] = []

    for key in keys:
        teacher_row = teacher_rows.get(key)
        candidate_row = candidate_rows.get(key)
        teacher_hit = row_in_scope(teacher_row, include_keep_existing=include_keep_existing)
        candidate_hit = row_in_scope(candidate_row, include_keep_existing=include_keep_existing)
        if teacher_hit:
            teacher_updated.add(key)
        if candidate_hit:
            candidate_updated.add(key)

        if teacher_hit and candidate_hit:
            if masked_account(teacher_row) and masked_account(teacher_row) == masked_account(candidate_row):
                aligned.add(key)
            else:
                disagreements.add(key)
                delta_rows.append(
                    build_delta_row(
                        candidate_label=candidate_label,
                        scope=scope,
                        relation="account_disagreement",
                        key=key,
                        teacher_row=teacher_row,
                        candidate_row=candidate_row,
                    )
                )
        elif teacher_hit and not candidate_hit:
            teacher_missed.add(key)
            delta_rows.append(
                build_delta_row(
                    candidate_label=candidate_label,
                    scope=scope,
                    relation="teacher_missed",
                    key=key,
                    teacher_row=teacher_row,
                    candidate_row=candidate_row,
                )
            )
        elif candidate_hit and not teacher_hit:
            candidate_only.add(key)
            delta_rows.append(
                build_delta_row(
                    candidate_label=candidate_label,
                    scope=scope,
                    relation="candidate_only",
                    key=key,
                    teacher_row=teacher_row,
                    candidate_row=candidate_row,
                )
            )

    counts = {
        "teacher_updated": len(teacher_updated),
        "candidate_updated": len(candidate_updated),
        "aligned_updated": len(aligned),
        "teacher_missed": len(teacher_missed),
        "candidate_only": len(candidate_only),
        "account_disagreement": len(disagreements),
        "teacher_recall": ratio(len(aligned), len(teacher_updated)),
        "candidate_alignment_precision": ratio(len(aligned), len(candidate_updated)),
    }
    return counts, delta_rows


def build_candidate_alignment(
    candidate_label: str,
    teacher_rows: dict[str, dict[str, str]],
    candidate_rows: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    all_counts, all_deltas = classify_scope(
        candidate_label=candidate_label,
        scope="all_rows",
        teacher_rows=teacher_rows,
        candidate_rows=candidate_rows,
        include_keep_existing=True,
    )
    ocr_counts, ocr_deltas = classify_scope(
        candidate_label=candidate_label,
        scope="ocr_only",
        teacher_rows=teacher_rows,
        candidate_rows=candidate_rows,
        include_keep_existing=False,
    )
    return {"all_rows": all_counts, "ocr_only": ocr_counts}, ocr_deltas or all_deltas


def build_teacher_alignment_summary(
    teacher: tuple[str, Path],
    candidates: list[tuple[str, Path]],
) -> dict[str, Any]:
    teacher_label, teacher_root = teacher
    teacher_rows = read_final_rows(teacher_root)
    candidate_summaries: dict[str, Any] = {}
    delta_rows: list[dict[str, str]] = []
    roots = {teacher_label: str(teacher_root)}

    for candidate_label, candidate_root in candidates:
        roots[candidate_label] = str(candidate_root)
        candidate_rows = read_final_rows(candidate_root)
        summary, candidate_deltas = build_candidate_alignment(candidate_label, teacher_rows, candidate_rows)
        candidate_summaries[candidate_label] = summary
        delta_rows.extend(candidate_deltas)

    return {
        "teacher": teacher_label,
        "roots": roots,
        "candidates": candidate_summaries,
        "delta_rows": delta_rows,
        "policy_note": (
            "ocr_only excludes keep_existing rows so the metric focuses on OCR-derived account recovery. "
            "Rows are masked-only and should feed policy/manual review, not blind auto-fill."
        ),
    }


def write_delta_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DELTA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected label=/path/to/output_root")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("Model label cannot be empty")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize candidate settlement OCR rows against a teacher output root.")
    parser.add_argument("--teacher", required=True, type=parse_model_arg, help="label=/path/to/teacher_output_root")
    parser.add_argument("--candidate", action="append", required=True, type=parse_model_arg, help="label=/path/to/candidate_output_root")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_teacher_alignment_summary(args.teacher, args.candidate)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_delta_csv(args.output_csv, summary["delta_rows"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
