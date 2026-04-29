#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FINAL_UPDATES = Path("final_workbook") / "account_updates_deepseek_resolution.csv"
SAFE_ROW_FIELDS = [
    "group",
    "no",
    "name",
    "cell",
    "status",
    "decision",
    "source",
    "account_masked",
    "candidate_count",
    "candidate_accounts_masked",
    "candidate_files",
]
DELTA_FIELDS = [
    "relation",
    "key",
    "name",
    "cell",
    "primary_status",
    "fallback_status",
    "primary_decision",
    "fallback_decision",
    "primary_source",
    "fallback_source",
    "primary_account_masked",
    "fallback_account_masked",
    "primary_candidate_accounts_masked",
    "fallback_candidate_accounts_masked",
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


def updated_rows(rows: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    return {key: row for key, row in rows.items() if row.get("status") == "updated"}


def safe_value(row: dict[str, str] | None, field: str) -> str:
    if row is None:
        return ""
    if field == "account":
        return ""
    return row.get(field, "")


def build_delta_row(
    relation: str,
    key: str,
    primary_row: dict[str, str] | None,
    fallback_row: dict[str, str] | None,
) -> dict[str, str]:
    visible = primary_row or fallback_row or {}
    return {
        "relation": relation,
        "key": key,
        "name": visible.get("name", ""),
        "cell": visible.get("cell", ""),
        "primary_status": safe_value(primary_row, "status"),
        "fallback_status": safe_value(fallback_row, "status"),
        "primary_decision": safe_value(primary_row, "decision"),
        "fallback_decision": safe_value(fallback_row, "decision"),
        "primary_source": safe_value(primary_row, "source"),
        "fallback_source": safe_value(fallback_row, "source"),
        "primary_account_masked": safe_value(primary_row, "account_masked"),
        "fallback_account_masked": safe_value(fallback_row, "account_masked"),
        "primary_candidate_accounts_masked": safe_value(primary_row, "candidate_accounts_masked"),
        "fallback_candidate_accounts_masked": safe_value(fallback_row, "candidate_accounts_masked"),
    }


def build_union_fallback_summary(primary: tuple[str, Path], fallback: tuple[str, Path]) -> dict[str, Any]:
    primary_label, primary_root = primary
    fallback_label, fallback_root = fallback
    primary_rows = read_final_rows(primary_root)
    fallback_rows = read_final_rows(fallback_root)
    primary_updated = updated_rows(primary_rows)
    fallback_updated = updated_rows(fallback_rows)

    primary_keys = set(primary_updated)
    fallback_keys = set(fallback_updated)
    primary_only = sorted(primary_keys - fallback_keys)
    fallback_only = sorted(fallback_keys - primary_keys)
    overlap = sorted(primary_keys & fallback_keys)
    union = sorted(primary_keys | fallback_keys)

    delta_rows = [
        build_delta_row("primary_only", key, primary_rows.get(key), fallback_rows.get(key))
        for key in primary_only
    ] + [
        build_delta_row("fallback_only", key, primary_rows.get(key), fallback_rows.get(key))
        for key in fallback_only
    ]

    return {
        "models": {
            "primary": primary_label,
            "fallback": fallback_label,
        },
        "roots": {
            primary_label: str(primary_root),
            fallback_label: str(fallback_root),
        },
        "counts": {
            "primary_updated": len(primary_keys),
            "fallback_updated": len(fallback_keys),
            "overlap_updated": len(overlap),
            "primary_only_updated": len(primary_only),
            "fallback_only_updated": len(fallback_only),
            "union_updated": len(union),
            "primary_gain_from_fallback": len(fallback_only),
        },
        "policy_note": (
            "Use fallback_only rows as reranker candidates, not blind auto-fill rows. "
            "primary_only rows should remain visible in manual/policy audit because they are not reproduced by fallback."
        ),
        "delta_rows": delta_rows,
        "safe_row_fields": SAFE_ROW_FIELDS,
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
    parser = argparse.ArgumentParser(description="Summarize primary/fallback union from final settlement OCR outputs.")
    parser.add_argument("--primary", required=True, type=parse_model_arg, help="label=/path/to/primary_output_root")
    parser.add_argument("--fallback", required=True, type=parse_model_arg, help="label=/path/to/fallback_output_root")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_union_fallback_summary(args.primary, args.fallback)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_delta_csv(args.output_csv, summary["delta_rows"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
