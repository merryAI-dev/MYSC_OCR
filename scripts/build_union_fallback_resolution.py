#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.apply_resolution_workbook import is_workbook_auto_fill_decision
from settlement_tool.core import write_csv


FIELDS = [
    "group",
    "no",
    "name",
    "prior_status",
    "prior_account_masked",
    "chosen_account",
    "chosen_account_masked",
    "decision",
    "source",
    "candidate_count",
    "candidate_accounts_masked",
    "candidate_files",
]

AUDIT_FIELDS = [
    "name",
    "relation",
    "primary_decision",
    "fallback_decision",
    "primary_account_masked",
    "fallback_account_masked",
    "primary_source",
    "fallback_source",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def account_key(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def normalized_row(row: dict[str, str]) -> dict[str, str]:
    return {field: str(row.get(field, "")) for field in FIELDS}


def conflict_audit_row(
    name: str,
    primary_row: dict[str, str],
    fallback_row: dict[str, str],
    *,
    relation: str,
) -> dict[str, str]:
    return {
        "name": name,
        "relation": relation,
        "primary_decision": primary_row.get("decision", ""),
        "fallback_decision": fallback_row.get("decision", ""),
        "primary_account_masked": primary_row.get("chosen_account_masked", ""),
        "fallback_account_masked": fallback_row.get("chosen_account_masked", ""),
        "primary_source": primary_row.get("source", ""),
        "fallback_source": fallback_row.get("source", ""),
    }


def fallback_can_fill(row: dict[str, str]) -> bool:
    return bool(row.get("chosen_account")) and is_workbook_auto_fill_decision(row.get("decision", ""))


def fill_from_fallback(
    primary_row: dict[str, str],
    fallback_row: dict[str, str],
    *,
    fallback_label: str,
) -> dict[str, str]:
    output = dict(primary_row)
    output["chosen_account"] = fallback_row.get("chosen_account", "")
    output["chosen_account_masked"] = fallback_row.get("chosen_account_masked", "")
    output["decision"] = "auto_fill_union_fallback"
    output["source"] = (
        f"union_fallback:{fallback_label}:{fallback_row.get('decision', '')}:{fallback_row.get('source', '')}"
    )
    output["candidate_count"] = fallback_row.get("candidate_count", "")
    output["candidate_accounts_masked"] = fallback_row.get("candidate_accounts_masked", "")
    output["candidate_files"] = fallback_row.get("candidate_files", "")
    return output


def build_union_resolution(
    primary_rows: list[dict[str, str]],
    fallback_rows: list[dict[str, str]],
    *,
    primary_label: str = "primary",
    fallback_label: str = "fallback",
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    fallback_by_name = {row.get("name", ""): normalized_row(row) for row in fallback_rows if row.get("name", "")}
    used_fallback_names: set[str] = set()
    output_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    fallback_added = 0
    primary_preserved = 0

    for row in primary_rows:
        primary_row = normalized_row(row)
        name = primary_row.get("name", "")
        fallback_row = fallback_by_name.get(name)
        if fallback_row:
            used_fallback_names.add(name)

        primary_account = primary_row.get("chosen_account", "")
        fallback_account = fallback_row.get("chosen_account", "") if fallback_row else ""
        if primary_account:
            primary_preserved += 1
            if fallback_account and account_key(primary_account) != account_key(fallback_account):
                audit_rows.append(
                    conflict_audit_row(
                        name,
                        primary_row,
                        fallback_row or {},
                        relation="conflict_preserved_primary",
                    )
                )
            output_rows.append(primary_row)
            continue

        if fallback_row and fallback_can_fill(fallback_row):
            output_rows.append(fill_from_fallback(primary_row, fallback_row, fallback_label=fallback_label))
            fallback_added += 1
            continue

        if fallback_row and fallback_account and not fallback_can_fill(fallback_row):
            audit_rows.append(
                conflict_audit_row(
                    name,
                    primary_row,
                    fallback_row,
                    relation="fallback_filled_but_not_autofill",
                )
            )
        output_rows.append(primary_row)

    for name, fallback_row in fallback_by_name.items():
        if name in used_fallback_names or not fallback_can_fill(fallback_row):
            continue
        output_rows.append(fill_from_fallback(fallback_row, fallback_row, fallback_label=fallback_label))
        fallback_added += 1

    summary = {
        "primary_label": primary_label,
        "fallback_label": fallback_label,
        "total_people": len(output_rows),
        "primary_filled": sum(1 for row in primary_rows if row.get("chosen_account")),
        "fallback_filled": sum(1 for row in fallback_rows if row.get("chosen_account")),
        "filled": sum(1 for row in output_rows if row.get("chosen_account")),
        "primary_preserved": primary_preserved,
        "fallback_added": fallback_added,
        "conflicts_preserved_primary": sum(
            1 for row in audit_rows if row.get("relation") == "conflict_preserved_primary"
        ),
        "audit_rows": len(audit_rows),
    }
    return output_rows, audit_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a conservative primary+fallback account resolution CSV.")
    parser.add_argument("--primary-resolution-csv", type=Path, required=True)
    parser.add_argument("--fallback-resolution-csv", type=Path, required=True)
    parser.add_argument("--primary-label", default="primary")
    parser.add_argument("--fallback-label", default="fallback")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows, audit_rows, summary = build_union_resolution(
        read_csv(args.primary_resolution_csv),
        read_csv(args.fallback_resolution_csv),
        primary_label=args.primary_label,
        fallback_label=args.fallback_label,
    )
    summary["output_dir"] = str(args.output_dir)
    write_csv(args.output_dir / "account_resolution_candidates.csv", output_rows, FIELDS)
    write_csv(args.output_dir / "union_fallback_audit.csv", audit_rows, AUDIT_FIELDS)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
