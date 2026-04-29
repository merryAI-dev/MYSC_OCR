#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from settlement_tool.core import extract_roster, normalize_text, write_csv  # noqa: E402
from scripts.evaluate_human_workbook_labels import load_review_names  # noqa: E402


RESOLUTION_FIELDS = [
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mask_account(value: object) -> str:
    return re.sub(r"\d(?=\d{3})", "*", normalize_text(value))


def account_key(value: object) -> str:
    return re.sub(r"\D", "", normalize_text(value))


def _decision_source(decision: dict[str, Any]) -> str:
    model = normalize_text(decision.get("model")) or "openai_structured_reranker"
    confidence = decision.get("confidence", "")
    reasons = ",".join(str(item) for item in decision.get("reason_codes") or [])
    parts = [model]
    if confidence != "":
        parts.append(f"confidence={float(confidence):.3f}")
    if reasons:
        parts.append(reasons)
    return "openai_reranker:" + ":".join(parts)


def _accepted_candidates(
    *,
    raw_maps: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    raw_by_key = {
        (normalize_text(row.get("source_id")), normalize_text(row.get("candidate_id"))): row
        for row in raw_maps
    }
    accepted_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        if decision.get("action") != "accept":
            continue
        source_id = normalize_text(decision.get("source_id"))
        candidate_id = normalize_text(decision.get("selected_candidate_id"))
        raw_row = raw_by_key.get((source_id, candidate_id))
        if not raw_row:
            continue
        name = normalize_text(raw_row.get("name"))
        if not name:
            continue
        accepted_by_name[name].append(
            {
                **raw_row,
                "decision_source": _decision_source(decision),
                "confidence": decision.get("confidence", ""),
            }
        )
    return accepted_by_name


def build_openai_resolution_candidates(
    *,
    source_workbook: Path,
    raw_map_jsonl: Path,
    decisions_jsonl: Path,
    output_dir: Path,
    manual_review_workbook: Path | None = None,
) -> dict[str, Any]:
    roster = extract_roster(source_workbook)
    raw_maps = read_jsonl(raw_map_jsonl)
    decisions = read_jsonl(decisions_jsonl)
    accepted_by_name = _accepted_candidates(raw_maps=raw_maps, decisions=decisions)
    hard_gate_names = load_review_names(manual_review_workbook) if manual_review_workbook else set()
    rows: list[dict[str, Any]] = []

    for person in roster.people:
        accepted = accepted_by_name.get(person.name, [])
        unique_by_account: dict[str, dict[str, Any]] = {}
        for row in accepted:
            key = account_key(row.get("candidate_raw"))
            if key and key not in unique_by_account:
                unique_by_account[key] = row
        unique = list(unique_by_account.values())

        chosen = ""
        source = ""
        if person.name in hard_gate_names:
            decision = "manual_review_hard_gate"
            source = "openai_reranker_hard_gate:manual_review_required"
        elif len(unique) == 1:
            chosen = normalize_text(unique[0].get("candidate_raw"))
            decision = "auto_fill_openai_reranker"
            source = normalize_text(unique[0].get("decision_source"))
        elif len(unique) > 1:
            decision = "multiple_openai_reranker_candidates"
            source = "openai_reranker:multiple_accepted_candidates"
        else:
            decision = "openai_reranker_no_candidate"
            source = "openai_reranker:no_accept_decision"

        rows.append(
            {
                "group": person.group,
                "no": person.no,
                "name": person.name,
                "prior_status": "",
                "prior_account_masked": "",
                "chosen_account": chosen,
                "chosen_account_masked": mask_account(chosen),
                "decision": decision,
                "source": source,
                "candidate_count": len(unique),
                "candidate_accounts_masked": "; ".join(mask_account(row.get("candidate_raw")) for row in unique),
                "candidate_files": "; ".join(normalize_text(row.get("source_name")) for row in unique),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    resolution_csv = output_dir / "account_resolution_candidates.csv"
    write_csv(resolution_csv, rows, RESOLUTION_FIELDS)
    summary = {
        "source_workbook": str(source_workbook),
        "raw_map_jsonl": str(raw_map_jsonl),
        "decisions_jsonl": str(decisions_jsonl),
        "resolution_csv": str(resolution_csv),
        "total_people": len(rows),
        "auto_fill_openai_reranker": sum(1 for row in rows if row["decision"] == "auto_fill_openai_reranker"),
        "manual_review_hard_gate": sum(1 for row in rows if row["decision"] == "manual_review_hard_gate"),
        "multiple_openai_reranker_candidates": sum(1 for row in rows if row["decision"] == "multiple_openai_reranker_candidates"),
        "openai_reranker_no_candidate": sum(1 for row in rows if row["decision"] == "openai_reranker_no_candidate"),
        "notes": [
            "This output is local-only because chosen_account contains raw account numbers.",
            "Names present in REMAINING_REVIEW are forced to manual_review_hard_gate even when OpenAI accepts a candidate.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply OpenAI structured reranker decisions through a local raw map and hard gate.")
    parser.add_argument("--source-workbook", type=Path, required=True)
    parser.add_argument("--raw-map-jsonl", type=Path, required=True)
    parser.add_argument("--decisions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manual-review-workbook", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_openai_resolution_candidates(
        source_workbook=args.source_workbook,
        raw_map_jsonl=args.raw_map_jsonl,
        decisions_jsonl=args.decisions_jsonl,
        output_dir=args.output_dir,
        manual_review_workbook=args.manual_review_workbook,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
