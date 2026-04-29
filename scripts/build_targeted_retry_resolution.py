#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from settlement_tool.account_policy import policy_score_candidate
from settlement_tool.core import write_csv
from scripts.build_deepseek_resolution import (
    RERANK_MIN_MARGIN,
    RERANK_POLICY_THRESHOLD,
    mask,
    policy_audit_rows_for_ocr_row,
    policy_rerank_resolution_candidates,
    read_ocr_text_for_policy,
)


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


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def account_key(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def policy_accepts_retry_row(row: dict[str, str], *, min_score: float = RERANK_POLICY_THRESHOLD) -> bool:
    account = row.get("account", "")
    if not account:
        return False
    text = read_ocr_text_for_policy(row)
    decision = policy_score_candidate(text, account)
    return decision.accepted and decision.score >= min_score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-resolution-csv", type=Path, required=True)
    parser.add_argument("--targeted-retry-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reranker-min-score", type=float, default=RERANK_POLICY_THRESHOLD)
    parser.add_argument("--reranker-min-margin", type=float, default=RERANK_MIN_MARGIN)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = read_csv(args.base_resolution_csv)
    retry_rows = read_csv(args.targeted_retry_csv)

    high_by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    all_candidates_by_name: dict[str, Counter[str]] = defaultdict(Counter)
    files_by_name: dict[str, set[str]] = defaultdict(set)
    policy_audit_rows_out: list[dict[str, str]] = []
    for row in retry_rows:
        policy_audit_rows_out.extend(policy_audit_rows_for_ocr_row(row))
        name = row["name"]
        if not name:
            continue
        files_by_name[name].add(row["filename_hint"])
        if row["confidence"] == "high" and row["account"] and policy_accepts_retry_row(row, min_score=args.reranker_min_score):
            high_by_name[name].append(row)
        for candidate in (row.get("candidates") or "").split("; "):
            key = account_key(candidate)
            if key:
                all_candidates_by_name[name][candidate] += 1

    output_rows = []
    for row in base_rows:
        current = dict(row)
        if current["decision"] in {"keep_existing_final_run", "auto_fill_single_candidate"}:
            output_rows.append(current)
            continue

        high_rows = high_by_name.get(current["name"], [])
        unique_high = []
        seen = set()
        for high in high_rows:
            key = account_key(high["account"])
            if not key or key in seen:
                continue
            seen.add(key)
            unique_high.append(high)

        if len(unique_high) == 1:
            chosen = unique_high[0]["account"]
            current["chosen_account"] = chosen
            current["chosen_account_masked"] = mask(chosen)
            current["decision"] = "auto_fill_targeted_deepseek"
            current["source"] = f"targeted_retry:{unique_high[0]['variant']}:{unique_high[0]['prompt_id']}"
            current["candidate_count"] = "1"
            current["candidate_accounts_masked"] = mask(chosen)
            current["candidate_files"] = unique_high[0]["filename_hint"]
        elif len(unique_high) > 1:
            reranked = policy_rerank_resolution_candidates(
                unique_high,
                min_score=args.reranker_min_score,
                min_margin=args.reranker_min_margin,
            )
            if reranked:
                selected = reranked["candidate"]
                chosen = selected["account"]
                current["chosen_account"] = chosen
                current["chosen_account_masked"] = mask(chosen)
                current["decision"] = "auto_fill_targeted_policy_reranker"
                current["source"] = (
                    f"targeted_retry_policy_reranker:min_score={args.reranker_min_score:.1f}:"
                    f"min_margin={args.reranker_min_margin:.1f}:{selected['variant']}:{selected['prompt_id']}:"
                    f"score={float(reranked['policy_score']):.1f}"
                )
                current["candidate_count"] = str(len(unique_high))
                current["candidate_accounts_masked"] = "; ".join(mask(item["account"]) for item in unique_high)
                current["candidate_files"] = "; ".join(sorted(files_by_name.get(current["name"], set())))
            else:
                current["decision"] = "multiple_targeted_high_candidates"
                current["source"] = "targeted_retry"
                current["candidate_count"] = str(len(unique_high))
                current["candidate_accounts_masked"] = "; ".join(mask(item["account"]) for item in unique_high)
                current["candidate_files"] = "; ".join(sorted(files_by_name.get(current["name"], set())))
        elif files_by_name.get(current["name"]):
            current["decision"] = "targeted_retry_no_candidate"
            current["source"] = "targeted_retry"
            candidate_counter = all_candidates_by_name.get(current["name"], Counter())
            current["candidate_count"] = str(len(candidate_counter))
            current["candidate_accounts_masked"] = "; ".join(
                mask(candidate) for candidate, _ in candidate_counter.most_common()
            )
            current["candidate_files"] = "; ".join(sorted(files_by_name.get(current["name"], set())))
        output_rows.append(current)

    write_csv(args.output_dir / "account_resolution_candidates.csv", output_rows, FIELDS)
    write_csv(
        args.output_dir / "targeted_policy_audit.csv",
        policy_audit_rows_out,
        [
            "source_id",
            "source_name",
            "filename_hint",
            "matched_name",
            "variant",
            "prompt_id",
            "ocr_confidence",
            "candidate_masked",
            "accepted",
            "policy_score",
            "policy_reasons",
            "has_prompt_leakage_context",
            "has_wrong_field_context",
            "has_direct_account_field_context",
            "has_structured_bankbook_context",
        ],
    )
    summary = {
        "total_people": len(output_rows),
        "filled": sum(1 for row in output_rows if row["chosen_account"]),
        "keep_existing": sum(1 for row in output_rows if row["decision"] == "keep_existing_final_run"),
        "auto_fill_single_candidate": sum(1 for row in output_rows if row["decision"] == "auto_fill_single_candidate"),
        "auto_fill_targeted_deepseek": sum(1 for row in output_rows if row["decision"] == "auto_fill_targeted_deepseek"),
        "auto_fill_targeted_policy_reranker": sum(1 for row in output_rows if row["decision"] == "auto_fill_targeted_policy_reranker"),
        "multiple_or_review": sum(1 for row in output_rows if row["decision"].startswith("multiple")),
        "targeted_retry_no_candidate": sum(1 for row in output_rows if row["decision"] == "targeted_retry_no_candidate"),
        "no_candidate": sum(1 for row in output_rows if row["decision"] == "no_candidate"),
        "policy_audit_rows": len(policy_audit_rows_out),
        "policy_rejected_rows": sum(1 for row in policy_audit_rows_out if row.get("accepted") == "0"),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
