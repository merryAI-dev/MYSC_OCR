#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.openai_structured_rerank import DECISION_SCHEMA_VERSION, read_jsonl, validate_decision, write_jsonl  # noqa: E402


def _visible_numbers_account_field_rescue(candidate: dict[str, Any]) -> bool:
    risk = candidate.get("risk_flags") or {}
    field = candidate.get("field_evidence") or {}
    bank_holder = candidate.get("bank_holder_evidence") or {}
    consensus = candidate.get("consensus_features") or {}
    shape = candidate.get("shape_features") or {}
    source = candidate.get("source_evidence") or {}
    return bool(
        source.get("source_kind") == "targeted_retry"
        and source.get("prompt_id") == "visible_numbers"
        and risk.get("has_wrong_field_context")
        and not risk.get("looks_like_phone")
        and not risk.get("has_prompt_leakage_context")
        and field.get("is_value_in_account_field")
        and field.get("same_line_label_type") == "account_number"
        and not field.get("is_value_in_customer_number_field")
        and bank_holder.get("bank_name_present")
        and shape.get("has_bank_style_hyphenation")
        and not shape.get("is_single_long_run")
        and int(consensus.get("unique_candidate_count_for_person") or 0) == 1
        and int(consensus.get("candidate_source_count_for_person") or 0) >= 2
        and (
            consensus.get("same_candidate_seen_across_variants")
            or int(consensus.get("variant_vote_count") or 0) >= 2
        )
    )


def v2_feature_score(candidate: dict[str, Any]) -> float:
    score = float(candidate.get("teacher_policy_score") or 0.0)
    shape = candidate.get("shape_features") or {}
    field = candidate.get("field_evidence") or {}
    bank_holder = candidate.get("bank_holder_evidence") or {}
    consensus = candidate.get("consensus_features") or {}
    risk = candidate.get("risk_flags") or {}
    source = candidate.get("source_evidence") or {}
    kie = candidate.get("kie_evidence") or {}
    kie_field = kie.get("field_type")
    has_kie_evidence = bool(kie.get("backend"))
    candidate_source_count = int(consensus.get("candidate_source_count_for_person") or 0)

    if kie_field == "account_number" and kie.get("confidence_bucket") == "high":
        score += 4.0
    if kie_field == "account_number" and bank_holder.get("holder_match_status") == "match":
        score += 2.0
    if kie_field == "account_number" and bank_holder.get("bank_name_present"):
        score += 2.0
    if field.get("is_value_in_account_field"):
        score += 4.0
    if field.get("same_line_label_type") == "account_number":
        score += 2.0
    if field.get("table_row_label_type") == "account_number":
        score += 2.0
    if bank_holder.get("bank_name_present"):
        score += 2.0
    if bank_holder.get("holder_match_status") == "match":
        score += 2.0
    if bank_holder.get("bankbook_doc_type_confidence") == "high":
        score += 2.0
    if shape.get("pattern_family") == "bank_account_like":
        score += 1.0
    if consensus.get("same_candidate_seen_across_variants"):
        score += 2.0
    if consensus.get("seen_in_full_ocr") and consensus.get("seen_in_targeted_retry"):
        score += 2.0
    if int(consensus.get("variant_vote_count") or 0) >= 2:
        score += 1.0
    if (
        source.get("source_kind") == "targeted_retry"
        and source.get("prompt_id") == "visible_numbers"
        and shape.get("has_bank_style_hyphenation")
        and int(consensus.get("unique_candidate_count_for_person") or 0) == 1
        and not any(bool(value) for value in risk.values())
    ):
        score += 4.0
    if (
        source.get("source_kind") == "targeted_retry"
        and source.get("prompt_id") == "account_structured_ko"
        and risk.get("has_prompt_leakage_context")
        and field.get("is_value_in_account_field")
        and field.get("same_line_label_type") == "account_number"
        and bank_holder.get("bank_name_present")
        and shape.get("has_bank_style_hyphenation")
        and not shape.get("is_single_long_run")
        and 1 <= candidate_source_count <= 2
        and int(consensus.get("unique_candidate_count_for_person") or 0) <= 2
    ):
        score += 2.0
    if (
        source.get("source_kind") == "targeted_retry"
        and source.get("prompt_id") == "account_structured_ko"
        and risk.get("has_prompt_leakage_context")
        and field.get("is_value_in_account_field")
        and field.get("same_line_label_type") == "account_number"
        and bank_holder.get("bank_name_present")
        and shape.get("has_bank_style_hyphenation")
        and not shape.get("is_single_long_run")
        and int(consensus.get("unique_candidate_count_for_person") or 0) == 1
        and int(consensus.get("candidate_source_count_for_person") or 0) >= 4
        and int(consensus.get("variant_vote_count") or 0) >= 4
    ):
        score += 1.0
    if _visible_numbers_account_field_rescue(candidate):
        score += 32.0

    if has_kie_evidence and kie_field != "account_number":
        score -= 30.0
    if (
        has_kie_evidence
        and kie_field == "account_number"
        and int(consensus.get("unique_candidate_count_for_person") or 0) >= 5
        and bank_holder.get("holder_match_status") != "match"
    ):
        score -= 30.0
    if kie_field in {"customer_number", "phone", "date", "amount"}:
        score -= 20.0
    if has_kie_evidence and kie_field == "unknown" and not consensus.get("same_candidate_seen_across_variants"):
        score -= 8.0
    if field.get("is_value_in_customer_number_field"):
        score -= 10.0
    if shape.get("is_single_long_run") and not field.get("is_value_in_account_field"):
        score -= 4.0
    if (
        risk.get("has_prompt_leakage_context")
        and source.get("source_kind") == "targeted_retry"
        and source.get("prompt_id") == "account_structured_ko"
        and bank_holder.get("holder_match_status") != "match"
    ):
        score -= 12.0
    if (
        shape.get("is_single_long_run")
        and risk.get("has_prompt_leakage_context")
        and source.get("source_kind") == "targeted_retry"
    ):
        score -= 4.0
    if (
        int(consensus.get("unique_candidate_count_for_person") or 0) >= 4
        and bank_holder.get("holder_match_status") != "match"
    ):
        score -= 20.0
    if risk.get("looks_like_phone") or risk.get("has_wrong_field_context"):
        score -= 40.0
    if risk.get("has_prompt_leakage_context") and not field.get("is_value_in_account_field"):
        score -= 10.0
    return score


def _hard_risk(candidate: dict[str, Any]) -> bool:
    risk = candidate.get("risk_flags") or {}
    field = candidate.get("field_evidence") or {}
    kie = candidate.get("kie_evidence") or {}
    wrong_field_hard_risk = risk.get("has_wrong_field_context") and not _visible_numbers_account_field_rescue(candidate)
    return bool(
        risk.get("looks_like_phone")
        or wrong_field_hard_risk
        or (bool(kie.get("backend")) and kie.get("field_type") != "account_number")
        or (
            bool(kie.get("backend"))
            and kie.get("field_type") == "account_number"
            and int((candidate.get("consensus_features") or {}).get("unique_candidate_count_for_person") or 0) >= 5
            and (candidate.get("bank_holder_evidence") or {}).get("holder_match_status") != "match"
        )
        or (risk.get("has_prompt_leakage_context") and not field.get("is_value_in_account_field"))
    )


def rerank_payload_with_v2_feature_score(
    payload: dict[str, Any],
    *,
    threshold: float = 10.0,
    min_margin: float = 2.0,
) -> dict[str, Any]:
    candidates = list(payload.get("candidates") or [])
    source_id = str(payload.get("source_id") or "")
    if not candidates:
        return {
            "schema_version": DECISION_SCHEMA_VERSION,
            "source_id": source_id,
            "action": "reject",
            "selected_candidate_id": None,
            "confidence": 0.0,
            "reason_codes": ["no_candidates"],
            "risk_flags": [],
            "model": "v2_feature_score",
        }

    ranked = sorted(((candidate, v2_feature_score(candidate)) for candidate in candidates), key=lambda item: item[1], reverse=True)
    top, top_score = ranked[0]
    runner_score = ranked[1][1] if len(ranked) > 1 else -999.0
    margin = top_score - runner_score
    risk_flags = sorted(key for key, value in (top.get("risk_flags") or {}).items() if value)

    if _hard_risk(top):
        action = "reject"
        selected_candidate_id = None
        reason_codes = ["top_candidate_has_hard_risk"]
    elif top_score >= threshold and (len(ranked) == 1 or margin >= min_margin):
        action = "accept"
        selected_candidate_id = top.get("candidate_id")
        reason_codes = ["v2_feature_score_above_threshold", "margin_clear"]
    else:
        action = "review"
        selected_candidate_id = None
        reason_codes = ["v2_feature_score_below_threshold_or_margin_low"]

    decision = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "source_id": source_id,
        "action": action,
        "selected_candidate_id": selected_candidate_id,
        "confidence": round(1.0 / (1.0 + math.exp(-max(min(top_score / 10.0, 20.0), -20.0))), 4),
        "reason_codes": reason_codes,
        "risk_flags": risk_flags,
        "model": "v2_feature_score",
        "feature_score": round(top_score, 4),
        "runner_up_score": round(runner_score, 4) if runner_score > -900 else None,
        "score_margin": round(margin, 4) if runner_score > -900 else None,
    }
    validate_decision(payload, decision)
    return decision


def run_local_feature_score_rerank(
    *,
    payload_jsonl: Path,
    output_jsonl: Path,
    threshold: float = 10.0,
    min_margin: float = 2.0,
) -> dict[str, Any]:
    payloads = read_jsonl(payload_jsonl)
    decisions = [
        rerank_payload_with_v2_feature_score(payload, threshold=threshold, min_margin=min_margin)
        for payload in payloads
    ]
    write_jsonl(output_jsonl, decisions)
    summary = {
        "payload_jsonl": str(payload_jsonl),
        "output_jsonl": str(output_jsonl),
        "threshold": threshold,
        "min_margin": min_margin,
        "decision_count": len(decisions),
        "action_counts": {
            action: sum(1 for decision in decisions if decision.get("action") == action)
            for action in ("accept", "review", "reject")
        },
    }
    (output_jsonl.parent / "local_feature_score_reranker_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic v2 redacted-feature scoring over account-candidate payloads.")
    parser.add_argument("--payload-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--min-margin", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_local_feature_score_rerank(
        payload_jsonl=args.payload_jsonl,
        output_jsonl=args.output_jsonl,
        threshold=args.threshold,
        min_margin=args.min_margin,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
