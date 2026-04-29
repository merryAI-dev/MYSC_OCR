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


DEFAULT_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY = (
    "Select the target Korean bank account candidate from OCR features. "
    "Prefer direct account fields, structured bankbook context, bank keywords, and account keywords. "
    "Reject phone numbers, prompt leakage, wrong table fields, and customer-number metadata."
)


def candidate_document(candidate: dict[str, Any]) -> str:
    context = candidate.get("context_flags") or {}
    risks = candidate.get("risk_flags") or {}
    shape = candidate.get("shape_features") or {}
    field = candidate.get("field_evidence") or {}
    bank_holder = candidate.get("bank_holder_evidence") or {}
    consensus = candidate.get("consensus_features") or {}
    source = candidate.get("source_evidence") or {}
    positive = [key for key, value in context.items() if value]
    negative = [key for key, value in risks.items() if value]
    return (
        f"candidate_id={candidate.get('candidate_id')} "
        f"account_shape={candidate.get('account_shape')} "
        f"digit_count={candidate.get('digit_count')} "
        f"hyphen_count={candidate.get('hyphen_count')} "
        f"group_count={candidate.get('group_count')} "
        f"repeat_count={candidate.get('repeat_count')} "
        f"local_policy_score={candidate.get('teacher_policy_score')} "
        f"positive_evidence={','.join(positive) or 'none'} "
        f"risk_flags={','.join(negative) or 'none'} "
        f"group_lengths={shape.get('group_lengths')} "
        f"digit_count_bucket={shape.get('digit_count_bucket')} "
        f"pattern_family={shape.get('pattern_family')} "
        f"prefix_class={shape.get('prefix_class')} "
        f"has_bank_style_hyphenation={shape.get('has_bank_style_hyphenation')} "
        f"same_line_label_type={field.get('same_line_label_type')} "
        f"table_row_label_type={field.get('table_row_label_type')} "
        f"is_value_in_account_field={field.get('is_value_in_account_field')} "
        f"is_value_in_customer_number_field={field.get('is_value_in_customer_number_field')} "
        f"bank_name_present={bank_holder.get('bank_name_present')} "
        f"bank_name_normalized={bank_holder.get('bank_name_normalized')} "
        f"holder_field_present={bank_holder.get('holder_field_present')} "
        f"holder_match_status={bank_holder.get('holder_match_status')} "
        f"bankbook_doc_type_confidence={bank_holder.get('bankbook_doc_type_confidence')} "
        f"variant_vote_count={consensus.get('variant_vote_count')} "
        f"prompt_vote_count={consensus.get('prompt_vote_count')} "
        f"seen_in_full_ocr={consensus.get('seen_in_full_ocr')} "
        f"seen_in_targeted_retry={consensus.get('seen_in_targeted_retry')} "
        f"same_candidate_seen_across_variants={consensus.get('same_candidate_seen_across_variants')} "
        f"unique_candidate_count_for_person={consensus.get('unique_candidate_count_for_person')} "
        f"source_kind={source.get('source_kind')} "
        f"variant={source.get('variant')} "
        f"prompt_id={source.get('prompt_id')}"
    )


def _score_pairs(model_id: str, pairs: list[tuple[str, str]], *, local_files_only: bool = True) -> list[float]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=local_files_only)
    model.eval()
    encoded = tokenizer(
        [query for query, _ in pairs],
        [document for _, document in pairs],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits.squeeze(-1)
    return [float(score) for score in logits.reshape(-1).tolist()]


def _hard_risk(candidate: dict[str, Any]) -> bool:
    risks = candidate.get("risk_flags") or {}
    context = candidate.get("context_flags") or {}
    if risks.get("looks_like_phone") or risks.get("has_wrong_field_context"):
        return True
    return bool(risks.get("has_prompt_leakage_context") and not context.get("has_direct_account_field_context"))


def rerank_payload_with_scores(
    payload: dict[str, Any],
    *,
    scores: list[float],
    model_id: str,
    policy_threshold: float = 10.0,
    min_score: float = -999.0,
    min_margin: float = 0.0,
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
            "model": f"local_cross_encoder:{model_id}",
        }
    ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
    top, top_score = ranked[0]
    runner_score = ranked[1][1] if len(ranked) > 1 else -999.0
    margin = top_score - runner_score
    top_policy_score = float(top.get("teacher_policy_score") or 0.0)
    top_risks = sorted(key for key, value in (top.get("risk_flags") or {}).items() if value)

    if _hard_risk(top):
        action = "reject"
        selected_candidate_id = None
        reason_codes = ["top_candidate_has_hard_risk"]
        confidence = 0.1
    elif top_policy_score >= policy_threshold and top_score >= min_score and (len(ranked) == 1 or margin >= min_margin):
        action = "accept"
        selected_candidate_id = top.get("candidate_id")
        reason_codes = ["cross_encoder_top_rank", "policy_score_above_threshold"]
        confidence = 1.0 / (1.0 + math.exp(-max(min(top_score, 20.0), -20.0)))
    elif top_policy_score >= policy_threshold:
        action = "review"
        selected_candidate_id = None
        reason_codes = ["policy_score_above_threshold_but_cross_encoder_margin_low"]
        confidence = 0.5
    else:
        action = "review"
        selected_candidate_id = None
        reason_codes = ["policy_score_below_threshold"]
        confidence = 0.25

    decision = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "source_id": source_id,
        "action": action,
        "selected_candidate_id": selected_candidate_id,
        "confidence": round(confidence, 4),
        "reason_codes": reason_codes,
        "risk_flags": top_risks,
        "model": f"local_cross_encoder:{model_id}",
        "reranker_score": round(top_score, 6),
        "runner_up_score": round(runner_score, 6) if runner_score > -900 else None,
        "score_margin": round(margin, 6) if runner_score > -900 else None,
    }
    validate_decision(payload, decision)
    return decision


def run_local_cross_encoder_rerank(
    *,
    payload_jsonl: Path,
    output_jsonl: Path,
    model_id: str = DEFAULT_MODEL_ID,
    policy_threshold: float = 10.0,
    min_score: float = -999.0,
    min_margin: float = 0.0,
    local_files_only: bool = True,
) -> dict[str, Any]:
    payloads = read_jsonl(payload_jsonl)
    pairs: list[tuple[str, str]] = []
    candidate_counts: list[int] = []
    for payload in payloads:
        candidates = list(payload.get("candidates") or [])
        candidate_counts.append(len(candidates))
        for candidate in candidates:
            pairs.append((QUERY, candidate_document(candidate)))

    scores = _score_pairs(model_id, pairs, local_files_only=local_files_only) if pairs else []
    decisions = []
    offset = 0
    for payload, count in zip(payloads, candidate_counts):
        payload_scores = scores[offset : offset + count]
        offset += count
        decisions.append(
            rerank_payload_with_scores(
                payload,
                scores=payload_scores,
                model_id=model_id,
                policy_threshold=policy_threshold,
                min_score=min_score,
                min_margin=min_margin,
            )
        )

    write_jsonl(output_jsonl, decisions)
    summary = {
        "payload_jsonl": str(payload_jsonl),
        "output_jsonl": str(output_jsonl),
        "model_id": model_id,
        "decision_count": len(decisions),
        "candidate_count": len(scores),
        "policy_threshold": policy_threshold,
        "min_score": min_score,
        "min_margin": min_margin,
        "action_counts": {
            action: sum(1 for decision in decisions if decision.get("action") == action)
            for action in ("accept", "review", "reject")
        },
    }
    (output_jsonl.parent / "local_cross_encoder_reranker_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local Hugging Face cross-encoder reranker over redacted candidate payloads.")
    parser.add_argument("--payload-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--policy-threshold", type=float, default=10.0)
    parser.add_argument("--min-score", type=float, default=-999.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_local_cross_encoder_rerank(
        payload_jsonl=args.payload_jsonl,
        output_jsonl=args.output_jsonl,
        model_id=args.model_id,
        policy_threshold=args.policy_threshold,
        min_score=args.min_score,
        min_margin=args.min_margin,
        local_files_only=not args.allow_download,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
