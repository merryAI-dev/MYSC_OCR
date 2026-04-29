#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DECISION_SCHEMA_VERSION = "openai_account_rerank_decision_v1"
DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string"},
        "source_id": {"type": "string"},
        "action": {"type": "string", "enum": ["accept", "review", "reject"]},
        "selected_candidate_id": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason_codes": {"type": "array", "items": {"type": "string"}},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "schema_version",
        "source_id",
        "action",
        "selected_candidate_id",
        "confidence",
        "reason_codes",
        "risk_flags",
    ],
}

SYSTEM_PROMPT = """You are a privacy-preserving reranker for bank-account OCR candidates.
You receive only redacted numeric-shape features and policy flags. Do not infer or output
any personal data. Select exactly one candidate only when the top candidate is clearly a
target bank account. Otherwise return review or reject."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _risk_flag_names(candidate: dict[str, Any]) -> list[str]:
    risk_flags = candidate.get("risk_flags") or {}
    return sorted(key for key, value in risk_flags.items() if value)


def dry_run_rerank_payload(
    payload: dict[str, Any],
    *,
    threshold: float = 10.0,
    min_margin: float = 3.0,
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
        }

    ranked = sorted(candidates, key=lambda row: float(row.get("teacher_policy_score") or 0.0), reverse=True)
    top = ranked[0]
    runner_up_score = float(ranked[1].get("teacher_policy_score") or 0.0) if len(ranked) > 1 else -999.0
    top_score = float(top.get("teacher_policy_score") or 0.0)
    margin = top_score - runner_up_score
    top_risk_flags = _risk_flag_names(top)
    hard_risks = {"looks_like_phone", "has_wrong_field_context"}
    prompt_leak_without_direct_field = (
        "has_prompt_leakage_context" in top_risk_flags
        and not bool((top.get("context_flags") or {}).get("has_direct_account_field_context"))
    )

    if hard_risks.intersection(top_risk_flags) or prompt_leak_without_direct_field:
        action = "reject"
        selected_candidate_id = None
        confidence = 0.15
        reason_codes = ["top_candidate_has_hard_risk"]
    elif top_score >= threshold and (len(ranked) == 1 or margin >= min_margin):
        action = "accept"
        selected_candidate_id = top.get("candidate_id")
        confidence = min(0.99, 0.55 + max(top_score - threshold, 0.0) / 20.0 + max(margin, 0.0) / 20.0)
        reason_codes = ["top_score_above_threshold", "margin_clear"]
    elif top_score >= threshold:
        action = "review"
        selected_candidate_id = None
        confidence = 0.55
        reason_codes = ["top_score_above_threshold_but_margin_low"]
    else:
        action = "review"
        selected_candidate_id = None
        confidence = 0.35
        reason_codes = ["top_score_below_threshold"]

    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "source_id": source_id,
        "action": action,
        "selected_candidate_id": selected_candidate_id,
        "confidence": round(confidence, 4),
        "reason_codes": reason_codes,
        "risk_flags": top_risk_flags,
    }


def validate_decision(payload: dict[str, Any], decision: dict[str, Any]) -> None:
    source_id = str(payload.get("source_id") or "")
    if decision.get("source_id") != source_id:
        raise ValueError(f"decision source_id mismatch: {decision.get('source_id')} != {source_id}")
    action = decision.get("action")
    if action not in {"accept", "review", "reject"}:
        raise ValueError(f"invalid action: {action}")
    confidence = float(decision.get("confidence") or 0.0)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"invalid confidence: {confidence}")
    candidate_ids = {candidate.get("candidate_id") for candidate in payload.get("candidates") or []}
    selected = decision.get("selected_candidate_id")
    if action == "accept" and selected not in candidate_ids:
        raise ValueError(f"accepted decision selected an unknown candidate_id: {selected}")
    if action != "accept" and selected not in ("", None):
        raise ValueError("review/reject decisions must not select a candidate")


def _extract_output_text(response: dict[str, Any]) -> str:
    if response.get("output_text"):
        return str(response["output_text"])
    for output in response.get("output", []):
        for content in output.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                return str(content["text"])
    raise RuntimeError("OpenAI response did not contain output text")


def call_openai_structured_reranker(
    payload: dict[str, Any],
    *,
    model: str,
    api_key: str | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is used")

    request_body = {
        "model": model,
        "store": False,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "account_candidate_rerank_decision",
                "strict": True,
                "schema": DECISION_SCHEMA,
            }
        },
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed: {exc.code} {body}") from exc

    decision = json.loads(_extract_output_text(response_payload))
    validate_decision(payload, decision)
    return decision


def run_openai_structured_rerank(
    *,
    payload_jsonl: Path,
    output_jsonl: Path,
    model: str,
    dry_run: bool = False,
    threshold: float = 10.0,
    min_margin: float = 3.0,
) -> dict[str, Any]:
    payloads = read_jsonl(payload_jsonl)
    decisions: list[dict[str, Any]] = []
    for payload in payloads:
        decision = (
            dry_run_rerank_payload(payload, threshold=threshold, min_margin=min_margin)
            if dry_run
            else call_openai_structured_reranker(payload, model=model)
        )
        validate_decision(payload, decision)
        decision["model"] = "dry_run_policy_proxy" if dry_run else model
        decisions.append(decision)
    write_jsonl(output_jsonl, decisions)
    summary = {
        "payload_jsonl": str(payload_jsonl),
        "output_jsonl": str(output_jsonl),
        "model": "dry_run_policy_proxy" if dry_run else model,
        "decision_count": len(decisions),
        "action_counts": {
            action: sum(1 for decision in decisions if decision.get("action") == action)
            for action in ("accept", "review", "reject")
        },
    }
    (output_jsonl.parent / "openai_reranker_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a structured OpenAI reranker over redacted account-candidate payloads.")
    parser.add_argument("--payload-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--min-margin", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_openai_structured_rerank(
        payload_jsonl=args.payload_jsonl,
        output_jsonl=args.output_jsonl,
        model=args.model,
        dry_run=args.dry_run,
        threshold=args.threshold,
        min_margin=args.min_margin,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
