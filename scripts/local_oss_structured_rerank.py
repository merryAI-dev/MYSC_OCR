#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.openai_structured_rerank import (  # noqa: E402
    DECISION_SCHEMA,
    DECISION_SCHEMA_VERSION,
    SYSTEM_PROMPT,
    dry_run_rerank_payload,
    read_jsonl,
    validate_decision,
    write_jsonl,
)


USER_PROMPT = """Return only a JSON object matching the schema.

Rules:
- action=accept only when exactly one candidate is clearly the target bank account.
- action=review when there is ambiguity, weak evidence, or close candidates.
- action=reject when the top candidate has hard risk flags.
- selected_candidate_id must be null unless action is accept.
- Never output raw account numbers, names, OCR text, or paths.

Payload:
{payload_json}
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in model output: {text[:200]}")
    return json.loads(text[start : end + 1])


def _repair_decision(payload: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    repaired = {
        "schema_version": str(decision.get("schema_version") or DECISION_SCHEMA_VERSION),
        "source_id": str(decision.get("source_id") or payload.get("source_id") or ""),
        "action": str(decision.get("action") or "review").lower(),
        "selected_candidate_id": decision.get("selected_candidate_id"),
        "confidence": float(decision.get("confidence") or 0.0),
        "reason_codes": decision.get("reason_codes") if isinstance(decision.get("reason_codes"), list) else [],
        "risk_flags": decision.get("risk_flags") if isinstance(decision.get("risk_flags"), list) else [],
    }
    if repaired["action"] not in {"accept", "review", "reject"}:
        repaired["action"] = "review"
    if repaired["action"] != "accept":
        repaired["selected_candidate_id"] = None
    repaired["confidence"] = max(0.0, min(1.0, repaired["confidence"]))
    return repaired


def call_ollama_structured_reranker(
    payload: dict[str, Any],
    *,
    model: str,
    host: str = "http://127.0.0.1:11434",
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    request_body = {
        "model": model,
        "stream": False,
        "format": DECISION_SCHEMA,
        "options": {
            "temperature": 0,
            "num_predict": 256,
            "num_gpu": 0,
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(payload_json=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            },
        ],
    }
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama server is not reachable. Start it with `ollama serve`.") from exc

    content = str((response_payload.get("message") or {}).get("content") or "")
    decision = _repair_decision(payload, _extract_json_object(content))
    validate_decision(payload, decision)
    return decision


def run_local_oss_structured_rerank(
    *,
    payload_jsonl: Path,
    output_jsonl: Path,
    model: str,
    provider: str = "ollama",
    host: str = "http://127.0.0.1:11434",
    fallback_policy_on_error: bool = False,
    threshold: float = 10.0,
    min_margin: float = 3.0,
) -> dict[str, Any]:
    payloads = read_jsonl(payload_jsonl)
    decisions: list[dict[str, Any]] = []
    error_count = 0
    for payload in payloads:
        try:
            if provider != "ollama":
                raise ValueError(f"unsupported provider: {provider}")
            decision = call_ollama_structured_reranker(payload, model=model, host=host)
        except Exception as exc:
            if not fallback_policy_on_error:
                raise
            error_count += 1
            decision = dry_run_rerank_payload(payload, threshold=threshold, min_margin=min_margin)
            decision["fallback_error"] = str(exc)
        validate_decision(payload, decision)
        decision["model"] = f"{provider}:{model}"
        decisions.append(decision)

    write_jsonl(output_jsonl, decisions)
    summary = {
        "payload_jsonl": str(payload_jsonl),
        "output_jsonl": str(output_jsonl),
        "provider": provider,
        "model": model,
        "decision_count": len(decisions),
        "error_count": error_count,
        "action_counts": {
            action: sum(1 for decision in decisions if decision.get("action") == action)
            for action in ("accept", "review", "reject")
        },
    }
    (output_jsonl.parent / "local_oss_reranker_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local open-weight structured reranker over redacted account-candidate payloads.")
    parser.add_argument("--payload-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--provider", default="ollama", choices=["ollama"])
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--fallback-policy-on-error", action="store_true")
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--min-margin", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_local_oss_structured_rerank(
        payload_jsonl=args.payload_jsonl,
        output_jsonl=args.output_jsonl,
        provider=args.provider,
        model=args.model,
        host=args.host,
        fallback_policy_on_error=args.fallback_policy_on_error,
        threshold=args.threshold,
        min_margin=args.min_margin,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
