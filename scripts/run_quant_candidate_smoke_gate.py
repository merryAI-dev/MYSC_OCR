#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from settlement_tool.free_running_gate import free_running_degeneration_metrics, tokenize_free_running_output
except ImportError:  # pragma: no cover - fallback is for standalone portability.
    free_running_degeneration_metrics = None
    tokenize_free_running_output = None


TEXT_FIELDS = ("text", "raw_text", "response", "output", "ocr_text")
REPETITION_REASONS = {
    "low_unique_token_ratio",
    "top_token_dominance",
    "long_token_run",
    "high_trigram_repetition",
}


def _fallback_metrics(text: str, *, require_account_candidate: bool) -> dict[str, object]:
    normalized = " ".join((text or "").split())
    tokens = normalized.split()
    token_count = len(tokens)
    token_counts = Counter(token.casefold() for token in tokens)
    unique_token_ratio = len(token_counts) / token_count if token_count else 0.0
    top_token_share = max(token_counts.values()) / token_count if token_counts else 0.0
    max_token_run = 0
    current_run = 0
    previous = None
    for token in tokens:
        current = token.casefold()
        current_run = current_run + 1 if current == previous else 1
        previous = current
        max_token_run = max(max_token_run, current_run)
    account_candidate_presence = any(char.isdigit() for char in normalized)

    reasons: list[str] = []
    if token_count == 0:
        reasons.append("empty_output")
    if token_count >= 10 and unique_token_ratio < 0.18:
        reasons.append("low_unique_token_ratio")
    if token_count >= 10 and top_token_share > 0.45:
        reasons.append("top_token_dominance")
    if max_token_run > 8:
        reasons.append("long_token_run")
    if require_account_candidate and not account_candidate_presence:
        reasons.append("missing_account_candidate")

    return {
        "token_count": token_count,
        "unique_token_ratio": unique_token_ratio,
        "top_token_share": top_token_share,
        "max_token_run": max_token_run,
        "repetition_trigram_ratio": 0.0,
        "digit_group_count": int(account_candidate_presence),
        "account_candidate_count": int(account_candidate_presence),
        "account_candidate_presence": account_candidate_presence,
        "hangul_presence": False,
        "degeneration_pass": not reasons,
        "degeneration_reason": ",".join(reasons) if reasons else "ok",
    }


def row_text(row: dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        value = row.get(field)
        if value is not None:
            return str(value)
    text_path = row.get("text_path")
    if text_path:
        path = Path(str(text_path))
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def degeneration_reasons(metrics: dict[str, object]) -> list[str]:
    reason = str(metrics.get("degeneration_reason") or "")
    if not reason or reason == "ok":
        return []
    return [part for part in reason.split(",") if part]


def repeated_token_signature(text: str, reasons: Iterable[str]) -> str:
    if not REPETITION_REASONS.intersection(reasons):
        return ""
    if tokenize_free_running_output is not None:
        tokens = tokenize_free_running_output(text)
    else:
        tokens = (text or "").split()
    token_counts = Counter(token.casefold() for token in tokens if token.strip())
    if not token_counts:
        return ""
    token, count = token_counts.most_common(1)[0]
    return f"{token} x{count}"


def classify_output(row: dict[str, Any], *, require_account_candidate: bool = True) -> dict[str, object]:
    text = row_text(row)
    error = str(row.get("error") or "")
    if free_running_degeneration_metrics is not None:
        metrics = free_running_degeneration_metrics(text, require_account_candidate=require_account_candidate)
    else:
        metrics = _fallback_metrics(text, require_account_candidate=require_account_candidate)
    reasons = degeneration_reasons(metrics)
    if error:
        reasons.append("backend_error")
    signature = repeated_token_signature(text, reasons)
    degenerate = bool(reasons)
    return {
        "id": row.get("id", row.get("item_id", "")),
        "name": row.get("name", ""),
        "prompt_id": row.get("prompt_id", ""),
        "degenerate": degenerate,
        "reasons": reasons,
        "reason": ",".join(reasons) if reasons else "ok",
        "missing_account_candidate": "missing_account_candidate" in reasons,
        "repeated_token_signature": signature,
        "metrics": metrics,
        "error": error,
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    max_degenerate_outputs: int = 0,
    require_account_candidate: bool = True,
) -> dict[str, object]:
    classified = [classify_output(row, require_account_candidate=require_account_candidate) for row in rows]
    degenerate_rows = [row for row in classified if row["degenerate"]]
    reason_counts = Counter(reason for row in degenerate_rows for reason in row["reasons"])
    signature_counts = Counter(str(row["repeated_token_signature"]) for row in degenerate_rows if row["repeated_token_signature"])
    degenerate_outputs = len(degenerate_rows)
    failed_reason = ""
    if not rows:
        failed_reason = "no_rows"
    elif degenerate_outputs > max_degenerate_outputs:
        failed_reason = f"degenerate_outputs {degenerate_outputs} > max_degenerate_outputs {max_degenerate_outputs}"
    status = "fail" if failed_reason else "pass"
    return {
        "status": status,
        "total": len(rows),
        "degenerate_outputs": degenerate_outputs,
        "missing_account_candidate": int(reason_counts.get("missing_account_candidate", 0)),
        "repeated_token_signatures": dict(sorted(signature_counts.items())),
        "repeated_token_reasons": dict(sorted(reason_counts.items())),
        "max_degenerate_outputs": max_degenerate_outputs,
        "failed_reason": failed_reason,
        "rows": classified,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def live_ocr_rows(
    manifest: Path,
    api_url: str,
    *,
    limit: int = 0,
    timeout: int = 900,
    prompt: str = "",
) -> list[dict[str, Any]]:
    from scripts.evaluate_ocr_server import PROMPT_PRESETS, post_ocr

    selected_prompt = prompt or PROMPT_PRESETS["bank_fields"]
    rows: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if limit and len(rows) >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            image_path = Path(str(item.get("image_path", "")))
            row: dict[str, Any] = {
                "id": item.get("id", ""),
                "name": item.get("name", ""),
                "image_path": str(image_path),
            }
            try:
                row["text"] = post_ocr(image_path, api_url, timeout, selected_prompt)
            except Exception as exc:  # live smoke should summarize backend failures.
                row["text"] = ""
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-fast smoke gate for a quantized OCR candidate.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--responses-jsonl", type=Path, help="Offline JSONL with OCR text/response rows.")
    mode.add_argument("--manifest", type=Path, help="Manifest JSONL for live OCR smoke.")
    parser.add_argument("--api-url", default="", help="OCR server URL. Required with --manifest.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-degenerate-outputs", type=int, default=0)
    parser.add_argument("--allow-missing-account-candidate", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--prompt", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.responses_jsonl:
        rows = read_jsonl(args.responses_jsonl)
    else:
        if not args.api_url:
            raise SystemExit("--api-url is required with --manifest")
        rows = live_ocr_rows(args.manifest, args.api_url, limit=args.limit, timeout=args.timeout, prompt=args.prompt)

    summary = summarize_rows(
        rows,
        max_degenerate_outputs=args.max_degenerate_outputs,
        require_account_candidate=not args.allow_missing_account_candidate,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
