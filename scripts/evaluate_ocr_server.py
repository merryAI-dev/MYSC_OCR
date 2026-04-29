#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from settlement_tool.core import classify_account_candidates, safe_filename_part, write_csv  # noqa: E402
from settlement_tool.free_running_gate import free_running_degeneration_metrics  # noqa: E402
from settlement_tool.ocrbench_v2_bankbook import (  # noqa: E402
    BankbookGold,
    BankbookPrediction,
    bankbook_ocrbench_v2_metrics,
)


CSV_FIELDS = [
    "id",
    "split",
    "name",
    "image_name",
    "label_account_masked",
    "predicted_account_masked",
    "candidate_accounts_masked",
    "exact_match",
    "candidate_exact_match",
    "false_positive",
    "ocrbench_recognition_score",
    "account_digit_edit_similarity",
    "ocrbench_extraction_f1",
    "ocrbench_basic_vqa_score",
    "ocrbench_composite_score",
    "free_running_gate_pass",
    "surface_gate_pass",
    "degeneration_reason",
    "token_count",
    "unique_token_ratio",
    "top_token_share",
    "max_token_run",
    "repetition_trigram_ratio",
    "digit_group_count",
    "account_candidate_count",
    "account_candidate_presence",
    "hangul_presence",
    "confidence",
    "reason",
    "elapsed_seconds",
    "error",
]


PROMPT_PRESETS = {
    "bank_zip_full_success": (
        "<image>\n"
        "OCR this Korean bank account image. Copy only visible text and numbers. "
        "Focus on 계좌번호, 예금주, 은행명. "
        "Return lines like: bank: ..., account_holder: ..., account_number: ... "
        "If unknown, write UNKNOWN. Do not explain."
    ),
    "bank_fields": (
        "<image>\n"
        "한국 은행 통장사본 OCR입니다. 보이는 텍스트만 읽으세요. "
        "계좌번호/예금주/은행명을 찾고 추측하지 마세요. "
        "주민등록번호, 운전면허번호, 전화번호, 날짜는 계좌번호가 아닙니다. "
        "출력: bank: ...\\naccount_holder: ...\\naccount_number: ..."
    ),
    "number_inventory": (
        "<image>\n"
        "한국 은행 통장사본에서 보이는 숫자 묶음을 빠짐없이 OCR하세요. "
        "각 숫자 옆에는 반드시 주변 라벨을 붙이세요. 예: 계좌번호: 000-000-000000. "
        "계좌번호, 입금계좌, 예금주, 은행, 통장 주변 숫자를 우선하세요. "
        "전화번호, 날짜, 시간, 주민등록번호, 운전면허번호, 카드번호는 계좌번호가 아니라고 표시하세요. "
        "추측하지 말고 보이는 숫자만 줄 단위로 출력하세요."
    ),
    "account_candidates": (
        "<image>\n"
        "한국 은행 통장사본 OCR입니다. 보이는 계좌번호 후보만 찾으세요. "
        "하이픈과 공백을 가능한 보이는 그대로 유지하세요. "
        "계좌번호 후보는 반드시 '계좌번호 후보: <number>' 형식으로 줄마다 출력하세요. "
        "은행명과 예금주가 보이면 각각 '은행명:', '예금주:'로 출력하세요. "
        "전화번호, 날짜, 시간, 주민등록번호, 운전면허번호, 카드번호는 제외하세요. 추측하지 마세요."
    ),
}


@dataclass(frozen=True)
class EvaluationItem:
    item_id: str
    split: str
    name: str
    image_path: Path
    label_account_number: str
    label_bank: str = ""
    label_account_holder: str = ""


def account_key(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def mask_account(value: str) -> str:
    digits_total = sum(1 for char in value or "" if char.isdigit())
    visible_after = max(digits_total - 4, 0)
    seen = 0
    masked = []
    for char in value or "":
        if char.isdigit():
            seen += 1
            masked.append("*" if seen <= visible_after else char)
        else:
            masked.append(char)
    return "".join(masked)


def read_manifest(path: Path, splits: set[str]) -> list[EvaluationItem]:
    items: list[EvaluationItem] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            split = row.get("split", "")
            if splits and split not in splits:
                continue
            label = row.get("label") or {}
            account = label.get("account_number", "")
            bank = label.get("bank", "")
            account_holder = label.get("account_holder", "") or row.get("account_holder", "")
            image_path = Path(row.get("image_path", ""))
            if not account or not image_path.exists():
                continue
            items.append(
                EvaluationItem(
                    item_id=row.get("id", ""),
                    split=split,
                    name=row.get("name", ""),
                    image_path=image_path,
                    label_account_number=account,
                    label_bank=bank,
                    label_account_holder=account_holder,
                )
            )
    return items


def run(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout)


def render_for_api(path: Path, output_dir: Path) -> Path:
    if path.suffix.lower() != ".pdf":
        return path
    if not shutil.which("pdftoppm"):
        raise RuntimeError("pdftoppm is required for PDF evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / safe_filename_part(path.stem)
    proc = run(["pdftoppm", "-png", "-f", "1", "-singlefile", "-r", "220", str(path), str(prefix)], timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "pdftoppm failed")
    rendered = prefix.with_suffix(".png")
    if not rendered.exists():
        raise RuntimeError(f"pdftoppm did not create {rendered}")
    return rendered


def post_ocr(
    image_path: Path,
    api_url: str,
    timeout: int,
    prompt: str,
    *,
    content_type: str = "Scene",
    subcategory: str = "Verification",
    complexity: str = "Tiny",
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    repetition_context_size: int | None = None,
    early_stop_account: bool = False,
    prefix_salvage: bool = False,
) -> str:
    boundary = f"----codex-eval-{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    fields = {
        "content_type": content_type,
        "subcategory": subcategory,
        "complexity": complexity,
        "prompt": prompt,
    }
    optional_fields = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "repetition_context_size": repetition_context_size,
    }
    fields.update({key: str(value) for key, value in optional_fields.items() if value is not None})
    if early_stop_account:
        fields["early_stop_account"] = "1"
    if prefix_salvage:
        fields["prefix_salvage"] = "1"
    body = bytearray()
    for key, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n".encode()
    )
    body.extend(image_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/ocr",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("success"):
        raise RuntimeError(payload.get("error") or "OCR server returned success=false")
    return payload.get("text", "")


def check_health(api_url: str) -> None:
    request = urllib.request.Request(f"{api_url.rstrip('/')}/api/health")
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"health check failed: HTTP {response.status}")


def evaluate_items(
    items: list[EvaluationItem],
    output_dir: Path,
    ocr_fn: Callable[[Path], str],
    render_dir: Path | None = None,
    require_account_candidate: bool = False,
    account_only_gold: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir = render_dir or output_dir / "rendered"
    rows: list[dict[str, object]] = []
    exact_count = 0
    candidate_exact_count = 0
    false_positive_count = 0
    error_count = 0
    latencies: list[float] = []
    recognition_scores: list[float] = []
    digit_similarity_scores: list[float] = []
    extraction_f1_scores: list[float] = []
    basic_vqa_scores: list[float] = []
    composite_scores: list[float] = []
    free_running_gate_pass_count = 0
    surface_gate_pass_count = 0
    unique_token_ratios: list[float] = []
    top_token_shares: list[float] = []
    max_token_runs: list[float] = []
    repetition_trigram_ratios: list[float] = []
    account_candidate_presence_count = 0

    for index, item in enumerate(items, start=1):
        started = time.time()
        text = ""
        error = ""
        try:
            image_for_api = render_for_api(item.image_path, render_dir)
            text = ocr_fn(image_for_api)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            error_count += 1

        elapsed = time.time() - started
        latencies.append(elapsed)
        result = classify_account_candidates(text)
        label_key = account_key(item.label_account_number)
        predicted = result.value or ""
        predicted_key = account_key(predicted)
        candidate_keys = {account_key(candidate) for candidate in result.candidates}
        exact_match = bool(predicted_key and predicted_key == label_key)
        candidate_exact = bool(label_key and label_key in candidate_keys)
        false_positive = bool(predicted_key and predicted_key != label_key)
        degeneration = free_running_degeneration_metrics(
            text,
            require_account_candidate=require_account_candidate,
        )
        free_running_gate_pass = bool(degeneration["degeneration_pass"]) and not error
        surface_gate_pass = free_running_gate_pass and not false_positive
        ocrbench = bankbook_ocrbench_v2_metrics(
            BankbookGold(
                account_number=item.label_account_number,
                bank="" if account_only_gold else item.label_bank,
                account_holder="" if account_only_gold else item.label_account_holder or item.name,
            ),
            BankbookPrediction(
                raw_text=text,
                account_number=predicted,
                candidate_accounts=tuple(result.candidates),
            ),
        )
        recognition = ocrbench["recognition"]
        extraction = ocrbench["extraction"]
        basic_vqa = ocrbench["basic_vqa"]
        recognition_scores.append(float(recognition["score"]))
        digit_similarity_scores.append(float(recognition["account_digit_edit_similarity"]))
        extraction_f1_scores.append(float(extraction["f1"]))
        basic_vqa_scores.append(float(basic_vqa["score"]))
        composite_scores.append(float(ocrbench["composite_score"]))

        exact_count += int(exact_match)
        candidate_exact_count += int(candidate_exact)
        false_positive_count += int(false_positive)
        free_running_gate_pass_count += int(free_running_gate_pass)
        surface_gate_pass_count += int(surface_gate_pass)
        unique_token_ratios.append(float(degeneration["unique_token_ratio"]))
        top_token_shares.append(float(degeneration["top_token_share"]))
        max_token_runs.append(float(degeneration["max_token_run"]))
        repetition_trigram_ratios.append(float(degeneration["repetition_trigram_ratio"]))
        account_candidate_presence_count += int(bool(degeneration["account_candidate_presence"]))

        rows.append(
            {
                "id": item.item_id,
                "split": item.split,
                "name": item.name,
                "image_name": item.image_path.name,
                "label_account_masked": mask_account(item.label_account_number),
                "predicted_account_masked": mask_account(predicted),
                "candidate_accounts_masked": "; ".join(mask_account(candidate) for candidate in result.candidates),
                "exact_match": int(exact_match),
                "candidate_exact_match": int(candidate_exact),
                "false_positive": int(false_positive),
                "ocrbench_recognition_score": f"{float(recognition['score']):.4f}",
                "account_digit_edit_similarity": f"{float(recognition['account_digit_edit_similarity']):.4f}",
                "ocrbench_extraction_f1": f"{float(extraction['f1']):.4f}",
                "ocrbench_basic_vqa_score": f"{float(basic_vqa['score']):.4f}",
                "ocrbench_composite_score": f"{float(ocrbench['composite_score']):.4f}",
                "free_running_gate_pass": int(free_running_gate_pass),
                "surface_gate_pass": int(surface_gate_pass),
                "degeneration_reason": degeneration["degeneration_reason"],
                "token_count": degeneration["token_count"],
                "unique_token_ratio": f"{float(degeneration['unique_token_ratio']):.4f}",
                "top_token_share": f"{float(degeneration['top_token_share']):.4f}",
                "max_token_run": degeneration["max_token_run"],
                "repetition_trigram_ratio": f"{float(degeneration['repetition_trigram_ratio']):.4f}",
                "digit_group_count": degeneration["digit_group_count"],
                "account_candidate_count": degeneration["account_candidate_count"],
                "account_candidate_presence": int(bool(degeneration["account_candidate_presence"])),
                "hangul_presence": int(bool(degeneration["hangul_presence"])),
                "confidence": result.confidence,
                "reason": result.reason,
                "elapsed_seconds": f"{elapsed:.2f}",
                "error": error,
            }
        )
        write_csv(output_dir / "evaluation_results.csv", rows, CSV_FIELDS)
        print(
            json.dumps(
                {
                    "event": "evaluated",
                    "index": index,
                    "total": len(items),
                    "id": item.item_id,
                    "exact_match": exact_match,
                    "candidate_exact_match": candidate_exact,
                    "free_running_gate_pass": free_running_gate_pass,
                    "surface_gate_pass": surface_gate_pass,
                    "error": bool(error),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    sorted_latencies = sorted(latencies)

    def percentile(fraction: float) -> float:
        if not sorted_latencies:
            return 0.0
        idx = min(len(sorted_latencies) - 1, int(round((len(sorted_latencies) - 1) * fraction)))
        return sorted_latencies[idx]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary: dict[str, object] = {
        "total": len(items),
        "account_exact_match": exact_count,
        "account_exact_match_rate": exact_count / len(items) if items else 0,
        "candidate_exact_match": candidate_exact_count,
        "candidate_exact_match_rate": candidate_exact_count / len(items) if items else 0,
        "false_positive_count": false_positive_count,
        "error_count": error_count,
        "free_running_gate_pass": free_running_gate_pass_count,
        "free_running_gate_pass_rate": free_running_gate_pass_count / len(items) if items else 0,
        "surface_gate_pass": surface_gate_pass_count,
        "surface_gate_pass_rate": surface_gate_pass_count / len(items) if items else 0,
        "free_running_degeneration": {
            "require_account_candidate": require_account_candidate,
            "unique_token_ratio_mean": mean(unique_token_ratios),
            "top_token_share_mean": mean(top_token_shares),
            "max_token_run_mean": mean(max_token_runs),
            "repetition_trigram_ratio_mean": mean(repetition_trigram_ratios),
            "account_candidate_presence_rate": account_candidate_presence_count / len(items) if items else 0,
            "notes": [
                "The gate detects free-running collapse before semantic scoring.",
                "Surface gate additionally rejects gold-known false-positive account selection.",
            ],
        },
        "latency_p50_seconds": round(percentile(0.50), 3),
        "latency_p95_seconds": round(percentile(0.95), 3),
        "ocrbench_v2_adapted": {
            "recognition_score_mean": mean(recognition_scores),
            "account_digit_edit_similarity_mean": mean(digit_similarity_scores),
            "extraction_f1_mean": mean(extraction_f1_scores),
            "basic_vqa_score_mean": mean(basic_vqa_scores),
            "composite_score_mean": mean(composite_scores),
            "notes": [
                "Recognition follows exact account match plus normalized digit edit similarity.",
                "Extraction follows OCRBench v2 key-value F1 over bank/account_holder/account_number when labels exist.",
                "Basic VQA proxy penalizes false positive account-number selection.",
            ],
        },
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:5001")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated manifest splits to evaluate.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prompt-preset", choices=sorted(PROMPT_PRESETS), default="bank_fields")
    parser.add_argument("--prompt", default="", help="Custom OCR prompt. Overrides --prompt-preset when provided.")
    parser.add_argument("--content-type", default="Scene")
    parser.add_argument("--subcategory", default="Verification")
    parser.add_argument("--complexity", default="Tiny")
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--repetition-context-size", type=int, default=0)
    parser.add_argument("--early-stop-account", action="store_true")
    parser.add_argument("--prefix-salvage", action="store_true")
    parser.add_argument(
        "--require-account-candidate",
        action="store_true",
        help="Require at least one account-number candidate for the free-running gate.",
    )
    parser.add_argument(
        "--account-only-gold",
        action="store_true",
        help="Score extraction only against account_number. Use until bank/account_holder labels are independently verified.",
    )
    args = parser.parse_args()

    check_health(args.api_url)
    splits = {split.strip() for split in args.splits.split(",") if split.strip()}
    items = read_manifest(args.manifest, splits)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("No evaluation items found.")

    prompt = args.prompt or PROMPT_PRESETS[args.prompt_preset]
    summary = evaluate_items(
        items,
        output_dir=args.output_dir,
        ocr_fn=lambda image_path: post_ocr(
            image_path,
            args.api_url,
            args.timeout,
            prompt,
            content_type=args.content_type,
            subcategory=args.subcategory,
            complexity=args.complexity,
            max_tokens=args.max_tokens or None,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            repetition_context_size=args.repetition_context_size or None,
            early_stop_account=args.early_stop_account,
            prefix_salvage=args.prefix_salvage,
        ),
        require_account_candidate=args.require_account_candidate,
        account_only_gold=args.account_only_gold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
