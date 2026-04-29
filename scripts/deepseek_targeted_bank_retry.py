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
from pathlib import Path

import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from settlement_tool.core import (  # noqa: E402
    AccountResult,
    classify_account_candidates,
    normalize_text,
    safe_filename_part,
    write_csv,
)

try:  # noqa: SIM105
    from scripts.build_deepseek_resolution import MANUAL_NAME_HINTS, SOURCE_NAME_HINTS, mask  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - keeps direct script execution working
    from build_deepseek_resolution import MANUAL_NAME_HINTS, SOURCE_NAME_HINTS, mask  # type: ignore  # noqa: E402


PROMPTS = {
    "account_structured_ko": (
        "<image>\n"
        "한국 은행 통장사본 또는 계좌확인 이미지입니다. 보이는 글자만 읽으세요. "
        "계좌번호, 예금주, 은행명만 찾고 추측하지 마세요. "
        "출력 형식: bank: ... / account_holder: ... / account_number: ... "
        "계좌번호가 안 보이면 account_number: UNKNOWN"
    ),
    "visible_numbers": (
        "<image>\n"
        "Copy every visible number group exactly as written, one per line. "
        "Prioritize bank account numbers near labels like 계좌번호, 입금계좌, 예금주, 통장, 은행. "
        "Do not explain."
    ),
    "account_only": (
        "<image>\n"
        "Find the bank account number in this Korean bankbook copy. "
        "Return only: account_number: <number>. If it is not visible, return account_number: UNKNOWN. "
        "Do not return phone numbers, dates, or resident registration numbers."
    ),
}


CSV_FIELDS = [
    "name",
    "filename_hint",
    "source_name",
    "variant",
    "prompt_id",
    "account",
    "confidence",
    "candidates",
    "account_reason",
    "elapsed_seconds",
    "ocr_text_path",
    "image_path",
    "error",
]
DEFAULT_TARGET_DECISIONS = (
    "no_candidate",
    "multiple_candidates_review",
    "openai_reranker_no_candidate",
    "multiple_openai_reranker_candidates",
    "targeted_retry_no_candidate",
)
RETRY_TARGET_FIELDS = [
    "name",
    "decision",
    "filename_hint",
    "source_name",
    "extracted_path",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def default_target_decisions() -> set[str]:
    return set(DEFAULT_TARGET_DECISIONS)


def target_decisions_from_args(values: list[str] | None) -> set[str]:
    return set(values or DEFAULT_TARGET_DECISIONS)


def run(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout)


def render_pdf(path: Path, output_dir: Path) -> list[Path]:
    if path.suffix.lower() != ".pdf":
        return [path]
    if not shutil.which("pdftoppm"):
        raise RuntimeError("pdftoppm is required for PDF retry OCR")
    prefix = output_dir / path.stem
    proc = run(["pdftoppm", "-png", "-r", "300", str(path), str(prefix)], timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "pdftoppm failed")
    return sorted(output_dir.glob(f"{path.stem}-*.png"))[:1]


def make_variants(image_path: Path, output_dir: Path) -> list[tuple[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants: list[tuple[str, Path]] = [("original", image_path)]
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            width, height = img.size
            scale = 2 if max(width, height) < 1800 else 1
            if scale > 1:
                img = img.resize((width * scale, height * scale), Image.Resampling.LANCZOS)

            gray = ImageOps.grayscale(img)
            gray = ImageOps.autocontrast(gray)
            gray = ImageEnhance.Contrast(gray).enhance(1.8)
            gray = gray.filter(ImageFilter.SHARPEN)
            contrast_path = output_dir / f"{safe_filename_part(image_path.stem)}__contrast.png"
            gray.save(contrast_path)
            variants.append(("contrast", contrast_path))

            width, height = gray.size
            if height > 900:
                top = gray.crop((0, 0, width, int(height * 0.62)))
                top_path = output_dir / f"{safe_filename_part(image_path.stem)}__top.png"
                top.save(top_path)
                variants.append(("top_crop", top_path))
    except Exception as exc:
        print(f"variant_error {image_path}: {type(exc).__name__}: {exc}", flush=True)
    return variants


def deepseek_ocr(image_path: Path, prompt: str, api_url: str, timeout: int) -> str:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    with image_path.open("rb") as handle:
        response = requests.post(
            f"{api_url.rstrip('/')}/api/ocr",
            files={"file": (image_path.name, handle, mime)},
            data={
                "content_type": "Scene",
                "subcategory": "Verification",
                "complexity": "Tiny",
                "mode": "basic",
                "prompt": prompt,
            },
            timeout=timeout,
        )
    response.raise_for_status()
    data = response.json()
    if not data.get("success"):
        raise RuntimeError(data.get("error") or "DeepSeek OCR failed")
    return data.get("text", "")


def target_name(row: dict[str, str], target_names: set[str]) -> str:
    source_manual = SOURCE_NAME_HINTS.get(Path(row.get("source_name", "")).stem, "")
    if source_manual in target_names:
        return source_manual
    matched = row.get("matched_name", "")
    if matched in target_names:
        return matched
    hint = row.get("filename_hint", "")
    manual = MANUAL_NAME_HINTS.get(hint, "")
    if manual in target_names:
        return manual
    for name in target_names:
        if name and name in row.get("source_name", ""):
            return name
    return ""


def target_decision_by_name(resolution_rows: list[dict[str, str]], target_decisions: set[str]) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for row in resolution_rows:
        name = row.get("name", "")
        decision = row.get("decision", "")
        if name and decision in target_decisions and name not in decisions:
            decisions[name] = decision
    return decisions


def build_retry_targets(
    *,
    resolution_rows: list[dict[str, str]],
    deepseek_rows: list[dict[str, str]],
    target_decisions: set[str],
    limit: int = 0,
) -> list[tuple[str, dict[str, str]]]:
    target_names = set(target_decision_by_name(resolution_rows, target_decisions))
    targets: list[tuple[str, dict[str, str]]] = []
    seen_sources = set()
    for row in deepseek_rows:
        name = target_name(row, target_names)
        if not name:
            continue
        source = row.get("source_name", "")
        if source in seen_sources:
            continue
        seen_sources.add(source)
        targets.append((name, row))
        if limit and len(targets) >= limit:
            break
    return targets


def write_retry_target_manifest(
    *,
    output_dir: Path,
    targets: list[tuple[str, dict[str, str]]],
    decisions_by_name: dict[str, str],
) -> Path:
    rows = []
    for name, row in targets:
        rows.append(
            {
                "name": name,
                "decision": decisions_by_name.get(name, ""),
                "filename_hint": row.get("filename_hint", ""),
                "source_name": row.get("source_name", ""),
                "extracted_path": row.get("extracted_path", ""),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "retry_targets.csv"
    write_csv(path, rows, RETRY_TARGET_FIELDS)
    return path


def retry_row_key(name: str, source_name: str, variant: str, prompt_id: str) -> tuple[str, str, str, str]:
    return (name, source_name, variant, prompt_id)


def completed_retry_keys(rows: list[dict[str, str]]) -> set[tuple[str, str, str, str]]:
    completed = set()
    for row in rows:
        if row.get("confidence") == "error":
            continue
        completed.add(
            retry_row_key(
                row.get("name", ""),
                row.get("source_name", ""),
                row.get("variant", ""),
                row.get("prompt_id", ""),
            )
        )
    return completed


def is_backend_down_error(error: str) -> bool:
    lowered = error.lower()
    return "connectionerror" in lowered or "connection refused" in lowered or "readtimeout" in lowered or "timed out" in lowered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution-csv", type=Path, required=True)
    parser.add_argument("--deepseek-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:5001")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--target-decision",
        action="append",
        help="Resolution decision to retry. Can be repeated. Defaults include no_candidate and OpenAI reranker no-candidate decisions.",
    )
    parser.add_argument("--plan-only", action="store_true", help="Write retry_targets.csv and summary.json without calling OCR.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing non-error rows in targeted_retry_ocr.csv and retry only missing/error rows.")
    parser.add_argument(
        "--stop-on-backend-down",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop after the first connection/timeout error instead of filling the rest of the batch with repeated errors.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = args.output_dir / "rendered"
    variants_dir = args.output_dir / "variants"
    text_dir = args.output_dir / "deepseek_text"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    variants_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    resolution_rows = read_csv(args.resolution_csv)
    deepseek_rows = read_csv(args.deepseek_csv)
    target_decisions = target_decisions_from_args(args.target_decision)
    decisions_by_name = target_decision_by_name(resolution_rows, target_decisions)
    targets = build_retry_targets(
        resolution_rows=resolution_rows,
        deepseek_rows=deepseek_rows,
        target_decisions=target_decisions,
        limit=args.limit,
    )
    retry_target_manifest = write_retry_target_manifest(
        output_dir=args.output_dir,
        targets=targets,
        decisions_by_name=decisions_by_name,
    )

    if args.plan_only:
        summary = {
            "targets": len(targets),
            "runs": 0,
            "high_accounts": 0,
            "names_with_high": [],
            "target_decisions": sorted(target_decisions),
            "retry_targets_csv": str(retry_target_manifest),
            "output_dir": str(args.output_dir),
            "plan_only": True,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    health = requests.get(f"{args.api_url.rstrip('/')}/api/health", timeout=10)
    health.raise_for_status()

    output_csv = args.output_dir / "targeted_retry_ocr.csv"
    rows = read_csv(output_csv) if args.resume and output_csv.exists() else []
    completed_keys = completed_retry_keys(rows)
    skipped_completed_runs = 0
    backend_down = False
    for target_index, (name, row) in enumerate(targets, start=1):
        if backend_down:
            break
        extracted = Path(row["extracted_path"])
        print(f"Retry target {target_index}/{len(targets)} {name} {extracted.name}", flush=True)
        try:
            pages = render_pdf(extracted, rendered_dir)
        except Exception as exc:
            pages = []
            rows.append(
                {
                    "name": name,
                    "filename_hint": row["filename_hint"],
                    "source_name": row["source_name"],
                    "variant": "",
                    "prompt_id": "",
                    "account": "",
                    "confidence": "error",
                    "candidates": "",
                    "account_reason": "",
                    "elapsed_seconds": "0.00",
                    "ocr_text_path": "",
                    "image_path": str(extracted),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            write_csv(args.output_dir / "targeted_retry_ocr.csv", rows, CSV_FIELDS)
            continue

        for page_path in pages:
            if backend_down:
                break
            for variant_name, variant_path in make_variants(page_path, variants_dir):
                if backend_down:
                    break
                for prompt_id, prompt in PROMPTS.items():
                    key = retry_row_key(name, row.get("source_name", ""), variant_name, prompt_id)
                    if key in completed_keys:
                        skipped_completed_runs += 1
                        continue
                    started = time.time()
                    text = ""
                    error = ""
                    result = AccountResult(None, "none", [], "not_run", "mlx-deepseek")
                    try:
                        text = deepseek_ocr(variant_path, prompt, args.api_url, args.timeout)
                        result = classify_account_candidates(text)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        result = AccountResult(None, "error", [], error, "mlx-deepseek")
                        if args.stop_on_backend_down and is_backend_down_error(error):
                            backend_down = True
                    elapsed = time.time() - started
                    text_stem = safe_filename_part(
                        f"{target_index:02d}_{name}_{Path(row['source_name']).stem}_{variant_name}_{prompt_id}"
                    )
                    text_path = text_dir / f"{text_stem}.txt"
                    text_path.write_text(text, encoding="utf-8")
                    rows.append(
                        {
                            "name": name,
                            "filename_hint": row["filename_hint"],
                            "source_name": row["source_name"],
                            "variant": variant_name,
                            "prompt_id": prompt_id,
                            "account": result.value or "",
                            "confidence": result.confidence,
                            "candidates": "; ".join(result.candidates),
                            "account_reason": result.reason,
                            "elapsed_seconds": f"{elapsed:.2f}",
                            "ocr_text_path": str(text_path),
                            "image_path": str(variant_path),
                            "error": error,
                        }
                    )
                    write_csv(args.output_dir / "targeted_retry_ocr.csv", rows, CSV_FIELDS)
                    if backend_down:
                        break

    summary = {
        "targets": len(targets),
        "runs": len(rows),
        "new_runs": len(rows) - len(completed_keys),
        "skipped_completed_runs": skipped_completed_runs,
        "backend_down": backend_down,
        "high_accounts": sum(1 for row in rows if row["confidence"] == "high"),
        "names_with_high": sorted({row["name"] for row in rows if row["confidence"] == "high"}),
        "target_decisions": sorted(target_decisions),
        "retry_targets_csv": str(retry_target_manifest),
        "output_dir": str(args.output_dir),
        "plan_only": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    masked_rows = []
    for row in rows:
        masked = dict(row)
        masked["account"] = mask(masked["account"])
        masked["candidates"] = mask(masked["candidates"])
        masked_rows.append(masked)
    write_csv(args.output_dir / "targeted_retry_ocr_masked.csv", masked_rows, CSV_FIELDS)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
