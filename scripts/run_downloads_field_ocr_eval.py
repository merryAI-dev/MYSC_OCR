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
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from settlement_tool.core import classify_account_candidates, decode_zip_name, normalize_text, safe_filename_part  # noqa: E402
from scripts.run_mlx4bit_practical_eval import (  # noqa: E402
    PracticalEvalConfig,
    is_healthy,
    start_server,
    stop_server,
    wait_for_server,
)


DIGIT_GROUP_RE = re.compile(r"(?<!\d)(?:\d[\d -]{4,22}\d)(?!\d)")
RESULT_FIELDS = [
    "model",
    "item_id",
    "purpose",
    "category",
    "source_name",
    "source_type",
    "prompt_id",
    "status",
    "error",
    "elapsed_seconds",
    "text_chars",
    "line_count",
    "digit_group_count",
    "account_confidence",
    "account_candidate_count",
    "account_masked",
    "masked_text_path",
    "baseline_digit_groups",
    "overlap_digit_groups",
    "digit_group_recall",
    "digit_group_precision",
]


def digit_groups(text: str | Iterable[str]) -> set[str]:
    if not isinstance(text, str):
        return {re.sub(r"\D", "", value) for value in text if re.sub(r"\D", "", value)}
    groups: set[str] = set()
    for match in DIGIT_GROUP_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if digits:
            groups.add(digits)
    return groups


def mask_sensitive_text(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        value = match.group(0)
        digit_indexes = [index for index, char in enumerate(value) if char.isdigit()]
        keep = set(digit_indexes[-4:])
        chars = list(value)
        for index in digit_indexes:
            if index not in keep:
                chars[index] = "*"
        return "".join(chars)

    return DIGIT_GROUP_RE.sub(repl, text)


def compare_digit_groups(candidate: set[str], baseline: set[str]) -> dict[str, float | int]:
    overlap = candidate & baseline
    recall = len(overlap) / len(baseline) if baseline else (1.0 if not candidate else 0.0)
    precision = len(overlap) / len(candidate) if candidate else (1.0 if not baseline else 0.0)
    return {
        "candidate_digit_groups": len(candidate),
        "baseline_digit_groups": len(baseline),
        "overlap_digit_groups": len(overlap),
        "digit_group_recall": round(recall, 6),
        "digit_group_precision": round(precision, 6),
    }


def summarize_results(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"models": {}}

    def add(scope: dict[str, object], row: dict[str, object]) -> None:
        text = str(row.get("text", ""))
        error = str(row.get("error", "")).strip()
        confidence = str(row.get("account_confidence", ""))
        category = str(row.get("category", ""))
        groups = digit_groups(text)

        scope["items"] = int(scope.get("items", 0)) + 1
        scope["errors"] = int(scope.get("errors", 0)) + (1 if error else 0)
        scope["non_empty"] = int(scope.get("non_empty", 0)) + (1 if text.strip() else 0)
        scope["high_accounts"] = int(scope.get("high_accounts", 0)) + (1 if confidence == "high" else 0)
        scope["low_accounts"] = int(scope.get("low_accounts", 0)) + (1 if confidence == "low" else 0)
        scope["digit_groups"] = int(scope.get("digit_groups", 0)) + len(groups)
        scope["text_chars"] = int(scope.get("text_chars", 0)) + len(text)
        categories = scope.setdefault("categories", defaultdict(int))
        assert isinstance(categories, defaultdict)
        categories[category] += 1

    for row in rows:
        model = str(row.get("model") or "unknown")
        purpose = str(row.get("purpose") or "unknown")
        model_summary = summary["models"].setdefault(model, {})  # type: ignore[index]
        for scope_name in ("all", purpose):
            scope = model_summary.setdefault(scope_name, {})  # type: ignore[union-attr]
            add(scope, row)

    for model_summary in summary["models"].values():  # type: ignore[union-attr]
        for scope in model_summary.values():
            items = int(scope.get("items", 0))
            non_empty = int(scope.get("non_empty", 0))
            errors = int(scope.get("errors", 0))
            scope["non_empty_rate"] = round(non_empty / items, 6) if items else 0.0
            scope["error_rate"] = round(errors / items, 6) if items else 0.0
            categories = scope.get("categories", {})
            scope["categories"] = dict(sorted(categories.items()))
    return summary


def load_manifest(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def limit_manifest_per_purpose(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0:
        return rows
    counts: dict[str, int] = defaultdict(int)
    limited: list[dict[str, str]] = []
    for row in rows:
        purpose = row.get("purpose", "")
        if counts[purpose] >= limit:
            continue
        limited.append(row)
        counts[purpose] += 1
    return limited


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be LABEL=/path/to/model")
    label, path = value.split("=", 1)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    if not label:
        raise argparse.ArgumentTypeError("model label is empty")
    return label, Path(path).expanduser()


def run(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout)


def render_for_api(path: Path, output_dir: Path, max_pages: int) -> list[Path]:
    if path.suffix.lower() != ".pdf":
        return [path]
    if not shutil.which("pdftoppm"):
        raise RuntimeError("pdftoppm is required for PDF OCR field eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / path.stem
    proc = run(["pdftoppm", "-png", "-r", "220", "-f", "1", "-l", str(max_pages), str(path), str(prefix)], timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "pdftoppm failed")
    return sorted(output_dir.glob(f"{path.stem}-*.png"))[:max_pages]


def extract_manifest_item(row: dict[str, str], work_dir: Path) -> Path:
    source_path = Path(row["source_path"]).expanduser()
    if row.get("source_type") != "zip_member":
        return source_path

    member_name = normalize_text(row["member_name"])
    destination = work_dir / "zip_members" / f"{row['item_id']}_{safe_filename_part(Path(member_name).name)}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_path) as archive:
        for info in archive.infolist():
            if decode_zip_name(info.filename) == member_name:
                with archive.open(info) as source, destination.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                return destination
    raise FileNotFoundError(member_name)


def api_config_for_category(category: str) -> dict[str, str]:
    if category == "bankbook_zip_member":
        return {"content_type": "Scene", "subcategory": "Verification", "complexity": "Tiny", "mode": "basic"}
    if category in {"receipt_tax", "business_registry", "business_document", "resume_form", "investment_report", "report_proposal"}:
        return {"content_type": "Document", "subcategory": "Business", "complexity": "Medium"}
    if category == "diagram_screenshot":
        return {"content_type": "Document", "subcategory": "Content", "complexity": "Medium"}
    if category == "paper_pdf":
        return {"content_type": "Document", "subcategory": "Academic", "complexity": "Medium"}
    return {"content_type": "Document", "subcategory": "Academic", "complexity": "Medium"}


def ocr_image(image_path: Path, row: dict[str, str], api_url: str, timeout: int) -> str:
    import requests

    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    data = api_config_for_category(row["category"])
    data["prompt"] = row["prompt"]
    with image_path.open("rb") as handle:
        response = requests.post(
            f"{api_url.rstrip('/')}/api/ocr",
            files={"file": (image_path.name, handle, mime)},
            data=data,
            timeout=timeout,
        )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(payload.get("error") or "OCR failed")
    return str(payload.get("text", ""))


def write_masked_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mask_sensitive_text(text), encoding="utf-8")


def clean_generated_outputs(output_root: Path) -> None:
    for name in ("masked_text", "work", "rendered", "server_logs"):
        path = output_root / name
        if path.exists():
            shutil.rmtree(path)
    for name in ("field_ocr_results_masked.csv", "field_ocr_summary.json"):
        path = output_root / name
        if path.exists():
            path.unlink()


def result_row_for_output(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in RESULT_FIELDS}


def run_model(
    *,
    model_label: str,
    api_url: str,
    timeout: int,
    manifest_rows: list[dict[str, str]],
    output_root: Path,
    max_pages: int,
) -> list[dict[str, object]]:
    work_dir = output_root / "work" / model_label
    render_dir = output_root / "rendered" / model_label
    text_dir = output_root / "masked_text" / model_label
    rows: list[dict[str, object]] = []

    for index, manifest_row in enumerate(manifest_rows, start=1):
        print(f"{model_label} OCR {index}/{len(manifest_rows)} {manifest_row['source_name']}", flush=True)
        started = time.time()
        text = ""
        error = ""
        status = "ok"
        try:
            local_path = extract_manifest_item(manifest_row, work_dir)
            page_paths = render_for_api(local_path, render_dir / manifest_row["item_id"], max_pages=max_pages)
            text = "\n".join(ocr_image(page_path, manifest_row, api_url, timeout) for page_path in page_paths)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            status = "error"

        account = classify_account_candidates(text) if not error else None
        groups = digit_groups(text)
        masked_text_path = text_dir / f"{manifest_row['item_id']}.txt"
        write_masked_text(masked_text_path, text)
        elapsed = time.time() - started

        rows.append(
            {
                "model": model_label,
                "item_id": manifest_row["item_id"],
                "purpose": manifest_row["purpose"],
                "category": manifest_row["category"],
                "source_name": manifest_row["source_name"],
                "source_type": manifest_row["source_type"],
                "prompt_id": manifest_row["prompt_id"],
                "status": status,
                "error": error,
                "elapsed_seconds": round(elapsed, 2),
                "text": text,
                "text_chars": len(text),
                "line_count": len([line for line in text.splitlines() if line.strip()]),
                "digit_group_count": len(groups),
                "account_confidence": account.confidence if account else "error",
                "account_candidate_count": len(account.candidates) if account else 0,
                "account_masked": mask_sensitive_text(account.value or "") if account and account.value else "",
                "masked_text_path": str(masked_text_path),
            }
        )
    return rows


def write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(result_row_for_output(row))


def add_baseline_comparison(rows: list[dict[str, object]], baseline_label: str) -> None:
    baseline_by_item = {
        str(row["item_id"]): digit_groups(str(row.get("text", "")))
        for row in rows
        if row.get("model") == baseline_label
    }
    for row in rows:
        baseline = baseline_by_item.get(str(row.get("item_id")), set())
        comparison = compare_digit_groups(digit_groups(str(row.get("text", ""))), baseline)
        row["baseline_digit_groups"] = comparison["baseline_digit_groups"]
        row["overlap_digit_groups"] = comparison["overlap_digit_groups"]
        row["digit_group_recall"] = comparison["digit_group_recall"]
        row["digit_group_precision"] = comparison["digit_group_precision"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unlabeled Downloads OCR field eval for depth and generality.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", action="append", type=parse_model, required=True, help="Repeatable LABEL=/path/to/mlx_model")
    parser.add_argument("--baseline-label", default="")
    parser.add_argument("--api-url", default=PracticalEvalConfig.api_url)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--wait-seconds", type=int, default=180)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit-per-purpose", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_rows = load_manifest(args.manifest)
    manifest_rows = limit_manifest_per_purpose(manifest_rows, args.limit_per_purpose)
    if args.limit:
        manifest_rows = manifest_rows[: args.limit]

    plan = {
        "manifest": str(args.manifest),
        "items": len(manifest_rows),
        "models": {label: str(path) for label, path in args.model},
        "baseline_label": args.baseline_label,
        "output_root": str(args.output_root),
        "max_pages": args.max_pages,
        "limit_per_purpose": args.limit_per_purpose,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    clean_generated_outputs(args.output_root)

    all_rows: list[dict[str, object]] = []
    for label, model_path in args.model:
        if not model_path.exists():
            raise SystemExit(f"Missing model path for {label}: {model_path}")
        if is_healthy(args.api_url):
            raise SystemExit(f"{args.api_url} already has a healthy server. Stop it first to avoid mixing models.")
        config = PracticalEvalConfig(
            model_path=model_path,
            api_url=args.api_url,
            timeout=args.timeout,
            wait_seconds=args.wait_seconds,
            output_root=args.output_root / "server_logs" / label,
        )
        server, server_log = start_server(config)
        try:
            wait_for_server(config, server, server_log)
            all_rows.extend(
                run_model(
                    model_label=label,
                    api_url=args.api_url,
                    timeout=args.timeout,
                    manifest_rows=manifest_rows,
                    output_root=args.output_root,
                    max_pages=args.max_pages,
                )
            )
        finally:
            stop_server(server)

    if args.baseline_label:
        add_baseline_comparison(all_rows, args.baseline_label)

    summary = summarize_results(all_rows)
    summary["plan"] = plan
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_results_csv(args.output_root / "field_ocr_results_masked.csv", all_rows)
    (args.output_root / "field_ocr_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
