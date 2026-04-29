#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from settlement_tool.core import decode_zip_name  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DOC_SUFFIXES = {".pdf"}
FIELDNAMES = [
    "item_id",
    "purpose",
    "category",
    "source_type",
    "source_path",
    "member_name",
    "source_name",
    "prompt_id",
    "prompt",
]


PROMPTS = {
    "bankbook_account": (
        "<image>\n"
        "OCR this Korean bank account image. Copy only visible text and numbers. "
        "Focus on 계좌번호, 예금주, 은행명. "
        "Return lines like: bank: ..., account_holder: ..., account_number: ... "
        "If unknown, write UNKNOWN. Do not explain."
    ),
    "business_field": (
        "<image>\n"
        "OCR this Korean business or financial document. Preserve visible headings, key-value fields, dates, and number groups. "
        "Do not summarize beyond visible text."
    ),
    "layout_text": (
        "<image>\n"
        "OCR all visible text in reading order. Preserve labels, table cells, diagram node text, and number groups. "
        "Do not explain."
    ),
    "paper_text": (
        "<image>\n"
        "OCR the visible page text. Prioritize title, authors, section headings, equations, table labels, and number groups. "
        "Do not explain."
    ),
}


def normalize_name(path: Path) -> str:
    return path.as_posix().lower()


def classify_download_path(path: Path) -> str:
    name = normalize_name(path)
    stem = path.stem.lower()
    if any(token in name for token in ["세금계산서", "invoice", "receipt", "영수증"]):
        return "receipt_tax"
    if any(token in name for token in ["사업자등록증", "business_registration"]):
        return "business_registry"
    if any(token in name for token in ["이력서", "resume", "cv"]):
        return "resume_form"
    if any(token in name for token in ["mermaid", "flowchart", "diagram", "스크린샷", "screenshot", "desktop", "mobile", "tablet"]):
        return "diagram_screenshot"
    if re.fullmatch(r"\d{4}\.\d+(v\d+)?", stem):
        return "paper_pdf"
    if any(token in name for token in ["투심", "투자심사"]):
        return "investment_report"
    if any(token in name for token in ["재무제표", "등기부", "정관", "명부", "인증서"]):
        return "business_document"
    if any(token in name for token in ["제안", "보고서", "proposal", "report"]):
        return "report_proposal"
    if path.suffix.lower() == ".pdf":
        return "document_pdf"
    return "image_text"


def prompt_id_for_category(category: str) -> str:
    if category == "bankbook_zip_member":
        return "bankbook_account"
    if category in {"receipt_tax", "business_registry", "business_document", "resume_form", "investment_report", "report_proposal"}:
        return "business_field"
    if category == "paper_pdf":
        return "paper_text"
    return "layout_text"


def iter_download_files(downloads_root: Path) -> Iterable[Path]:
    suffixes = IMAGE_SUFFIXES | DOC_SUFFIXES
    for path in sorted(downloads_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        relative = path.relative_to(downloads_root)
        parts = set(relative.parts)
        if "hackathon_settlement_output" in parts:
            continue
        yield path


def round_robin_by_category(paths: list[Path], limit: int) -> list[Path]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(classify_download_path(path), []).append(path)

    selected: list[Path] = []
    categories = sorted(grouped)
    while len(selected) < limit and any(grouped.values()):
        for category in categories:
            if grouped[category]:
                selected.append(grouped[category].pop(0))
                if len(selected) >= limit:
                    break
    return selected


def bankbook_zip_rows(downloads_root: Path, bankbook_limit: int, bankbook_zip: Path | None = None) -> list[dict[str, str]]:
    zip_path = bankbook_zip or downloads_root / "5. 통장 사본 업로드 (File responses)-20260424T051727Z-3-001.zip"
    if not zip_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            decode_zip_name(info.filename)
            for info in archive.infolist()
            if not info.is_dir() and Path(decode_zip_name(info.filename)).name != ".DS_Store"
        ]
    for index, member in enumerate(members[:bankbook_limit], start=1):
        prompt_id = prompt_id_for_category("bankbook_zip_member")
        rows.append(
            {
                "item_id": f"depth_bankbook_{index:03d}",
                "purpose": "1_depth_bankbook",
                "category": "bankbook_zip_member",
                "source_type": "zip_member",
                "source_path": str(zip_path),
                "member_name": member,
                "source_name": Path(member).name,
                "prompt_id": prompt_id,
                "prompt": PROMPTS[prompt_id],
            }
        )
    return rows


def generality_rows(downloads_root: Path, generality_limit: int) -> list[dict[str, str]]:
    candidates = [
        path
        for path in iter_download_files(downloads_root)
        if classify_download_path(path) != "image_text" or path.suffix.lower() in IMAGE_SUFFIXES
    ]
    selected = round_robin_by_category(candidates, generality_limit)
    rows: list[dict[str, str]] = []
    for index, path in enumerate(selected, start=1):
        category = classify_download_path(path)
        prompt_id = prompt_id_for_category(category)
        rows.append(
            {
                "item_id": f"generality_{index:03d}",
                "purpose": "2_generality_ocr",
                "category": category,
                "source_type": "file",
                "source_path": str(path),
                "member_name": "",
                "source_name": str(path.relative_to(downloads_root)),
                "prompt_id": prompt_id,
                "prompt": PROMPTS[prompt_id],
            }
        )
    return rows


def build_manifest(downloads_root: Path, *, bankbook_limit: int = 12, generality_limit: int = 36) -> list[dict[str, str]]:
    return bankbook_zip_rows(downloads_root, bankbook_limit) + generality_rows(downloads_root, generality_limit)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an unlabeled Downloads OCR field-test manifest.")
    parser.add_argument("--downloads-root", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bankbook-limit", type=int, default=12)
    parser.add_argument("--generality-limit", type=int, default=36)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_manifest(args.downloads_root, bankbook_limit=args.bankbook_limit, generality_limit=args.generality_limit)
    write_jsonl(args.output_jsonl, rows)
    write_csv(args.output_csv, rows)
    summary = {
        "items": len(rows),
        "purposes": sorted({row["purpose"] for row in rows}),
        "categories": sorted({row["category"] for row in rows}),
        "output_jsonl": str(args.output_jsonl),
        "output_csv": str(args.output_csv),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
