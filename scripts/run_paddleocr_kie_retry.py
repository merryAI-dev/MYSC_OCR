#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from settlement_tool.core import _normalize_account_candidate, compact_text, write_csv  # noqa: E402
from settlement_tool.kie_evidence import infer_kie_field_type, normalize_kie_row, redacted_kie_evidence  # noqa: E402
from settlement_tool.teacher_distill import ACCOUNT_CANDIDATE_RE, account_key  # noqa: E402


CSV_FIELDS = [
    "name",
    "matched_name",
    "source_id",
    "source_name",
    "backend",
    "kie_backend",
    "kie_field_type",
    "kie_label_masked",
    "kie_holder_match_status",
    "kie_holder_field_present",
    "kie_bank_name_present",
    "candidate_raw",
    "candidate_masked",
    "confidence",
    "kie_confidence",
    "kie_confidence_bucket",
    "bbox_json",
    "page_width",
    "page_height",
    "layout_json",
    "error",
]
PROGRESS_FILE = "kie_processed_sources.jsonl"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def completed_source_names(output_dir: Path) -> set[str]:
    path = output_dir / PROGRESS_FILE
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_name = str(row.get("source_name", ""))
            if source_name and row.get("status") in {"completed", "error"}:
                completed.add(source_name)
    return completed


def _read_existing_local_rows(output_dir: Path) -> list[dict[str, object]]:
    path = output_dir / "kie_candidates_local.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _fixture_rows(path: Path) -> list[dict[str, object]]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for source_index, source in enumerate(fixture, start=1):
        source_name = str(source.get("source_name", f"fixture-{source_index}"))
        page_width = float(source.get("page_width") or 0)
        page_height = float(source.get("page_height") or 0)
        for item_index, item in enumerate(source.get("items", []), start=1):
            rows.append(
                normalize_kie_row(
                    source_id=f"paddleocr_kie:{source_index}:{item_index}",
                    source_name=source_name,
                    backend="paddleocr_kie",
                    text=str(item.get("text_masked", "")),
                    raw_text_local=str(item.get("raw_text_local", "")),
                    label_text=str(item.get("label_text", "")),
                    bbox=list(item.get("bbox", [])),
                    page_width=page_width,
                    page_height=page_height,
                    confidence=float(item.get("confidence") or 0.0),
                )
            )
    return rows


def _bbox4(value: object) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return [0.0, 0.0, 0.0, 0.0]
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        x1, y1, x2, y2 = [float(item) for item in value]
        return [x1, y1, x2, y2]
    points = [point for point in value if isinstance(point, (list, tuple)) and len(point) >= 2]
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _nearest_label_text(candidate_bbox: list[float], items: list[dict[str, Any]]) -> str:
    best: tuple[float, str] | None = None
    cx, cy = _center(candidate_bbox)
    for item in items:
        text = str(item.get("text", ""))
        if infer_kie_field_type(text) == "unknown":
            continue
        bbox = _bbox4(item.get("bbox"))
        ix, iy = _center(bbox)
        horizontal_gap = max(candidate_bbox[0] - bbox[2], 0)
        vertical_gap = max(candidate_bbox[1] - bbox[3], 0)
        same_line = abs(iy - cy) <= max(candidate_bbox[3] - candidate_bbox[1], bbox[3] - bbox[1], 24)
        above = bbox[3] <= candidate_bbox[1] and abs(ix - cx) <= max(candidate_bbox[2] - candidate_bbox[0], 160)
        if same_line and bbox[2] <= candidate_bbox[0]:
            distance = horizontal_gap
        elif above:
            distance = 40.0 + vertical_gap
        else:
            continue
        if best is None or distance < best[0]:
            best = (distance, text)
    return best[1] if best else ""


def rows_from_ocr_items(
    items: list[dict[str, Any]],
    *,
    source_id_prefix: str,
    source_name: str,
    page_width: float,
    page_height: float,
    target_name: str = "",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    joined_text = "\n".join(str(item.get("text", "")) for item in items)
    holder_match_status = "match" if target_name and compact_text(target_name) in compact_text(joined_text) else "not_present"
    holder_field_present = holder_match_status == "match" or any(
        infer_kie_field_type(str(item.get("text", ""))) == "holder" for item in items
    )
    bank_name_present = any(infer_kie_field_type(str(item.get("text", ""))) == "bank" for item in items)
    for item in items:
        text = str(item.get("text", ""))
        item_field_type = infer_kie_field_type(text)
        bbox = _bbox4(item.get("bbox"))
        label_text = text if item_field_type != "unknown" else _nearest_label_text(bbox, items)
        for raw in ACCOUNT_CANDIDATE_RE.findall(text):
            candidate = _normalize_account_candidate(raw)
            digits = account_key(candidate)
            if len(digits) < 9 or len(digits) > 16 or digits in seen:
                continue
            seen.add(digits)
            row = normalize_kie_row(
                source_id=f"{source_id_prefix}:{len(rows) + 1}",
                source_name=source_name,
                backend="paddleocr_kie",
                text="",
                raw_text_local=candidate,
                label_text=label_text,
                bbox=bbox,
                page_width=page_width,
                page_height=page_height,
                confidence=float(item.get("confidence") or 0.0),
            )
            row["kie_holder_match_status"] = holder_match_status
            row["kie_holder_field_present"] = holder_field_present
            row["kie_bank_name_present"] = bank_name_present
            rows.append(row)
    return rows


def _flatten_ocr_result(result: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(result, dict):
        texts = result.get("rec_texts")
        if texts is None:
            texts = result.get("texts")
        if texts is None:
            texts = []
        scores = result.get("rec_scores")
        if scores is None:
            scores = result.get("scores")
        if scores is None:
            scores = []
        boxes = result.get("rec_boxes")
        if boxes is None:
            boxes = result.get("rec_polys")
        if boxes is None:
            boxes = result.get("boxes")
        if boxes is None:
            boxes = []
        for index, text in enumerate(texts):
            items.append(
                {
                    "text": str(text),
                    "bbox": _bbox4(boxes[index] if index < len(boxes) else []),
                    "confidence": float(scores[index] if index < len(scores) else 0.0),
                }
            )
        return items
    if isinstance(result, list):
        if len(result) == 2 and isinstance(result[1], tuple):
            text = result[1][0] if result[1] else ""
            confidence = result[1][1] if len(result[1]) > 1 else 0.0
            return [{"text": str(text), "bbox": _bbox4(result[0]), "confidence": float(confidence or 0.0)}]
        for item in result:
            items.extend(_flatten_ocr_result(item))
    return items


def _image_size(path: Path) -> tuple[float, float]:
    from PIL import Image

    with Image.open(path) as image:
        return float(image.width), float(image.height)


def _prepare_image_for_ocr(path: Path, output_dir: Path, *, max_side: int = 1800) -> Path:
    from PIL import Image, ImageOps

    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        scale = min(1.0, max_side / max(width, height)) if max_side else 1.0
        if scale < 1.0:
            image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
        output_path = output_dir / f"{path.stem}__paddleocr.png"
        image.save(output_path)
    return output_path


def _render_pdf(path: Path, output_dir: Path) -> list[Path]:
    if path.suffix.lower() != ".pdf":
        return [path]
    if not shutil.which("pdftoppm"):
        raise RuntimeError("pdftoppm is required for PDF PaddleOCR retry")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / path.stem
    proc = subprocess.run(
        ["pdftoppm", "-png", "-r", "300", str(path), str(prefix)],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "pdftoppm failed")
    return sorted(output_dir.glob(f"{path.stem}-*.png"))[:1]


def _ensure_paddleocr_available() -> None:
    try:
        import paddleocr  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional local install
        raise RuntimeError(f"paddleocr is not available: {type(exc).__name__}: {exc}") from exc


def _paddleocr_rows(targets_csv: Path, *, output_dir: Path, limit: int = 0, resume: bool = False) -> list[dict[str, object]]:
    _ensure_paddleocr_available()
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(lang="korean")
    rows: list[dict[str, object]] = _read_existing_local_rows(output_dir) if resume else []
    processed = completed_source_names(output_dir) if resume else set()
    target_rows = read_csv(targets_csv)
    if limit:
        target_rows = target_rows[:limit]
    rendered_dir = output_dir / "rendered"
    prepared_dir = output_dir / "prepared"
    for target_index, target in enumerate(target_rows, start=1):
        source_path = Path(target.get("extracted_path", ""))
        source_name = target.get("source_name", "") or source_path.name
        target_name = target.get("name", "")
        if source_name in processed:
            continue
        target_candidate_count = 0
        target_error = ""
        try:
            image_paths = _render_pdf(source_path, rendered_dir)
            for page_index, image_path in enumerate(image_paths, start=1):
                prepared_path = _prepare_image_for_ocr(image_path, prepared_dir)
                width, height = _image_size(prepared_path)
                if hasattr(ocr, "predict"):
                    result = ocr.predict(str(prepared_path))
                else:
                    result = ocr.ocr(str(prepared_path))
                items = _flatten_ocr_result(result)
                new_rows = rows_from_ocr_items(
                    items,
                    source_id_prefix=f"paddleocr_kie:{target_index}:{page_index}",
                    source_name=source_name,
                    page_width=width,
                    page_height=height,
                    target_name=target_name,
                )
                target_candidate_count += len(new_rows)
                for row in new_rows:
                    row["name"] = target_name
                    row["matched_name"] = target_name
                rows.extend(new_rows)
        except Exception as exc:
            target_error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "source_id": f"paddleocr_kie:{target_index}:error",
                    "name": target_name,
                    "matched_name": target_name,
                    "source_name": source_name,
                    "backend": "paddleocr_kie",
                    "kie_backend": "paddleocr_kie",
                    "error": target_error,
                }
            )
        status = "error" if target_error else "completed"
        append_jsonl(
            output_dir / PROGRESS_FILE,
            {
                "source_name": source_name,
                "status": status,
                "candidate_count": target_candidate_count,
                "error_category": target_error.split(":", 1)[0] if target_error else "",
            },
        )
        processed.add(source_name)
        _write_outputs(
            output_dir,
            rows,
            {
                "plan_only": False,
                "retry_targets_csv": str(targets_csv),
                "target_count": len(target_rows),
                "processed_count": len(processed),
                "backend": "paddleocr_kie",
                "resume": resume,
            },
        )
    return rows


def _write_outputs(output_dir: Path, rows: list[dict[str, object]], summary: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / "kie_candidates_local.csv"
    redacted_path = output_dir / "kie_evidence_redacted.jsonl"
    write_csv(local_path, [{key: row.get(key, "") for key in CSV_FIELDS} for row in rows], CSV_FIELDS)
    write_jsonl(redacted_path, [redacted_kie_evidence(row) for row in rows if not row.get("error")])
    summary = {
        **summary,
        "candidate_count": len([row for row in rows if not row.get("error")]),
        "outputs": {
            "kie_candidates_local": str(local_path),
            "kie_evidence_redacted": str(redacted_path),
        },
        "notes": [
            "kie_candidates_local.csv is local-only and may contain raw account candidates.",
            "kie_evidence_redacted.jsonl must not contain raw account candidates, source paths, names, or OCR text.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def run_paddleocr_kie_retry(
    *,
    output_dir: Path,
    retry_targets_csv: Path | None = None,
    fixture_json: Path | None = None,
    plan_only: bool = False,
    limit: int = 0,
    resume: bool = False,
) -> dict[str, Any]:
    if fixture_json:
        rows = _fixture_rows(fixture_json)
        return _write_outputs(
            output_dir,
            rows,
            {
                "plan_only": False,
                "fixture_json": str(fixture_json),
                "target_count": 0,
                "backend": "paddleocr_kie_fixture",
            },
        )

    target_rows = read_csv(retry_targets_csv) if retry_targets_csv else []
    if limit:
        target_rows = target_rows[:limit]
    if plan_only:
        return _write_outputs(
            output_dir,
            [],
            {
                "plan_only": True,
                "retry_targets_csv": str(retry_targets_csv or ""),
                "target_count": len(target_rows),
                "backend": "paddleocr_kie",
            },
        )

    if not retry_targets_csv:
        raise RuntimeError("--retry-targets-csv is required unless --fixture-json or --plan-only is used")
    rows = _paddleocr_rows(retry_targets_csv, output_dir=output_dir, limit=limit, resume=resume)
    return _write_outputs(
        output_dir,
        rows,
        {
            "plan_only": False,
            "retry_targets_csv": str(retry_targets_csv),
            "target_count": len(target_rows),
            "backend": "paddleocr_kie",
            "processed_count": len(completed_source_names(output_dir)),
            "resume": resume,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaddleOCR KIE retry extraction for unresolved account candidates.")
    parser.add_argument("--retry-targets-csv", type=Path, default=None)
    parser.add_argument("--fixture-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_paddleocr_kie_retry(
            output_dir=args.output_dir,
            retry_targets_csv=args.retry_targets_csv,
            fixture_json=args.fixture_json,
            plan_only=args.plan_only,
            limit=args.limit,
            resume=args.resume,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
