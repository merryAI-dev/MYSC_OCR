from __future__ import annotations

import json
import re
from typing import Any

from .core import normalize_text


ACCOUNT_LABEL_RE = re.compile(r"계좌\s*번호|입금\s*계좌|account\s*no", re.IGNORECASE)
HOLDER_LABEL_RE = re.compile(r"예금주|받는\s*분|계좌주|고객명|성명")
CUSTOMER_NUMBER_LABEL_RE = re.compile(r"고객\s*번호|customer\s*number", re.IGNORECASE)
PHONE_LABEL_RE = re.compile(r"연락처|전화|휴대폰|mobile|phone", re.IGNORECASE)
DATE_LABEL_RE = re.compile(r"생년월일|발급일|거래일|date", re.IGNORECASE)
AMOUNT_LABEL_RE = re.compile(r"금액|원\b|amount", re.IGNORECASE)
BANK_LABEL_RE = re.compile(r"은행명|금융기관|은행|농협|신협|수협|새마을|우체국|카카오|토스")


def infer_kie_field_type(label_text: object) -> str:
    text = normalize_text(label_text)
    if ACCOUNT_LABEL_RE.search(text):
        return "account_number"
    if HOLDER_LABEL_RE.search(text):
        return "holder"
    if CUSTOMER_NUMBER_LABEL_RE.search(text):
        return "customer_number"
    if PHONE_LABEL_RE.search(text):
        return "phone"
    if DATE_LABEL_RE.search(text):
        return "date"
    if AMOUNT_LABEL_RE.search(text):
        return "amount"
    if BANK_LABEL_RE.search(text):
        return "bank"
    return "unknown"


def _bucket_ratio(value: float, *, low: float, high: float, labels: tuple[str, str, str]) -> str:
    if value < low:
        return labels[0]
    if value < high:
        return labels[1]
    return labels[2]


def bbox_bucket(bbox: list[float] | tuple[float, ...], *, page_width: float, page_height: float) -> dict[str, str]:
    if len(bbox) < 4 or page_width <= 0 or page_height <= 0:
        return {
            "x_bucket": "unknown",
            "y_bucket": "unknown",
            "width_bucket": "unknown",
            "height_bucket": "unknown",
        }
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    center_x = ((x1 + x2) / 2) / page_width
    center_y = ((y1 + y2) / 2) / page_height
    width = max(x2 - x1, 0) / page_width
    height = max(y2 - y1, 0) / page_height
    return {
        "x_bucket": _bucket_ratio(center_x, low=0.34, high=0.67, labels=("left", "center", "right")),
        "y_bucket": _bucket_ratio(center_y, low=0.34, high=0.67, labels=("top", "middle", "bottom")),
        "width_bucket": _bucket_ratio(width, low=0.20, high=0.50, labels=("narrow", "medium", "wide")),
        "height_bucket": _bucket_ratio(height, low=0.08, high=0.20, labels=("short", "medium", "tall")),
    }


def _confidence_bucket(confidence: float) -> str:
    if confidence >= 0.90:
        return "high"
    if confidence >= 0.70:
        return "medium"
    if confidence > 0:
        return "low"
    return "unknown"


def _mask_candidate(value: str) -> str:
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


def _json_list(values: list[float] | tuple[float, ...]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _json_object(values: dict[str, Any]) -> str:
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_kie_row(
    *,
    source_id: str,
    source_name: str,
    backend: str,
    text: str,
    raw_text_local: str,
    label_text: str,
    bbox: list[float] | tuple[float, ...],
    page_width: float,
    page_height: float,
    confidence: float,
) -> dict[str, object]:
    field_type = infer_kie_field_type(label_text)
    layout = bbox_bucket(bbox, page_width=page_width, page_height=page_height)
    raw_candidate = normalize_text(raw_text_local)
    masked_candidate = normalize_text(text) or _mask_candidate(raw_candidate)
    return {
        "source_id": normalize_text(source_id),
        "source_name": normalize_text(source_name),
        "backend": normalize_text(backend),
        "kie_backend": normalize_text(backend),
        "kie_field_type": field_type,
        "kie_label_masked": normalize_text(label_text),
        "candidate_raw": raw_candidate,
        "candidate_masked": masked_candidate,
        "confidence": float(confidence or 0.0),
        "kie_confidence": float(confidence or 0.0),
        "kie_confidence_bucket": _confidence_bucket(float(confidence or 0.0)),
        "bbox_json": _json_list(bbox),
        "page_width": float(page_width or 0),
        "page_height": float(page_height or 0),
        "layout_json": _json_object(layout),
        "layout_evidence": layout,
        "error": "",
    }


def redacted_kie_evidence(row: dict[str, object]) -> dict[str, object]:
    kie_backend = normalize_text(row.get("kie_backend"))
    kie_field_type = normalize_text(row.get("kie_field_type"))
    if not kie_backend and not kie_field_type:
        return {
            "backend": "",
            "field_type": "unknown",
            "confidence_bucket": "unknown",
            "layout": {
                "x_bucket": "unknown",
                "y_bucket": "unknown",
                "width_bucket": "unknown",
                "height_bucket": "unknown",
            },
        }
    layout = row.get("layout_evidence")
    if not isinstance(layout, dict):
        try:
            layout = json.loads(normalize_text(row.get("layout_json")))
        except json.JSONDecodeError:
            layout = {}
    return {
        "backend": kie_backend,
        "field_type": kie_field_type or "unknown",
        "confidence_bucket": normalize_text(row.get("kie_confidence_bucket"))
        or _confidence_bucket(float(row.get("kie_confidence") or row.get("confidence") or 0.0)),
        "layout": {
            "x_bucket": normalize_text(layout.get("x_bucket")) if isinstance(layout, dict) else "unknown",
            "y_bucket": normalize_text(layout.get("y_bucket")) if isinstance(layout, dict) else "unknown",
            "width_bucket": normalize_text(layout.get("width_bucket")) if isinstance(layout, dict) else "unknown",
            "height_bucket": normalize_text(layout.get("height_bucket")) if isinstance(layout, dict) else "unknown",
        },
    }
