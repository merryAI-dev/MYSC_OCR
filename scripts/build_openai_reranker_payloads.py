#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from settlement_tool.core import normalize_text  # noqa: E402
from settlement_tool.kie_evidence import redacted_kie_evidence  # noqa: E402


PAYLOAD_SCHEMA_VERSION = "openai_account_reranker_redacted_v3"
CONTEXT_FLAG_KEYS = [
    "has_account_keyword_context",
    "has_bank_keyword_context",
    "has_structured_bankbook_context",
    "has_direct_account_field_context",
    "has_customer_number_metadata_context",
]
RISK_FLAG_KEYS = [
    "looks_like_phone",
    "has_negative_keyword_context",
    "has_prompt_leakage_context",
    "has_wrong_field_context",
]
NUMERIC_FEATURE_KEYS = [
    "digit_count",
    "hyphen_count",
    "group_count",
    "repeat_count",
    "teacher_policy_score",
]
BANK_ALIASES = {
    "KB국민은행": ("국민", "KB", "KB국민"),
    "신한은행": ("신한",),
    "하나은행": ("하나",),
    "우리은행": ("우리",),
    "NH농협은행": ("농협", "NH"),
    "IBK기업은행": ("기업", "IBK"),
    "카카오뱅크": ("카카오",),
    "토스뱅크": ("토스",),
    "MG새마을금고": ("새마을", "MG"),
    "신협": ("신협",),
    "수협": ("수협",),
    "우체국": ("우체국",),
}
ACCOUNT_LABEL_RE = re.compile(r"계좌\s*번호|입금\s*계좌|Account\s*No", re.IGNORECASE)
HOLDER_LABEL_RE = re.compile(r"예금주|받는\s*분|계좌주|고객명|성명")
BANK_LABEL_RE = re.compile(r"은행명|금융기관|은행|농협|신협|수협|새마을|우체국|카카오|토스")
CUSTOMER_NUMBER_LABEL_RE = re.compile(r"고객\s*번호|customer\s*number", re.IGNORECASE)
PHONE_LABEL_RE = re.compile(r"연락처|전화|휴대폰|mobile|phone", re.IGNORECASE)
DATE_LABEL_RE = re.compile(r"생년월일|발급일|거래일|date", re.IGNORECASE)
AMOUNT_LABEL_RE = re.compile(r"금액|원\\b|amount", re.IGNORECASE)


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


def account_shape(value: object) -> str:
    text = normalize_text(value)
    if not re.search(r"\d", text):
        return "unknown"
    return re.sub(r"\d", "d", text)


def account_key(value: object) -> str:
    return re.sub(r"\D", "", normalize_text(value))


def compact_text(value: object) -> str:
    return re.sub(r"\s+", "", normalize_text(value))


def group_lengths(value: object) -> list[int]:
    return [len(part) for part in re.findall(r"\d+", normalize_text(value))]


def digit_count_bucket(digit_count: int) -> str:
    if digit_count <= 8:
        return "lt_9"
    if digit_count <= 10:
        return "9_10"
    if digit_count <= 13:
        return "11_13"
    if digit_count <= 16:
        return "14_16"
    return "gt_16"


def normalize_bank_name(context: object) -> str:
    text = normalize_text(context)
    for normalized, aliases in BANK_ALIASES.items():
        if any(alias in text for alias in aliases):
            return normalized
    return ""


def label_type(context: object) -> str:
    text = normalize_text(context)
    if ACCOUNT_LABEL_RE.search(text):
        return "account_number"
    if HOLDER_LABEL_RE.search(text):
        return "holder"
    if CUSTOMER_NUMBER_LABEL_RE.search(text):
        return "customer_number"
    if PHONE_LABEL_RE.search(text):
        return "phone"
    if BANK_LABEL_RE.search(text):
        return "bank"
    if DATE_LABEL_RE.search(text):
        return "date"
    if AMOUNT_LABEL_RE.search(text):
        return "amount"
    return "unknown"


def pattern_family(feature: dict[str, Any]) -> str:
    if feature.get("looks_like_phone"):
        return "phone_like"
    context = feature.get("teacher_context_masked", "")
    if DATE_LABEL_RE.search(normalize_text(context)):
        return "date_like"
    if feature.get("has_customer_number_metadata_context") or CUSTOMER_NUMBER_LABEL_RE.search(normalize_text(context)):
        return "customer_id_like"
    digit_count = int(feature.get("digit_count") or 0)
    if 9 <= digit_count <= 16 and (feature.get("hyphen_count") or feature.get("has_bank_keyword_context") or feature.get("has_account_keyword_context")):
        return "bank_account_like"
    return "unknown"


def prefix_class(feature: dict[str, Any]) -> str:
    digits = account_key(feature.get("candidate_raw"))
    if digits.startswith("010"):
        return "mobile_prefix"
    if digits.startswith(("19", "20")) and len(digits) in {8, 10, 12, 14}:
        return "date_year_prefix"
    if feature.get("has_customer_number_metadata_context"):
        return "customer_number_prefix"
    if feature.get("has_bank_keyword_context") or feature.get("has_account_keyword_context"):
        return "known_bank_prefix"
    return "unknown"


def shape_features(feature: dict[str, Any]) -> dict[str, Any]:
    candidate = feature.get("candidate_raw") or feature.get("candidate_masked")
    digit_count = int(feature.get("digit_count") or 0)
    return {
        "group_lengths": group_lengths(candidate),
        "digit_count_bucket": digit_count_bucket(digit_count),
        "pattern_family": pattern_family(feature),
        "prefix_class": prefix_class(feature),
        "has_bank_style_hyphenation": bool(feature.get("hyphen_count")) and 2 <= int(feature.get("group_count") or 0) <= 4,
        "is_single_long_run": int(feature.get("hyphen_count") or 0) == 0 and digit_count >= 9,
    }


def field_evidence(feature: dict[str, Any]) -> dict[str, Any]:
    context = feature.get("teacher_context_masked", "")
    detected = label_type(context)
    return {
        "nearest_left_label_type": detected,
        "nearest_above_label_type": "unknown",
        "same_line_label_type": detected,
        "table_row_label_type": detected if feature.get("has_structured_bankbook_context") else "unknown",
        "is_value_in_account_field": detected == "account_number" or bool(feature.get("has_direct_account_field_context")),
        "is_value_in_holder_field": detected == "holder" or bool(feature.get("has_wrong_field_context")),
        "is_value_in_customer_number_field": detected == "customer_number" or bool(feature.get("has_customer_number_metadata_context")),
        "candidate_line_position_bucket": "unknown",
        "candidate_order_in_document": "unknown",
    }


def bank_holder_evidence(feature: dict[str, Any]) -> dict[str, Any]:
    context = normalize_text(feature.get("teacher_context_masked"))
    matched_name = normalize_text(feature.get("matched_name"))
    holder_present = HOLDER_LABEL_RE.search(context) is not None
    if holder_present and matched_name and compact_text(matched_name) in compact_text(context):
        holder_status = "match"
    elif holder_present:
        holder_status = "unknown"
    else:
        holder_status = "not_present"
    bank_name = normalize_bank_name(context)
    if feature.get("kie_holder_match_status"):
        holder_status = normalize_text(feature.get("kie_holder_match_status"))
    if feature.get("kie_holder_field_present"):
        holder_present = True
    bank_present = bool(bank_name or feature.get("has_bank_keyword_context") or feature.get("kie_bank_name_present"))
    return {
        "bank_name_present": bank_present,
        "bank_name_normalized": bank_name,
        "holder_field_present": holder_present,
        "holder_match_status": holder_status,
        "bankbook_doc_type_confidence": "high"
        if feature.get("has_structured_bankbook_context") or feature.get("kie_bank_name_present")
        else "medium"
        if feature.get("has_bank_keyword_context")
        else "low",
    }


def _source_id_for_feature(feature: dict[str, Any], index: int) -> str:
    return normalize_text(feature.get("source_id")) or f"source_{index:06d}"


def _group_features_by_source(features: list[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    source_person_slots: OrderedDict[str, OrderedDict[str, int]] = OrderedDict()
    for index, feature in enumerate(features, start=1):
        source_id = _source_id_for_feature(feature, index)
        person = _person_key(feature)
        slots = source_person_slots.setdefault(source_id, OrderedDict())
        if person not in slots:
            slots[person] = len(slots) + 1

    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for index, feature in enumerate(features, start=1):
        source_id = _source_id_for_feature(feature, index)
        person = _person_key(feature)
        slots = source_person_slots[source_id]
        payload_source_id = source_id
        if len(slots) > 1:
            payload_source_id = f"{source_id}#person_{slots[person]}"
        grouped.setdefault(payload_source_id, []).append(feature)
    return grouped


def _person_key(feature: dict[str, Any]) -> str:
    return normalize_text(feature.get("matched_name")) or normalize_text(feature.get("source_name")) or normalize_text(feature.get("source_id"))


def _source_kind(feature: dict[str, Any]) -> str:
    source_id = normalize_text(feature.get("source_id")).lower()
    if "targeted_retry" in source_id:
        return "targeted_retry"
    if "deepseek_bank_zip_full" in source_id or "full_ocr" in source_id:
        return "full_ocr"
    return "other"


def source_evidence(feature: dict[str, Any]) -> dict[str, str]:
    return {
        "source_kind": _source_kind(feature),
        "variant": normalize_text(feature.get("variant")) or "unknown",
        "prompt_id": normalize_text(feature.get("prompt_id")) or "unknown",
    }


def layout_evidence(feature: dict[str, Any]) -> dict[str, str]:
    layout = feature.get("layout_evidence")
    if not isinstance(layout, dict):
        return {
            "x_bucket": "unknown",
            "y_bucket": "unknown",
            "width_bucket": "unknown",
            "height_bucket": "unknown",
        }
    return {
        "x_bucket": normalize_text(layout.get("x_bucket")) or "unknown",
        "y_bucket": normalize_text(layout.get("y_bucket")) or "unknown",
        "width_bucket": normalize_text(layout.get("width_bucket")) or "unknown",
        "height_bucket": normalize_text(layout.get("height_bucket")) or "unknown",
    }


def consensus_index(features: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    unique_by_person: dict[str, set[str]] = defaultdict(set)
    for feature in features:
        person = _person_key(feature)
        candidate_key = account_key(feature.get("candidate_raw"))
        if not candidate_key:
            continue
        grouped[(person, candidate_key)].append(feature)
        unique_by_person[person].add(candidate_key)

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        variants = {normalize_text(row.get("variant")) for row in rows if normalize_text(row.get("variant"))}
        prompts = {normalize_text(row.get("prompt_id")) for row in rows if normalize_text(row.get("prompt_id"))}
        source_kinds = {_source_kind(row) for row in rows}
        person = key[0]
        index[key] = {
            "seen_in_full_ocr": "full_ocr" in source_kinds,
            "seen_in_targeted_retry": "targeted_retry" in source_kinds,
            "variant_vote_count": max(len(variants), len(rows)),
            "prompt_vote_count": max(len(prompts), 1),
            "same_candidate_seen_across_variants": len(variants) > 1 or len(rows) > 1,
            "candidate_source_count_for_person": len(rows),
            "unique_candidate_count_for_person": len(unique_by_person[person]),
        }
    return index


def _candidate_payload(feature: dict[str, Any], candidate_id: str, consensus: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "account_shape": account_shape(feature.get("candidate_raw") or feature.get("candidate_masked")),
        "context_flags": {key: bool(feature.get(key)) for key in CONTEXT_FLAG_KEYS},
        "risk_flags": {key: bool(feature.get(key)) for key in RISK_FLAG_KEYS},
        "shape_features": shape_features(feature),
        "field_evidence": field_evidence(feature),
        "bank_holder_evidence": bank_holder_evidence(feature),
        "source_evidence": source_evidence(feature),
        "kie_evidence": redacted_kie_evidence(feature),
        "layout_evidence": layout_evidence(feature),
        "consensus_features": consensus
        or {
            "seen_in_full_ocr": _source_kind(feature) == "full_ocr",
            "seen_in_targeted_retry": _source_kind(feature) == "targeted_retry",
            "variant_vote_count": 1,
            "prompt_vote_count": 1,
            "same_candidate_seen_across_variants": False,
            "candidate_source_count_for_person": 1,
            "unique_candidate_count_for_person": 1,
        },
    }
    for key in NUMERIC_FEATURE_KEYS:
        value = feature.get(key)
        if isinstance(value, bool):
            payload[key] = int(value)
        elif isinstance(value, (int, float)):
            payload[key] = value
        elif value not in (None, ""):
            try:
                payload[key] = float(value)
            except (TypeError, ValueError):
                payload[key] = 0
        else:
            payload[key] = 0
    return payload


def _local_raw_map(feature: dict[str, Any], source_id: str, candidate_id: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "candidate_id": candidate_id,
        "candidate_raw": normalize_text(feature.get("candidate_raw")),
        "candidate_masked": normalize_text(feature.get("candidate_masked")),
        "name": normalize_text(feature.get("matched_name")),
        "group": normalize_text(feature.get("matched_group")),
        "no": normalize_text(feature.get("matched_no")),
        "source_name": normalize_text(feature.get("source_name")),
        "backend": normalize_text(feature.get("backend")),
        "variant": normalize_text(feature.get("variant")),
        "prompt_id": normalize_text(feature.get("prompt_id")),
    }


def build_openai_reranker_payloads(
    features: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []
    raw_maps: list[dict[str, Any]] = []
    consensus = consensus_index(features)
    for source_id, source_features in _group_features_by_source(features).items():
        candidates = []
        for index, feature in enumerate(source_features, start=1):
            candidate_id = f"acct_{index}"
            candidates.append(
                _candidate_payload(
                    feature,
                    candidate_id,
                    consensus.get((_person_key(feature), account_key(feature.get("candidate_raw")))),
                )
            )
            raw_maps.append(_local_raw_map(feature, source_id, candidate_id))
        payloads.append(
            {
                "schema_version": PAYLOAD_SCHEMA_VERSION,
                "task": "select_target_bank_account_candidate",
                "source_id": source_id,
                "privacy_contract": {
                    "raw_account_numbers": "omitted",
                    "person_names": "omitted",
                    "source_paths": "omitted",
                    "ocr_text": "omitted",
                    "local_raw_mapping": "candidate_raw_map_local.jsonl",
                },
                "candidates": candidates,
            }
        )
    return payloads, raw_maps


def build_openai_reranker_files(*, features_jsonl: Path, output_dir: Path) -> dict[str, Any]:
    features = read_jsonl(features_jsonl)
    payloads, raw_maps = build_openai_reranker_payloads(features)
    redacted_path = output_dir / "candidate_features_redacted.jsonl"
    raw_map_path = output_dir / "candidate_raw_map_local.jsonl"
    write_jsonl(redacted_path, payloads)
    write_jsonl(raw_map_path, raw_maps)
    summary = {
        "features_jsonl": str(features_jsonl),
        "payload_count": len(payloads),
        "candidate_count": len(raw_maps),
        "outputs": {
            "candidate_features_redacted": str(redacted_path),
            "candidate_raw_map_local": str(raw_map_path),
        },
        "notes": [
            "candidate_features_redacted.jsonl is the only artifact intended for OpenAI reranking.",
            "candidate_raw_map_local.jsonl contains raw account mappings and must remain local-only.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build redacted OpenAI reranker payloads plus a local-only raw candidate map.")
    parser.add_argument("--features-jsonl", type=Path, required=True, help="Local candidate_features_local.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_openai_reranker_files(features_jsonl=args.features_jsonl, output_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
