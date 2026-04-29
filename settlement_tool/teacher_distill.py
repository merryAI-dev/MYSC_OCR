from __future__ import annotations

import re
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .account_policy import candidate_policy_features, policy_score_from_features
from .core import _looks_like_phone_number, _normalize_account_candidate, normalize_text


ACCOUNT_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[\d -]{7,22}\d)(?!\d)")
CONTEXT_DIGIT_RUN_RE = re.compile(r"(?<!\d)(?:\d[\d -]{2,22}\d)(?!\d)")


def account_key(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def mask_candidate(value: str) -> str:
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


def mask_digit_context(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        digits_total = sum(1 for char in value if char.isdigit())
        if digits_total < 4:
            return value
        if digits_total <= 4:
            return re.sub(r"\d", "*", value)
        return mask_candidate(value)

    return CONTEXT_DIGIT_RUN_RE.sub(replace, normalize_text(text))


def _candidate_context(text: str, candidate: str, window: int = 48) -> str:
    match = re.search(re.escape(candidate), text)
    if not match:
        digits = account_key(candidate)
        compact_digits = account_key(text)
        index = compact_digits.find(digits)
        if index < 0:
            return text[: window * 2]
        return text[: window * 2]
    return text[max(0, match.start() - window) : match.end() + window]


def _group_count(candidate: str) -> int:
    return len([part for part in re.split(r"[-\s]+", candidate) if part])


def _parse_json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(normalize_text(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _candidate_base_features(text: str, candidate: str) -> dict[str, Any]:
    context = _candidate_context(text, candidate)
    digits = account_key(candidate)
    policy_features = candidate_policy_features(text, candidate)
    return {
        "candidate_raw": candidate,
        "candidate_masked": mask_candidate(candidate),
        "digit_count": len(digits),
        "hyphen_count": candidate.count("-"),
        "group_count": _group_count(candidate),
        **policy_features,
        "teacher_context_masked": mask_digit_context(context),
    }


def build_candidate_features(
    text: str,
    *,
    source_id: str,
    source_name: str,
    gold_account: str = "",
    backend: str = "",
    variant: str = "",
    prompt_id: str = "",
    include_phone_like: bool = False,
) -> list[dict[str, Any]]:
    text = normalize_text(text)
    raw_candidates = ACCOUNT_CANDIDATE_RE.findall(text)
    counts = Counter(_normalize_account_candidate(raw) for raw in raw_candidates)
    features: list[dict[str, Any]] = []
    seen: set[str] = set()
    gold_key = account_key(gold_account)

    for raw in raw_candidates:
        candidate = _normalize_account_candidate(raw)
        digits = account_key(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if len(digits) < 9 or len(digits) > 16:
            continue
        if _looks_like_phone_number(candidate) and not include_phone_like:
            continue

        feature = _candidate_base_features(text, candidate)
        feature.update(
            {
                "source_id": source_id,
                "source_name": source_name,
                "backend": backend,
                "variant": variant,
                "prompt_id": prompt_id,
                "repeat_count": counts[candidate],
                "gold_label_available": bool(gold_key),
                "gold_exact_match": (account_key(candidate) == gold_key) if gold_key else None,
                "teacher_policy_score": policy_score(feature),
            }
        )
        features.append(feature)

    return features


def _read_text_for_row(row: dict[str, str]) -> str:
    for key in ("ocr_text_path", "text_path"):
        value = row.get(key, "")
        if value and Path(value).exists():
            return Path(value).read_text(encoding="utf-8", errors="replace")
    parts = [row.get("account", ""), row.get("candidates", "")]
    return "\n".join(part for part in parts if part)


def _source_gold(row: dict[str, str], gold_by_source_name: dict[str, str]) -> str:
    for key in ("source_name", "extracted_path", "image_path", "image_name"):
        value = row.get(key, "")
        if value in gold_by_source_name:
            return gold_by_source_name[value]
        basename = Path(value).name if value else ""
        if basename in gold_by_source_name:
            return gold_by_source_name[basename]
    return ""


def features_from_ocr_csv(
    csv_path: Path | str,
    *,
    gold_by_source_name: dict[str, str] | None = None,
    backend: str = "",
    include_phone_like: bool = False,
) -> list[dict[str, Any]]:
    csv_path = Path(csv_path)
    gold_by_source_name = gold_by_source_name or {}
    records: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            text = _read_text_for_row(row)
            source_name = row.get("source_name") or row.get("image_name") or f"{csv_path.name}:{index}"
            row_features = build_candidate_features(
                text,
                source_id=f"{csv_path.stem}:{index}",
                source_name=source_name,
                gold_account=_source_gold(row, gold_by_source_name),
                backend=backend,
                variant=row.get("variant", ""),
                prompt_id=row.get("prompt_id", ""),
                include_phone_like=include_phone_like,
            )
            for feature in row_features:
                feature.update(
                    {
                        "matched_name": row.get("matched_name", "") or row.get("name", ""),
                        "matched_group": row.get("matched_group", "") or row.get("roster_group", ""),
                        "matched_no": row.get("matched_no", "") or row.get("roster_no", ""),
                    }
                )
            records.extend(row_features)
    return records


def features_from_kie_csv(
    csv_path: Path | str,
    *,
    gold_by_source_name: dict[str, str] | None = None,
    backend: str = "",
) -> list[dict[str, Any]]:
    csv_path = Path(csv_path)
    gold_by_source_name = gold_by_source_name or {}
    records: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            if row.get("error"):
                continue
            candidate_raw = normalize_text(row.get("candidate_raw"))
            candidate_masked = normalize_text(row.get("candidate_masked"))
            if not candidate_raw and not candidate_masked:
                continue
            field_type = normalize_text(row.get("kie_field_type")) or "unknown"
            label = normalize_text(row.get("kie_label_masked"))
            context = "\n".join(part for part in (label, candidate_masked) if part)
            feature = _candidate_base_features(context, candidate_raw or candidate_masked)
            feature.update(
                {
                    "candidate_raw": candidate_raw,
                    "candidate_masked": candidate_masked or mask_candidate(candidate_raw),
                    "source_id": row.get("source_id") or f"{csv_path.stem}:{index}",
                    "source_name": row.get("source_name") or f"{csv_path.name}:{index}",
                    "backend": backend or row.get("backend", ""),
                    "variant": row.get("variant", ""),
                    "prompt_id": row.get("prompt_id", "kie"),
                    "repeat_count": 1,
                    "kie_backend": row.get("kie_backend") or row.get("backend", ""),
                    "kie_field_type": field_type,
                    "kie_label_masked": label,
                    "kie_confidence": float(row.get("kie_confidence") or row.get("confidence") or 0.0),
                    "kie_confidence_bucket": row.get("kie_confidence_bucket", ""),
                    "kie_holder_match_status": row.get("kie_holder_match_status", ""),
                    "kie_holder_field_present": str(row.get("kie_holder_field_present", "")).strip().lower() == "true",
                    "kie_bank_name_present": str(row.get("kie_bank_name_present", "")).strip().lower() == "true",
                    "layout_evidence": _parse_json_object(row.get("layout_json")),
                    "gold_label_available": False,
                    "gold_exact_match": None,
                    "matched_name": row.get("matched_name", "") or row.get("name", ""),
                    "matched_group": row.get("matched_group", "") or row.get("roster_group", ""),
                    "matched_no": row.get("matched_no", "") or row.get("roster_no", ""),
                }
            )
            if field_type == "account_number":
                feature["has_account_keyword_context"] = True
                feature["has_direct_account_field_context"] = True
            elif field_type == "customer_number":
                feature["has_customer_number_metadata_context"] = True
            elif field_type == "phone":
                feature["looks_like_phone"] = True
                feature["has_negative_keyword_context"] = True
            elif field_type in {"date", "amount"}:
                feature["has_negative_keyword_context"] = True
            gold_key = account_key(_source_gold(row, gold_by_source_name))
            if gold_key:
                feature["gold_label_available"] = True
                feature["gold_exact_match"] = account_key(candidate_raw) == gold_key
            feature["teacher_context_masked"] = mask_digit_context(context)
            feature["teacher_policy_score"] = policy_score(feature)
            records.append(feature)
    return records


def teacher_review_record(feature: dict[str, Any]) -> dict[str, Any]:
    blocked = {"candidate_raw"}
    return {key: value for key, value in feature.items() if key not in blocked}


def seed_teacher_policy_label(feature: dict[str, Any]) -> dict[str, Any]:
    review = teacher_review_record(feature)
    if feature.get("gold_label_available"):
        if feature.get("gold_exact_match"):
            label = "accept"
            requires_review = False
            reason = "gold_exact_match"
        else:
            label = "reject"
            requires_review = False
            reason = "gold_mismatch"
    elif feature.get("looks_like_phone") or feature.get("has_negative_keyword_context"):
        label = "reject"
        requires_review = False
        reason = "obvious_negative_context"
    elif (
        float(feature.get("teacher_policy_score") or 0.0) >= 12.0
        and feature.get("has_account_keyword_context")
        and feature.get("has_bank_keyword_context")
    ):
        label = "review_accept_candidate"
        requires_review = True
        reason = "strong_account_context_without_gold"
    else:
        label = "review"
        requires_review = True
        reason = "insufficient_local_evidence"

    review.update(
        {
            "suggested_label": label,
            "suggested_reason": reason,
            "requires_teacher_review": requires_review,
            "teacher_label": "",
            "teacher_reason": "",
            "teacher_evidence": "",
        }
    )
    return review


def draft_masked_context_teacher_label(
    row: dict[str, Any],
    *,
    teacher_id: str = "codex_draft_masked_context",
) -> dict[str, Any]:
    label = "reject"
    reason = "codex_draft_reject_insufficient_masked_context"
    suggested = row.get("suggested_label")
    score = float(row.get("teacher_policy_score") or 0.0)

    if suggested == "reject":
        label = "reject"
        reason = "codex_draft_reject_seed_negative"
    elif suggested == "accept":
        label = "accept"
        reason = "codex_draft_accept_seed_gold"
    elif (
        score >= 10.0
        and row.get("has_account_keyword_context")
        and row.get("has_bank_keyword_context")
        and not row.get("has_negative_keyword_context")
        and not row.get("looks_like_phone")
    ):
        label = "accept"
        reason = "codex_draft_accept_visible_account_bank_context"

    return {
        "source_id": row.get("source_id", ""),
        "candidate_masked": row.get("candidate_masked", ""),
        "variant": row.get("variant", ""),
        "prompt_id": row.get("prompt_id", ""),
        "teacher_label": label,
        "teacher_reason": reason,
        "teacher_evidence": row.get("teacher_context_masked", ""),
        "teacher_id": teacher_id,
    }


def evaluate_policy_labels(rows: list[dict[str, Any]], *, threshold: float = 10.0) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("teacher_label") in {"accept", "reject"}]
    tp = tn = fp = fn = 0
    for row in labeled:
        predicted_accept = float(row.get("teacher_policy_score") or 0.0) >= threshold
        actual_accept = row.get("teacher_label") == "accept"
        if predicted_accept and actual_accept:
            tp += 1
        elif predicted_accept and not actual_accept:
            fp += 1
        elif not predicted_accept and actual_accept:
            fn += 1
        else:
            tn += 1
    total = len(labeled)
    return {
        "threshold": threshold,
        "labeled_count": total,
        "true_positive_count": tp,
        "true_negative_count": tn,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def teacher_label_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("source_id", "")),
        str(row.get("candidate_masked", "")),
        str(row.get("variant", "")),
        str(row.get("prompt_id", "")),
    )


def merge_teacher_labels(seed_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = {teacher_label_key(row): row for row in label_rows}
    merged: list[dict[str, Any]] = []
    for row in seed_rows:
        current = dict(row)
        label = labels.get(teacher_label_key(row))
        if label:
            for key in ("teacher_label", "teacher_reason", "teacher_evidence", "teacher_id"):
                if key in label:
                    current[key] = label[key]
        merged.append(current)
    return merged


def summarize_label_coverage(seed_rows: list[dict[str, Any]], merged_rows: list[dict[str, Any]]) -> dict[str, Any]:
    review_keys = {teacher_label_key(row) for row in seed_rows if row.get("requires_teacher_review")}
    labeled_review_keys = {
        teacher_label_key(row)
        for row in merged_rows
        if teacher_label_key(row) in review_keys and row.get("teacher_label") in {"accept", "reject"}
    }
    label_counts: dict[str, int] = {}
    review_label_counts: dict[str, int] = {}
    teacher_id_counts: dict[str, int] = {}
    for row in merged_rows:
        label = str(row.get("teacher_label", ""))
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1
        if teacher_label_key(row) in review_keys and label:
            review_label_counts[label] = review_label_counts.get(label, 0) + 1
        teacher_id = str(row.get("teacher_id", ""))
        if teacher_id:
            teacher_id_counts[teacher_id] = teacher_id_counts.get(teacher_id, 0) + 1

    return {
        "review_row_count": len(review_keys),
        "labeled_review_row_count": len(labeled_review_keys),
        "unlabeled_review_row_count": len(review_keys - labeled_review_keys),
        "review_label_counts": review_label_counts,
        "all_label_counts": label_counts,
        "teacher_id_counts": teacher_id_counts,
    }


def calibrate_policy_threshold(
    rows: list[dict[str, Any]],
    *,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or [float(value) for value in range(0, 21)]
    results = [evaluate_policy_labels(rows, threshold=threshold) for threshold in thresholds]
    best = max(
        results,
        key=lambda item: (
            -int(item["false_positive_count"]),
            float(item["recall"]),
            float(item["accuracy"]),
            -abs(float(item["threshold"]) - 10.0),
        ),
    )
    return {
        "best_threshold": best["threshold"],
        "best": best,
        "thresholds": results,
    }


def evaluate_source_selection(rows: list[dict[str, Any]], *, threshold: float = 10.0) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("teacher_label") in {"accept", "reject"}]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in labeled:
        grouped.setdefault(str(row.get("source_name") or row.get("source_id") or ""), []).append(row)

    selected_count = selected_accept = selected_reject = missed_accept = 0
    source_with_accept = 0
    selected_reject_examples: list[dict[str, Any]] = []
    missed_accept_examples: list[dict[str, Any]] = []
    for source_rows in grouped.values():
        has_accept = any(row.get("teacher_label") == "accept" for row in source_rows)
        if has_accept:
            source_with_accept += 1
        eligible = [row for row in source_rows if float(row.get("teacher_policy_score") or 0.0) >= threshold]
        if not eligible:
            if has_accept:
                missed_accept += 1
                accept_row = next(row for row in source_rows if row.get("teacher_label") == "accept")
                if len(missed_accept_examples) < 10:
                    missed_accept_examples.append(
                        {
                            "source_id": accept_row.get("source_id", ""),
                            "candidate_masked": accept_row.get("candidate_masked", ""),
                            "teacher_policy_score": accept_row.get("teacher_policy_score", 0.0),
                        }
                    )
            continue
        selected = max(eligible, key=lambda row: float(row.get("teacher_policy_score") or 0.0))
        selected_count += 1
        if selected.get("teacher_label") == "accept":
            selected_accept += 1
        else:
            selected_reject += 1
            if len(selected_reject_examples) < 10:
                selected_reject_examples.append(
                    {
                        "source_id": selected.get("source_id", ""),
                        "candidate_masked": selected.get("candidate_masked", ""),
                        "teacher_policy_score": selected.get("teacher_policy_score", 0.0),
                        "teacher_reason": selected.get("teacher_reason", ""),
                    }
                )

    return {
        "threshold": threshold,
        "source_count": len(grouped),
        "source_with_accept_count": source_with_accept,
        "selected_count": selected_count,
        "selected_accept_count": selected_accept,
        "selected_reject_count": selected_reject,
        "missed_accept_source_count": missed_accept,
        "selection_precision": selected_accept / selected_count if selected_count else 0.0,
        "source_accept_recall": selected_accept / source_with_accept if source_with_accept else 0.0,
        "selected_reject_examples": selected_reject_examples,
        "missed_accept_examples": missed_accept_examples,
    }


def _score(row: dict[str, Any]) -> float:
    return float(row.get("teacher_policy_score") or 0.0)


def _masked_example(row: dict[str, Any], *, reason: str = "") -> dict[str, Any]:
    example = {
        "source_id": row.get("source_id", ""),
        "candidate_masked": row.get("candidate_masked", ""),
        "teacher_policy_score": row.get("teacher_policy_score", 0.0),
        "teacher_label": row.get("teacher_label", ""),
    }
    if reason:
        example["reason"] = reason
    return example


def _missed_accept_example(source_rows: list[dict[str, Any]], *, reason: str) -> dict[str, Any]:
    accept_rows = [row for row in source_rows if row.get("teacher_label") == "accept"]
    accept_row = max(accept_rows, key=_score)
    return _masked_example(accept_row, reason=reason)


def evaluate_source_reranker(
    rows: list[dict[str, Any]],
    *,
    threshold: float = 10.0,
    min_margin: float = 2.0,
) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("teacher_label") in {"accept", "reject"}]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in labeled:
        grouped.setdefault(str(row.get("source_name") or row.get("source_id") or ""), []).append(row)

    selected_count = selected_accept = selected_reject = missed_accept = 0
    deferred_conflict = source_with_accept = 0
    selected_examples: list[dict[str, Any]] = []
    selected_reject_examples: list[dict[str, Any]] = []
    deferred_conflict_examples: list[dict[str, Any]] = []
    missed_accept_examples: list[dict[str, Any]] = []

    for source_rows in grouped.values():
        has_accept = any(row.get("teacher_label") == "accept" for row in source_rows)
        if has_accept:
            source_with_accept += 1

        eligible = sorted(
            [row for row in source_rows if _score(row) >= threshold],
            key=_score,
            reverse=True,
        )
        if not eligible:
            if has_accept:
                missed_accept += 1
                if len(missed_accept_examples) < 10:
                    missed_accept_examples.append(_missed_accept_example(source_rows, reason="no_candidate_above_threshold"))
            continue

        selected = eligible[0]
        runner_up = eligible[1] if len(eligible) > 1 else None
        score_gap = _score(selected) - _score(runner_up) if runner_up else None
        if runner_up and score_gap is not None and score_gap < min_margin:
            deferred_conflict += 1
            if len(deferred_conflict_examples) < 10:
                example = _masked_example(selected, reason="score_gap_below_margin")
                example.update(
                    {
                        "runner_up_masked": runner_up.get("candidate_masked", ""),
                        "runner_up_score": runner_up.get("teacher_policy_score", 0.0),
                        "score_gap": score_gap,
                    }
                )
                deferred_conflict_examples.append(example)
            if has_accept:
                missed_accept += 1
                if len(missed_accept_examples) < 10:
                    missed_accept_examples.append(_missed_accept_example(source_rows, reason="score_gap_below_margin"))
            continue

        selected_count += 1
        selected_example = _masked_example(selected, reason="selected")
        if runner_up:
            selected_example.update(
                {
                    "runner_up_masked": runner_up.get("candidate_masked", ""),
                    "runner_up_score": runner_up.get("teacher_policy_score", 0.0),
                    "score_gap": score_gap,
                }
            )
        if len(selected_examples) < 10:
            selected_examples.append(selected_example)
        if selected.get("teacher_label") == "accept":
            selected_accept += 1
        else:
            selected_reject += 1
            if len(selected_reject_examples) < 10:
                selected_reject_examples.append(selected_example)

    return {
        "threshold": threshold,
        "min_margin": min_margin,
        "source_count": len(grouped),
        "source_with_accept_count": source_with_accept,
        "selected_count": selected_count,
        "selected_accept_count": selected_accept,
        "selected_reject_count": selected_reject,
        "deferred_conflict_count": deferred_conflict,
        "missed_accept_source_count": missed_accept,
        "selection_precision": selected_accept / selected_count if selected_count else 0.0,
        "source_accept_recall": selected_accept / source_with_accept if source_with_accept else 0.0,
        "selected_examples": selected_examples,
        "selected_reject_examples": selected_reject_examples,
        "deferred_conflict_examples": deferred_conflict_examples,
        "missed_accept_examples": missed_accept_examples,
    }


def sweep_source_reranker(
    rows: list[dict[str, Any]],
    *,
    thresholds: list[float] | None = None,
    margins: list[float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or [10.0, 12.0, 14.0, 18.0]
    margins = margins or [0.0, 1.0, 2.0, 3.0, 4.0]
    grid = [
        evaluate_source_reranker(rows, threshold=threshold, min_margin=margin)
        for threshold in thresholds
        for margin in margins
    ]
    best = max(
        grid,
        key=lambda item: (
            -int(item["selected_reject_count"]),
            float(item["source_accept_recall"]),
            float(item["selection_precision"]),
            int(item["selected_accept_count"]),
            -int(item["deferred_conflict_count"]),
            -abs(float(item["threshold"]) - 10.0),
            -float(item["min_margin"]),
        ),
    ) if grid else evaluate_source_reranker([], threshold=10.0, min_margin=0.0)
    return {
        "best_threshold": best["threshold"],
        "best_margin": best["min_margin"],
        "best": best,
        "grid": grid,
    }


def policy_score(feature: dict[str, Any]) -> float:
    return policy_score_from_features(feature)
