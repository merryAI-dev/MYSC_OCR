from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BankbookGold:
    account_number: str
    bank: str = ""
    account_holder: str = ""


@dataclass(frozen=True)
class BankbookPrediction:
    raw_text: str
    account_number: str = ""
    candidate_accounts: tuple[str, ...] = ()


FIELD_ALIASES = {
    "bank": ("bank", "은행", "은행명", "금융기관", "금융기관명"),
    "account_holder": ("account_holder", "holder", "예금주", "예금주명", "성명", "이름"),
    "account_number": ("account_number", "account", "계좌번호", "계좌", "입금계좌"),
}

UNKNOWN_VALUES = {"", "unknown", "not_found", "notfound", "n/a", "na", "none", "미상", "없음", "모름"}


def digits_only(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def normalize_match_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value or "").lower()
    return re.sub(r"[\s:：,，./\\|_\-(){}\[\]<>]+", "", value)


def is_known_label_value(value: str) -> bool:
    return normalize_match_text(value) not in UNKNOWN_VALUES


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + int(left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def normalized_edit_similarity(predicted: str, expected: str) -> float:
    if not predicted and not expected:
        return 1.0
    denominator = max(len(predicted), len(expected), 1)
    return max(0.0, 1.0 - (levenshtein_distance(predicted, expected) / denominator))


def _line_field_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(rf"^\s*{re.escape(alias)}\s*[:：=-]\s*(.+?)\s*$", flags=re.IGNORECASE)


def parse_structured_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for field, aliases in FIELD_ALIASES.items():
            if field in fields:
                continue
            for alias in aliases:
                match = _line_field_pattern(alias).match(line)
                if match:
                    value = match.group(1).strip()
                    if value and value.upper() != "NOT_FOUND":
                        fields[field] = value
                    break
    return fields


def field_match_score(predicted: str, expected: str) -> float:
    predicted_norm = normalize_match_text(predicted)
    expected_norm = normalize_match_text(expected)
    if not expected_norm:
        return 0.0
    if predicted_norm == expected_norm:
        return 1.0
    if len(expected_norm) < 5 and expected_norm in predicted_norm:
        return 1.0
    return normalized_edit_similarity(predicted_norm, expected_norm)


def key_value_f1(predicted: dict[str, str], expected: dict[str, str]) -> dict[str, float | int]:
    expected_pairs = {key: value for key, value in expected.items() if is_known_label_value(value)}
    predicted_pairs = {key: value for key, value in predicted.items() if is_known_label_value(value)}
    correct = 0
    for key, expected_value in expected_pairs.items():
        predicted_value = predicted_pairs.get(key, "")
        if field_match_score(predicted_value, expected_value) >= 0.95:
            correct += 1

    precision = correct / len(predicted_pairs) if predicted_pairs else 0.0
    recall = correct / len(expected_pairs) if expected_pairs else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "ground_truth_pairs": len(expected_pairs),
        "predicted_pairs": len(predicted_pairs),
        "correct_pairs": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _candidate_digit_similarities(candidates: Iterable[str], expected_digits: str) -> list[float]:
    return [
        normalized_edit_similarity(digits_only(candidate), expected_digits)
        for candidate in candidates
        if digits_only(candidate)
    ]


def bankbook_ocrbench_v2_metrics(gold: BankbookGold, prediction: BankbookPrediction) -> dict[str, object]:
    fields = parse_structured_fields(prediction.raw_text)
    predicted_account = prediction.account_number or fields.get("account_number", "")
    expected_digits = digits_only(gold.account_number)
    predicted_digits = digits_only(predicted_account)
    candidate_exact = expected_digits in {digits_only(candidate) for candidate in prediction.candidate_accounts}
    candidate_scores = _candidate_digit_similarities(prediction.candidate_accounts, expected_digits)
    best_candidate_similarity = max(candidate_scores, default=0.0)
    account_exact = bool(expected_digits and predicted_digits == expected_digits)
    false_positive = bool(predicted_digits and expected_digits and predicted_digits != expected_digits)

    expected_fields = {
        "bank": gold.bank,
        "account_holder": gold.account_holder,
        "account_number": gold.account_number,
    }
    predicted_fields = {
        "bank": fields.get("bank", ""),
        "account_holder": fields.get("account_holder", ""),
        "account_number": predicted_account,
    }
    extraction = key_value_f1(predicted_fields, expected_fields)
    account_digit_similarity = normalized_edit_similarity(predicted_digits, expected_digits)
    recognition_score = 1.0 if account_exact else max(account_digit_similarity, best_candidate_similarity)
    false_positive_penalty = 0.0 if false_positive else 1.0
    extraction_f1 = float(extraction["f1"])
    composite = (0.50 * recognition_score) + (0.35 * extraction_f1) + (0.15 * false_positive_penalty)

    return {
        "benchmark": "bankbook_ocrbench_v2_adapted",
        "metrics_source": "OCRBench v2 metric families adapted to Korean bankbook OCR",
        "recognition": {
            "account_exact": account_exact,
            "candidate_account_exact": candidate_exact,
            "account_digit_edit_similarity": account_digit_similarity,
            "best_candidate_digit_edit_similarity": best_candidate_similarity,
            "score": recognition_score,
        },
        "extraction": extraction,
        "basic_vqa": {
            "no_account_false_positive": not false_positive,
            "score": false_positive_penalty,
        },
        "composite_score": composite,
    }
