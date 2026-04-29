from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


BANK_KEYWORDS = (
    "은행",
    "카카오",
    "토스",
    "농협",
    "신협",
    "새마을",
    "수협",
    "우체국",
    "증권",
    "계좌",
    "통장",
    "예금",
)
ACCOUNT_KEYWORDS = ("계좌", "계좌번호", "입금계좌", "account", "Account", "ACCOUNT")
NEGATIVE_KEYWORDS = ("연락처", "전화", "휴대폰", "고객센터", "주민", "면허", "생년월일", "date", "Date")
PROMPT_LEAKAGE_KEYWORDS = ("KNOWN 계좌번호", "계좌번호가 보이면", "Use a clear and concise format", "UNKNOWN 계좌번호")
STRUCTURED_BANKBOOK_KEYWORDS = ("예금주", "상품종류", "상품명", "계좌종류", "은행명", "과목", "Account No.", "<tr", "<td", "|계좌번호|")


@dataclass(frozen=True)
class AccountPolicyDecision:
    candidate: str
    score: float
    accepted: bool
    reasons: tuple[str, ...]
    features: dict[str, bool | int | float]


@dataclass(frozen=True)
class AccountRerankResult:
    selected: AccountPolicyDecision | None
    decisions: tuple[AccountPolicyDecision, ...]
    status: str
    reason: str


def normalize_policy_text(value: object) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFC", str(value)).strip()


def account_digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def mask_account_candidate(value: str) -> str:
    digits_total = sum(1 for char in value or "" if char.isdigit())
    visible_after = max(digits_total - 4, 0)
    seen = 0
    masked: list[str] = []
    for char in value or "":
        if char.isdigit():
            seen += 1
            masked.append("*" if seen <= visible_after else char)
        else:
            masked.append(char)
    return "".join(masked)


def looks_like_phone_number(candidate: str) -> bool:
    parts = candidate.split("-")
    digits = candidate.replace("-", "")
    if re.fullmatch(r"010-?\d{4}-?\d{4}", candidate):
        return True
    if re.fullmatch(r"0(?:2|[3-6][1-5]|70|80)-?\d{3,4}-?\d{4}", candidate):
        return True
    if re.fullmatch(r"82-?\d{1,2}-?\d{3,4}-?\d{4}", candidate):
        return True
    if len(parts) == 3 and parts[0] in {"02", "070", "080"}:
        return True
    if len(parts) == 3 and re.fullmatch(r"0[3-6][1-5]", parts[0]):
        return True
    if len(digits) in {9, 10, 11} and digits.startswith(
        ("02", "070", "080", "050", "031", "032", "033", "041", "042", "043", "044", "051", "052", "053", "054", "055", "061", "062", "063", "064")
    ):
        return True
    return False


def candidate_windows(text: str, candidate: str, *, window: int = 48) -> list[str]:
    text = normalize_policy_text(text)
    matches = list(re.finditer(re.escape(candidate), text))
    if not matches:
        return [text[: window * 2]]
    return [text[max(0, match.start() - window) : match.end() + window] for match in matches]


def candidate_policy_features(text: str, candidate: str, *, repeat_count: int | None = None) -> dict[str, bool | int | float]:
    text = normalize_policy_text(text)
    windows = candidate_windows(text, candidate)
    joined = "\n".join(windows)
    digits = account_digits(candidate)
    repeat = repeat_count if repeat_count is not None else max(text.count(candidate), 1)
    direct_account_field = (
        f"계좌번호</td><td>{candidate}" in joined
        or f"|계좌번호|{candidate}" in joined
        or re.search(rf"계좌\s*번호\s*[:：]?\s*{re.escape(candidate)}", joined) is not None
        or re.search(rf"Account No\.\s*{re.escape(candidate)}", joined, re.IGNORECASE) is not None
    )
    return {
        "digit_count": len(digits),
        "hyphen_count": candidate.count("-"),
        "repeat_count": repeat,
        "looks_like_phone": looks_like_phone_number(candidate),
        "has_account_keyword_context": any(keyword in joined for keyword in ACCOUNT_KEYWORDS),
        "has_bank_keyword_context": any(keyword in joined for keyword in BANK_KEYWORDS),
        "has_negative_keyword_context": any(keyword in joined for keyword in NEGATIVE_KEYWORDS),
        "has_prompt_leakage_context": any(keyword in joined for keyword in PROMPT_LEAKAGE_KEYWORDS),
        "has_structured_bankbook_context": "계좌" in joined and any(keyword in joined for keyword in STRUCTURED_BANKBOOK_KEYWORDS),
        "has_direct_account_field_context": direct_account_field,
        "has_wrong_field_context": f"예금주</td><td>{candidate}</td>" in joined,
        "has_customer_number_metadata_context": re.search(r"고객\s*번호", joined) is not None,
    }


def policy_score_from_features(features: dict[str, bool | int | float]) -> float:
    score = 0.0
    digit_count = int(features.get("digit_count") or 0)
    if 10 <= digit_count <= 14:
        score += 3.0
    if 11 <= digit_count <= 13:
        score += 1.0
    score += 2.0 if features.get("hyphen_count", 0) else 0.0
    score += 5.0 if features.get("has_account_keyword_context") else 0.0
    score += 3.0 if features.get("has_bank_keyword_context") else 0.0
    score += min(float(features.get("repeat_count") or 1) - 1.0, 3.0)
    score += 2.0 if features.get("has_structured_bankbook_context") else 0.0
    score += 2.0 if features.get("has_direct_account_field_context") else 0.0
    score -= 8.0 if features.get("looks_like_phone") else 0.0
    score -= 4.0 if features.get("has_negative_keyword_context") else 0.0
    if features.get("has_prompt_leakage_context"):
        score -= 2.0 if features.get("has_structured_bankbook_context") else 6.0
    score -= 8.0 if features.get("has_wrong_field_context") else 0.0
    if features.get("has_customer_number_metadata_context") and not features.get("has_structured_bankbook_context"):
        score -= 7.0
    return score


def policy_reasons(features: dict[str, bool | int | float]) -> tuple[str, ...]:
    reasons: list[str] = []
    if features.get("has_structured_bankbook_context"):
        reasons.append("structured_bankbook_context")
    if features.get("has_account_keyword_context"):
        reasons.append("account_keyword_context")
    if features.get("has_bank_keyword_context"):
        reasons.append("bank_keyword_context")
    if features.get("has_prompt_leakage_context") and not features.get("has_structured_bankbook_context"):
        reasons.append("prompt_leakage_without_structured_context")
    elif features.get("has_prompt_leakage_context"):
        reasons.append("prompt_leakage_with_structured_context")
    if features.get("has_wrong_field_context"):
        reasons.append("wrong_field_context")
    if features.get("has_customer_number_metadata_context"):
        reasons.append("customer_number_metadata_context")
    if features.get("has_direct_account_field_context"):
        reasons.append("direct_account_field_context")
    if features.get("looks_like_phone"):
        reasons.append("phone_like")
    if features.get("has_negative_keyword_context"):
        reasons.append("negative_keyword_context")
    return tuple(reasons)


def policy_score_candidate(
    text: str,
    candidate: str,
    *,
    min_score: float = 10.0,
    repeat_count: int | None = None,
) -> AccountPolicyDecision:
    features = candidate_policy_features(text, candidate, repeat_count=repeat_count)
    score = policy_score_from_features(features)
    hard_reject = (
        features.get("has_wrong_field_context")
        or features.get("looks_like_phone")
        or (features.get("has_prompt_leakage_context") and not features.get("has_direct_account_field_context"))
    )
    bank_context_fallback = (
        10 <= int(features.get("digit_count") or 0) <= 14
        and bool(features.get("hyphen_count"))
        and bool(features.get("has_bank_keyword_context"))
        and not (
            features.get("has_customer_number_metadata_context")
            and not features.get("has_structured_bankbook_context")
        )
    )
    accepted = (
        not hard_reject
        and (
            score >= min_score
            or bank_context_fallback
        )
    )
    return AccountPolicyDecision(
        candidate=candidate,
        score=score,
        accepted=accepted,
        reasons=policy_reasons(features),
        features=features,
    )


def rank_account_candidates(text: str, candidates: list[str], *, min_score: float = 10.0) -> list[AccountPolicyDecision]:
    decisions = [policy_score_candidate(text, candidate, min_score=min_score) for candidate in candidates]
    return sorted(decisions, key=lambda item: (item.accepted, item.score), reverse=True)


def select_account_candidate(text: str, candidates: list[str], *, min_score: float = 10.0) -> AccountPolicyDecision | None:
    ranked = rank_account_candidates(text, candidates, min_score=min_score)
    accepted = [item for item in ranked if item.accepted]
    if len(accepted) == 1:
        return accepted[0]
    return None


def rerank_account_candidates(
    text: str,
    candidates: list[str],
    *,
    min_score: float = 10.0,
    min_margin: float = 2.0,
) -> AccountRerankResult:
    best_by_key: dict[str, AccountPolicyDecision] = {}
    for candidate in candidates:
        key = account_digits(candidate)
        if not key:
            continue
        decision = policy_score_candidate(text, candidate, min_score=min_score)
        current = best_by_key.get(key)
        if current is None or (decision.accepted, decision.score) > (current.accepted, current.score):
            best_by_key[key] = decision

    decisions = tuple(sorted(best_by_key.values(), key=lambda item: (item.accepted, item.score), reverse=True))
    accepted = [decision for decision in decisions if decision.accepted]
    if not accepted:
        return AccountRerankResult(
            selected=None,
            decisions=decisions,
            status="no_accepted_candidate",
            reason="ambiguous_or_low_score",
        )
    if len(accepted) == 1:
        return AccountRerankResult(
            selected=accepted[0],
            decisions=decisions,
            status="selected",
            reason="single_policy_accept",
        )

    score_gap = accepted[0].score - accepted[1].score
    if score_gap >= min_margin:
        return AccountRerankResult(
            selected=accepted[0],
            decisions=decisions,
            status="selected",
            reason="top_policy_margin",
        )
    return AccountRerankResult(
        selected=None,
        decisions=decisions,
        status="ambiguous_conflict",
        reason="accepted_candidates_within_margin",
    )


def _flag(value: bool | int | float | None) -> str:
    return "1" if bool(value) else "0"


def policy_audit_rows(
    text: str,
    candidates: list[str],
    *,
    source_id: str = "",
    source_name: str = "",
    min_score: float = 10.0,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = account_digits(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        decision = policy_score_candidate(text, candidate, min_score=min_score)
        rows.append(
            {
                "source_id": source_id,
                "source_name": source_name,
                "candidate_masked": mask_account_candidate(candidate),
                "accepted": "1" if decision.accepted else "0",
                "policy_score": f"{decision.score:.1f}",
                "policy_reasons": ";".join(decision.reasons),
                "has_prompt_leakage_context": _flag(decision.features.get("has_prompt_leakage_context")),
                "has_wrong_field_context": _flag(decision.features.get("has_wrong_field_context")),
                "has_direct_account_field_context": _flag(decision.features.get("has_direct_account_field_context")),
                "has_structured_bankbook_context": _flag(decision.features.get("has_structured_bankbook_context")),
            }
        )
    return rows
