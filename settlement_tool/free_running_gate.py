from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from .core import classify_account_candidates, normalize_text


TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+(?:[-_][A-Za-z0-9가-힣]+)*|[^\s]")
DIGIT_GROUP_RE = re.compile(r"(?<!\d)\d[\d -]{2,22}\d(?!\d)")
MARKUP_TAG_RE = re.compile(r"<[^>\n]{1,200}>")


def _strip_markup_tags(text: str) -> str:
    return MARKUP_TAG_RE.sub(" ", text or "")


def tokenize_free_running_output(text: str) -> list[str]:
    normalized = normalize_text(_strip_markup_tags(text))
    return TOKEN_RE.findall(normalized)


def _max_run(tokens: Iterable[str]) -> int:
    max_seen = 0
    current = 0
    previous = None
    for token in tokens:
        normalized = token.casefold()
        if normalized == previous:
            current += 1
        else:
            previous = normalized
            current = 1
        max_seen = max(max_seen, current)
    return max_seen


def _ngram_repetition_ratio(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    normalized = [token.casefold() for token in tokens]
    ngrams = [tuple(normalized[index : index + n]) for index in range(len(normalized) - n + 1)]
    if not ngrams:
        return 0.0
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)


def free_running_degeneration_metrics(
    text: str,
    *,
    require_account_candidate: bool = False,
) -> dict[str, object]:
    tokens = tokenize_free_running_output(text)
    token_count = len(tokens)
    token_counts = Counter(token.casefold() for token in tokens)
    unique_token_ratio = len(token_counts) / token_count if token_count else 0.0
    top_token_share = max(token_counts.values()) / token_count if token_counts else 0.0
    max_token_run = _max_run(tokens)
    repetition_trigram_ratio = _ngram_repetition_ratio(tokens, 3)
    candidate_result = classify_account_candidates(text)
    account_candidate_count = len(candidate_result.candidates)
    digit_group_count = len(DIGIT_GROUP_RE.findall(normalize_text(text)))
    hangul_presence = bool(re.search(r"[가-힣]", normalize_text(text)))

    reasons: list[str] = []
    if token_count == 0:
        reasons.append("empty_output")
    if token_count >= 10 and unique_token_ratio < 0.18:
        reasons.append("low_unique_token_ratio")
    if token_count >= 10 and top_token_share > 0.45:
        reasons.append("top_token_dominance")
    if max_token_run > 8:
        reasons.append("long_token_run")
    if token_count >= 12 and repetition_trigram_ratio > 0.35:
        reasons.append("high_trigram_repetition")
    if require_account_candidate and account_candidate_count == 0:
        reasons.append("missing_account_candidate")

    return {
        "token_count": token_count,
        "unique_token_ratio": unique_token_ratio,
        "top_token_share": top_token_share,
        "max_token_run": max_token_run,
        "repetition_trigram_ratio": repetition_trigram_ratio,
        "digit_group_count": digit_group_count,
        "account_candidate_count": account_candidate_count,
        "account_candidate_presence": account_candidate_count > 0,
        "hangul_presence": hangul_presence,
        "degeneration_pass": not reasons,
        "degeneration_reason": ",".join(reasons) if reasons else "ok",
    }
