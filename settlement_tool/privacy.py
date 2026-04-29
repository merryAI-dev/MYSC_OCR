from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PrivacySpan:
    label: str
    text: str
    score: float | None = None


def detect_privacy_spans(text: str) -> list[PrivacySpan]:
    """Detect PII spans in local OCR text.

    Uses openai/privacy-filter when transformers is installed. Falls back to
    conservative regex detection so the rest of the workflow remains usable.
    """
    try:
        from transformers import pipeline

        classifier = pipeline(
            task="token-classification",
            model="openai/privacy-filter",
            aggregation_strategy="simple",
        )
        spans = classifier(text)
        return [
            PrivacySpan(
                label=str(span.get("entity_group") or span.get("entity") or ""),
                text=str(span.get("word") or ""),
                score=float(span["score"]) if span.get("score") is not None else None,
            )
            for span in spans
        ]
    except Exception:
        return _regex_privacy_spans(text)


def _regex_privacy_spans(text: str) -> list[PrivacySpan]:
    spans: list[PrivacySpan] = []
    for match in re.finditer(r"\b010-?\d{4}-?\d{4}\b", text):
        spans.append(PrivacySpan("private_phone", match.group(0), None))
    for match in re.finditer(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text):
        spans.append(PrivacySpan("private_email", match.group(0), None))
    for match in re.finditer(r"(?<!\d)(?:\d[\d -]{7,22}\d)(?!\d)", text):
        spans.append(PrivacySpan("account_number", match.group(0), None))
    return spans
