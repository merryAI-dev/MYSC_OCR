---
title: MerryPII Wiki Index
type: index
status: active
updated: 2026-04-29
tags: [pii, wiki]
---

# MerryPII Wiki Index

This is the content-oriented catalog for the LLM-maintained MerryPII wiki.

Start here before answering project questions. Then drill into the linked pages.

## Overview

- [[wiki/overview|Overview]] - current synthesis of the MerryPII PII extraction loop.
- [[wiki/log|Log]] - append-only chronological record of ingests, queries, and maintenance.

## Experiments

- [[wiki/experiments/2026-04-29-pii-candidate-recall-loop|2026-04-29 PII Candidate Recall Loop]] - the main zero-FP recall loop that moved from 42/64 to 52/64.

## Concepts

- [[wiki/concepts/zero-fp-gate|Zero-FP Gate]] - why `wrong_positive=0` and `review_false_positive=0` dominate threshold decisions.
- [[wiki/concepts/redacted-reranker-payload|Redacted Reranker Payload]] - the local raw map plus redacted payload contract.
- [[wiki/concepts/kie-candidate-generation|KIE Candidate Generation]] - PaddleOCR/KIE candidate generation and why non-account KIE fields remain risky.
- [[wiki/concepts/manual-review-hard-gate|Manual Review Hard Gate]] - why release can pass PII/reranker gates and still be blocked.
- [[wiki/concepts/bank-name-as-metadata|Bank Name as Metadata]] - bank name extraction helps review, but is not the PII target label.

## Decisions

- [[wiki/decisions/2026-04-29-threshold-10-margin-2|Threshold 10 Margin 2]] - keep threshold 10 despite threshold 9 recovering one more correct case.
- [[wiki/decisions/2026-04-29-person-split-payloads|Person-Split Payloads]] - split same-source multi-person payloads into anonymous slots.
- [[wiki/decisions/2026-04-29-reject-kie-bank-rescue|Reject KIE Bank Rescue]] - do not auto-accept account-shaped numbers from KIE `bank` fields.

## Sources

- [[wiki/sources/2026-04-29-pii-recall-doc|PII Recall Doc Source Summary]] - sanitized summary of `docs/2026-04-29-pii-candidate-generation-wiki.md`.
- [[docs/2026-04-29-pii-candidate-generation-wiki|Original 2026-04-29 Wiki-Style Report]] - project report committed before the LLM wiki split.
- [[docs/release-policy|Release Policy]] - operational release policy and hard-gate rules.

## Open Questions

- [[wiki/questions/next-ocr-candidate-recall-loop|Next OCR Candidate Recall Loop]] - next bottleneck: exact candidate generation and field localization.
- [[wiki/questions/independent-validation-set|Independent Validation Set]] - what is needed before broader performance claims.

## Maintenance Notes

- Update this index on every ingest or synthesis.
- Keep source summaries sanitized.
- Link new durable ideas into `concepts/`, not only into experiment notes.
