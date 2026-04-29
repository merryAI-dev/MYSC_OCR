---
title: MerryPII Wiki Index
type: index
status: active
updated: 2026-04-29
tags: [pii, wiki, ocr, quantization]
---

# MerryPII Wiki Index

This is the content-oriented catalog for the LLM-maintained MerryPII / settlement OCR wiki.

Start here before answering project questions. Then drill into the linked pages.

## Overview

- [[wiki/overview|Overview]] - current synthesis of the MerryPII PII extraction loop.
- [[wiki/log|Log]] - append-only chronological record of ingests, queries, and maintenance.

## Paper Drafts

- [[wiki/paper/arxiv-iteration-narrative|arXiv Iteration Narrative]] - paper-oriented narrative of the failed Monday GS64 path, subsequent engineering changes, and final W8 all70 union candidate result.

## Experiments

- [[wiki/experiments/2026-04-29-pii-candidate-recall-loop|2026-04-29 PII Candidate Recall Loop]] - the main zero-FP recall loop that moved from 42/64 to 52/64.
- [[wiki/experiments/2026-04-27-uniform-6bit-gs64-failure|2026-04-27 Uniform 6bit GS64 Failure]] - Monday failure postmortem for the BF16-direct uniform 6bit GS64 DeepSeek-OCR path.
- [[wiki/experiments/2026-04-29-w8-all70-union-candidate|2026-04-29 W8 All70 Union Candidate]] - W8 all70 candidate that absorbs MLX8 primary accepted rows and adds W8-only accepted rows.

## Concepts

- [[wiki/concepts/zero-fp-gate|Zero-FP Gate]] - why `wrong_positive=0` and `review_false_positive=0` dominate threshold decisions.
- [[wiki/concepts/redacted-reranker-payload|Redacted Reranker Payload]] - the local raw map plus redacted payload contract.
- [[wiki/concepts/kie-candidate-generation|KIE Candidate Generation]] - PaddleOCR/KIE candidate generation and why non-account KIE fields remain risky.
- [[wiki/concepts/manual-review-hard-gate|Manual Review Hard Gate]] - why release can pass PII/reranker gates and still be blocked.
- [[wiki/concepts/bank-name-as-metadata|Bank Name as Metadata]] - bank name extraction helps review, but is not the PII target label.
- [[wiki/concepts/union-superset-candidate|Union Superset Candidate]] - a candidate artifact that preserves baseline accepted rows and adds complementary model discoveries without overwriting.
- [[wiki/concepts/product-gated-ocr-quantization|Product-Gated OCR Quantization]] - why quantized OCR is judged by policy-gated workbook outcomes rather than generic OCR text similarity alone.
- [[wiki/concepts/outlier-residual-sidecar|Outlier Residual Sidecar]] - activation-risk and protected-channel residual correction used in the W8/W4 experiments.

## Decisions

- [[wiki/decisions/2026-04-29-threshold-10-margin-2|Threshold 10 Margin 2]] - keep threshold 10 despite threshold 9 recovering one more correct case.
- [[wiki/decisions/2026-04-29-person-split-payloads|Person-Split Payloads]] - split same-source multi-person payloads into anonymous slots.
- [[wiki/decisions/2026-04-29-reject-kie-bank-rescue|Reject KIE Bank Rescue]] - do not auto-accept account-shaped numbers from KIE `bank` fields.

## Sources

- [[wiki/sources/2026-04-29-pii-recall-doc|PII Recall Doc Source Summary]] - sanitized summary of `docs/2026-04-29-pii-candidate-generation-wiki.md`.
- [[docs/2026-04-29-pii-candidate-generation-wiki|Original 2026-04-29 Wiki-Style Report]] - project report committed before the LLM wiki split.
- [[docs/release-policy|Release Policy]] - operational release policy and hard-gate rules.
- [GS64 6bit OCR Quantization Failure Postmortem](../docs/2026-04-27-gs64-6bit-failure-postmortem.md) - ignored local source doc, not public commit target.
- [W8 All70 Union Candidate](../docs/2026-04-29-mlx8-w8-union-fallback.md) - ignored local source doc, not public commit target.

## Open Questions

- [[wiki/questions/next-ocr-candidate-recall-loop|Next OCR Candidate Recall Loop]] - next bottleneck: exact candidate generation and field localization.
- [[wiki/questions/independent-validation-set|Independent Validation Set]] - what is needed before broader performance claims.

## Maintenance Notes

- Update this index on every ingest or synthesis.
- Keep source summaries sanitized.
- Link new durable ideas into `concepts/`, not only into experiment notes.
- Keep single-checkpoint, union-candidate, and product-pipeline claims separate.
