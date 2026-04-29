---
title: PII Recall Doc Source Summary
type: source
status: active
updated: 2026-04-29
tags: [pii, source, recall]
---

# PII Recall Doc Source Summary

Source: [[docs/2026-04-29-pii-candidate-generation-wiki|Original 2026-04-29 Wiki-Style Report]]

## Sanitized Summary

The source records a zero-FP PII candidate generation loop. It documents candidate generation, KIE evidence, payload redaction, release gating, threshold sweeps, and remaining misses.

## Extracted Claims

- The baseline had 42 correct positives out of 64 human positives.
- The final zero-FP run had 52 correct positives out of 64 human positives.
- Threshold lowering increased recall but introduced wrong positives.
- KIE helped recall, but non-account KIE fields remain unsafe for automatic rescue.
- Release passed PII/reranker gates but remained blocked by manual review.

## Pages Updated From This Source

- [[wiki/overview|Overview]]
- [[wiki/experiments/2026-04-29-pii-candidate-recall-loop|2026-04-29 PII Candidate Recall Loop]]
- [[wiki/concepts/zero-fp-gate|Zero-FP Gate]]
- [[wiki/concepts/redacted-reranker-payload|Redacted Reranker Payload]]
- [[wiki/concepts/kie-candidate-generation|KIE Candidate Generation]]
- [[wiki/decisions/2026-04-29-threshold-10-margin-2|Threshold 10 Margin 2]]

## Sanitization

This source summary intentionally omits raw accounts, real names, source filenames, local paths, and human workbook contents.
