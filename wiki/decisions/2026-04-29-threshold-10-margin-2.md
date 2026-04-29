---
title: Threshold 10 Margin 2
type: decision
status: active
updated: 2026-04-29
tags: [pii, reranker, threshold]
---

# Threshold 10 Margin 2

## Decision

Keep `threshold=10` and `min_margin=2` for the current deterministic feature scorer.

## Evidence

| Threshold | Margin | Correct | Wrong | Review FP | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 2 | 52 | 0 | 0 | keep |
| 9 | 2 | 53 | 1 | 0 | reject |
| 9 | 3 | 53 | 1 | 0 | reject |
| 8 | 2 | 53 | 2 | 0 | reject |

Threshold 9 recovers one more correct positive but violates the [[wiki/concepts/zero-fp-gate|Zero-FP Gate]].

## Implication

The next recall gains should come from better candidates and stronger redacted evidence, not from lowering the threshold.

## Related Pages

- [[wiki/experiments/2026-04-29-pii-candidate-recall-loop|2026-04-29 PII Candidate Recall Loop]]
- [[wiki/questions/next-ocr-candidate-recall-loop|Next OCR Candidate Recall Loop]]
