---
title: Zero-FP Gate
type: concept
status: active
updated: 2026-04-29
tags: [pii, safety, evaluation]
---

# Zero-FP Gate

The Zero-FP Gate is the main safety standard for MerryPII candidate selection.

An experiment is acceptable only when:

- `wrong_positive=0`
- `review_false_positive=0`
- `safe_selection_precision=1.0`

Recall matters only after these are satisfied.

## Why

The extracted value is bank-account PII. A wrong positive can map a payment or ledger row to the wrong account. That is a product and compliance failure, not just a model error.

## Current Consequence

The [[wiki/experiments/2026-04-29-pii-candidate-recall-loop|2026-04-29 loop]] rejected threshold 9 even though it recovered one more correct positive, because it also created one wrong positive.

## Related Pages

- [[wiki/decisions/2026-04-29-threshold-10-margin-2|Threshold 10 Margin 2]]
- [[wiki/concepts/manual-review-hard-gate|Manual Review Hard Gate]]
