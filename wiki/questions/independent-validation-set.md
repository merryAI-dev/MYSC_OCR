---
title: Independent Validation Set
type: question
status: active
updated: 2026-04-29
tags: [pii, eval, validation]
---

# Independent Validation Set

## Question

What validation is needed before MerryPII can make broader product performance claims?

## Current Limitation

The current eval is useful, but it is still one dataset and one workflow. Some labels and policies were developed while building the system. That is enough for internal iteration, but not enough for broad claims.

## Needed

- An independent human-labeled validation set.
- Clear separation between tuning data and final evaluation data.
- The same metrics as the current loop:
  - correct positives
  - missed positives
  - wrong positives
  - review false positives
  - safe selection precision
  - positive recall
- A fresh PII release scan over redacted artifacts.

## Success Bar

Broader product claims require preserving:

- `wrong_positive=0`
- `review_false_positive=0`
- no raw PII in release artifacts

Recall improvements are secondary to those gates.

## Related Pages

- [[wiki/concepts/zero-fp-gate|Zero-FP Gate]]
- [[wiki/overview|Overview]]
