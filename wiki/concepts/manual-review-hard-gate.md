---
title: Manual Review Hard Gate
type: concept
status: active
updated: 2026-04-29
tags: [pii, release, review]
---

# Manual Review Hard Gate

The Manual Review Hard Gate prevents high-confidence PII extraction from becoming automatic ledger or workbook update when operational review is required.

## Rule

If a row is in manual review scope, it is not auto-applied even if OCR or reranker confidence is high.

The row goes to a review queue. A human must confirm the original image, account value, bank metadata, holder evidence, and target row before operational use.

## Why It Matters

The [[wiki/experiments/2026-04-29-pii-candidate-recall-loop|2026-04-29 loop]] passed the PII scan and reranker gate, but release status remained blocked because manual review was pending. That is the correct outcome.

## Related Pages

- [[docs/release-policy|Release Policy]]
- [[wiki/concepts/zero-fp-gate|Zero-FP Gate]]
