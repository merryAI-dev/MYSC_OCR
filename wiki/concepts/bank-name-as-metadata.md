---
title: Bank Name as Metadata
type: concept
status: active
updated: 2026-04-29
tags: [pii, bank, review]
---

# Bank Name as Metadata

Bank name extraction helps account-candidate selection and human review, but it is not the target PII label in this loop.

## Current Role

Bank name evidence can support:

- bankbook document confidence
- account candidate scoring
- review queue ergonomics
- holder/account consistency checks

## Boundary

The target label is the account-number PII candidate. Bank name is operational metadata. It should not be treated as a separate PII label target for this eval.

## Related Pages

- [[wiki/concepts/kie-candidate-generation|KIE Candidate Generation]]
- [[wiki/concepts/redacted-reranker-payload|Redacted Reranker Payload]]
