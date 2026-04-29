---
title: Reject KIE Bank Rescue
type: decision
status: active
updated: 2026-04-29
tags: [pii, kie, safety]
---

# Reject KIE Bank Rescue

## Decision

Do not auto-rescue account-shaped numbers from KIE `bank` fields, even when holder or bank evidence looks strong.

## Why

The redacted pattern is not clean enough. Exact and non-exact cases can share the same broad evidence:

- KIE field type is `bank`
- holder evidence may be present
- account-like shape may appear
- bank evidence may be present

Accepting this class would risk wrong positives.

## Implication

KIE `bank` evidence can support document confidence, but it cannot directly select an account candidate today.

## Related Pages

- [[wiki/concepts/kie-candidate-generation|KIE Candidate Generation]]
- [[wiki/concepts/zero-fp-gate|Zero-FP Gate]]
