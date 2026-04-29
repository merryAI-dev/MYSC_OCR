---
title: KIE Candidate Generation
type: concept
status: active
updated: 2026-04-29
tags: [pii, ocr, kie]
---

# KIE Candidate Generation

KIE candidate generation uses document structure to find account-like values that plain OCR candidate extraction missed.

## What KIE Adds

KIE adds redacted evidence such as:

- field type
- confidence bucket
- layout bucket
- holder field presence
- bank name presence
- holder match status bucket

This evidence improved recall in the [[wiki/experiments/2026-04-29-pii-candidate-recall-loop|2026-04-29 loop]].

## Safety Finding

Only KIE `account_number` fields are eligible for positive scoring today.

Non-account KIE fields are risky because account-shaped numbers can appear in fields such as `bank`, `holder`, `phone`, or `amount`. The `bank` case looked tempting but was rejected in [[wiki/decisions/2026-04-29-reject-kie-bank-rescue|Reject KIE Bank Rescue]].

## Next Work

KIE is still promising if field correction improves. The next safe direction is to combine KIE bounding boxes with account-field crop OCR, not to trust broad non-account fields.

## Related Pages

- [[wiki/questions/next-ocr-candidate-recall-loop|Next OCR Candidate Recall Loop]]
- [[wiki/concepts/bank-name-as-metadata|Bank Name as Metadata]]
