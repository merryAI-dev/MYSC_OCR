---
title: Next OCR Candidate Recall Loop
type: question
status: active
updated: 2026-04-29
tags: [pii, ocr, roadmap]
---

# Next OCR Candidate Recall Loop

## Question

How can MerryPII recover the remaining missed positives without violating the [[wiki/concepts/zero-fp-gate|Zero-FP Gate]]?

## Current Miss Profile

| Category | Count |
| --- | ---: |
| exact candidate exists but safety rejected or reviewed | 6 |
| candidates exist but no exact account candidate | 6 |

## Hypothesis

The next meaningful gain comes from better OCR candidate generation and account-field localization, not from general reranker threshold changes.

## Candidate Actions

1. Generate account-field crops from reliable KIE/layout boxes.
2. Run targeted crop OCR with strict local-only raw handling.
3. Add redacted crop evidence to payloads.
4. Re-evaluate at `threshold=10, margin=2`.
5. Reject any feature that creates wrong positives.

## Related Pages

- [[wiki/concepts/kie-candidate-generation|KIE Candidate Generation]]
- [[wiki/decisions/2026-04-29-threshold-10-margin-2|Threshold 10 Margin 2]]
- [[wiki/questions/independent-validation-set|Independent Validation Set]]
