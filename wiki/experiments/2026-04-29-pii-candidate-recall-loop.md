---
title: 2026-04-29 PII Candidate Recall Loop
type: experiment
status: active
updated: 2026-04-29
tags: [pii, ocr, kie, reranker, zero-fp]
---

# 2026-04-29 PII Candidate Recall Loop

This experiment improved MerryPII's fully automatic account candidate extraction eval while preserving the [[wiki/concepts/zero-fp-gate|Zero-FP Gate]].

## Question

Can recall improve beyond the 42/64 baseline without increasing:

- `wrong_positive`
- `review_false_positive`

## Result

| Metric | Baseline | Final |
| --- | ---: | ---: |
| human positives | 64 | 64 |
| correct positives | 42 | 52 |
| missed positives | 22 | 12 |
| wrong positives | 0 | 0 |
| review false positives | 0 | 0 |
| positive recall | 0.65625 | 0.8125 |
| safe selection precision | 1.0 | 1.0 |

The release gate still blocked final use because manual review was pending. This is expected under [[wiki/concepts/manual-review-hard-gate|Manual Review Hard Gate]].

## What Worked

### KIE candidates

PaddleOCR/KIE added document-structure evidence and recovered candidates the prior OCR loop missed. The useful part was not raw text; it was redacted field/layout evidence. See [[wiki/concepts/kie-candidate-generation|KIE Candidate Generation]].

### Holder and bank evidence

Holder and bank evidence helped distinguish account candidates from generic numeric strings without exposing names or raw OCR.

### Person-split payloads

Some payloads contained candidates for more than one participant under one `source_id`. Since the reranker chooses one candidate per payload, this suppressed correct candidates. Splitting into anonymous person slots improved recall from 49/64 to 51/64. See [[wiki/decisions/2026-04-29-person-split-payloads|Person-Split Payloads]].

### Visible-numbers rescue

A narrow visible-numbers pattern recovered one additional correct candidate:

- targeted retry source
- visible-numbers prompt
- account-field evidence
- bank evidence present
- bank-style hyphenation
- repeated consensus
- no prompt leakage
- no phone-like shape

Generic wrong-field context remains hard risk.

## What Failed

### Lowering threshold

Threshold sweep:

| Threshold | Margin | Correct | Wrong | Review FP | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 2 | 52 | 0 | 0 | keep |
| 9 | 2 | 53 | 1 | 0 | reject |
| 9 | 3 | 53 | 1 | 0 | reject |
| 8 | 2 | 53 | 2 | 0 | reject |

See [[wiki/decisions/2026-04-29-threshold-10-margin-2|Threshold 10 Margin 2]].

### KIE bank-field rescue

KIE `bank` field rows sometimes contain account-shaped numbers, but exact and non-exact cases share similar redacted features. Auto-rescuing these would be unsafe. See [[wiki/decisions/2026-04-29-reject-kie-bank-rescue|Reject KIE Bank Rescue]].

## Remaining Misses

| Category | Count |
| --- | ---: |
| exact candidate exists but was rejected/reviewed by safety | 6 |
| candidates exist but no exact account candidate | 6 |

The first category needs stronger safety evidence. The second category needs better OCR/candidate generation.

## Compiled Learning

The bottleneck is still candidate generation and field localization, not general-purpose reranking. Lowering the threshold is not a safe substitute for generating the right candidate.

## Links

- [[wiki/overview|Overview]]
- [[wiki/concepts/redacted-reranker-payload|Redacted Reranker Payload]]
- [[wiki/questions/next-ocr-candidate-recall-loop|Next OCR Candidate Recall Loop]]
