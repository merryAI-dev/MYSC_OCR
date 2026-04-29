---
title: MerryPII Wiki Log
type: log
status: active
updated: 2026-04-29
tags: [pii, log]
---

# MerryPII Wiki Log

Append-only chronology. Use a consistent heading prefix so the log is grep-friendly.

## [2026-04-29] ingest | Initial PII Recall Loop

Created the initial LLM Wiki layer from the completed PII candidate-generation loop.

Pages added:

- [[wiki/overview|Overview]]
- [[wiki/experiments/2026-04-29-pii-candidate-recall-loop|2026-04-29 PII Candidate Recall Loop]]
- [[wiki/concepts/zero-fp-gate|Zero-FP Gate]]
- [[wiki/concepts/redacted-reranker-payload|Redacted Reranker Payload]]
- [[wiki/concepts/kie-candidate-generation|KIE Candidate Generation]]
- [[wiki/concepts/manual-review-hard-gate|Manual Review Hard Gate]]
- [[wiki/concepts/bank-name-as-metadata|Bank Name as Metadata]]
- [[wiki/decisions/2026-04-29-threshold-10-margin-2|Threshold 10 Margin 2]]
- [[wiki/decisions/2026-04-29-person-split-payloads|Person-Split Payloads]]
- [[wiki/decisions/2026-04-29-reject-kie-bank-rescue|Reject KIE Bank Rescue]]
- [[wiki/questions/next-ocr-candidate-recall-loop|Next OCR Candidate Recall Loop]]
- [[wiki/questions/independent-validation-set|Independent Validation Set]]

Sanitization rule applied: no raw account numbers, real participant names, raw OCR text, local paths, or human workbook contents.

## [2026-04-29] decision | Keep Zero-FP Threshold

Threshold sweep showed `threshold=9` recovered one more correct positive but introduced one wrong positive. `threshold=8` introduced two wrong positives. The wiki records `threshold=10, margin=2` as the current zero-FP operating point.
