---
title: Redacted Reranker Payload
type: concept
status: active
updated: 2026-04-29
tags: [pii, reranker, privacy]
---

# Redacted Reranker Payload

The redacted reranker payload is the privacy boundary between local OCR extraction and any model-based candidate selection.

## Contract

Allowed in the redacted payload:

- `source_id`
- `candidate_id`
- account shape, digit count, group count, hyphen count
- boolean context flags
- risk flags
- field/layout buckets
- KIE field type and confidence bucket
- consensus counts

Forbidden:

- raw account numbers
- real participant names
- raw OCR text
- local paths or source filenames
- human-label answers

## Local Raw Map

Raw account recovery happens through a local-only sidecar:

`candidate_raw_map_local.jsonl`

The reranker returns only `candidate_id`; local apply code resolves that ID to the raw value. This keeps the scoring layer PII-minimized.

## Related Pages

- [[wiki/concepts/zero-fp-gate|Zero-FP Gate]]
- [[wiki/decisions/2026-04-29-person-split-payloads|Person-Split Payloads]]
