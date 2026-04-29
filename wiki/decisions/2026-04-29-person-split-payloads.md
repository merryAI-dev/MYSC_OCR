---
title: Person-Split Payloads
type: decision
status: active
updated: 2026-04-29
tags: [pii, reranker, payload]
---

# Person-Split Payloads

## Decision

When a single `source_id` contains candidates for multiple participants, split the redacted payload into anonymous person slots.

## Problem

The reranker selects one candidate per payload. If one payload contains candidates for multiple people, the correct candidate for one person can be suppressed by another person's candidate.

## Fix

- Keep real names only in local raw mapping.
- Emit redacted source IDs with anonymous `person_N` suffixes.
- Do not include names, local paths, source filenames, or raw accounts in the redacted payload.

## Impact

| Run | Correct | Wrong | Review FP | Recall |
| --- | ---: | ---: | ---: | ---: |
| before person split | 49 | 0 | 0 | 0.765625 |
| after person split | 51 | 0 | 0 | 0.796875 |

## Related Pages

- [[wiki/concepts/redacted-reranker-payload|Redacted Reranker Payload]]
- [[wiki/experiments/2026-04-29-pii-candidate-recall-loop|2026-04-29 PII Candidate Recall Loop]]
