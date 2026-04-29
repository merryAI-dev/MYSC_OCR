---
title: MerryPII Overview
type: overview
status: active
updated: 2026-04-29
tags: [pii, ocr, reranker]
---

# MerryPII Overview

MerryPII is a local-first PII candidate extraction and candidate-selection system for Korean settlement bank-account workflows.

The core product boundary is:

> Extract and select bank-account PII candidates automatically, but do not auto-apply to a ledger or workbook when release gates require manual confirmation.

The current verified loop is summarized in [[wiki/experiments/2026-04-29-pii-candidate-recall-loop|2026-04-29 PII Candidate Recall Loop]].

## Current State

The current best eval result:

| Metric | Value |
| --- | ---: |
| correct positives | 52/64 |
| positive recall | 0.8125 |
| wrong positives | 0 |
| review false positives | 0 |
| safe selection precision | 1.0 |
| PII release scan | pass |
| reranker gate | pass |
| release status | blocked by manual review queue |

The system is therefore useful as a zero-FP candidate extraction engine, but not yet a fully automatic ledger update product.

## Architecture Thesis

The system has three separate layers:

1. Candidate generation
   - OCR and KIE produce account-like candidates.
   - This is currently the main bottleneck.

2. Candidate selection
   - Redacted features are scored or reranked.
   - The current deterministic scorer is safer than lowering thresholds.

3. Release control
   - Manual review hard gates and PII leak scans decide whether any output can be used operationally.

## Current Operating Principle

MerryPII follows the [[wiki/concepts/zero-fp-gate|Zero-FP Gate]]. Recall improvements are rejected if they create any wrong positive or review false positive.

## What Changed Recently

The biggest improvements came from:

- [[wiki/concepts/kie-candidate-generation|KIE Candidate Generation]]
- [[wiki/decisions/2026-04-29-person-split-payloads|Person-Split Payloads]]
- a narrow visible-numbers rescue recorded in [[wiki/experiments/2026-04-29-pii-candidate-recall-loop|the loop narrative]]

The strongest rejected path is documented in [[wiki/decisions/2026-04-29-threshold-10-margin-2|Threshold 10 Margin 2]].

## Next Direction

The next useful loop is [[wiki/questions/next-ocr-candidate-recall-loop|Next OCR Candidate Recall Loop]].

Before broader claims, build [[wiki/questions/independent-validation-set|Independent Validation Set]].
