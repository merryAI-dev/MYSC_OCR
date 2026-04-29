---
title: MerryPII Wiki Log
type: log
status: active
updated: 2026-04-29
tags: [pii, log, quantization]
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

## [2026-04-29] ingest | Monday GS64 failure and W8 all70 union candidate

Added the quantization/recovery branch of the wiki and an arXiv-oriented iteration narrative.

Pages added:

- [[wiki/paper/arxiv-iteration-narrative|arXiv Iteration Narrative]]
- [[wiki/experiments/2026-04-27-uniform-6bit-gs64-failure|2026-04-27 Uniform 6bit GS64 Failure]]
- [[wiki/experiments/2026-04-29-w8-all70-union-candidate|2026-04-29 W8 All70 Union Candidate]]
- [[wiki/concepts/union-superset-candidate|Union Superset Candidate]]
- [[wiki/concepts/product-gated-ocr-quantization|Product-Gated OCR Quantization]]
- [[wiki/concepts/outlier-residual-sidecar|Outlier Residual Sidecar]]

Sources ingested:

- `docs/2026-04-27-gs64-6bit-failure-postmortem.md`
- `docs/2026-04-29-mlx8-w8-union-fallback.md`
- `docs/2026-04-29-pii-candidate-generation-wiki.md`
- `/Users/boram/Downloads/hackathon_settlement_output/compare_bf16_6bit_vs_mlx8_full_20260427/comparison_summary.json`
- `/Users/boram/Downloads/hackathon_settlement_output/compare_bf16_6bit_vs_mlx8_full_20260427/fallback_union_summary.json`
- `/Users/boram/Downloads/hackathon_settlement_output/deepseek_ocr_gptq_ref_20260428/union_mlx8_default_primary_w8_all70_fallback_20260429/pipeline_summary.json`
- `/Users/boram/Downloads/hackathon_settlement_output/deepseek_ocr_gptq_ref_20260428/union_mlx8_default_primary_w8_all70_fallback_20260429/review/model_advantage_report.json`

Main synthesis:

- The Monday GS64 path failed due to generation reliability collapse, not only a small workbook-count deficit.
- The W8 all70 standalone checkpoint is not the claim. The fixed internal candidate is `w8_all70_union_candidate`.
- `w8_all70_union_candidate` absorbs 57 MLX8 primary accepted rows, adds 3 W8-native accepted rows, and produces 60 final workbook updates with 0 overwrites and 0 preserved-primary conflicts on the current sample.
