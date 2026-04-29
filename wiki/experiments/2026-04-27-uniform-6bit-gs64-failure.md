# 2026-04-27 Uniform 6bit GS64 Failure

## Summary

The Monday path evaluated a BF16-direct uniform 6bit GS64 DeepSeek-OCR quantization against the Korean settlement account-number task. It was retired because it failed the reliability gate, not because the final workbook count was only slightly below MLX8.

## Baseline Result

| Metric | Uniform 6bit GS64 | MLX8 reference | Delta |
| --- | ---: | ---: | ---: |
| Full OCR members | 69 | 69 | 0 |
| Full OCR high account detections | 10 | 19 | -9 |
| Full OCR matched names | 55 | 57 | -2 |
| Targeted final filled rows | 54 | 57 | -3 |
| Final skipped rows | 19 | 16 | +3 |

The surface delta was manageable. The failure mode was not.

## Reliability Failure

Targeted recovery showed image-independent repeated output:

| Prompt | Observed failure signature |
| --- | --- |
| `copy_all_text` | repeated phrase: `eye a` |
| `account_structured_ko_v2` | repeated phrase using a repeated CJK token |
| `account_only_negative_rules` | repeated phrase: `a phone` |
| `number_inventory_ko` | zero-placeholder repetition |

The dedicated recovery loop produced:

- unresolved rows: 19
- active targets with files: 11
- missing bankbook files: 9
- 99-attempt run: 0 evidence and 0 recovered accounts
- revised 4-attempt stop: all 4 attempts degenerate

## Important Negative Lesson

The model was not merely missing small account digits. It was free-running into prompt-conditioned, image-independent text. More retry prompts, contrast variants, or policy relaxation would not fix that. In an account-number workflow, relaxing policy under generation collapse would convert a model failure into unsafe autofill.

## What This Forced

This failure forced four design changes:

1. Stop treating a small final-row delta as sufficient. Collapse-free generation became a hard gate.
2. Stop targeting uniform low-bit language quantization as the next path.
3. Evaluate quantization under product workflow gates, not only conversion success or text-level smoke tests.
4. Separate raw model claims from union/product-pipeline claims.

## Sources

- [GS64 6bit OCR Quantization Failure Postmortem](../../docs/2026-04-27-gs64-6bit-failure-postmortem.md)
- `/Users/boram/Downloads/hackathon_settlement_output/compare_bf16_6bit_vs_mlx8_full_20260427/comparison_summary.json`
- `/Users/boram/Downloads/hackathon_settlement_output/compare_bf16_6bit_vs_mlx8_full_20260427/baseline_alignment_6bit_summary.json`
