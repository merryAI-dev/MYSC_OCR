# 2026-04-29 W8 All70 Union Candidate

## Summary

The fixed internal candidate is `w8_all70_union_candidate`, not the raw W8 all70 checkpoint. It absorbs the deployed/default MLX8 primary accepted rows and then adds W8 all70-only accepted rows where MLX8 had no chosen account.

## Current Result

| Metric | Value |
| --- | ---: |
| Absorbed MLX8 primary accepted rows | 57 |
| Raw W8 all70 standalone accepted rows | 57 |
| Overlap between MLX8 and W8 standalone | 54 |
| MLX8-only standalone rows | 3 |
| W8-only standalone rows | 3 |
| Final union candidate workbook updates | 60 |
| W8-native added rows after absorption | 3 |
| Primary overwrites | 0 |
| Preserved-primary conflicts | 0 |

## Why This Is Better Than the Monday Path

The Monday GS64 path tried to promote a weaker standalone checkpoint and then recover failures with retries. The W8 all70 union candidate changes both sides of that mistake:

- The base quantization is W8 for language layers instead of unstable uniform 6bit.
- Protected outlier residuals are consumed by runtime, not just recorded in a plan.
- Decode settings are fixed and auditable.
- Full OCR and targeted retry are evaluated under policy gates.
- The final promotion unit is a no-overwrite union candidate, not a raw checkpoint.

## Claim Boundary

Allowed:

> On the current settlement sample, `w8_all70_union_candidate` is better than the deployed/default MLX8 primary artifact under the fixed workbook-update gate: 60 updates versus 57, with 0 overwrites and 0 preserved-primary conflicts.

Not allowed:

> The raw W8 all70 checkpoint alone beats MLX8 as a standalone model.

The standalone checkpoint ties the deployed/default MLX8 reference at 57 final updates and has row regressions. The current advantage belongs to the fixed union candidate.

## Sources

- [W8 All70 Union Candidate](../../docs/2026-04-29-mlx8-w8-union-fallback.md)
- `/Users/boram/Downloads/hackathon_settlement_output/deepseek_ocr_gptq_ref_20260428/union_mlx8_default_primary_w8_all70_fallback_20260429/pipeline_summary.json`
- `/Users/boram/Downloads/hackathon_settlement_output/deepseek_ocr_gptq_ref_20260428/union_mlx8_default_primary_w8_all70_fallback_20260429/review/model_advantage_report.json`
- `/Users/boram/Downloads/hackathon_settlement_output/deepseek_ocr_gptq_ref_20260428/runtime_gate_w8_all70_practical_full_tok512_rp105_20260428/w8_all70_vs_mlx8_reference_comparison.json`
