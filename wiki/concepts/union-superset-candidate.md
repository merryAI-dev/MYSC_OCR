# Union Superset Candidate

## Definition

A union superset candidate is a reproducible model-output artifact that preserves all accepted rows from a trusted primary path and adds rows from a secondary model only when the primary path has no accepted answer.

In this project, the candidate is:

`w8_all70_union_candidate = absorbed MLX8 primary accepted rows + W8 all70-only accepted rows`

## Why This Exists

The raw W8 all70 checkpoint does not yet justify a standalone replacement claim. It ties the deployed/default MLX8 reference at 57 final workbook updates and has row-level regressions. The union candidate is stronger because it changes the promotion unit: instead of asking whether W8 alone beats MLX8, it asks whether the fixed candidate can absorb MLX8 behavior and add complementary W8 discoveries without destructive overwrites.

## Current Gate

For the 2026-04-29 artifact:

| Metric | Value |
| --- | ---: |
| Absorbed MLX8 primary updates | 57 |
| Native W8 added updates | 3 |
| Union candidate updates | 60 |
| Primary overwrites | 0 |
| Preserved-primary conflicts | 0 |

## Claim Boundary

Allowed:

> On this settlement sample, the fixed `w8_all70_union_candidate` outperforms the deployed/default MLX8 primary artifact under the workbook-update gate: 60 updates versus 57, with 0 primary overwrites and 0 preserved-primary conflicts.

Not allowed:

> The raw W8 all70 checkpoint is a standalone replacement for MLX8.

That claim is not supported by the current artifact.

## Sources

- [W8 All70 Union Candidate](../../docs/2026-04-29-mlx8-w8-union-fallback.md)
- `/Users/boram/Downloads/hackathon_settlement_output/deepseek_ocr_gptq_ref_20260428/union_mlx8_default_primary_w8_all70_fallback_20260429/review/model_advantage_report.json`
