# Product-Gated OCR Quantization

## Thesis

For this settlement task, quantized OCR quality is not measured by generic OCR readability alone. The relevant product gate is whether a model can safely produce account-number candidates that pass policy checks and update the workbook without accepting phone numbers, IDs, dates, prompt echoes, repeated placeholders, or wrong-person evidence.

## Gate Stack

The project evaluates candidate models through a layered gate:

1. OCR generation does not collapse into repeated or prompt-echo text.
2. Account candidates are extracted from OCR output.
3. Policy rejects phone-like, ID-like, date-like, customer-number-like, and weak-context values.
4. Targeted retry is used only for unresolved rows.
5. Workbook writing occurs only for auto-fill eligible decisions.
6. Manual/review artifacts stay masked and auditable.
7. Union candidates may add rows only where the absorbed primary has no chosen account.

## Why Monday Failed This Gate

The uniform 6bit GS64 path was not rejected because it was merely three workbook rows behind MLX8. It was rejected because targeted recovery emitted image-independent repeated text and produced 0 credible recovery evidence after retry. In this gate, generation collapse is a safety failure.

## Why the W8 Union Candidate Passes the Current Internal Gate

The fixed `w8_all70_union_candidate` passes as an internal candidate because the accepted MLX8 rows are preserved, W8 contributes three additional policy-accepted rows, and the final artifact records 0 overwrites and 0 conflicts.

## Sources

- [[experiments/2026-04-27-uniform-6bit-gs64-failure]]
- [[experiments/2026-04-29-w8-all70-union-candidate]]
- [Release Policy](../../docs/release-policy.md)
