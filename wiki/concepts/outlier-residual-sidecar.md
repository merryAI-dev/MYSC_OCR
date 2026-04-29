# Outlier Residual Sidecar

## Summary

The outlier residual sidecar is the mechanism used to make protected input channels operational rather than merely descriptive. A base quantized linear layer computes its usual output, and selected high-risk input channels receive a BF16 residual correction:

`output = base_linear(x) + x[..., protected_input_channels] @ residual_weight.T`

This lets the candidate keep most weights in the quantized base while restoring selected channel contributions where quantization error or activation risk is highest.

## Current W8 All70 Sidecar

The W8 all70 sidecar materialized residuals for:

| Field | Value |
| --- | ---: |
| protected modules | 70 |
| protected input channels | 940 |
| base bits | 8 |
| group size | 128 |

The sidecar summary reports near-zero selected-column MSE after residual materialization for protected columns. That does not by itself prove OCR quality, but it closes the earlier implementation gap where `protected_input_channels` existed in planning artifacts without being consumed by runtime inference.

## Lesson

The key improvement was not just "protect outliers" as a paper idea. The important engineering step was making the runtime wrapper actually consume the sidecar, then evaluating the resulting model under decode, drift, smoke, full OCR, and workbook gates.

## Sources

- `/Users/boram/Downloads/hackathon_settlement_output/deepseek_ocr_gptq_ref_20260428/outlier_residual_w8_all70_sidecar_summary_20260428.json`
- `/Users/boram/MLX-Video-OCR-DeepSeek-Apple-Silicon/scripts/outlier_residual_runtime.py`
- `/Users/boram/MLX-Video-OCR-DeepSeek-Apple-Silicon/scripts/materialize_outlier_residual_sidecar.py`
