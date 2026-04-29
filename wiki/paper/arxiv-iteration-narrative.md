# arXiv Draft Notes: From Failed Uniform Low-Bit OCR Quantization to a Product-Gated W8 Union Candidate

## Working Title

Product-Gated Quantization for Private OCR Workflows: A Failure-Oriented Case Study on Korean Settlement Account Extraction

## Abstract Draft

We study post-training quantization for a private Korean settlement OCR workflow in which the model must recover bank account numbers from heterogeneous document images and update a workbook only when policy evidence is strong. A naive uniform 6-bit quantization of DeepSeek-OCR appeared close to a community MLX 8-bit reference by final workbook count, but failed a targeted recovery gate by emitting image-independent repeated text under multiple prompts. We therefore shifted from final-count-only evaluation to a product-gated protocol that includes collapse detection, policy-safe account extraction, targeted retry, masked audit artifacts, and no-overwrite union promotion. The resulting fixed candidate, `w8_all70_union_candidate`, absorbs 57 accepted rows from the deployed MLX8 primary artifact and adds 3 W8-only accepted rows, producing 60 workbook updates with 0 overwrites and 0 preserved-primary conflicts on the current sample. The result is not a standalone raw-checkpoint claim; it is a reproducible union-candidate claim showing how quantized OCR evaluation changes when deployment safety, privacy, and workflow correctness are first-class gates.

## 1. Problem Setting

The task is not generic OCR. The system must:

- read Korean settlement documents and bankbook images
- extract account-number candidates
- reject phone numbers, resident IDs, dates, customer numbers, amounts, and prompt leakage
- match evidence to the correct participant
- update the settlement workbook only for policy-eligible decisions
- produce masked review/audit artifacts suitable for internal inspection

This turns OCR quantization into a product-gated problem. A model that produces readable text but causes unsafe workbook autofill is a failure. A model that produces a few additional candidates but overwrites accepted baseline rows is also a failure.

## 2. Monday Failure: Uniform 6bit GS64 Was Not a Small Miss

The first failed path used BF16-direct uniform 6bit GS64 quantization from the local DeepSeek-OCR BF16 checkpoint. Against the MLX8 reference, the final workbook delta looked small:

| Metric | Uniform 6bit GS64 | MLX8 reference | Delta |
| --- | ---: | ---: | ---: |
| Full OCR high account detections | 10 | 19 | -9 |
| Full OCR matched names | 55 | 57 | -2 |
| Final workbook updates | 54 | 57 | -3 |
| Final skipped rows | 19 | 16 | +3 |

If we had evaluated only the final workbook count, the model might have looked salvageable. The recovery loop showed otherwise. A 99-attempt recovery run produced 0 evidence and 0 recovered accounts. A revised DeepSeek-only stop condition halted after four attempts because all four were degenerate. The model emitted repeated prompt-conditioned text rather than image-grounded OCR.

This matters because the failure is not a parser failure. Relaxing the account policy would not recover truthful evidence; it would accept hallucinated or placeholder numbers. The correct interpretation is that uniform low-bit quantization damaged free-running generation reliability enough to fail the trust gate.

## 3. Lessons From the Failure

The Monday failure changed the protocol in five ways.

First, promotion requires collapse-free generation. A candidate cannot be promoted if targeted recovery produces repeated non-OCR text, prompt echoes, or placeholder loops.

Second, the gate moved from "does quantization run" to "does the model survive the settlement workflow." Conversion success is not evidence of account-extraction reliability.

Third, W4/W6 language quantization was treated as generation-risky rather than merely lower precision. The project stopped assuming that MLP-only fixes would be enough; W4 language quant itself can perturb free-running behavior.

Fourth, calibration needed to reflect generated continuations, not only locked prompt/image inputs. The later protocol used canonical BF16-generated continuation manifests and sharded layer 1-10 activation collection to avoid both GPU watchdog pressure and calibration-distribution mismatch.

Fifth, "our model" could no longer mean a raw checkpoint unless that checkpoint passed the same gates. The safer unit became a reproducible candidate artifact with explicit claim boundaries.

## 4. Engineering Changes After Monday

### 4.1 W8 Base Instead of Uniform 6bit

The successful branch moved to a W8 language-layer base for layers 1-10 while keeping sensitive regions protected by policy: vision/projector/head and boundary language layers were not treated as generic low-bit targets. This choice prioritized generation stability over compression novelty.

### 4.2 Generated-Continuation Calibration and Sharding

The activation analysis was rebuilt around canonical BF16-generated continuations. Layer 1-10 activation capture was sharded, which reduced long-running Metal/GPU watchdog risk and made the calibration artifacts reproducible. This was a direct response to the earlier problem where calibration did not sufficiently represent free-running decode behavior.

### 4.3 Runtime-Consumed Outlier Residuals

The `protected_input_channels` plan was converted into a runtime sidecar. For W8 all70, the sidecar materialized 70 protected modules and 940 protected input channels. The runtime wrapper applies:

```text
output = base_linear(x) + x[..., protected_input_channels] @ residual_weight.T
```

The important improvement is not the existence of an outlier list. It is that the quantized runtime consumes the residual correction during inference, closing the implementation gap between risk analysis and actual model behavior.

### 4.4 Decode and Smoke Gates

The practical evaluation fixed decode controls:

- max OCR tokens: 512
- repetition penalty: 1.05
- repetition context size: 64

The free-running gate was also corrected so structured HTML/table output would not be falsely counted as degenerate repetition. This separated true model collapse from harmless markup repetition.

### 4.5 Conservative Union Promotion

The final promotion unit became `w8_all70_union_candidate`. It absorbs MLX8 primary accepted rows first, then uses W8 only for rows with no primary chosen account. The rule is no overwrite by construction.

## 5. Results

### 5.1 Monday Path

| Candidate | Updates | Main Failure |
| --- | ---: | --- |
| Uniform 6bit GS64 | 54 | targeted recovery collapse; 0 recovered evidence |
| MLX8 reference | 57 | stronger baseline |
| GS64 + MLX8 union ceiling | 58 | useful complement, but GS64 itself unsafe |

The GS64 experiment did contain one complementary row, but the model was not safe enough to use as a recovery engine.

### 5.2 W8 All70 Standalone

The raw W8 all70 checkpoint tied the deployed/default MLX8 reference at 57 final updates. Under same decode settings it was +1 over a separate MLX8 tok512/rp105 run, but it still had row regressions and a worse full-output collapse-only audit than MLX8 in that comparison.

Therefore, the raw W8 all70 checkpoint is not the claim.

### 5.3 Fixed W8 Union Candidate

| Metric | MLX8 primary | W8 all70 union candidate |
| --- | ---: | ---: |
| Absorbed primary updates | 57 | 57 |
| Native W8 additions after absorption | 0 | 3 |
| Final workbook updates | 57 | 60 |
| Primary overwrites | n/a | 0 |
| Preserved-primary conflicts | n/a | 0 |

The fixed candidate is superior on this sample under the workbook-update gate because it preserves the MLX8 accepted behavior and adds three W8-only policy-accepted rows.

## 6. What Improved Compared With Monday

| Axis | Monday uniform 6bit path | W8 all70 union candidate path |
| --- | --- | --- |
| Promotion unit | raw quantized checkpoint | reproducible union candidate artifact |
| Quantization strategy | uniform 6bit GS64 | W8 base plus protected residual sidecar |
| Calibration signal | insufficient for free-running risk | canonical BF16-generated continuations and sharded layer capture |
| Runtime protection | planning-level protection only | actual residual correction consumed by Linear wrapper |
| Decode control | unstable recovery prompts exposed collapse | fixed decode controls and collapse-aware smoke gates |
| Failure handling | more retries initially attempted | retry stops on degeneration; no policy relaxation |
| Evaluation target | final count plus recovery attempts | policy-gated workbook update, drift/smoke/audit, no overwrite |
| Result | 54 updates and generation collapse | 60 updates as union candidate, 0 overwrites, 0 conflicts |

The main scientific lesson is that quantization quality for deployment OCR cannot be inferred from static conversion or final aggregate count alone. It must be evaluated under the actual recovery and policy loop.

## 7. Privacy and Audit Design

The workflow separates local raw artifacts from reviewable artifacts:

- raw account values remain local-only in resolution/workbook artifacts needed for actual workbook writing
- review artifacts use masked account forms
- paper-facing pages should use aggregate counts and anonymized row labels
- release gates scan for raw PII leakage

This is important for any paper claim: the dataset is private, and the result should be reported as a workflow case study unless an anonymized validation set is created.

## 8. Claim Wording

Strong but accurate:

> On the current private settlement sample, `w8_all70_union_candidate` outperforms the deployed/default MLX8 primary artifact under a fixed policy-gated workbook-update metric: it absorbs 57 MLX8 accepted rows, adds 3 W8-native accepted rows, and produces 60 final updates with 0 overwrites and 0 preserved-primary conflicts.

Too strong:

> W8 all70 is a standalone replacement for MLX8.

The second claim is not supported by the current artifact because raw W8 all70 ties the deployed/default MLX8 reference and has row-level regressions.

## 9. Limitations

This is a private, small-sample workflow result. It does not establish general OCR superiority. It does not prove that the raw W8 all70 checkpoint is universally better than MLX8. It also currently requires two model outputs to construct the union candidate. The correct next step for a publishable paper would be an anonymized evaluation set, independent labels, and a separate ablation table that isolates W8 base, outlier residuals, decode controls, and union policy.

## 10. Suggested Paper Figures and Tables

1. Failure waterfall: uniform 6bit final count looked close, but targeted recovery collapsed.
2. Gate diagram: OCR decode -> candidate extraction -> policy -> targeted retry -> workbook update -> masked audit.
3. Candidate composition diagram: MLX8 accepted rows absorbed first; W8-only rows added only when primary is empty.
4. Table: Monday GS64 vs MLX8 vs W8 standalone vs W8 union candidate.
5. Table: privacy artifacts, local-only artifacts, and public/reportable artifacts.

## Sources

- [[experiments/2026-04-27-uniform-6bit-gs64-failure]]
- [[experiments/2026-04-29-w8-all70-union-candidate]]
- [[concepts/union-superset-candidate]]
- [[concepts/product-gated-ocr-quantization]]
- [[concepts/outlier-residual-sidecar]]
