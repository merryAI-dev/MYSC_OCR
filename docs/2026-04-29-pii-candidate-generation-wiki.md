# PII Candidate Generation Recall Wiki

Date: 2026-04-29  
Status: zero-FP candidate extraction loop verified, ledger auto-apply still gated

## TL;DR

The loop improved the fully automatic Korean bank-account PII candidate extraction engine from `42/64` to `52/64` correct positives while keeping:

| Gate | Result |
| --- | ---: |
| `wrong_positive` | 0 |
| `review_false_positive` | 0 |
| `safe_selection_precision` | 1.0 |
| positive recall | 0.8125 |
| PII release scan | pass |
| reranker quality gate | pass |
| release status | blocked by pending manual review |

The useful gain came from candidate generation and payload grouping, not from lowering the decision threshold. Lower thresholds recovered more rows but immediately introduced wrong positives.

## Product Boundary

This loop is a PII extraction and candidate-selection engine. It is not a ledger auto-apply release.

The defensible release claim is:

> Redacted candidate selection can automatically identify 52 of 64 human-positive account labels with zero wrong positives on this eval set. Any actual workbook or ledger write remains blocked until the manual review queue is confirmed.

## Baseline

Control artifact alias:

`<eval_root>/no_candidate_retry_full27_resume_v2_safetyfix`

| Metric | Baseline |
| --- | ---: |
| human positives | 64 |
| correct positives | 42 |
| missed positives | 22 |
| wrong positives | 0 |
| review false positives | 0 |
| positive recall | 0.65625 |

Interpretation: the previous bottleneck was not the reranker. Most failures had no usable account candidate reaching the selection layer.

## Loop Timeline

### 1. KIE Candidate Generation

Added a PaddleOCR KIE retry path that writes two classes of artifact:

- local-only raw sidecar: `kie_candidates_local.csv`
- redacted evidence: `kie_evidence_redacted.jsonl`

Important contract:

- raw account strings, raw OCR text, names, source paths, and human workbook values stay local-only
- redacted payloads carry only shape, field type, layout bucket, confidence bucket, source kind, and boolean evidence

First KIE result improved recall, but raw KIE field labels were not safe enough for automatic acceptance. In particular, KIE `bank` field rows sometimes contained account-shaped numbers. Those rows looked useful but created false-positive risk.

Decision: only KIE `account_number` is eligible for positive scoring. Non-account KIE fields are hard-risk unless a future feature proves a narrower safe exception.

### 2. Holder and Bank Evidence

Added redacted holder/bank evidence:

- holder field present
- holder match status bucket
- bank name present
- bankbook document confidence bucket

This helped separate high-signal account candidates from generic numeric strings without sending names to the reranker.

Important policy decision: bank name extraction is operational metadata, not a PII label target. It helps review and scoring, but it is not itself the account-number label.

### 3. Release Gate Expansion

Expanded release scanning to cover:

- redacted reranker payloads
- decisions/reports
- KIE redacted evidence
- local-only KIE raw artifacts
- sensitive workbook terms without echoing matched terms in reports

The release gate intentionally blocks raw local artifacts such as candidate raw maps and KIE candidate CSVs from any release bundle.

### 4. Person-Split Payload Grouping

Found a structural loss: a small number of redacted payloads grouped candidates for more than one participant under the same `source_id`. The reranker contract chooses one candidate per payload, so the second participant could be silently suppressed even when their exact candidate existed.

Fix:

- keep names only in local raw mapping
- split same-source multi-person payloads into anonymous `person_N` slots
- do not include real names or local paths in the redacted payload

Impact:

| Run | Correct | Wrong | Review FP | Recall |
| --- | ---: | ---: | ---: | ---: |
| before person split | 49 | 0 | 0 | 0.765625 |
| after person split | 51 | 0 | 0 | 0.796875 |

### 5. Visible-Numbers Rescue

A targeted retry path sometimes marked an otherwise good account candidate with `wrong_field_context`, even though redacted evidence showed:

- targeted retry source
- visible-numbers prompt
- account-number field evidence
- bank evidence present
- bank-style hyphenation
- repeated consensus across variants
- no prompt leakage
- no phone-like shape

Fix: add a narrow rescue for this exact condition and keep generic wrong-field cases as hard risk.

Impact:

| Run | Correct | Wrong | Review FP | Recall |
| --- | ---: | ---: | ---: | ---: |
| after person split | 51 | 0 | 0 | 0.796875 |
| after visible-numbers rescue | 52 | 0 | 0 | 0.8125 |

### 6. Structured Retry Repeat Check

Threshold sweep showed a tempting path:

| Threshold | Margin | Correct | Wrong | Review FP | Gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 2 | 52 | 0 | 0 | pass, manual review pending |
| 9 | 2 | 53 | 1 | 0 | blocked |
| 9 | 3 | 53 | 1 | 0 | blocked |
| 8 | 2 | 53 | 2 | 0 | blocked |

Analysis of threshold 9 showed that `account_structured_ko` prompt-leakage candidates with four repeated sightings were clean in this eval, while three repeated sightings already mixed exact and non-exact candidates.

Implemented a narrow +1 score for the four-repeat pattern, but it did not increase person-level recall at threshold 10. It stays as a documented conservative feature, not a release claim driver.

## Final Verified Result

Final artifact alias:

`<eval_root>/experiment_paddleocr_kie_holder_personsplit_dual_rescue_20260429`

Comparison artifact alias:

`<eval_root>/compare_baseline_vs_final_52_zero_fp_20260429`

| Metric | Baseline | Final |
| --- | ---: | ---: |
| human positives | 64 | 64 |
| correct positives | 42 | 52 |
| missed positives | 22 | 12 |
| wrong positives | 0 | 0 |
| review false positives | 0 | 0 |
| positive recall | 0.65625 | 0.8125 |
| safe selection precision | 1.0 | 1.0 |

Release gate interpretation:

- PII scan passed
- reranker gate passed
- release remains blocked by the manual autofill review queue

## Remaining Misses

There are 12 missed positives after the final zero-FP run.

| Remaining miss category | Count |
| --- | ---: |
| exact candidate exists but safety rejected or reviewed | 6 |
| candidates exist but no exact account candidate | 6 |

The next bottleneck is still OCR/candidate generation and account-field localization. Blind threshold lowering is not acceptable because it introduces wrong positives.

## Failed or Rejected Ideas

| Idea | Result | Decision |
| --- | --- | --- |
| Lower threshold to 9 | +1 correct, +1 wrong positive | reject |
| Lower threshold to 8 | no additional correct over threshold 9, more wrong positives | reject |
| Trust KIE `bank` field when holder matched | exact and non-exact rows share the same redacted pattern | reject |
| Treat bank name as a label target | bank name is useful metadata but not the PII target | reject |
| Send raw candidates to a reranker | violates PII contract | reject |

## PII Handling Rules Learned

1. Redacted payloads must be sufficient for selection, but never contain raw account numbers, names, raw OCR text, file paths, or human-label answers.
2. Local raw mapping is allowed only as a local sidecar and must not enter release bundles.
3. KIE evidence is useful only when it is reduced to field buckets, confidence buckets, and layout buckets.
4. Manual-review rows remain hard-gated even when the reranker is confident.
5. Any threshold or feature change must be evaluated against `wrong_positive=0` and `review_false_positive=0`, not only recall.

## Next Useful Loop

Priority order:

1. Improve account-field candidate generation for the 6 cases where no exact candidate exists.
2. Add account-field crop OCR once bounding boxes are reliable enough.
3. Improve KIE field correction so account-shaped values inside non-account KIE fields can be safely reclassified only with stronger local evidence.
4. Build a separate independent human-labeled validation set before claiming general product performance.
5. Keep release status blocked until manual review queue confirmation is represented in the release gate.

## Verification

Fresh verification command:

```bash
python3 -m pytest -q
```

Observed result:

```text
181 passed
```

