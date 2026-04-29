# MerryPII

MerryPII is a local-first Korean settlement document OCR and bank-account PII candidate extraction toolkit.

The core design goal is strict separation between:

- local-only raw artifacts needed to recover account numbers
- redacted candidate features that can be evaluated or reranked safely
- release gates that prevent ledger/workbook auto-apply unless risk criteria are met

## Scope

This project can assist with:

- matching settlement documents to a roster
- extracting bank-account candidates from OCR/KIE outputs
- building redacted reranker payloads
- comparing candidate-selection policies against human labels
- scanning release bundles for PII leaks

This project does not treat high-confidence OCR as permission to update a ledger automatically. Manual-review hard gates and release gates remain separate product controls.

## Quick Start

```bash
python3 -m pip install -e .
python3 -m pytest -q
```

Run the CLI with a local config:

```bash
python3 -m settlement_tool.cli --config config.json analyze --ocr-backend none
python3 -m settlement_tool.cli --config config.json run --mode copy --ocr-backend tesseract
```

Create `config.json` from `config.example.json` and point it at local workbooks, ZIP files, and output directories. Do not commit `config.json`.

## PII Contract

Redacted payloads must not contain:

- raw account numbers
- person names
- raw OCR text
- source file names or local paths
- human-label answers

Raw sidecars such as `candidate_raw_map_local.jsonl`, `candidate_features_local.jsonl`, and `kie_candidates_local.csv` are local-only and must not be included in release bundles or commits.

## Local Reranker Flow

Build local features and redacted payloads:

```bash
python3 scripts/build_codex_teacher_distill.py \
  --input-csv /path/to/full_ocr.csv \
  --input-csv /path/to/targeted_retry.csv \
  --input-kie-csv /path/to/kie_candidates_local.csv \
  --output-dir /path/to/payloads \
  --backend mixed_candidate_generation
```

Run the deterministic zero-FP-oriented scorer:

```bash
python3 scripts/local_feature_score_rerank.py \
  --payload-jsonl /path/to/payloads/candidate_features_redacted.jsonl \
  --output-jsonl /path/to/decisions.jsonl \
  --threshold 10 \
  --min-margin 2
```

Apply decisions through the local raw map:

```bash
python3 scripts/apply_openai_reranker_decisions.py \
  --source-workbook /path/to/source.xlsx \
  --raw-map-jsonl /path/to/payloads/candidate_raw_map_local.jsonl \
  --decisions-jsonl /path/to/decisions.jsonl \
  --output-dir /path/to/resolution \
  --manual-review-workbook /path/to/source.xlsx
```

Evaluate against a human-label workbook:

```bash
python3 scripts/evaluate_human_workbook_labels.py \
  --human-workbook /path/to/human_labels.xlsx \
  --resolution-csv /path/to/resolution/account_resolution_candidates.csv \
  --output-dir /path/to/eval
```

Build the release gate report:

```bash
python3 scripts/build_release_gate.py \
  --resolution-csv /path/to/resolution/account_resolution_candidates.csv \
  --reranker-eval /path/to/eval/human_label_eval.json \
  --bundle-path /path/to/release_bundle \
  --output-dir /path/to/release_gate \
  --sensitive-workbook /path/to/human_labels.xlsx
```

## Current Verified Loop

The 2026-04-29 loop improved the zero-FP account candidate extraction eval from `42/64` to `52/64` correct positives while keeping:

- `wrong_positive=0`
- `review_false_positive=0`
- `safe_selection_precision=1.0`
- PII release scan passing

The release remained blocked by pending manual review, which is expected. See [docs/2026-04-29-pii-candidate-generation-wiki.md](docs/2026-04-29-pii-candidate-generation-wiki.md).

## LLM Wiki

The maintained project knowledge base lives in [wiki/index.md](wiki/index.md).

Use it as the first stop for accumulated decisions, experiments, and open questions. The schema for future LLM agents is [AGENTS.md](AGENTS.md).

## Release Policy

See [docs/release-policy.md](docs/release-policy.md).

Short version:

- manual-review rows are never auto-applied
- lowering threshold is not acceptable if it creates any wrong positive
- redacted reranker payloads are allowed
- local raw maps are not release artifacts
- bank name is review metadata, not the target PII label
