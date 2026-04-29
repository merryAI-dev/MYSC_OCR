# MerryPII LLM Wiki Schema

This repository uses an LLM-maintained wiki pattern. Future agents should treat Markdown knowledge files as a persistent, compounding artifact, not as disposable chat notes.

## Layers

1. Raw sources
   - Raw source files, OCR outputs, workbooks, ZIPs, and local eval artifacts are immutable local inputs.
   - They may be read for analysis when available, but must not be committed when they contain PII, local paths, or raw account data.

2. Wiki
   - The wiki lives under `wiki/`.
   - The LLM owns this layer: create pages, update cross-references, revise stale summaries, and append to the log.
   - Humans read and steer the wiki; the LLM does maintenance.

3. Schema
   - This file is the schema.
   - Update it when the wiki conventions change.

## Privacy Rules

Never add these to committed Markdown:

- raw bank account numbers
- participant names from real eval data
- raw OCR text containing PII
- local file paths
- human-label workbook contents
- source filenames if they identify a person

Allowed:

- aggregate metrics
- redacted artifact names
- abstract aliases such as `<eval_root>`
- policy names, script names, and page links
- synthetic fixture values inside tests

If a source contains PII, summarize it into aggregate or categorical form only.

## Directory Convention

- `wiki/index.md`: content-oriented catalog of wiki pages
- `wiki/log.md`: append-only chronological activity log
- `wiki/overview.md`: current high-level synthesis
- `wiki/concepts/`: durable concepts and policies
- `wiki/decisions/`: dated decisions and rejected alternatives
- `wiki/experiments/`: dated experiment narratives and outcomes
- `wiki/sources/`: source summaries, sanitized
- `wiki/questions/`: open questions and next research loops

## Page Format

Every wiki page should start with YAML frontmatter:

```yaml
---
title: Short Title
type: concept | decision | experiment | source | question | overview
status: draft | active | superseded
updated: YYYY-MM-DD
tags: [pii, ocr]
---
```

Use wiki links liberally:

- `[[wiki/concepts/zero-fp-gate|Zero-FP Gate]]`
- `[[wiki/experiments/2026-04-29-pii-candidate-recall-loop|PII Candidate Recall Loop]]`

## Ingest Workflow

When ingesting a new source:

1. Read the source.
2. Identify whether it contains PII or local paths.
3. Write or update a sanitized source summary under `wiki/sources/`.
4. Update relevant concept, decision, experiment, and question pages.
5. Update `wiki/index.md`.
6. Append one entry to `wiki/log.md`.
7. Run a privacy scan before committing.

## Query Workflow

When answering a project question:

1. Read `wiki/index.md` first.
2. Read the relevant linked pages.
3. Answer from compiled wiki knowledge and cite page links.
4. If the answer creates a useful synthesis, file it back into the wiki and update the log.

## Lint Workflow

Periodically check for:

- contradictions between pages
- stale metrics superseded by newer runs
- orphan pages with no inbound links
- important concepts mentioned without their own page
- policy claims not backed by a decision page
- PII or local path leakage

## Current Operating Principle

MerryPII prioritizes zero wrong positives over higher recall. A recall gain is not accepted if it creates `wrong_positive > 0` or `review_false_positive > 0`.
