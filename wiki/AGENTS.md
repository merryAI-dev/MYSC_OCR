# LLM Wiki Maintenance Schema

This wiki is the maintained synthesis layer for the Korean settlement OCR / quantization project.

## Layers

- Raw sources: immutable experiment artifacts, logs, workbooks, and existing docs. They live outside `wiki/` or in `docs/`. Do not rewrite them from the wiki workflow.
- Wiki: LLM-maintained markdown pages under `wiki/`. These pages synthesize, cross-reference, and prepare paper/product narratives.
- Schema: this file. Update it only when the wiki conventions change.

## Page Rules

- Use stable Obsidian-style links like `[[experiments/2026-04-27-uniform-6bit-gs64-failure]]`.
- Every page should include `Sources` with local artifact/doc links.
- Do not include raw account numbers, phone numbers, resident IDs, API tokens, or private local raw OCR text.
- For arXiv/public-draft pages, avoid participant names. Use anonymized row labels or aggregate counts.
- Internal-only pages may reference local artifact paths and masked accounts when the source artifact already uses masking.
- Distinguish three claim types:
  - `single-checkpoint`: a standalone model claim.
  - `union-candidate`: a reproducible artifact that composes model outputs.
  - `product-pipeline`: policy, PII, workflow, and audit claims.

## Ingest Workflow

1. Read the new source or artifact summary.
2. Add a chronological entry to `log.md`.
3. Update `index.md`.
4. Update the relevant experiment/concept pages.
5. If the source changes the paper narrative, update `paper/arxiv-iteration-narrative.md`.

## Query Workflow

1. Read `index.md` first.
2. Read the most relevant pages.
3. Answer with citations to wiki pages and local artifacts.
4. If the answer creates durable synthesis, add or update a wiki page and append to `log.md`.

## Lint Workflow

Periodically check for:

- pages missing sources
- stale claims superseded by newer artifacts
- contradiction between standalone model claims and union-candidate claims
- public-draft pages that leak raw PII
- orphan concepts that should be linked from `index.md`
