# Crop OCR Recall Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover additional Korean bank-account PII candidates by adding local-only account-field crop OCR while preserving `wrong_positive=0`, `review_false_positive=0`, and redacted release artifacts.

**Architecture:** Add a local miss analyzer first so every recall loop explains its failure modes without exposing PII. Then add a crop evidence layer that turns KIE/layout boxes into local crop OCR candidates plus redacted crop/layout evidence. Feed crop candidates through the existing teacher-distill, redacted payload, deterministic scorer, eval, and release-gate pipeline without lowering `threshold=10, min_margin=2`.

**Tech Stack:** Python 3.10+, pytest, PIL/Pillow, optional `pdftoppm`, existing PaddleOCR/KIE outputs, CSV/JSONL artifacts, `build_codex_teacher_distill.py`, `local_feature_score_rerank.py`, `run_candidate_recall_experiment.py`, `build_release_gate.py`.

---

## Current Baseline For This Plan

- Current zero-FP best: `52/64`
- Positive recall: `0.8125`
- Wrong positives: `0`
- Review false positives: `0`
- Remaining miss profile:
  - exact candidate exists but safety rejected or reviewed: `6`
  - candidates exist but no exact candidate: `6`

Do not lower the threshold to chase recall. Threshold sweeps already showed wrong positives at lower thresholds. The plan targets better candidate generation and stronger redacted evidence.

## File Structure

- Create `scripts/analyze_candidate_misses.py`
  - Reads human eval details, redacted payloads, decisions, and local raw map.
  - Emits aggregate miss categories only.
  - Never prints names, raw account values, source paths, or raw OCR text.

- Create `settlement_tool/crop_evidence.py`
  - Owns bbox expansion, crop layout bucket normalization, local crop row normalization, and redacted crop evidence.
  - Does not read or write files.

- Create `scripts/run_account_field_crop_retry.py`
  - Reads retry target CSVs plus KIE local CSVs.
  - Creates local crop images under an output directory.
  - Runs OCR over crops only when an OCR backend is available.
  - Supports fixture mode and plan-only mode for tests.
  - Writes `crop_candidates_local.csv`, `crop_evidence_redacted.jsonl`, `crop_processed_sources.jsonl`, and `summary.json`.

- Modify `settlement_tool/teacher_distill.py`
  - Add `features_from_crop_csv`.
  - Convert local crop candidates into the existing feature shape.

- Modify `scripts/build_codex_teacher_distill.py`
  - Add repeatable `--input-crop-csv`.
  - Merge crop features with OCR and KIE features.

- Modify `scripts/build_openai_reranker_payloads.py`
  - Bump schema to `openai_account_reranker_redacted_v4`.
  - Add `crop_evidence` to candidates.
  - Ensure crop evidence includes only buckets and booleans.

- Modify `scripts/local_feature_score_rerank.py`
  - Add conservative crop evidence scoring.
  - Hard-risk crop candidates when crop field evidence is not account-like.

- Modify `scripts/build_release_gate.py`
  - Treat `crop_candidates_local.csv` and crop images as local-only raw artifacts.
  - Allow `crop_evidence_redacted.jsonl` only when it contains no forbidden keys or account-like digit runs.

- Modify `scripts/run_candidate_recall_experiment.py`
  - Add repeatable `--payload-input-crop-csv`.
  - Copy `crop_evidence_redacted.jsonl` into the release bundle when provided via `--redacted-artifact`.

- Add tests:
  - `tests/test_candidate_miss_analyzer.py`
  - `tests/test_crop_evidence.py`
  - `tests/test_account_field_crop_retry.py`
  - Extend `tests/test_teacher_distill.py`
  - Extend `tests/test_openai_reranker_pipeline.py`
  - Extend `tests/test_release_gate.py`
  - Extend `tests/test_candidate_recall_experiment.py`

---

## Task 1: Aggregate Miss Analyzer

**Files:**

- Create: `scripts/analyze_candidate_misses.py`
- Test: `tests/test_candidate_miss_analyzer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_candidate_miss_analyzer.py`:

```python
import csv
import json
import subprocess
import sys
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_miss_analyzer_outputs_aggregate_categories_without_pii(tmp_path: Path):
    account = "110" + "-123" + "-456789"
    eval_dir = tmp_path / "eval"
    payload_dir = tmp_path / "payloads"
    reranker_dir = tmp_path / "reranker"
    eval_dir.mkdir()
    payload_dir.mkdir()
    reranker_dir.mkdir()

    (eval_dir / "human_label_eval_details.csv").write_text(
        "name,outcome,decision,source\n"
        "LOCAL_NAME_SENTINEL,missed_positive,openai_reranker_no_candidate,\n",
        encoding="utf-8",
    )
    write_jsonl(
        payload_dir / "candidate_raw_map_local.jsonl",
        [
            {
                "source_id": "source-1",
                "candidate_id": "acct_1",
                "name": "LOCAL_NAME_SENTINEL",
                "candidate_raw": account,
            }
        ],
    )
    write_jsonl(
        payload_dir / "candidate_features_redacted.jsonl",
        [
            {
                "source_id": "source-1",
                "candidates": [
                    {
                        "candidate_id": "acct_1",
                        "source_evidence": {"source_kind": "targeted_retry", "prompt_id": "visible_numbers"},
                        "kie_evidence": {"backend": "", "field_type": "unknown"},
                        "field_evidence": {"is_value_in_account_field": True, "same_line_label_type": "account_number"},
                        "risk_flags": {"has_wrong_field_context": True},
                    }
                ],
            }
        ],
    )
    write_jsonl(
        reranker_dir / "decisions.jsonl",
        [
            {
                "source_id": "source-1",
                "action": "reject",
                "selected_candidate_id": None,
                "reason_codes": ["top_candidate_has_hard_risk"],
                "feature_score": -16,
            }
        ],
    )

    output_dir = tmp_path / "miss_analysis"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_candidate_misses.py",
            "--eval-details-csv",
            str(eval_dir / "human_label_eval_details.csv"),
            "--raw-map-jsonl",
            str(payload_dir / "candidate_raw_map_local.jsonl"),
            "--payload-jsonl",
            str(payload_dir / "candidate_features_redacted.jsonl"),
            "--decisions-jsonl",
            str(reranker_dir / "decisions.jsonl"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    report = json.loads((output_dir / "miss_analysis_summary.json").read_text(encoding="utf-8"))
    report_text = json.dumps(report, ensure_ascii=False)

    assert proc.returncode == 0
    assert report["missed_people_count"] == 1
    assert report["coverage_counts"] == {"candidates_present": 1}
    assert report["decision_counts"] == {"reject": 1}
    assert "LOCAL_NAME_SENTINEL" not in report_text
    assert account not in report_text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_candidate_miss_analyzer.py -q
```

Expected: fails because `scripts/analyze_candidate_misses.py` does not exist.

- [ ] **Step 3: Implement `scripts/analyze_candidate_misses.py`**

Implement these functions:

```python
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze_candidate_misses(
    *,
    eval_details_csv: Path,
    raw_map_jsonl: Path,
    payload_jsonl: Path,
    decisions_jsonl: Path,
    output_dir: Path,
) -> dict[str, Any]:
    details = read_csv(eval_details_csv)
    raw_rows = read_jsonl(raw_map_jsonl)
    payloads = {row["source_id"]: row for row in read_jsonl(payload_jsonl)}
    decisions = {row["source_id"]: row for row in read_jsonl(decisions_jsonl)}
    missed_names = {row.get("name", "") for row in details if row.get("outcome") == "missed_positive"}

    raw_by_name: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        name = str(row.get("name", ""))
        if name in missed_names:
            raw_by_name.setdefault(name, []).append(row)

    coverage_counts: dict[str, int] = {"candidates_present": 0, "no_candidates": 0}
    decision_counts: dict[str, int] = {}
    feature_patterns: dict[str, int] = {}

    for name in missed_names:
        rows = raw_by_name.get(name, [])
        if rows:
            coverage_counts["candidates_present"] += 1
        else:
            coverage_counts["no_candidates"] += 1
        for source_id in sorted({str(row.get("source_id", "")) for row in rows if row.get("source_id")}):
            decision = decisions.get(source_id, {})
            action = str(decision.get("action", "missing_decision"))
            decision_counts[action] = decision_counts.get(action, 0) + 1
            payload = payloads.get(source_id, {})
            candidate = (payload.get("candidates") or [{}])[0]
            source = candidate.get("source_evidence") or {}
            kie = candidate.get("kie_evidence") or {}
            field = candidate.get("field_evidence") or {}
            risk = candidate.get("risk_flags") or {}
            key = "|".join(
                [
                    str(source.get("source_kind", "unknown")),
                    str(source.get("prompt_id", "unknown")),
                    str(kie.get("field_type", "unknown")),
                    f"account_field={bool(field.get('is_value_in_account_field'))}",
                    f"wrong_field={bool(risk.get('has_wrong_field_context'))}",
                    action,
                ]
            )
            feature_patterns[key] = feature_patterns.get(key, 0) + 1

    report = {
        "missed_people_count": len(missed_names),
        "coverage_counts": {key: value for key, value in coverage_counts.items() if value},
        "decision_counts": dict(sorted(decision_counts.items())),
        "feature_patterns": dict(sorted(feature_patterns.items())),
        "privacy_contract": {
            "names": "omitted",
            "raw_accounts": "omitted",
            "source_paths": "omitted",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "miss_analysis_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
```

Wire it to argparse with required paths matching the test.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m pytest tests/test_candidate_miss_analyzer.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_candidate_misses.py tests/test_candidate_miss_analyzer.py
git commit -m "feat: add aggregate miss analyzer"
```

---

## Task 2: Crop Evidence Normalizer

**Files:**

- Create: `settlement_tool/crop_evidence.py`
- Test: `tests/test_crop_evidence.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_crop_evidence.py`:

```python
from settlement_tool.crop_evidence import (
    crop_bbox_around_field,
    normalize_crop_row,
    redacted_crop_evidence,
)


def test_crop_bbox_around_field_expands_right_side_and_clamps_to_page():
    bbox = crop_bbox_around_field(
        [100, 50, 200, 90],
        page_width=500,
        page_height=300,
        expand_left=0.5,
        expand_right=3.0,
        expand_y=1.0,
    )

    assert bbox == [50.0, 10.0, 500.0, 130.0]


def test_normalize_crop_row_keeps_raw_local_only_and_redacts_evidence():
    row = normalize_crop_row(
        source_id="crop:1",
        source_name="/private/source.png",
        crop_image_path="/private/crop.png",
        backend="crop_tesseract",
        raw_candidate_local="RAW_ACCOUNT_SENTINEL",
        candidate_masked="MASKED_ACCOUNT_SENTINEL",
        crop_bbox=[50, 10, 500, 130],
        page_width=500,
        page_height=300,
        label_type="account_number",
        ocr_confidence=0.91,
        crop_variant="right_expand",
    )

    evidence = redacted_crop_evidence(row)
    serialized = str(evidence)

    assert row["candidate_raw"] == "RAW_ACCOUNT_SENTINEL"
    assert evidence["backend"] == "crop_tesseract"
    assert evidence["label_type"] == "account_number"
    assert evidence["confidence_bucket"] == "high"
    assert evidence["layout"]["x_bucket"] in {"center", "right"}
    assert "RAW_ACCOUNT_SENTINEL" not in serialized
    assert "/private" not in serialized
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_crop_evidence.py -q
```

Expected: import failure for `settlement_tool.crop_evidence`.

- [ ] **Step 3: Implement `settlement_tool/crop_evidence.py`**

Implement:

```python
from __future__ import annotations

import json
from typing import Any

from .core import normalize_text
from .kie_evidence import bbox_bucket


def _confidence_bucket(confidence: float) -> str:
    if confidence >= 0.90:
        return "high"
    if confidence >= 0.70:
        return "medium"
    if confidence > 0:
        return "low"
    return "unknown"


def crop_bbox_around_field(
    bbox: list[float] | tuple[float, ...],
    *,
    page_width: float,
    page_height: float,
    expand_left: float = 0.5,
    expand_right: float = 3.0,
    expand_y: float = 1.0,
) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    return [
        max(0.0, x1 - width * expand_left),
        max(0.0, y1 - height * expand_y),
        min(float(page_width), x2 + width * expand_right),
        min(float(page_height), y2 + height * expand_y),
    ]


def _json_object(values: dict[str, Any]) -> str:
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_crop_row(
    *,
    source_id: str,
    source_name: str,
    crop_image_path: str,
    backend: str,
    raw_candidate_local: str,
    candidate_masked: str,
    crop_bbox: list[float] | tuple[float, ...],
    page_width: float,
    page_height: float,
    label_type: str,
    ocr_confidence: float,
    crop_variant: str,
) -> dict[str, object]:
    layout = bbox_bucket(crop_bbox, page_width=page_width, page_height=page_height)
    return {
        "source_id": normalize_text(source_id),
        "source_name": normalize_text(source_name),
        "crop_image_path": normalize_text(crop_image_path),
        "backend": normalize_text(backend),
        "crop_backend": normalize_text(backend),
        "crop_variant": normalize_text(crop_variant),
        "crop_label_type": normalize_text(label_type) or "unknown",
        "candidate_raw": normalize_text(raw_candidate_local),
        "candidate_masked": normalize_text(candidate_masked),
        "crop_confidence": float(ocr_confidence or 0.0),
        "crop_confidence_bucket": _confidence_bucket(float(ocr_confidence or 0.0)),
        "crop_bbox_json": json.dumps(list(crop_bbox), ensure_ascii=False, separators=(",", ":")),
        "page_width": float(page_width or 0),
        "page_height": float(page_height or 0),
        "crop_layout_json": _json_object(layout),
        "layout_evidence": layout,
        "error": "",
    }


def redacted_crop_evidence(row: dict[str, object]) -> dict[str, object]:
    layout = row.get("layout_evidence")
    if not isinstance(layout, dict):
        try:
            layout = json.loads(normalize_text(row.get("crop_layout_json")))
        except json.JSONDecodeError:
            layout = {}
    return {
        "backend": normalize_text(row.get("crop_backend")),
        "variant": normalize_text(row.get("crop_variant")) or "unknown",
        "label_type": normalize_text(row.get("crop_label_type")) or "unknown",
        "confidence_bucket": normalize_text(row.get("crop_confidence_bucket"))
        or _confidence_bucket(float(row.get("crop_confidence") or 0.0)),
        "layout": {
            "x_bucket": normalize_text(layout.get("x_bucket")) if isinstance(layout, dict) else "unknown",
            "y_bucket": normalize_text(layout.get("y_bucket")) if isinstance(layout, dict) else "unknown",
            "width_bucket": normalize_text(layout.get("width_bucket")) if isinstance(layout, dict) else "unknown",
            "height_bucket": normalize_text(layout.get("height_bucket")) if isinstance(layout, dict) else "unknown",
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m pytest tests/test_crop_evidence.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add settlement_tool/crop_evidence.py tests/test_crop_evidence.py
git commit -m "feat: add redacted crop evidence helpers"
```

---

## Task 3: Account-Field Crop Retry Harness

**Files:**

- Create: `scripts/run_account_field_crop_retry.py`
- Test: `tests/test_account_field_crop_retry.py`
- Fixture: `tests/fixtures/account_field_crop_sample.json`

- [ ] **Step 1: Write the fixture**

Create `tests/fixtures/account_field_crop_sample.json`:

```json
[
  {
    "source_name": "sample.png",
    "page_width": 500,
    "page_height": 300,
    "items": [
      {
        "label_type": "account_number",
        "label_bbox": [100, 50, 200, 90],
        "crop_variant": "right_expand",
        "candidate_masked": "MASKED_ACCOUNT_SENTINEL",
        "raw_candidate_local": "RAW_ACCOUNT_SENTINEL",
        "ocr_confidence": 0.93
      }
    ]
  }
]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_account_field_crop_retry.py`:

```python
import csv
import json
import subprocess
import sys
from pathlib import Path


def test_crop_retry_fixture_mode_writes_local_and_redacted_outputs(tmp_path: Path):
    output_dir = tmp_path / "crop_retry"
    fixture = Path("tests/fixtures/account_field_crop_sample.json")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_account_field_crop_retry.py",
            "--fixture-json",
            str(fixture),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    local_rows = list(csv.DictReader((output_dir / "crop_candidates_local.csv").open(encoding="utf-8-sig")))
    redacted_text = (output_dir / "crop_evidence_redacted.jsonl").read_text(encoding="utf-8")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert proc.returncode == 0
    assert summary["candidate_count"] == 1
    assert local_rows[0]["candidate_raw"] == "RAW_ACCOUNT_SENTINEL"
    assert "RAW_ACCOUNT_SENTINEL" not in redacted_text
    assert "sample.png" not in redacted_text


def test_crop_retry_plan_only_reads_kie_account_targets(tmp_path: Path):
    kie_csv = tmp_path / "kie_candidates_local.csv"
    kie_csv.write_text(
        "source_id,source_name,kie_field_type,kie_label_masked,bbox_json,page_width,page_height,candidate_raw,candidate_masked,error\n"
        "kie:1,source.png,account_number,계좌번호,\"[100,50,200,90]\",500,300,,,\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "plan"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_account_field_crop_retry.py",
            "--input-kie-csv",
            str(kie_csv),
            "--output-dir",
            str(output_dir),
            "--plan-only",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert proc.returncode == 0
    assert summary["plan_only"] is True
    assert summary["target_count"] == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_account_field_crop_retry.py -q
```

Expected: fails because `scripts/run_account_field_crop_retry.py` does not exist.

- [ ] **Step 4: Implement fixture and plan-only modes**

Implement `scripts/run_account_field_crop_retry.py` with:

- `CSV_FIELDS`:
  - `source_id`
  - `source_name`
  - `crop_image_path`
  - `backend`
  - `crop_backend`
  - `crop_variant`
  - `crop_label_type`
  - `candidate_raw`
  - `candidate_masked`
  - `crop_confidence`
  - `crop_confidence_bucket`
  - `crop_bbox_json`
  - `page_width`
  - `page_height`
  - `crop_layout_json`
  - `error`
- `read_csv(path: Path) -> list[dict[str, str]]`
- `write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None`
- `write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None`
- `fixture_rows(path: Path) -> list[dict[str, object]]`
- `target_rows_from_kie_csv(path: Path) -> list[dict[str, str]]`
- `run_crop_retry(args: argparse.Namespace) -> dict[str, Any]`

In fixture mode, call `normalize_crop_row` for each fixture item and write both local and redacted outputs.

In plan-only mode, count only KIE rows where:

```python
row.get("kie_field_type") == "account_number" and not row.get("error")
```

Write `summary.json` with:

```python
{
    "plan_only": bool(args.plan_only),
    "target_count": target_count,
    "candidate_count": len(local_rows),
    "outputs": {
        "crop_candidates_local": str(output_dir / "crop_candidates_local.csv"),
        "crop_evidence_redacted": str(output_dir / "crop_evidence_redacted.jsonl"),
    },
}
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
python3 -m pytest tests/test_account_field_crop_retry.py -q
```

Expected: pass.

- [ ] **Step 6: Add real image crop implementation behind tests**

Add optional real mode:

- accept `--input-kie-csv`
- accept `--source-root`
- accept `--ocr-backend tesseract`
- create crop images under `output_dir / "crops_local"`
- use PIL to crop images with `crop_bbox_around_field`
- run local OCR only on crop images
- extract candidates with existing `ACCOUNT_CANDIDATE_RE`

Keep crop image paths local-only. Do not copy crop images into release bundles.

- [ ] **Step 7: Add focused real-mode test without OCR dependency**

Extend `tests/test_account_field_crop_retry.py`:

```python
from PIL import Image


def test_crop_retry_real_mode_creates_crop_manifest_without_ocr_when_disabled(tmp_path: Path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (500, 300), "white").save(image_path)
    kie_csv = tmp_path / "kie_candidates_local.csv"
    kie_csv.write_text(
        "source_id,source_name,kie_field_type,kie_label_masked,bbox_json,page_width,page_height,candidate_raw,candidate_masked,error\n"
        f"kie:1,{image_path.name},account_number,계좌번호,\"[100,50,200,90]\",500,300,,,\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "crop"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_account_field_crop_retry.py",
            "--input-kie-csv",
            str(kie_csv),
            "--source-root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--ocr-backend",
            "none",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert proc.returncode == 0
    assert summary["target_count"] == 1
    assert summary["candidate_count"] == 0
    assert summary["crop_count"] == 1
    assert (output_dir / "crops_local").exists()
```

Run:

```bash
python3 -m pytest tests/test_account_field_crop_retry.py -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/run_account_field_crop_retry.py tests/test_account_field_crop_retry.py tests/fixtures/account_field_crop_sample.json
git commit -m "feat: add account-field crop retry harness"
```

---

## Task 4: Ingest Crop Candidates Into Teacher Distill

**Files:**

- Modify: `settlement_tool/teacher_distill.py`
- Modify: `scripts/build_codex_teacher_distill.py`
- Test: extend `tests/test_teacher_distill.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_teacher_distill.py`:

```python
def test_features_from_crop_csv_preserves_crop_evidence_and_local_raw(tmp_path: Path):
    from settlement_tool.teacher_distill import features_from_crop_csv

    candidate = "110" + "-123" + "-456789"
    crop_csv = tmp_path / "crop_candidates_local.csv"
    crop_csv.write_text(
        "source_id,source_name,crop_backend,crop_variant,crop_label_type,candidate_raw,candidate_masked,crop_confidence,crop_confidence_bucket,crop_layout_json,error\n"
        f"crop:1,source.png,crop_tesseract,right_expand,account_number,{candidate},MASKED_ACCOUNT_SENTINEL,0.93,high,"
        "\"{\"\"x_bucket\"\":\"\"right\"\",\"\"y_bucket\"\":\"\"top\"\",\"\"width_bucket\"\":\"\"wide\"\",\"\"height_bucket\"\":\"\"short\"\"}\",\n",
        encoding="utf-8",
    )

    [feature] = features_from_crop_csv(crop_csv, backend="mixed_candidate_generation")

    assert feature["candidate_raw"] == candidate
    assert feature["crop_backend"] == "crop_tesseract"
    assert feature["crop_label_type"] == "account_number"
    assert feature["has_account_keyword_context"] is True
    assert feature["has_direct_account_field_context"] is True
```

Add a CLI integration dry-run test:

```python
def test_build_teacher_distill_accepts_crop_csv(tmp_path: Path):
    crop_csv = tmp_path / "crop_candidates_local.csv"
    candidate = "110" + "-123" + "-456789"
    crop_csv.write_text(
        "source_id,source_name,crop_backend,crop_variant,crop_label_type,candidate_raw,candidate_masked,crop_confidence,crop_confidence_bucket,crop_layout_json,error\n"
        f"crop:1,source.png,crop_tesseract,right_expand,account_number,{candidate},MASKED_ACCOUNT_SENTINEL,0.93,high,\"{}\",\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "distill"

    summary = build_teacher_distill_outputs(
        input_csvs=[],
        input_kie_csvs=[],
        input_crop_csvs=[crop_csv],
        output_dir=output_dir,
        backend="mixed_candidate_generation",
    )

    assert summary["crop_candidate_count"] == 1
    assert summary["candidate_count"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_teacher_distill.py -q
```

Expected: fails because `features_from_crop_csv` and `input_crop_csvs` do not exist.

- [ ] **Step 3: Implement `features_from_crop_csv`**

In `settlement_tool/teacher_distill.py`, add:

```python
def features_from_crop_csv(
    csv_path: Path | str,
    *,
    gold_by_source_name: dict[str, str] | None = None,
    backend: str = "",
) -> list[dict[str, Any]]:
    csv_path = Path(csv_path)
    records: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            if row.get("error"):
                continue
            candidate_raw = normalize_text(row.get("candidate_raw"))
            candidate_masked = normalize_text(row.get("candidate_masked"))
            if not candidate_raw and not candidate_masked:
                continue
            label = normalize_text(row.get("crop_label_type")) or "unknown"
            context = "\n".join(part for part in (label, candidate_masked) if part)
            feature = _candidate_base_features(context, candidate_raw or candidate_masked)
            feature.update(
                {
                    "candidate_raw": candidate_raw,
                    "candidate_masked": candidate_masked or mask_candidate(candidate_raw),
                    "source_id": row.get("source_id") or f"{csv_path.stem}:{index}",
                    "source_name": row.get("source_name") or f"{csv_path.name}:{index}",
                    "backend": backend or row.get("backend", ""),
                    "variant": row.get("crop_variant", ""),
                    "prompt_id": "account_field_crop",
                    "repeat_count": 1,
                    "crop_backend": row.get("crop_backend", ""),
                    "crop_variant": row.get("crop_variant", ""),
                    "crop_label_type": label,
                    "crop_confidence": float(row.get("crop_confidence") or 0.0),
                    "crop_confidence_bucket": row.get("crop_confidence_bucket", ""),
                    "crop_layout_evidence": _parse_json_object(row.get("crop_layout_json")),
                    "gold_label_available": False,
                    "gold_exact_match": None,
                    "matched_name": row.get("matched_name", "") or row.get("name", ""),
                    "matched_group": row.get("matched_group", "") or row.get("roster_group", ""),
                    "matched_no": row.get("matched_no", "") or row.get("roster_no", ""),
                }
            )
            if label == "account_number":
                feature["has_account_keyword_context"] = True
                feature["has_direct_account_field_context"] = True
            elif label in {"phone", "date", "amount", "customer_number"}:
                feature["has_negative_keyword_context"] = True
            feature["teacher_context_masked"] = mask_digit_context(context)
            feature["teacher_policy_score"] = policy_score(feature)
            records.append(feature)
    return records
```

- [ ] **Step 4: Add `--input-crop-csv` to build script**

In `scripts/build_codex_teacher_distill.py`:

- import `features_from_crop_csv`
- add `input_crop_csvs` parameter to `build_teacher_distill_outputs`
- loop over crop CSVs and extend `features`
- add `crop_candidate_count` to summary
- add CLI arg:

```python
parser.add_argument("--input-crop-csv", action="append", type=Path, default=[])
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_teacher_distill.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add settlement_tool/teacher_distill.py scripts/build_codex_teacher_distill.py tests/test_teacher_distill.py
git commit -m "feat: ingest crop candidates into distill"
```

---

## Task 5: Redacted Crop Payload and Scoring

**Files:**

- Modify: `scripts/build_openai_reranker_payloads.py`
- Modify: `scripts/local_feature_score_rerank.py`
- Test: extend `tests/test_openai_reranker_pipeline.py`

- [ ] **Step 1: Write failing payload tests**

Append to `tests/test_openai_reranker_pipeline.py`:

```python
def test_v4_payload_includes_crop_evidence_without_raw_values():
    feature = {
        "source_id": "crop:1",
        "source_name": "/private/source.png",
        "candidate_raw": "RAW_ACCOUNT_SENTINEL",
        "candidate_masked": "MASKED_ACCOUNT_SENTINEL",
        "digit_count": 12,
        "hyphen_count": 2,
        "group_count": 3,
        "teacher_policy_score": 10,
        "repeat_count": 1,
        "crop_backend": "crop_tesseract",
        "crop_variant": "right_expand",
        "crop_label_type": "account_number",
        "crop_confidence": 0.93,
        "crop_confidence_bucket": "high",
        "crop_layout_evidence": {"x_bucket": "right", "y_bucket": "top", "width_bucket": "wide"},
        "has_account_keyword_context": True,
        "has_direct_account_field_context": True,
        "matched_name": "LOCAL_NAME_SENTINEL",
    }

    payloads, raw_maps = build_openai_reranker_payloads([feature])
    payload_text = json.dumps(payloads, ensure_ascii=False)
    candidate = payloads[0]["candidates"][0]

    assert payloads[0]["schema_version"] == "openai_account_reranker_redacted_v4"
    assert candidate["crop_evidence"]["backend"] == "crop_tesseract"
    assert candidate["crop_evidence"]["label_type"] == "account_number"
    assert candidate["crop_evidence"]["confidence_bucket"] == "high"
    assert "RAW_ACCOUNT_SENTINEL" not in payload_text
    assert "LOCAL_NAME_SENTINEL" not in payload_text
    assert "/private" not in payload_text
    assert raw_maps[0]["candidate_raw"] == "RAW_ACCOUNT_SENTINEL"
```

Append scorer tests:

```python
def test_v2_feature_score_rewards_high_confidence_account_crop():
    candidate = {
        "teacher_policy_score": 7,
        "risk_flags": {
            "looks_like_phone": False,
            "has_prompt_leakage_context": False,
            "has_wrong_field_context": False,
        },
        "field_evidence": {
            "is_value_in_account_field": True,
            "same_line_label_type": "account_number",
            "is_value_in_customer_number_field": False,
        },
        "shape_features": {
            "pattern_family": "bank_account_like",
            "has_bank_style_hyphenation": True,
            "is_single_long_run": False,
        },
        "bank_holder_evidence": {
            "bank_name_present": True,
            "holder_match_status": "not_present",
        },
        "consensus_features": {
            "unique_candidate_count_for_person": 1,
            "candidate_source_count_for_person": 1,
            "variant_vote_count": 1,
        },
        "source_evidence": {
            "source_kind": "other",
            "prompt_id": "account_field_crop",
        },
        "crop_evidence": {
            "backend": "crop_tesseract",
            "label_type": "account_number",
            "confidence_bucket": "high",
            "layout": {"width_bucket": "wide"},
        },
    }

    decision = rerank_payload_with_v2_feature_score(
        {"source_id": "source-1", "candidates": [candidate]},
        threshold=10.0,
        min_margin=2.0,
    )

    assert v2_feature_score(candidate) >= 10
    assert decision["action"] == "accept"


def test_v2_feature_score_blocks_non_account_crop_label():
    candidate = {
        "teacher_policy_score": 20,
        "risk_flags": {
            "looks_like_phone": False,
            "has_prompt_leakage_context": False,
            "has_wrong_field_context": False,
        },
        "field_evidence": {
            "is_value_in_account_field": False,
            "is_value_in_customer_number_field": False,
        },
        "shape_features": {
            "pattern_family": "bank_account_like",
            "has_bank_style_hyphenation": True,
            "is_single_long_run": False,
        },
        "bank_holder_evidence": {
            "bank_name_present": True,
            "holder_match_status": "not_present",
        },
        "consensus_features": {"unique_candidate_count_for_person": 1},
        "source_evidence": {"source_kind": "other", "prompt_id": "account_field_crop"},
        "crop_evidence": {
            "backend": "crop_tesseract",
            "label_type": "phone",
            "confidence_bucket": "high",
        },
    }

    decision = rerank_payload_with_v2_feature_score(
        {"source_id": "source-1", "candidates": [candidate]},
        threshold=10.0,
        min_margin=2.0,
    )

    assert decision["action"] == "reject"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_openai_reranker_pipeline.py::test_v4_payload_includes_crop_evidence_without_raw_values tests/test_openai_reranker_pipeline.py::test_v2_feature_score_rewards_high_confidence_account_crop tests/test_openai_reranker_pipeline.py::test_v2_feature_score_blocks_non_account_crop_label -q
```

Expected: fail because crop evidence and schema v4 do not exist.

- [ ] **Step 3: Implement redacted crop payload**

In `scripts/build_openai_reranker_payloads.py`:

- change:

```python
PAYLOAD_SCHEMA_VERSION = "openai_account_reranker_redacted_v4"
```

- import:

```python
from settlement_tool.crop_evidence import redacted_crop_evidence
```

- add to `_candidate_payload`:

```python
"crop_evidence": redacted_crop_evidence(feature),
```

- extend `layout_evidence` to prefer crop layout when KIE layout is absent:

```python
layout = feature.get("layout_evidence") or feature.get("crop_layout_evidence")
```

- [ ] **Step 4: Implement crop scoring**

In `scripts/local_feature_score_rerank.py`:

- read:

```python
crop = candidate.get("crop_evidence") or {}
crop_label = crop.get("label_type")
has_crop_evidence = bool(crop.get("backend"))
```

- add conservative bonuses:

```python
if crop_label == "account_number" and crop.get("confidence_bucket") == "high":
    score += 4.0
if crop_label == "account_number" and field.get("is_value_in_account_field"):
    score += 2.0
```

- add penalties:

```python
if has_crop_evidence and crop_label != "account_number":
    score -= 40.0
```

- update `_hard_risk`:

```python
crop = candidate.get("crop_evidence") or {}
crop_hard_risk = bool(crop.get("backend")) and crop.get("label_type") != "account_number"
```

Include `crop_hard_risk` in the returned boolean.

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_openai_reranker_pipeline.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_openai_reranker_payloads.py scripts/local_feature_score_rerank.py tests/test_openai_reranker_pipeline.py
git commit -m "feat: add redacted crop evidence to reranker"
```

---

## Task 6: Release Gate Crop Artifact Policy

**Files:**

- Modify: `scripts/build_release_gate.py`
- Test: extend `tests/test_release_gate.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_release_gate.py`:

```python
def test_scan_release_bundle_flags_raw_crop_local_artifacts(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "crop_candidates_local.csv").write_text(
        "candidate_raw,candidate_masked\nRAW_ACCOUNT_SENTINEL,MASKED_ACCOUNT_SENTINEL\n",
        encoding="utf-8",
    )
    (bundle / "crops_local").mkdir()
    (bundle / "crops_local" / "crop.png").write_bytes(b"not-an-image")

    scan = scan_release_bundle(bundle)

    assert scan["blocked_artifact_count"] == 2
    assert any(path.endswith("crop_candidates_local.csv") for path in scan["blocked_artifact_paths"])
    assert any(path.endswith("crops_local/crop.png") for path in scan["blocked_artifact_paths"])


def test_scan_release_bundle_allows_crop_redacted_evidence(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "crop_evidence_redacted.jsonl").write_text(
        '{"backend":"crop_tesseract","label_type":"account_number","confidence_bucket":"high","layout":{"x_bucket":"right"}}\n',
        encoding="utf-8",
    )

    scan = scan_release_bundle(bundle)

    assert scan["blocked_artifact_count"] == 0
    assert scan["pii_match_count"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_release_gate.py::test_scan_release_bundle_flags_raw_crop_local_artifacts tests/test_release_gate.py::test_scan_release_bundle_allows_crop_redacted_evidence -q
```

Expected: first test fails because crop local artifacts are not blocked yet.

- [ ] **Step 3: Implement release gate updates**

In `scripts/build_release_gate.py`:

- add blocked names:

```python
"crop_candidates_local.csv",
```

- treat files under a `crops_local` path component as blocked:

```python
if "crops_local" in relative.parts:
    blocked_artifact_paths.append(relative.as_posix())
```

- ensure forbidden redacted keys include:

```python
"crop_image_path",
"crop_source_name",
```

Do not flag the key name `crop_evidence` itself.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_release_gate.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_release_gate.py tests/test_release_gate.py
git commit -m "test: gate crop OCR local artifacts"
```

---

## Task 7: Experiment Harness Crop Input

**Files:**

- Modify: `scripts/run_candidate_recall_experiment.py`
- Test: extend `tests/test_candidate_recall_experiment.py`

- [ ] **Step 1: Write failing test**

Extend `tests/test_candidate_recall_experiment.py`:

```python
def test_candidate_recall_experiment_accepts_crop_csv_in_dry_run(tmp_path: Path):
    output_dir = tmp_path / "experiment"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_candidate_recall_experiment.py",
            "--payload-input-csv",
            "full.csv",
            "--payload-input-crop-csv",
            "crop.csv",
            "--human-workbook",
            "human.xlsx",
            "--source-workbook",
            "human.xlsx",
            "--data-zip",
            "data.zip",
            "--output-dir",
            str(output_dir),
            "--threshold",
            "10",
            "--min-margin",
            "2",
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert proc.returncode == 0
    assert summary["dry_run"] is True
    assert "build_codex_teacher_distill" in " ".join(summary["planned_steps"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_candidate_recall_experiment.py::test_candidate_recall_experiment_accepts_crop_csv_in_dry_run -q
```

Expected: fails because `--payload-input-crop-csv` is not recognized.

- [ ] **Step 3: Implement crop CSV plumbing**

In `scripts/run_candidate_recall_experiment.py`:

- add `args.payload_input_crop_csv` to `build_commands`
- extend `build_payloads` with `--input-crop-csv`
- add parser arg:

```python
parser.add_argument("--payload-input-crop-csv", action="append", type=Path, default=[])
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m pytest tests/test_candidate_recall_experiment.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_candidate_recall_experiment.py tests/test_candidate_recall_experiment.py
git commit -m "feat: add crop input to recall experiment"
```

---

## Task 8: End-to-End Crop Recall Evaluation

**Files:**

- No code changes required if Tasks 1-7 are complete.
- Outputs are local-only under `$EVAL_ROOT`.
- Wiki updates after eval:
  - Modify `wiki/log.md`
  - Create `wiki/experiments/YYYY-MM-DD-crop-ocr-recall-loop.md`
  - Update `wiki/index.md`
  - Update `wiki/questions/next-ocr-candidate-recall-loop.md`

- [ ] **Step 1: Run miss analyzer on current best**

Run:

```bash
python3 scripts/analyze_candidate_misses.py \
  --eval-details-csv "$EVAL_ROOT/experiment_paddleocr_kie_holder_personsplit_dual_rescue_20260429/eval/human_label_eval_details.csv" \
  --raw-map-jsonl "$EVAL_ROOT/experiment_paddleocr_kie_holder_personsplit_dual_rescue_20260429/payloads/candidate_raw_map_local.jsonl" \
  --payload-jsonl "$EVAL_ROOT/experiment_paddleocr_kie_holder_personsplit_dual_rescue_20260429/payloads/candidate_features_redacted.jsonl" \
  --decisions-jsonl "$EVAL_ROOT/experiment_paddleocr_kie_holder_personsplit_dual_rescue_20260429/reranker/decisions.jsonl" \
  --output-dir "$EVAL_ROOT/miss_analysis_crop_loop_seed_$(date +%Y%m%d)"
```

Expected:

- output JSON exists
- no names or raw accounts appear in the report
- categories distinguish no-candidate from safety-blocked candidates

- [ ] **Step 2: Plan crop targets from KIE rows**

Run:

```bash
python3 scripts/run_account_field_crop_retry.py \
  --input-kie-csv "$EVAL_ROOT/paddleocr_kie_retry_holder_20260429/kie_candidates_local.csv" \
  --output-dir "$EVAL_ROOT/account_field_crop_retry_plan_$(date +%Y%m%d)" \
  --plan-only
```

Expected:

- `summary.json.plan_only == true`
- target count is nonzero only for account-number KIE rows

- [ ] **Step 3: Run crop retry locally**

Run only after confirming `--source-root` points to the local extracted document/image root:

```bash
python3 scripts/run_account_field_crop_retry.py \
  --input-kie-csv "$EVAL_ROOT/paddleocr_kie_retry_holder_20260429/kie_candidates_local.csv" \
  --source-root "$LOCAL_EXTRACTED_IMAGE_ROOT" \
  --output-dir "$EVAL_ROOT/account_field_crop_retry_$(date +%Y%m%d)" \
  --ocr-backend tesseract
```

Expected:

- `crop_candidates_local.csv` exists and stays local-only
- `crop_evidence_redacted.jsonl` exists and contains no raw candidates or source paths
- `crops_local/` exists and is not copied into release bundles

- [ ] **Step 4: Run recall experiment with crop candidates**

Run:

```bash
python3 scripts/run_candidate_recall_experiment.py \
  --payload-input-csv "$EVAL_ROOT/full_ocr/deepseek_bank_zip_full_ocr.csv" \
  --payload-input-csv "$EVAL_ROOT/targeted_retry/targeted_retry_ocr.csv" \
  --payload-input-csv "$EVAL_ROOT/no_candidate_retry_full27/targeted_retry_ocr.csv" \
  --payload-input-kie-csv "$EVAL_ROOT/paddleocr_kie_retry_holder_20260429/kie_candidates_local.csv" \
  --payload-input-crop-csv "$EVAL_ROOT/account_field_crop_retry_$(date +%Y%m%d)/crop_candidates_local.csv" \
  --redacted-artifact "$EVAL_ROOT/paddleocr_kie_retry_holder_20260429/kie_evidence_redacted.jsonl" \
  --redacted-artifact "$EVAL_ROOT/account_field_crop_retry_$(date +%Y%m%d)/crop_evidence_redacted.jsonl" \
  --human-workbook "$HUMAN_WORKBOOK" \
  --source-workbook "$HUMAN_WORKBOOK" \
  --data-zip "$DATA_ZIP" \
  --output-dir "$EVAL_ROOT/experiment_crop_ocr_recall_$(date +%Y%m%d)" \
  --threshold 10 \
  --min-margin 2
```

Expected pass criteria:

- `wrong_positive == 0`
- `review_false_positive == 0`
- `safe_selection_precision == 1.0`
- `positive_recall >= 0.8125`
- PII gate passes
- reranker gate passes
- overall release may remain `blocked_manual_review`

- [ ] **Step 5: Reject unsafe result immediately**

If any of these happen, do not keep crop scoring as an accepted feature:

- `wrong_positive > 0`
- `review_false_positive > 0`
- release bundle PII scan fails
- crop redacted evidence contains forbidden raw keys

Revert or hard-gate the unsafe scoring rule, then rerun Step 4.

- [ ] **Step 6: Update LLM Wiki**

If the run is safe, add:

- `wiki/experiments/YYYY-MM-DD-crop-ocr-recall-loop.md`
- a log entry in `wiki/log.md`
- an index link in `wiki/index.md`
- an updated status in `wiki/questions/next-ocr-candidate-recall-loop.md`

Record:

- aggregate metrics only
- miss category deltas
- accepted/rejected scoring changes
- release gate status

Do not record names, raw accounts, raw OCR, source filenames, or local paths.

- [ ] **Step 7: Run full verification**

Run:

```bash
git diff --check
python3 - <<'PY'
import re
from pathlib import Path

paths = [Path("AGENTS.md"), Path("README.md"), Path("wiki"), Path("docs/superpowers/plans/2026-04-29-crop-ocr-recall-loop.md")]
patterns = [
    re.compile(re.escape("/Users/" + "boram")),
    re.compile("Down" + "loads"),
    re.compile(r"[0-9]{2,4}-[0-9]{2,4}-[0-9]{3,}"),
    re.compile(r"[가-힣]{2,4},account_number"),
]
hits = []
for root in paths:
    files = [root] if root.is_file() else sorted(root.rglob("*.md"))
    for path in files:
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in patterns):
                hits.append(f"{path}:{lineno}")
if hits:
    raise SystemExit("\n".join(hits))
PY
python3 -m pytest -q
```

Expected:

- `git diff --check` has no output
- privacy scan has no raw/private hits outside synthetic tests
- all tests pass

- [ ] **Step 8: Commit**

```bash
git add wiki docs/superpowers/plans/2026-04-29-crop-ocr-recall-loop.md
git commit -m "docs: record crop OCR recall loop results"
```

---

## Final Acceptance Criteria

The implementation is accepted only if:

- unit tests pass for each task
- full `python3 -m pytest -q` passes
- release gate PII scan passes with crop redacted artifacts included
- `crop_candidates_local.csv` and `crops_local/` are blocked from release bundles
- final eval keeps `wrong_positive=0`
- final eval keeps `review_false_positive=0`
- final eval keeps `safe_selection_precision=1.0`
- recall is at least the current `52/64` baseline
- LLM Wiki is updated with aggregate-only results

## Expected First Safe Outcome

The most likely first safe outcome is modest: recover 0-2 additional positives without changing threshold. If the crop harness produces no recall improvement, keep the miss analyzer and crop artifact policy, but do not claim performance gain.
