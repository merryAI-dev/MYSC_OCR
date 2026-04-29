import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.run_paddleocr_kie_retry import _flatten_ocr_result, completed_source_names, rows_from_ocr_items


class AmbiguousBoxes:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return [10, 10, 80, 40]

    def __bool__(self):
        raise ValueError("ambiguous")


def test_flatten_ocr_result_handles_paddleocr_array_like_boxes_without_bool_eval():
    result = {
        "rec_texts": ["110-123-456789"],
        "rec_scores": [0.95],
        "rec_boxes": AmbiguousBoxes(),
    }

    items = _flatten_ocr_result(result)

    assert items == [{"text": "110-123-456789", "bbox": [10.0, 10.0, 80.0, 40.0], "confidence": 0.95}]


def test_completed_source_names_reads_progress_manifest(tmp_path: Path):
    output_dir = tmp_path / "kie"
    output_dir.mkdir()
    (output_dir / "kie_processed_sources.jsonl").write_text(
        '{"source_name":"a.png","status":"completed"}\n'
        '{"source_name":"b.png","status":"error"}\n',
        encoding="utf-8",
    )

    assert completed_source_names(output_dir) == {"a.png", "b.png"}


def test_rows_from_ocr_items_pairs_candidate_with_nearest_account_label():
    rows = rows_from_ocr_items(
        [
            {"text": "홍길동", "bbox": [10, 50, 80, 80], "confidence": 0.98},
            {"text": "은행명", "bbox": [10, 90, 80, 120], "confidence": 0.98},
            {"text": "계좌번호", "bbox": [10, 10, 80, 40], "confidence": 0.98},
            {"text": "110-123-456789", "bbox": [120, 12, 280, 42], "confidence": 0.95},
        ],
        source_id_prefix="paddleocr_kie:1",
        source_name="sample.png",
        page_width=1000,
        page_height=1000,
        target_name="홍길동",
    )

    assert len(rows) == 1
    assert rows[0]["candidate_raw"] == "110-123-456789"
    assert rows[0]["candidate_masked"] == "***-***-**6789"
    assert rows[0]["kie_field_type"] == "account_number"
    assert rows[0]["kie_label_masked"] == "계좌번호"
    assert rows[0]["kie_holder_match_status"] == "match"
    assert rows[0]["kie_bank_name_present"] is True


def test_rows_from_ocr_items_marks_phone_label_as_non_account_candidate():
    rows = rows_from_ocr_items(
        [
            {"text": "연락처", "bbox": [10, 10, 80, 40], "confidence": 0.98},
            {"text": "010-1234-5678", "bbox": [120, 12, 280, 42], "confidence": 0.95},
        ],
        source_id_prefix="paddleocr_kie:1",
        source_name="sample.png",
        page_width=1000,
        page_height=1000,
    )

    assert len(rows) == 1
    assert rows[0]["kie_field_type"] == "phone"


def test_paddleocr_kie_retry_fixture_mode_writes_local_and_redacted_outputs(tmp_path: Path):
    fixture = Path("tests/fixtures/paddleocr_kie_sample.json")
    output_dir = tmp_path / "kie"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_paddleocr_kie_retry.py",
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

    local_rows = list(csv.DictReader((output_dir / "kie_candidates_local.csv").open(encoding="utf-8-sig")))
    redacted_text = (output_dir / "kie_evidence_redacted.jsonl").read_text(encoding="utf-8")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert proc.returncode == 0
    assert summary["candidate_count"] == 1
    assert local_rows[0]["candidate_raw"] == "RAW_ACCOUNT_SENTINEL"
    assert "RAW_ACCOUNT_SENTINEL" not in redacted_text
    assert "/Users/boram" not in redacted_text


def test_paddleocr_kie_retry_plan_only_without_dependency(tmp_path: Path):
    targets = tmp_path / "retry_targets.csv"
    targets.write_text("name,decision,source_name,extracted_path\nA,openai_reranker_no_candidate,a.png,/tmp/a.png\n", encoding="utf-8")
    output_dir = tmp_path / "plan"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_paddleocr_kie_retry.py",
            "--retry-targets-csv",
            str(targets),
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
