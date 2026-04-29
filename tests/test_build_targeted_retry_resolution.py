import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.deepseek_targeted_bank_retry import (
    build_retry_targets,
    completed_retry_keys,
    default_target_decisions,
    is_backend_down_error,
    retry_row_key,
)
from scripts.build_targeted_retry_resolution import policy_accepts_retry_row


def test_policy_accepts_retry_row_rejects_prompt_leakage_high_row(tmp_path: Path):
    text_path = tmp_path / "ocr.txt"
    text_path.write_text("KNOWN 계좌번호가 보이면 account_number: 110-123-456789", encoding="utf-8")
    row = {
        "account": "110-123-456789",
        "confidence": "high",
        "ocr_text_path": str(text_path),
    }

    assert policy_accepts_retry_row(row) is False


def test_policy_accepts_retry_row_accepts_structured_high_row(tmp_path: Path):
    text_path = tmp_path / "ocr.txt"
    text_path.write_text(
        "<tr><td>계좌번호</td><td>110-123-456789</td></tr><tr><td>예금주</td><td>홍길동</td></tr>",
        encoding="utf-8",
    )
    row = {
        "account": "110-123-456789",
        "confidence": "high",
        "ocr_text_path": str(text_path),
    }

    assert policy_accepts_retry_row(row) is True


def test_policy_accepts_retry_row_requires_reranker_threshold_not_bank_fallback(tmp_path: Path):
    text_path = tmp_path / "ocr.txt"
    text_path.write_text("신한은행 110-123-456789", encoding="utf-8")
    row = {
        "account": "110-123-456789",
        "confidence": "high",
        "ocr_text_path": str(text_path),
    }

    assert policy_accepts_retry_row(row) is False


def test_policy_accepts_retry_row_allows_supplied_threshold(tmp_path: Path):
    text_path = tmp_path / "ocr.txt"
    text_path.write_text("신한은행 110-123-456789", encoding="utf-8")
    row = {
        "account": "110-123-456789",
        "confidence": "high",
        "ocr_text_path": str(text_path),
    }

    assert policy_accepts_retry_row(row, min_score=8.0) is True


def test_default_target_decisions_include_openai_reranker_no_candidate():
    assert "no_candidate" in default_target_decisions()
    assert "openai_reranker_no_candidate" in default_target_decisions()
    assert "targeted_retry_no_candidate" in default_target_decisions()


def test_build_retry_targets_includes_openai_reranker_no_candidate_and_dedupes_source():
    resolution_rows = [
        {"name": "홍길동", "decision": "auto_fill_openai_reranker"},
        {"name": "김철수", "decision": "openai_reranker_no_candidate"},
        {"name": "박영희", "decision": "targeted_retry_no_candidate"},
    ]
    deepseek_rows = [
        {
            "matched_name": "김철수",
            "filename_hint": "a.png",
            "source_name": "a.png",
            "extracted_path": "/tmp/a.png",
        },
        {
            "matched_name": "김철수",
            "filename_hint": "a-copy.png",
            "source_name": "a.png",
            "extracted_path": "/tmp/a-copy.png",
        },
        {
            "matched_name": "박영희",
            "filename_hint": "b.png",
            "source_name": "b.png",
            "extracted_path": "/tmp/b.png",
        },
        {
            "matched_name": "홍길동",
            "filename_hint": "c.png",
            "source_name": "c.png",
            "extracted_path": "/tmp/c.png",
        },
    ]

    targets = build_retry_targets(
        resolution_rows=resolution_rows,
        deepseek_rows=deepseek_rows,
        target_decisions=default_target_decisions(),
    )

    assert [(name, row["source_name"]) for name, row in targets] == [
        ("김철수", "a.png"),
        ("박영희", "b.png"),
    ]


def test_deepseek_targeted_bank_retry_plan_only_writes_retry_manifest_without_ocr_server(tmp_path: Path):
    resolution_csv = tmp_path / "resolution.csv"
    deepseek_csv = tmp_path / "deepseek.csv"
    output_dir = tmp_path / "retry"
    resolution_csv.write_text(
        "name,decision\n"
        "김철수,openai_reranker_no_candidate\n"
        "홍길동,auto_fill_openai_reranker\n",
        encoding="utf-8",
    )
    deepseek_csv.write_text(
        "matched_name,filename_hint,source_name,extracted_path\n"
        "김철수,a.png,a.png,/tmp/a.png\n"
        "홍길동,b.png,b.png,/tmp/b.png\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/deepseek_targeted_bank_retry.py",
            "--resolution-csv",
            str(resolution_csv),
            "--deepseek-csv",
            str(deepseek_csv),
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
    rows = list(csv.DictReader((output_dir / "retry_targets.csv").open(encoding="utf-8-sig")))

    assert proc.returncode == 0
    assert summary["plan_only"] is True
    assert summary["targets"] == 1
    assert rows[0]["name"] == "김철수"
    assert rows[0]["decision"] == "openai_reranker_no_candidate"


def test_completed_retry_keys_skip_successful_rows_but_retry_errors():
    rows = [
        {
            "name": "김철수",
            "source_name": "a.png",
            "variant": "original",
            "prompt_id": "visible_numbers",
            "confidence": "none",
        },
        {
            "name": "김철수",
            "source_name": "a.png",
            "variant": "contrast",
            "prompt_id": "visible_numbers",
            "confidence": "error",
        },
    ]

    completed = completed_retry_keys(rows)

    assert retry_row_key("김철수", "a.png", "original", "visible_numbers") in completed
    assert retry_row_key("김철수", "a.png", "contrast", "visible_numbers") not in completed


def test_is_backend_down_error_detects_connection_failures():
    assert is_backend_down_error("ConnectionError: connection refused") is True
    assert is_backend_down_error("ReadTimeout: timed out") is True
    assert is_backend_down_error("ValueError: bad OCR JSON") is False
