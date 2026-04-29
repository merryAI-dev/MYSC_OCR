import json
from pathlib import Path

from scripts.run_quant_candidate_smoke_gate import read_jsonl, summarize_rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_smoke_gate_passes_concise_account_outputs(tmp_path: Path):
    rows = [
        {"id": "ok-1", "text": "bank: 국민은행\naccount_holder: 홍길동\naccount_number: 123-456-789012"},
        {"id": "ok-2", "text": "계좌번호: 987-654-321098\n예금주: 김영희"},
    ]

    summary = summarize_rows(rows, max_degenerate_outputs=0)

    assert summary["status"] == "pass"
    assert summary["total"] == 2
    assert summary["degenerate_outputs"] == 0
    assert summary["missing_account_candidate"] == 0
    assert summary["failed_reason"] == ""


def test_smoke_gate_fails_repeated_collapse_from_offline_jsonl(tmp_path: Path):
    responses = tmp_path / "responses.jsonl"
    write_jsonl(
        responses,
        [
            {"id": "bad-1", "name": "collapse", "text": "Background " * 20},
            {"id": "ok-1", "name": "ok", "text": "계좌번호: 123-456-789012"},
        ],
    )

    summary = summarize_rows(read_jsonl(responses), max_degenerate_outputs=0)

    assert summary["status"] == "fail"
    assert summary["total"] == 2
    assert summary["degenerate_outputs"] == 1
    assert summary["missing_account_candidate"] == 1
    assert summary["repeated_token_signatures"] == {"background x20": 1}
    assert summary["repeated_token_reasons"]["top_token_dominance"] == 1
    assert "degenerate_outputs 1 > max_degenerate_outputs 0" == summary["failed_reason"]


def test_smoke_gate_threshold_allows_configured_degenerate_outputs():
    rows = [
        {"id": "bad-1", "text": "是 " * 12},
        {"id": "ok-1", "text": "계좌번호: 123-456-789012"},
    ]

    summary = summarize_rows(rows, max_degenerate_outputs=1)

    assert summary["status"] == "pass"
    assert summary["degenerate_outputs"] == 1
    assert summary["max_degenerate_outputs"] == 1
    assert summary["failed_reason"] == ""


def test_smoke_gate_fails_empty_input():
    summary = summarize_rows([], max_degenerate_outputs=0)

    assert summary["status"] == "fail"
    assert summary["total"] == 0
    assert summary["failed_reason"] == "no_rows"
