import csv
from pathlib import Path

from scripts.summarize_eval_union_fallback import build_union_fallback_summary


FIELDNAMES = [
    "group",
    "no",
    "name",
    "cell",
    "status",
    "decision",
    "source",
    "account",
    "account_masked",
    "candidate_count",
    "candidate_accounts_masked",
    "candidate_files",
]


def write_updates(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_build_union_fallback_summary_reports_masked_delta_rows(tmp_path: Path):
    primary = tmp_path / "primary"
    fallback = tmp_path / "fallback"
    write_updates(
        primary / "final_workbook" / "account_updates_deepseek_resolution.csv",
        [
            {
                "group": "g1",
                "no": "1",
                "name": "공통",
                "cell": "J1",
                "status": "updated",
                "decision": "keep_existing",
                "source": "final_run",
                "account": "111-222-3333",
                "account_masked": "***3333",
                "candidate_count": "0",
                "candidate_accounts_masked": "",
                "candidate_files": "",
            },
            {
                "group": "g1",
                "no": "2",
                "name": "육비트",
                "cell": "J2",
                "status": "updated",
                "decision": "auto_fill_targeted_deepseek",
                "source": "targeted_retry",
                "account": "222-333-4444",
                "account_masked": "***4444",
                "candidate_count": "1",
                "candidate_accounts_masked": "***4444",
                "candidate_files": "six.png",
            },
            {
                "group": "g1",
                "no": "3",
                "name": "팔비트",
                "cell": "J3",
                "status": "skipped",
                "decision": "no_candidate",
                "source": "",
                "account": "",
                "account_masked": "",
                "candidate_count": "0",
                "candidate_accounts_masked": "",
                "candidate_files": "",
            },
        ],
    )
    write_updates(
        fallback / "final_workbook" / "account_updates_deepseek_resolution.csv",
        [
            {
                "group": "g1",
                "no": "1",
                "name": "공통",
                "cell": "J1",
                "status": "updated",
                "decision": "keep_existing",
                "source": "final_run",
                "account": "111-222-3333",
                "account_masked": "***3333",
                "candidate_count": "0",
                "candidate_accounts_masked": "",
                "candidate_files": "",
            },
            {
                "group": "g1",
                "no": "2",
                "name": "육비트",
                "cell": "J2",
                "status": "skipped",
                "decision": "no_candidate",
                "source": "",
                "account": "",
                "account_masked": "",
                "candidate_count": "0",
                "candidate_accounts_masked": "",
                "candidate_files": "",
            },
            {
                "group": "g1",
                "no": "3",
                "name": "팔비트",
                "cell": "J3",
                "status": "updated",
                "decision": "deepseek_ocr_high_policy",
                "source": "deepseek",
                "account": "333-444-5555",
                "account_masked": "***5555",
                "candidate_count": "1",
                "candidate_accounts_masked": "***5555",
                "candidate_files": "eight.png",
            },
        ],
    )

    summary = build_union_fallback_summary(("bf16_direct_6bit", primary), ("mlx8bit", fallback))

    assert summary["counts"] == {
        "primary_updated": 2,
        "fallback_updated": 2,
        "overlap_updated": 1,
        "primary_only_updated": 1,
        "fallback_only_updated": 1,
        "union_updated": 3,
        "primary_gain_from_fallback": 1,
    }
    assert summary["delta_rows"][0]["relation"] == "primary_only"
    assert summary["delta_rows"][0]["primary_account_masked"] == "***4444"
    assert summary["delta_rows"][1]["relation"] == "fallback_only"
    assert summary["delta_rows"][1]["fallback_account_masked"] == "***5555"
    assert "account" not in summary["delta_rows"][0]
