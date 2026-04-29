import csv
from pathlib import Path

from scripts.summarize_teacher_alignment import build_teacher_alignment_summary


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


def test_teacher_alignment_splits_all_rows_from_ocr_derived_rows(tmp_path: Path):
    teacher = tmp_path / "teacher"
    candidate = tmp_path / "candidate"
    write_updates(
        teacher / "final_workbook" / "account_updates_deepseek_resolution.csv",
        [
            {
                "group": "g",
                "no": "1",
                "name": "기존",
                "cell": "J1",
                "status": "updated",
                "decision": "keep_existing_final_run",
                "source": "final_run",
                "account": "111-222-3333",
                "account_masked": "***3333",
                "candidate_count": "0",
                "candidate_accounts_masked": "",
                "candidate_files": "",
            },
            {
                "group": "g",
                "no": "2",
                "name": "선생님만",
                "cell": "J2",
                "status": "updated",
                "decision": "auto_fill_single_candidate",
                "source": "deepseek_ocr_high_policy",
                "account": "222-333-4444",
                "account_masked": "***4444",
                "candidate_count": "1",
                "candidate_accounts_masked": "***4444",
                "candidate_files": "teacher.png",
            },
            {
                "group": "g",
                "no": "3",
                "name": "후보만",
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
            {
                "group": "g",
                "no": "4",
                "name": "불일치",
                "cell": "J4",
                "status": "updated",
                "decision": "auto_fill_targeted_deepseek",
                "source": "targeted_retry",
                "account": "444-555-6666",
                "account_masked": "***6666",
                "candidate_count": "1",
                "candidate_accounts_masked": "***6666",
                "candidate_files": "teacher2.png",
            },
        ],
    )
    write_updates(
        candidate / "final_workbook" / "account_updates_deepseek_resolution.csv",
        [
            {
                "group": "g",
                "no": "1",
                "name": "기존",
                "cell": "J1",
                "status": "updated",
                "decision": "keep_existing_final_run",
                "source": "final_run",
                "account": "111-222-3333",
                "account_masked": "***3333",
                "candidate_count": "0",
                "candidate_accounts_masked": "",
                "candidate_files": "",
            },
            {
                "group": "g",
                "no": "2",
                "name": "선생님만",
                "cell": "J2",
                "status": "skipped",
                "decision": "targeted_retry_no_candidate",
                "source": "targeted_retry",
                "account": "",
                "account_masked": "",
                "candidate_count": "0",
                "candidate_accounts_masked": "",
                "candidate_files": "",
            },
            {
                "group": "g",
                "no": "3",
                "name": "후보만",
                "cell": "J3",
                "status": "updated",
                "decision": "auto_fill_single_candidate",
                "source": "deepseek_ocr_high_policy",
                "account": "333-444-5555",
                "account_masked": "***5555",
                "candidate_count": "1",
                "candidate_accounts_masked": "***5555",
                "candidate_files": "candidate.png",
            },
            {
                "group": "g",
                "no": "4",
                "name": "불일치",
                "cell": "J4",
                "status": "updated",
                "decision": "auto_fill_targeted_deepseek",
                "source": "targeted_retry",
                "account": "444-555-7777",
                "account_masked": "***7777",
                "candidate_count": "1",
                "candidate_accounts_masked": "***7777",
                "candidate_files": "candidate2.png",
            },
        ],
    )

    summary = build_teacher_alignment_summary(("teacher_bf16", teacher), [("A", candidate)])

    all_counts = summary["candidates"]["A"]["all_rows"]
    assert all_counts["teacher_updated"] == 3
    assert all_counts["candidate_updated"] == 3
    assert all_counts["aligned_updated"] == 1
    assert all_counts["teacher_missed"] == 1
    assert all_counts["candidate_only"] == 1
    assert all_counts["account_disagreement"] == 1

    ocr_counts = summary["candidates"]["A"]["ocr_only"]
    assert ocr_counts["teacher_updated"] == 2
    assert ocr_counts["candidate_updated"] == 2
    assert ocr_counts["aligned_updated"] == 0
    assert ocr_counts["teacher_recall"] == 0.0

    relations = [row["relation"] for row in summary["delta_rows"]]
    assert relations == ["teacher_missed", "candidate_only", "account_disagreement"]
    assert summary["delta_rows"][0]["teacher_account_masked"] == "***4444"
    assert "account" not in summary["delta_rows"][0]
