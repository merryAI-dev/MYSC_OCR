import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.build_union_fallback_resolution import build_union_resolution
from scripts.apply_resolution_workbook import is_workbook_auto_fill_decision


FIELDS = [
    "group",
    "no",
    "name",
    "prior_status",
    "prior_account_masked",
    "chosen_account",
    "chosen_account_masked",
    "decision",
    "source",
    "candidate_count",
    "candidate_accounts_masked",
    "candidate_files",
]


def row(name: str, account: str, decision: str, source: str = "") -> dict[str, str]:
    return {
        "group": "g",
        "no": name,
        "name": name,
        "prior_status": "skipped" if not account else "updated",
        "prior_account_masked": "",
        "chosen_account": account,
        "chosen_account_masked": "***" + account[-4:] if account else "",
        "decision": decision,
        "source": source,
        "candidate_count": "1" if account else "0",
        "candidate_accounts_masked": "***" + account[-4:] if account else "",
        "candidate_files": f"{name}.png" if account else "",
    }


def write_resolution(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_build_union_resolution_adds_fallback_only_without_overwriting_primary():
    primary_rows = [
        row("primary-only", "111-222-3333", "auto_fill_single_candidate", "primary"),
        row("fallback-add", "", "targeted_retry_no_candidate"),
        row("conflict", "333-444-5555", "auto_fill_targeted_deepseek", "primary"),
    ]
    fallback_rows = [
        row("primary-only", "", "no_candidate"),
        row("fallback-add", "222-333-4444", "auto_fill_targeted_deepseek", "w8-targeted"),
        row("conflict", "999-888-7777", "auto_fill_targeted_deepseek", "w8-conflict"),
    ]

    output_rows, audit_rows, summary = build_union_resolution(
        primary_rows,
        fallback_rows,
        primary_label="mlx8",
        fallback_label="w8_all70",
    )
    by_name = {item["name"]: item for item in output_rows}

    assert by_name["primary-only"]["chosen_account"] == "111-222-3333"
    assert by_name["fallback-add"]["chosen_account"] == "222-333-4444"
    assert by_name["fallback-add"]["decision"] == "auto_fill_union_fallback"
    assert by_name["fallback-add"]["source"] == "union_fallback:w8_all70:auto_fill_targeted_deepseek:w8-targeted"
    assert by_name["conflict"]["chosen_account"] == "333-444-5555"
    assert audit_rows == [
        {
            "name": "conflict",
            "relation": "conflict_preserved_primary",
            "primary_decision": "auto_fill_targeted_deepseek",
            "fallback_decision": "auto_fill_targeted_deepseek",
            "primary_account_masked": "***5555",
            "fallback_account_masked": "***7777",
            "primary_source": "primary",
            "fallback_source": "w8-conflict",
        }
    ]
    assert summary["filled"] == 3
    assert summary["fallback_added"] == 1
    assert summary["conflicts_preserved_primary"] == 1
    assert is_workbook_auto_fill_decision("auto_fill_union_fallback") is True


def test_build_union_fallback_resolution_cli_writes_resolution_and_audit(tmp_path: Path):
    primary_csv = tmp_path / "primary.csv"
    fallback_csv = tmp_path / "fallback.csv"
    output_dir = tmp_path / "union"
    write_resolution(primary_csv, [row("fallback-add", "", "no_candidate")])
    write_resolution(fallback_csv, [row("fallback-add", "222-333-4444", "auto_fill_targeted_deepseek", "w8")])

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/build_union_fallback_resolution.py",
            "--primary-resolution-csv",
            str(primary_csv),
            "--fallback-resolution-csv",
            str(fallback_csv),
            "--fallback-label",
            "w8_all70",
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    rows = list(csv.DictReader((output_dir / "account_resolution_candidates.csv").open(encoding="utf-8-sig")))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert proc.returncode == 0
    assert rows[0]["decision"] == "auto_fill_union_fallback"
    assert rows[0]["chosen_account"] == "222-333-4444"
    assert summary["fallback_added"] == 1
