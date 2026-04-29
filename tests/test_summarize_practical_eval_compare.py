import json
from pathlib import Path

from scripts.summarize_practical_eval_compare import build_comparison


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_build_comparison_reads_core_stage_metrics(tmp_path: Path):
    six = tmp_path / "six"
    eight = tmp_path / "eight"
    write_json(six / "full_ocr" / "summary.json", {"members": 73, "high_accounts": 10, "matched_names": 60})
    write_json(six / "policy_resolution" / "summary.json", {"auto_fill_single_candidate": 7, "policy_rejected_rows": 1})
    write_json(six / "targeted_retry" / "summary.json", {"targets": 20, "high_accounts": 3})
    write_json(six / "policy_resolution_targeted" / "summary.json", {"filled": 51, "policy_rejected_rows": 4})
    write_json(six / "final_workbook" / "summary.json", {"updated": 51, "skipped": 22})

    write_json(eight / "full_ocr" / "summary.json", {"members": 73, "high_accounts": 12, "matched_names": 61})
    write_json(eight / "policy_resolution" / "summary.json", {"auto_fill_single_candidate": 8, "policy_rejected_rows": 0})
    write_json(eight / "targeted_retry" / "summary.json", {"targets": 19, "high_accounts": 4})
    write_json(eight / "policy_resolution_targeted" / "summary.json", {"filled": 53, "policy_rejected_rows": 2})
    write_json(eight / "final_workbook" / "summary.json", {"updated": 53, "skipped": 20})

    comparison = build_comparison(("bf16_direct_6bit", six), ("mlx8bit", eight))

    rows = {(row["stage"], row["metric"]): row for row in comparison["rows"]}
    assert rows[("full_ocr", "high_accounts")]["bf16_direct_6bit"] == 10
    assert rows[("full_ocr", "high_accounts")]["mlx8bit"] == 12
    assert rows[("full_ocr", "high_accounts")]["delta"] == -2
    assert rows[("final_workbook", "updated")]["delta"] == -2
    assert comparison["models"] == ["bf16_direct_6bit", "mlx8bit"]
