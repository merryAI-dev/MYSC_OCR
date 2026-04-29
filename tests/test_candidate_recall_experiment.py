import json
import subprocess
import sys
from pathlib import Path


def test_candidate_recall_experiment_builds_commands_in_dry_run(tmp_path: Path):
    output_dir = tmp_path / "experiment"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_candidate_recall_experiment.py",
            "--payload-input-csv",
            "full.csv",
            "--payload-input-csv",
            "retry.csv",
            "--payload-input-kie-csv",
            "kie.csv",
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
    assert summary["threshold"] == 10.0
    assert summary["min_margin"] == 2.0
    assert "build_codex_teacher_distill" in " ".join(summary["planned_steps"])
