#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _command_name(command: list[str]) -> str:
    script = Path(command[1]).stem if len(command) > 1 else Path(command[0]).stem
    return script


def _run(command: list[str], *, cwd: Path, allowed_exit_codes: tuple[int, ...] = (0,)) -> None:
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if proc.returncode not in allowed_exit_codes:
        raise RuntimeError(
            f"{_command_name(command)} failed with exit {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_commands(args: argparse.Namespace, *, repo_root: Path) -> list[list[str]]:
    payload_dir = args.output_dir / "payloads"
    reranker_dir = args.output_dir / "reranker"
    resolution_dir = args.output_dir / "resolution"
    eval_dir = args.output_dir / "eval"
    compare_dir = args.output_dir / "compare"
    release_gate_dir = args.output_dir / "release_gate"
    redacted_bundle_dir = args.output_dir / "redacted_bundle"

    build_payloads = [
        sys.executable,
        "scripts/build_codex_teacher_distill.py",
        "--output-dir",
        str(payload_dir),
        "--backend",
        args.backend,
    ]
    for path in args.payload_input_csv:
        build_payloads.extend(["--input-csv", str(path)])
    for path in args.payload_input_kie_csv:
        build_payloads.extend(["--input-kie-csv", str(path)])

    return [
        build_payloads,
        [
            sys.executable,
            "scripts/local_feature_score_rerank.py",
            "--payload-jsonl",
            str(payload_dir / "candidate_features_redacted.jsonl"),
            "--output-jsonl",
            str(reranker_dir / "decisions.jsonl"),
            "--threshold",
            str(args.threshold),
            "--min-margin",
            str(args.min_margin),
        ],
        [
            sys.executable,
            "scripts/apply_openai_reranker_decisions.py",
            "--source-workbook",
            str(args.source_workbook),
            "--raw-map-jsonl",
            str(payload_dir / "candidate_raw_map_local.jsonl"),
            "--decisions-jsonl",
            str(reranker_dir / "decisions.jsonl"),
            "--output-dir",
            str(resolution_dir),
            "--manual-review-workbook",
            str(args.human_workbook),
        ],
        [
            sys.executable,
            "scripts/evaluate_human_workbook_labels.py",
            "--human-workbook",
            str(args.human_workbook),
            "--resolution-csv",
            str(resolution_dir / "account_resolution_candidates.csv"),
            "--output-dir",
            str(eval_dir),
            "--data-zip",
            str(args.data_zip),
        ],
        [
            sys.executable,
            "scripts/compare_human_eval_reports.py",
            "--report",
            f"experiment={eval_dir / 'human_label_eval.json'}",
            "--output-dir",
            str(compare_dir),
        ],
        [
            sys.executable,
            "scripts/build_release_gate.py",
            "--resolution-csv",
            str(resolution_dir / "account_resolution_candidates.csv"),
            "--reranker-eval",
            str(eval_dir / "human_label_eval.json"),
            "--bundle-path",
            str(redacted_bundle_dir),
            "--output-dir",
            str(release_gate_dir),
            "--sensitive-workbook",
            str(args.human_workbook),
        ],
    ]


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_candidate_recall_experiment(args: argparse.Namespace, *, repo_root: Path) -> dict[str, Any]:
    commands = build_commands(args, repo_root=repo_root)
    planned_steps = [_command_name(command) for command in commands]
    summary: dict[str, Any] = {
        "dry_run": bool(args.dry_run),
        "threshold": float(args.threshold),
        "min_margin": float(args.min_margin),
        "planned_steps": planned_steps,
        "output_dir": str(args.output_dir),
    }

    if args.dry_run:
        write_summary(args.output_dir, summary)
        return summary

    for directory in ("payloads", "reranker", "resolution", "eval", "compare", "release_gate", "redacted_bundle"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)

    for command in commands[:4]:
        _run(command, cwd=repo_root)

    redacted_bundle = args.output_dir / "redacted_bundle"
    _copy_if_exists(args.output_dir / "payloads" / "candidate_features_redacted.jsonl", redacted_bundle / "candidate_features_redacted.jsonl")
    _copy_if_exists(args.output_dir / "reranker" / "decisions.jsonl", redacted_bundle / "decisions.jsonl")
    for path in args.redacted_artifact:
        _copy_if_exists(path, redacted_bundle / path.name)

    for command in commands[4:]:
        allowed = (0, 1) if _command_name(command) == "build_release_gate" else (0,)
        _run(command, cwd=repo_root, allowed_exit_codes=allowed)

    eval_report = json.loads((args.output_dir / "eval" / "human_label_eval.json").read_text(encoding="utf-8"))
    release_report = json.loads((args.output_dir / "release_gate" / "release_gate_report.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "dry_run": False,
            "eval_summary": eval_report.get("summary", {}),
            "eval_metrics": eval_report.get("metrics", {}),
            "release_gate": {
                "overall_status": release_report.get("overall_status"),
                "pii_passed": release_report.get("pii_gate", {}).get("passed"),
                "reranker_passed": release_report.get("reranker_gate", {}).get("passed"),
            },
        }
    )
    write_summary(args.output_dir, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a candidate-generation recall experiment through rerank, eval, and release gates.")
    parser.add_argument("--payload-input-csv", action="append", type=Path, default=[])
    parser.add_argument("--payload-input-kie-csv", action="append", type=Path, default=[])
    parser.add_argument("--redacted-artifact", action="append", type=Path, default=[])
    parser.add_argument("--human-workbook", type=Path, required=True)
    parser.add_argument("--source-workbook", type=Path, required=True)
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--min-margin", type=float, default=2.0)
    parser.add_argument("--backend", default="mixed_candidate_generation")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args()
    try:
        summary = run_candidate_recall_experiment(args, repo_root=repo_root)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
