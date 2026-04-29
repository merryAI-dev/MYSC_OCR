#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %z")


def read_text(path: Path, limit_bytes: int = 200_000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) > limit_bytes:
        data = data[-limit_bytes:]
    return data.decode("utf-8", errors="replace")


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return max(0, sum(1 for _ in csv.reader(handle)) - 1)
    except Exception:
        return 0


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return sum(1 for _ in handle)
    except Exception:
        return 0


def compact_log_tail(text: str, max_lines: int = 10) -> list[str]:
    # tqdm writes carriage-return progress lines. Keep the latest readable view.
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = []
    for line in lines[-80:]:
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
        if len(line) > 240:
            line = line[:237] + "..."
        cleaned.append(line)
    return cleaned[-max_lines:]


def event_lines(log_text: str) -> list[dict[str, object]]:
    events = []
    for line in log_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("event"):
            events.append(payload)
    return events


def status_from(log_text: str, summary: dict[str, object]) -> str:
    if summary:
        if summary.get("skipped_total") == 0:
            return "all_resolved_or_complete"
        return "running_cycle_reports_available"
    if "Loading weights:" in log_text:
        return "loading_chandra_weights"
    if "Starting hypothesis-driven" in log_text:
        return "running_first_cycle"
    if "Waiting for DeepSeek server" in log_text:
        return "waiting_deepseek_server"
    if "deepseek_health_error" in log_text:
        return "deepseek_startup_retry"
    return "starting_or_no_log_yet"


def ensure_header(wiki_file: Path, output_dir: Path) -> None:
    if wiki_file.exists() and wiki_file.stat().st_size > 0:
        return
    wiki_file.parent.mkdir(parents=True, exist_ok=True)
    wiki_file.write_text(
        "\n".join(
            [
                "---",
                "type: ocr-autoresearch-log",
                f"created: {now()}",
                "privacy: masked-no-account-numbers",
                "---",
                "",
                "# Hackathon Settlement OCR Autoresearch",
                "",
                f"- output_dir: `{output_dir}`",
                "- cadence: 1 minute",
                "- note: 계좌번호 원문은 기록하지 않고 진행 상태와 실패 유형만 기록한다.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def append_report(output_dir: Path, wiki_file: Path) -> bool:
    log_text = read_text(output_dir / "tmux_driver.log")
    summary = read_json(output_dir / "summary.json")
    events = event_lines(log_text)
    latest_event = events[-1] if events else {}
    status = status_from(log_text, summary)

    attempts = count_csv_rows(output_dir / "attempts.csv")
    failures = count_lines(output_dir / "failure_corpus.jsonl")
    ocr_texts = len(list((output_dir / "ocr_text").glob("*.txt"))) if (output_dir / "ocr_text").exists() else 0
    workbooks = sorted(output_dir.glob("*_계좌번호입력_autoresearch.xlsx"))
    resolution_exists = (output_dir / "account_resolution_candidates.csv").exists()

    lines = [
        f"## {now()}",
        "",
        f"- status: `{status}`",
        f"- latest_event: `{latest_event.get('event', '')}`",
        f"- cycle: `{summary.get('cycle', latest_event.get('cycle', ''))}`",
        f"- updated_total: `{summary.get('updated_total', '')}`",
        f"- skipped_total: `{summary.get('skipped_total', '')}`",
        f"- autoresearch_chosen: `{summary.get('autoresearch_chosen', '')}`",
        f"- targets: `{summary.get('targets', latest_event.get('targets', ''))}`",
        f"- missing_files: `{summary.get('missing_files', '')}`",
        f"- attempts_rows: `{attempts}`",
        f"- failure_entries: `{failures}`",
        f"- ocr_text_files: `{ocr_texts}`",
        f"- resolution_csv: `{'yes' if resolution_exists else 'no'}`",
        f"- latest_workbook: `{workbooks[-1] if workbooks else ''}`",
        "",
        "Recent log:",
        "",
        "```text",
        *compact_log_tail(log_text),
        "```",
        "",
    ]
    with wiki_file.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return latest_event.get("event") in {"finish", "all_resolved"} or "finished_at:" in log_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wiki-file", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--max-hours", type=float, default=13)
    args = parser.parse_args()

    ensure_header(args.wiki_file, args.output_dir)
    deadline = time.time() + args.max_hours * 3600
    while time.time() < deadline:
        done = append_report(args.output_dir, args.wiki_file)
        if done:
            break
        time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
