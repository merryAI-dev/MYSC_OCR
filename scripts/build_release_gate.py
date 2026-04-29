#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from settlement_tool.core import extract_roster, write_csv  # noqa: E402


AUTOFILL_REVIEW_DECISIONS = {
    "auto_fill_openai_reranker",
    "auto_fill_targeted_deepseek",
    "auto_fill_targeted_policy_reranker",
}
MANUAL_REVIEW_FIELDS = [
    "name",
    "decision",
    "chosen_account_masked",
    "source",
    "candidate_files",
    "bank_name",
    "bank_evidence",
    "bank_confidence",
    "review_status",
    "reviewer_id",
    "review_notes",
]
MANUAL_REVIEW_PRESERVED_FIELDS = [
    "bank_name",
    "bank_evidence",
    "bank_confidence",
    "review_status",
    "reviewer_id",
    "review_notes",
]
TEXT_SCAN_EXTENSIONS = {".csv", ".json", ".jsonl", ".txt", ".log", ".md", ".py", ".sh", ".command"}
RAW_ARTIFACT_NAMES = {
    "candidate_features_local.jsonl",
    "candidate_raw_map_local.jsonl",
    "kie_candidates_local.csv",
    "kie_evidence_local.jsonl",
}
RAW_ARTIFACT_PATTERNS = (
    re.compile(r".*계좌번호입력.*\.xlsx$"),
    re.compile(r".*account_updates.*\.csv$"),
    re.compile(r".*account_resolution_candidates\.csv$"),
    re.compile(r".*targeted_retry_ocr\.csv$"),
    re.compile(r".*deepseek_bank_zip_full_ocr\.csv$"),
)
RAW_ARTIFACT_DIRS = {
    "documents",
    "bank_zip_extracted",
    "deepseek_text",
    "ocr_text",
    "rendered",
    "variants",
    "work",
}
PII_RE = re.compile(
    r"010-?[0-9]{4}-?[0-9]{4}"
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"|(?<![0-9.])[0-9]{6}-?[1-4][0-9]{6}(?![0-9.])"
    r"|(?<![0-9.])(?:[0-9][0-9 -]{7,22}[0-9])(?![0-9.])"
)
LOCAL_PATH_OR_TOKEN_RE = re.compile(r"/Users/boram|Downloads|HF_TOKEN|hf_[A-Za-z0-9_]+")
REDACTED_ARTIFACT_NAMES = {
    "candidate_features_redacted.jsonl",
    "kie_evidence_redacted.jsonl",
    "decisions.jsonl",
    "openai_reranker_decisions.jsonl",
    "openai_reranker_report.json",
    "summary.json",
}
FORBIDDEN_REDACTED_KEY_RE = re.compile(r'"(?:candidate_raw|raw_text_local|source_name|ocr_text_path|image_path)"\s*:') 


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def _manual_review_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("name", ""),
        row.get("decision", ""),
        row.get("chosen_account_masked", ""),
        row.get("source", ""),
        row.get("candidate_files", ""),
    )


def _report_file_name(path: Path) -> str:
    return path.name


def build_manual_autofill_review_queue(
    *,
    resolution_csv: Path,
    output_dir: Path,
    decisions: set[str] | None = None,
) -> dict[str, Any]:
    decisions = decisions or AUTOFILL_REVIEW_DECISIONS
    queue_path = output_dir / "manual_autofill_review_queue.csv"
    existing_rows = read_csv(queue_path) if queue_path.exists() else []
    existing_by_key = {_manual_review_key(row): row for row in existing_rows}
    rows = []
    for row in read_csv(resolution_csv):
        if row.get("decision") not in decisions:
            continue
        queue_row = {
            "name": row.get("name", ""),
            "decision": row.get("decision", ""),
            "chosen_account_masked": row.get("chosen_account_masked", ""),
            "source": row.get("source", ""),
            "candidate_files": row.get("candidate_files", ""),
            "bank_name": "",
            "bank_evidence": "",
            "bank_confidence": "",
            "review_status": "pending",
            "reviewer_id": "",
            "review_notes": "",
        }
        existing = existing_by_key.get(_manual_review_key(queue_row))
        if existing:
            for field in MANUAL_REVIEW_PRESERVED_FIELDS:
                value = existing.get(field, "")
                if value:
                    queue_row[field] = value
        rows.append(queue_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(queue_path, rows, MANUAL_REVIEW_FIELDS)
    confirmed_count = sum(1 for row in rows if row.get("review_status", "").strip().lower() == "confirmed")
    return {
        "queue_path": _report_file_name(queue_path),
        "required_count": len(rows),
        "pending_count": len(rows) - confirmed_count,
        "confirmed_count": confirmed_count,
        "decisions": sorted(decisions),
    }


def _is_raw_artifact(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if path.name in RAW_ARTIFACT_NAMES:
        return True
    if any(part in RAW_ARTIFACT_DIRS for part in rel.parts[:-1]):
        return True
    return any(pattern.fullmatch(path.name) for pattern in RAW_ARTIFACT_PATTERNS)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _normalize_sensitive_terms(terms: set[str] | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    normalized = {str(term).strip() for term in terms or [] if str(term).strip()}
    return tuple(sorted(term for term in normalized if len(term) >= 2))


def _contains_sensitive_term(text: str, sensitive_terms: tuple[str, ...]) -> bool:
    return bool(sensitive_terms) and any(term in text for term in sensitive_terms)


def load_sensitive_terms_from_workbook(workbook_path: Path) -> set[str]:
    return {person.name for person in extract_roster(workbook_path).people if person.name}


def _xlsx_contains_pii(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as workbook:
            for name in workbook.namelist():
                if not (name == "xl/sharedStrings.xml" or name.startswith("xl/worksheets/")):
                    continue
                text = workbook.read(name).decode("utf-8", errors="ignore")
                if PII_RE.search(text) or LOCAL_PATH_OR_TOKEN_RE.search(text):
                    return True
    except zipfile.BadZipFile:
        return False
    return False


def _xlsx_contains_sensitive_term(path: Path, sensitive_terms: tuple[str, ...]) -> bool:
    if not sensitive_terms:
        return False
    try:
        with zipfile.ZipFile(path) as workbook:
            for name in workbook.namelist():
                if not (name == "xl/sharedStrings.xml" or name.startswith("xl/worksheets/")):
                    continue
                text = workbook.read(name).decode("utf-8", errors="ignore")
                if _contains_sensitive_term(text, sensitive_terms):
                    return True
    except zipfile.BadZipFile:
        return False
    return False


def scan_release_bundle(bundle_path: Path, *, sensitive_terms: set[str] | list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    bundle_path = bundle_path.resolve()
    sensitive_terms_normalized = _normalize_sensitive_terms(sensitive_terms)
    blocked_artifact_paths: list[str] = []
    pii_match_paths: list[str] = []
    local_path_or_token_paths: list[str] = []
    sensitive_term_match_paths: list[str] = []
    forbidden_redacted_key_paths: list[str] = []

    for path in sorted(item for item in bundle_path.rglob("*") if item.is_file()):
        rel = str(path.relative_to(bundle_path))
        if _is_raw_artifact(path, bundle_path):
            blocked_artifact_paths.append(rel)

        if path.suffix in TEXT_SCAN_EXTENSIONS:
            text = _read_text(path)
            if PII_RE.search(text):
                pii_match_paths.append(rel)
            if LOCAL_PATH_OR_TOKEN_RE.search(text):
                local_path_or_token_paths.append(rel)
            if _contains_sensitive_term(text, sensitive_terms_normalized):
                sensitive_term_match_paths.append(rel)
            if path.name in REDACTED_ARTIFACT_NAMES and FORBIDDEN_REDACTED_KEY_RE.search(text):
                forbidden_redacted_key_paths.append(rel)
        elif path.suffix == ".xlsx":
            if _xlsx_contains_pii(path):
                pii_match_paths.append(rel)
            if _xlsx_contains_sensitive_term(path, sensitive_terms_normalized):
                sensitive_term_match_paths.append(rel)

    return {
        "bundle_path": _report_file_name(bundle_path),
        "blocked_artifact_count": len(blocked_artifact_paths),
        "blocked_artifact_paths": blocked_artifact_paths,
        "pii_match_count": len(pii_match_paths),
        "pii_match_paths": pii_match_paths,
        "local_path_or_token_count": len(local_path_or_token_paths),
        "local_path_or_token_paths": local_path_or_token_paths,
        "sensitive_term_match_count": len(sensitive_term_match_paths),
        "sensitive_term_match_paths": sensitive_term_match_paths,
        "forbidden_redacted_key_count": len(forbidden_redacted_key_paths),
        "forbidden_redacted_key_paths": forbidden_redacted_key_paths,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reranker_gate(reranker_eval_path: Path) -> dict[str, Any]:
    payload = _load_json(reranker_eval_path)
    if "summary" in payload and "metrics" in payload:
        summary = payload.get("summary", {})
        metrics = payload.get("metrics", {})
        wrong_positive = int(summary.get("wrong_positive") or 0)
        review_false_positive = int(summary.get("review_false_positive") or 0)
        safe_selection_precision = float(metrics.get("safe_selection_precision") or 0.0)
        positive_recall = float(metrics.get("positive_recall") or 0.0)
        passed = wrong_positive == 0 and review_false_positive == 0 and safe_selection_precision >= 1.0
        return {
            "path": _report_file_name(reranker_eval_path),
            "eval_format": "human_label_eval",
            "passed": passed,
            "wrong_positive": wrong_positive,
            "review_false_positive": review_false_positive,
            "safe_selection_precision": safe_selection_precision,
            "positive_recall": positive_recall,
            "correct_positive": int(summary.get("correct_positive") or 0),
            "human_positive_count": int(summary.get("human_positive_count") or 0),
        }

    best = payload.get("best", {})
    passed = (
        int(best.get("selected_reject_count") or 0) == 0
        and float(best.get("selection_precision") or 0.0) >= 1.0
    )
    return {
        "path": _report_file_name(reranker_eval_path),
        "eval_format": "sweep_summary",
        "passed": passed,
        "best_threshold": payload.get("best_threshold"),
        "best_margin": payload.get("best_margin"),
        "selected_reject_count": int(best.get("selected_reject_count") or 0),
        "selection_precision": float(best.get("selection_precision") or 0.0),
        "source_accept_recall": float(best.get("source_accept_recall") or 0.0),
    }


def _overall_status(*, manual_review: dict[str, Any], pii_gate: dict[str, Any], reranker_gate: dict[str, Any]) -> str:
    if not pii_gate["passed"]:
        return "blocked_pii"
    if not reranker_gate["passed"]:
        return "blocked_reranker"
    if int(manual_review["pending_count"]) > 0:
        return "blocked_manual_review"
    return "passed"


def build_release_gate_outputs(
    *,
    resolution_csv: Path,
    reranker_eval_path: Path,
    bundle_path: Path,
    output_dir: Path,
    sensitive_terms: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_review = build_manual_autofill_review_queue(resolution_csv=resolution_csv, output_dir=output_dir)
    pii_scan = scan_release_bundle(bundle_path, sensitive_terms=sensitive_terms)
    pii_gate = {
        "passed": (
            pii_scan["blocked_artifact_count"] == 0
            and pii_scan["pii_match_count"] == 0
            and pii_scan["local_path_or_token_count"] == 0
            and pii_scan["sensitive_term_match_count"] == 0
            and pii_scan["forbidden_redacted_key_count"] == 0
        ),
        **pii_scan,
    }
    reranker = _reranker_gate(reranker_eval_path)
    report = {
        "overall_status": _overall_status(
            manual_review=manual_review,
            pii_gate=pii_gate,
            reranker_gate=reranker,
        ),
        "manual_review": manual_review,
        "pii_gate": pii_gate,
        "reranker_gate": reranker,
    }
    (output_dir / "release_gate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build release-gate review queue, PII scan, and reranker gate report.")
    parser.add_argument("--resolution-csv", type=Path, required=True)
    parser.add_argument("--reranker-eval", type=Path, required=True)
    parser.add_argument("--bundle-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sensitive-workbook", type=Path, help="Optional workbook used only to load exact-match sensitive person names for leak scanning.")
    parser.add_argument("--sensitive-term", action="append", default=[], help="Optional exact sensitive term to scan for without echoing the term in reports.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sensitive_terms = set(args.sensitive_term or [])
    if args.sensitive_workbook:
        sensitive_terms.update(load_sensitive_terms_from_workbook(args.sensitive_workbook))
    report = build_release_gate_outputs(
        resolution_csv=args.resolution_csv,
        reranker_eval_path=args.reranker_eval,
        bundle_path=args.bundle_path,
        output_dir=args.output_dir,
        sensitive_terms=sensitive_terms,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
