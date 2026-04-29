#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_ocr_server import mask_account  # noqa: E402
from scripts.run_ocrbench_mini_settlement_eval import CANONICAL_MODEL_NAMES, metric_block  # noqa: E402
from settlement_tool.free_running_gate import free_running_degeneration_metrics  # noqa: E402
from settlement_tool.ocrbench_v2_bankbook import (  # noqa: E402
    BankbookGold,
    BankbookPrediction,
    bankbook_ocrbench_v2_metrics,
    digits_only,
)


RESULT_FIELDS = [
    "model_short_label",
    "model_name",
    "id",
    "split",
    "name",
    "source_name",
    "label_account_masked",
    "full_ocr_account_masked",
    "policy_account_masked",
    "candidate_accounts_masked",
    "full_ocr_exact_match",
    "policy_exact_match",
    "candidate_exact_match",
    "full_ocr_false_positive",
    "policy_false_positive",
    "ocrbench_recognition_score",
    "account_digit_edit_similarity",
    "ocrbench_extraction_f1",
    "ocrbench_basic_vqa_score",
    "ocrbench_composite_score",
    "free_running_gate_pass",
    "surface_gate_pass",
    "degeneration_reason",
    "decision",
    "source",
    "raw_output_root",
]


def parse_model(value: str) -> tuple[str, str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be SHORT_LABEL=/path/to/practical_output_root")
    short_label, raw_path = value.split("=", 1)
    short_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", short_label.strip())
    if not short_label:
        raise argparse.ArgumentTypeError("model short label is empty")
    return short_label, CANONICAL_MODEL_NAMES.get(short_label, short_label), Path(raw_path).expanduser()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_matches(row_value: str, source_name: str) -> bool:
    return row_value == source_name or row_value.endswith("/" + source_name) or row_value.endswith(source_name)


def find_full_ocr_row(rows: list[dict[str, str]], source_name: str) -> dict[str, str]:
    for row in rows:
        if source_matches(row.get("source_name", ""), source_name) or source_matches(row.get("extracted_path", ""), source_name):
            return row
    return {}


def find_final_row(rows: list[dict[str, str]], name: str) -> dict[str, str]:
    for row in rows:
        if row.get("name") == name:
            return row
    return {}


def load_raw_text(row: dict[str, str]) -> str:
    text_path = Path(row.get("ocr_text_path", ""))
    if text_path.exists():
        return text_path.read_text(encoding="utf-8", errors="replace")
    return ""


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    exact = sum(int(row["full_ocr_exact_match"]) for row in rows)
    policy_exact = sum(int(row["policy_exact_match"]) for row in rows)
    candidate_exact = sum(int(row["candidate_exact_match"]) for row in rows)
    false_positive = sum(int(row["full_ocr_false_positive"]) for row in rows)
    policy_false_positive = sum(int(row["policy_false_positive"]) for row in rows)
    free_gate = sum(int(row["free_running_gate_pass"]) for row in rows)
    surface_gate = sum(int(row["surface_gate_pass"]) for row in rows)
    recognition_scores = [float(row["ocrbench_recognition_score"]) for row in rows]
    digit_scores = [float(row["account_digit_edit_similarity"]) for row in rows]
    extraction_scores = [float(row["ocrbench_extraction_f1"]) for row in rows]
    basic_scores = [float(row["ocrbench_basic_vqa_score"]) for row in rows]
    composite_scores = [float(row["ocrbench_composite_score"]) for row in rows]
    return {
        "total": total,
        "account_exact_match": exact,
        "account_exact_match_rate": exact / total if total else 0,
        "policy_pipeline_exact_match": policy_exact,
        "policy_pipeline_exact_match_rate": policy_exact / total if total else 0,
        "candidate_exact_match": candidate_exact,
        "candidate_exact_match_rate": candidate_exact / total if total else 0,
        "false_positive_count": false_positive,
        "policy_pipeline_false_positive_count": policy_false_positive,
        "error_count": 0,
        "free_running_gate_pass": free_gate,
        "free_running_gate_pass_rate": free_gate / total if total else 0,
        "surface_gate_pass": surface_gate,
        "surface_gate_pass_rate": surface_gate / total if total else 0,
        "ocrbench_v2_adapted": {
            "recognition_score_mean": mean(recognition_scores),
            "account_digit_edit_similarity_mean": mean(digit_scores),
            "extraction_f1_mean": mean(extraction_scores),
            "basic_vqa_score_mean": mean(basic_scores),
            "composite_score_mean": mean(composite_scores),
            "notes": [
                "Recognition follows exact account match plus normalized digit edit similarity.",
                "Extraction follows OCRBench v2 key-value F1 over bank/account_holder/account_number when labels exist.",
                "Basic VQA proxy penalizes false positive account-number selection.",
            ],
        },
    }


def evaluate_practical_root(
    *,
    short_label: str,
    model_name: str,
    root: Path,
    manifest_rows: list[dict[str, Any]],
    require_account_candidate: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_rows = read_csv(root / "full_ocr" / "deepseek_bank_zip_full_ocr.csv")
    final_rows = read_csv(root / "final_workbook" / "account_updates_deepseek_resolution.csv")
    output_rows: list[dict[str, Any]] = []

    for item in manifest_rows:
        label = item.get("label", {})
        source_name = item.get("source_name", "")
        name = item.get("name", "")
        full = find_full_ocr_row(full_rows, source_name)
        final = find_final_row(final_rows, name)
        raw_text = load_raw_text(full)
        full_account = full.get("account", "")
        final_account = final.get("account", "")
        candidates = [value.strip() for value in (full.get("candidates", "") or "").split(";") if value.strip()]
        if final.get("candidate_accounts_masked") and not candidates:
            candidates = []

        gold = BankbookGold(
            account_number=label.get("account_number", ""),
            bank="",
            account_holder="",
        )
        prediction = BankbookPrediction(raw_text=raw_text, account_number=full_account, candidate_accounts=tuple(candidates))
        metrics = bankbook_ocrbench_v2_metrics(gold, prediction)
        recognition = metrics["recognition"]
        extraction = metrics["extraction"]
        basic_vqa = metrics["basic_vqa"]

        expected_digits = digits_only(label.get("account_number", ""))
        full_exact = bool(expected_digits and digits_only(full_account) == expected_digits)
        policy_exact = bool(expected_digits and digits_only(final_account) == expected_digits)
        full_false_positive = bool(digits_only(full_account) and not full_exact)
        policy_false_positive = bool(digits_only(final_account) and not policy_exact)
        degeneration = free_running_degeneration_metrics(raw_text, require_account_candidate=require_account_candidate)
        free_gate_pass = bool(degeneration["degeneration_pass"])
        surface_gate_pass = free_gate_pass and not full_false_positive

        output_rows.append(
            {
                "model_short_label": short_label,
                "model_name": model_name,
                "id": item.get("id", ""),
                "split": item.get("split", ""),
                "name": name,
                "source_name": source_name,
                "label_account_masked": mask_account(label.get("account_number", "")),
                "full_ocr_account_masked": mask_account(full_account),
                "policy_account_masked": mask_account(final_account),
                "candidate_accounts_masked": "; ".join(mask_account(candidate) for candidate in candidates),
                "full_ocr_exact_match": int(full_exact),
                "policy_exact_match": int(policy_exact),
                "candidate_exact_match": int(bool(recognition["candidate_account_exact"])),
                "full_ocr_false_positive": int(full_false_positive),
                "policy_false_positive": int(policy_false_positive),
                "ocrbench_recognition_score": f"{float(recognition['score']):.4f}",
                "account_digit_edit_similarity": f"{float(recognition['account_digit_edit_similarity']):.4f}",
                "ocrbench_extraction_f1": f"{float(extraction['f1']):.4f}",
                "ocrbench_basic_vqa_score": f"{float(basic_vqa['score']):.4f}",
                "ocrbench_composite_score": f"{float(metrics['composite_score']):.4f}",
                "free_running_gate_pass": int(free_gate_pass),
                "surface_gate_pass": int(surface_gate_pass),
                "degeneration_reason": degeneration["degeneration_reason"],
                "decision": final.get("decision", ""),
                "source": final.get("source", ""),
                "raw_output_root": str(root),
            }
        )

    summary = summarize_rows(output_rows)
    summary["model_name"] = model_name
    summary["model_short_label"] = short_label
    summary["output_dir"] = str(root)
    return output_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize existing practical OCR outputs as an OCRBench v2-style mini eval.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", action="append", type=parse_model, required=True, help="SHORT_LABEL=/path/to/practical_output_root")
    parser.add_argument("--require-account-candidate", action="store_true")
    args = parser.parse_args()

    manifest_rows = read_manifest(args.manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_result_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for short_label, model_name, root in args.model:
        rows, summary = evaluate_practical_root(
            short_label=short_label,
            model_name=model_name,
            root=root,
            manifest_rows=manifest_rows,
            require_account_candidate=args.require_account_candidate,
        )
        all_result_rows.extend(rows)
        summaries[short_label] = summary

    write_csv(args.output_root / "offline_ocrbench_mini_results_masked.csv", all_result_rows)
    report = {
        "benchmark": "SettlementOCRBench-mini-offline",
        "manifest": str(args.manifest),
        "models": {
            label: {
                "model_name": summary["model_name"],
                "metric_block": metric_block(summary),
                "policy_pipeline": {
                    "exact_match": summary["policy_pipeline_exact_match"],
                    "exact_match_rate": summary["policy_pipeline_exact_match_rate"],
                    "false_positive_count": summary["policy_pipeline_false_positive_count"],
                    "note": "Workbook final policy output; kept separate from full_ocr model-surface scoring.",
                },
                "raw_output_dir": summary["output_dir"],
            }
            for label, summary in summaries.items()
        },
        "notes": [
            "This offline report recomputes OCRBench v2-style metrics from existing practical eval outputs.",
            "Rows are masked-only in the CSV output.",
            "Current labels come from previous high-confidence successes, so this is a schema smoke benchmark until independent review labels are added.",
        ],
    }
    (args.output_root / "offline_ocrbench_mini_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
