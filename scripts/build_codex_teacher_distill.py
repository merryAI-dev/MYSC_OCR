#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from settlement_tool.teacher_distill import features_from_kie_csv, features_from_ocr_csv, seed_teacher_policy_label, teacher_review_record  # noqa: E402
from scripts.build_openai_reranker_payloads import build_openai_reranker_payloads, write_jsonl as write_openai_jsonl  # noqa: E402


def read_gold_manifest(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    gold: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("label", {})
            account = label.get("account_number", "")
            if not account:
                continue
            for key in (row.get("source_name", ""), Path(row.get("image_path", "")).name):
                if key:
                    gold[key] = account
    return gold


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_teacher_distill_outputs(
    *,
    input_csvs: list[Path],
    output_dir: Path,
    input_kie_csvs: list[Path] | None = None,
    gold_by_source_name: dict[str, str] | None = None,
    backend: str = "",
    include_phone_like: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    features: list[dict[str, Any]] = []
    for input_csv in input_csvs:
        features.extend(
            features_from_ocr_csv(
                input_csv,
                gold_by_source_name=gold_by_source_name or {},
                backend=backend,
                include_phone_like=include_phone_like,
            )
        )
    kie_features: list[dict[str, Any]] = []
    for input_kie_csv in input_kie_csvs or []:
        kie_features.extend(
            features_from_kie_csv(
                input_kie_csv,
                gold_by_source_name=gold_by_source_name or {},
                backend=backend,
            )
        )
    features.extend(kie_features)

    review_rows = [teacher_review_record(feature) for feature in features]
    seed_labels = [seed_teacher_policy_label(feature) for feature in features]
    openai_payloads, raw_maps = build_openai_reranker_payloads(features)
    write_jsonl(output_dir / "candidate_features_local.jsonl", features)
    write_openai_jsonl(output_dir / "candidate_features_redacted.jsonl", openai_payloads)
    write_openai_jsonl(output_dir / "candidate_raw_map_local.jsonl", raw_maps)
    write_jsonl(output_dir / "teacher_review_queue.jsonl", review_rows)
    write_jsonl(output_dir / "codex_policy_label_seed.jsonl", seed_labels)

    labeled = [feature for feature in features if feature.get("gold_label_available")]
    exact = sum(1 for feature in labeled if feature.get("gold_exact_match"))
    requires_review = sum(1 for row in seed_labels if row.get("requires_teacher_review"))
    label_counts: dict[str, int] = {}
    for row in seed_labels:
        label = str(row.get("suggested_label", ""))
        label_counts[label] = label_counts.get(label, 0) + 1
    summary = {
        "input_csvs": [str(path) for path in input_csvs],
        "input_kie_csvs": [str(path) for path in input_kie_csvs or []],
        "candidate_count": len(features),
        "kie_candidate_count": len(kie_features),
        "gold_labeled_candidate_count": len(labeled),
        "gold_exact_match_count": exact,
        "gold_exact_match_rate": exact / len(labeled) if labeled else 0.0,
        "suggested_label_counts": label_counts,
        "requires_teacher_review_count": requires_review,
        "backend": backend,
        "outputs": {
            "candidate_features_local": str(output_dir / "candidate_features_local.jsonl"),
            "candidate_features_redacted": str(output_dir / "candidate_features_redacted.jsonl"),
            "candidate_raw_map_local": str(output_dir / "candidate_raw_map_local.jsonl"),
            "teacher_review_queue": str(output_dir / "teacher_review_queue.jsonl"),
            "codex_policy_label_seed": str(output_dir / "codex_policy_label_seed.jsonl"),
        },
        "notes": [
            "candidate_features_local may contain raw account candidates and must stay local.",
            "candidate_features_redacted is the only candidate feature artifact intended for OpenAI reranking.",
            "candidate_raw_map_local contains raw candidate-to-id mappings and must stay local.",
            "teacher_review_queue removes candidate_raw and masks long digit groups for Codex/manual policy review.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PII-minimized Codex teacher distillation records from OCR CSV outputs.")
    parser.add_argument("--input-csv", action="append", type=Path, default=[])
    parser.add_argument("--input-kie-csv", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gold-manifest", type=Path, default=None)
    parser.add_argument("--backend", default="")
    parser.add_argument("--include-phone-like", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_teacher_distill_outputs(
        input_csvs=args.input_csv,
        input_kie_csvs=args.input_kie_csv,
        output_dir=args.output_dir,
        gold_by_source_name=read_gold_manifest(args.gold_manifest),
        backend=args.backend,
        include_phone_like=args.include_phone_like,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
