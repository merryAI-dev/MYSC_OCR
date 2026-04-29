#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_ocr_server import (  # noqa: E402
    EvaluationItem,
    PROMPT_PRESETS,
    check_health,
    evaluate_items,
    post_ocr,
    read_manifest,
    render_for_api,
)
from settlement_tool.image_variants import parse_variant_ids, render_image_variants  # noqa: E402


PROMPT_SWEEP_PRESETS = {
    "bank_zip_full_success": PROMPT_PRESETS["bank_zip_full_success"],
    "account_only_en": "<image> Read the account number only.",
    "account_only_ko": "<image> 계좌번호만 읽어줘. 보이는 숫자 그대로.",
    "candidate_lines": (
        "<image>\n"
        "Output only visible Korean bank account candidates. "
        "Use one line per candidate: account_candidate: <number>. Do not repeat."
    ),
    "copy_constraint": (
        "<image>\n"
        "Do not explain. Do not repeat. Copy only visible account-number digits and hyphens."
    ),
    "bank_fields": PROMPT_PRESETS["bank_fields"],
    "account_candidates": PROMPT_PRESETS["account_candidates"],
}


SURFACE_CONFIGS = {
    "scene_photo_small": ("Scene", "Photo", "Small"),
    "scene_photo_medium": ("Scene", "Photo", "Medium"),
    "scene_verification_tiny": ("Scene", "Verification", "Tiny"),
    "scene_verification_medium": ("Scene", "Verification", "Medium"),
}


DECODING_CONFIGS = {
    "baseline": {},
    "short48": {"max_tokens": 48},
    "short64_rep110": {"max_tokens": 64, "repetition_penalty": 1.10, "repetition_context_size": 32},
    "short64_rep115": {"max_tokens": 64, "repetition_penalty": 1.15, "repetition_context_size": 32},
    "short64_rep125": {"max_tokens": 64, "repetition_penalty": 1.25, "repetition_context_size": 32},
    "short64_rep115_salvage": {
        "max_tokens": 64,
        "repetition_penalty": 1.15,
        "repetition_context_size": 32,
        "prefix_salvage": True,
    },
    "short64_rep115_salvage_stop": {
        "max_tokens": 64,
        "repetition_penalty": 1.15,
        "repetition_context_size": 32,
        "prefix_salvage": True,
        "early_stop_account": True,
    },
}


@dataclass(frozen=True)
class SweepVariant:
    prompt_id: str
    prompt: str
    config_id: str
    content_type: str
    subcategory: str
    complexity: str
    decoding_id: str
    decoding_config: dict[str, object]

    @property
    def variant_id(self) -> str:
        return f"{self.prompt_id}__{self.config_id}__{self.decoding_id}"


def base_item_id(item_id: str) -> str:
    return item_id.split("::", 1)[0]


def _parse_csv_bool(row: dict[str, str], key: str) -> bool:
    return row.get(key, "0") in {"1", "true", "True"}


def _parse_csv_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0)
    except ValueError:
        return 0.0


def reranker_score(row: dict[str, str]) -> float:
    score = 0.0
    score += 100.0 if _parse_csv_bool(row, "surface_gate_pass") else 0.0
    score += 25.0 if _parse_csv_bool(row, "free_running_gate_pass") else 0.0
    score += 15.0 if _parse_csv_bool(row, "account_candidate_presence") else 0.0
    score += 5.0 * _parse_csv_float(row, "unique_token_ratio")
    score -= 3.0 * _parse_csv_float(row, "top_token_share")
    score -= 0.2 * _parse_csv_float(row, "max_token_run")
    return score


def read_rows(path: Path, variant_id: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["variant_id"] = variant_id
        row["reranker_score"] = f"{reranker_score(row):.4f}"
    return rows


def expand_items_with_image_variants(
    items: list[EvaluationItem],
    output_dir: Path,
    variant_ids: list[str],
) -> list[EvaluationItem]:
    if variant_ids == ["full"]:
        return items
    expanded: list[EvaluationItem] = []
    render_dir = output_dir / "_rendered_for_variants"
    variant_dir = output_dir / "_image_variants"
    for item in items:
        source_image = render_for_api(item.image_path, render_dir)
        rendered_variants = render_image_variants(
            source_image,
            variant_dir / item.item_id,
            item_id=item.item_id,
            variant_ids=variant_ids,
        )
        for rendered in rendered_variants:
            expanded.append(
                EvaluationItem(
                    item_id=f"{item.item_id}::{rendered.variant_id}",
                    split=item.split,
                    name=item.name,
                    image_path=rendered.path,
                    label_account_number=item.label_account_number,
                    label_bank=item.label_bank,
                    label_account_holder=item.label_account_holder,
                )
            )
    return expanded


def summarize_reranker(rows_by_id: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    selected: list[dict[str, str]] = []
    for item_id, rows in sorted(rows_by_id.items()):
        if not rows:
            continue
        best = max(rows, key=reranker_score)
        selected.append(best | {"id": item_id})

    total = len(selected)
    exact = sum(int(_parse_csv_bool(row, "exact_match")) for row in selected)
    candidate_exact = sum(int(_parse_csv_bool(row, "candidate_exact_match")) for row in selected)
    false_positive = sum(int(_parse_csv_bool(row, "false_positive")) for row in selected)
    free_running_gate = sum(int(_parse_csv_bool(row, "free_running_gate_pass")) for row in selected)
    surface_gate = sum(int(_parse_csv_bool(row, "surface_gate_pass")) for row in selected)
    variant_counts: dict[str, int] = {}
    for row in selected:
        variant_counts[row["variant_id"]] = variant_counts.get(row["variant_id"], 0) + 1

    return {
        "reranker_id": "gate_candidate_density_v1",
        "total": total,
        "account_exact_match": exact,
        "account_exact_match_rate": exact / total if total else 0.0,
        "candidate_exact_match": candidate_exact,
        "candidate_exact_match_rate": candidate_exact / total if total else 0.0,
        "false_positive_count": false_positive,
        "free_running_gate_pass": free_running_gate,
        "free_running_gate_pass_rate": free_running_gate / total if total else 0.0,
        "surface_gate_pass": surface_gate,
        "surface_gate_pass_rate": surface_gate / total if total else 0.0,
        "selected_variant_counts": variant_counts,
    }


def summarize_variant_harvest(rows: list[dict[str, str]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(base_item_id(row["id"]), []).append(row)

    selected: list[dict[str, str]] = []
    for item_id, item_rows in sorted(grouped.items()):
        best = max(item_rows, key=reranker_score)
        selected.append(best | {"base_id": item_id})

    total = len(selected)
    candidate_presence = sum(int(_parse_csv_bool(row, "account_candidate_presence")) for row in selected)
    exact = sum(int(_parse_csv_bool(row, "exact_match")) for row in selected)
    candidate_exact = sum(int(_parse_csv_bool(row, "candidate_exact_match")) for row in selected)
    false_positive = sum(int(_parse_csv_bool(row, "false_positive")) for row in selected)
    selected_variant_counts: dict[str, int] = {}
    for row in selected:
        image_variant = row["id"].split("::", 1)[1] if "::" in row["id"] else "full"
        selected_variant_counts[image_variant] = selected_variant_counts.get(image_variant, 0) + 1

    return {
        "reranker_id": "variant_candidate_vote_v1",
        "total": total,
        "account_candidate_presence": candidate_presence,
        "account_candidate_presence_rate": candidate_presence / total if total else 0.0,
        "account_exact_match": exact,
        "account_exact_match_rate": exact / total if total else 0.0,
        "candidate_exact_match": candidate_exact,
        "candidate_exact_match_rate": candidate_exact / total if total else 0.0,
        "false_positive_count": false_positive,
        "selected_image_variant_counts": selected_variant_counts,
        "notes": [
            "This summary groups crop/variant rows by original item id.",
            "Selection uses gate/candidate/repetition metrics only, not the gold account value.",
            "Candidate values remain masked in persisted CSV outputs.",
        ],
    }


def build_variants(prompt_ids: list[str], config_ids: list[str], decoding_ids: list[str]) -> list[SweepVariant]:
    variants: list[SweepVariant] = []
    for prompt_id in prompt_ids:
        if prompt_id not in PROMPT_SWEEP_PRESETS:
            raise SystemExit(f"Unknown prompt id: {prompt_id}")
        for config_id in config_ids:
            if config_id not in SURFACE_CONFIGS:
                raise SystemExit(f"Unknown surface config id: {config_id}")
            content_type, subcategory, complexity = SURFACE_CONFIGS[config_id]
            for decoding_id in decoding_ids:
                if decoding_id not in DECODING_CONFIGS:
                    raise SystemExit(f"Unknown decoding id: {decoding_id}")
                variants.append(
                    SweepVariant(
                        prompt_id=prompt_id,
                        prompt=PROMPT_SWEEP_PRESETS[prompt_id],
                        config_id=config_id,
                        content_type=content_type,
                        subcategory=subcategory,
                        complexity=complexity,
                        decoding_id=decoding_id,
                        decoding_config=DECODING_CONFIGS[decoding_id],
                    )
                )
    return variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:5001")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--prompt-ids",
        default="account_only_en,account_only_ko,candidate_lines,copy_constraint",
        help="Comma-separated prompt ids.",
    )
    parser.add_argument("--config-ids", default="scene_photo_small", help="Comma-separated surface config ids.")
    parser.add_argument("--decoding-ids", default="baseline", help="Comma-separated decoding config ids.")
    parser.add_argument("--variant-ids", default="full", help="Comma-separated image variant ids, or 'default'.")
    parser.add_argument("--require-account-candidate", action="store_true")
    args = parser.parse_args()

    check_health(args.api_url)
    splits = {split.strip() for split in args.splits.split(",") if split.strip()}
    items = read_manifest(args.manifest, splits)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("No evaluation items found.")
    variant_ids = parse_variant_ids(args.variant_ids)
    eval_items = expand_items_with_image_variants(items, args.output_dir, variant_ids)

    prompt_ids = [value.strip() for value in args.prompt_ids.split(",") if value.strip()]
    config_ids = [value.strip() for value in args.config_ids.split(",") if value.strip()]
    decoding_ids = [value.strip() for value in args.decoding_ids.split(",") if value.strip()]
    variants = build_variants(prompt_ids, config_ids, decoding_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, object] = {}
    rows_by_id: dict[str, list[dict[str, str]]] = {}
    for variant in variants:
        variant_dir = args.output_dir / variant.variant_id
        summary = evaluate_items(
            eval_items,
            output_dir=variant_dir,
            ocr_fn=lambda image_path, selected=variant: post_ocr(
                image_path,
                args.api_url,
                args.timeout,
                selected.prompt,
                content_type=selected.content_type,
                subcategory=selected.subcategory,
                complexity=selected.complexity,
                max_tokens=selected.decoding_config.get("max_tokens"),
                temperature=selected.decoding_config.get("temperature"),
                top_p=selected.decoding_config.get("top_p"),
                repetition_penalty=selected.decoding_config.get("repetition_penalty"),
                repetition_context_size=selected.decoding_config.get("repetition_context_size"),
                early_stop_account=bool(selected.decoding_config.get("early_stop_account", False)),
                prefix_salvage=bool(selected.decoding_config.get("prefix_salvage", False)),
            ),
            require_account_candidate=args.require_account_candidate,
        )
        summaries[variant.variant_id] = {
            "prompt_id": variant.prompt_id,
            "surface_config": {
                "config_id": variant.config_id,
                "content_type": variant.content_type,
                "subcategory": variant.subcategory,
                "complexity": variant.complexity,
            },
            "decoding_id": variant.decoding_id,
            "decoding_config": variant.decoding_config,
            "summary": summary,
        }
        for row in read_rows(variant_dir / "evaluation_results.csv", variant.variant_id):
            rows_by_id.setdefault(row["id"], []).append(row)

    reranker_summary = summarize_reranker(rows_by_id)
    all_rows = [row for rows in rows_by_id.values() for row in rows]
    harvest_summary = summarize_variant_harvest(all_rows)
    sweep_summary = {
        "variants": summaries,
        "reranker": reranker_summary,
        "candidate_harvest": harvest_summary,
        "image_variant_ids": variant_ids,
        "input_item_count": len(items),
        "expanded_item_count": len(eval_items),
        "notes": [
            "Reranker uses only gate/candidate/repetition metrics, not the gold account value.",
            "Exact-match numbers in this summary evaluate the selected variant after reranking.",
        ],
    }
    (args.output_dir / "sweep_summary.json").write_text(
        json.dumps(sweep_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sweep_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
