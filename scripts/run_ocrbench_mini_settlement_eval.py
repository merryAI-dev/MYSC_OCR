#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_ocr_server import PROMPT_PRESETS, evaluate_items, post_ocr, read_manifest  # noqa: E402
from scripts.run_mlx4bit_practical_eval import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    PracticalEvalConfig,
    is_healthy,
    start_server,
    stop_server,
    wait_for_server,
)


CANONICAL_MODEL_NAMES = {
    "bf16direct_mlx6_gs64": "DeepSeek-OCR-BF16Direct-MLX6-GS64",
    "mlx8bit": "DeepSeek-OCR-MLXCommunity-8bit",
}


def parse_model(value: str) -> tuple[str, str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be SHORT_LABEL=/path/to/model")
    short_label, raw_path = value.split("=", 1)
    short_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", short_label.strip())
    if not short_label:
        raise argparse.ArgumentTypeError("model short label is empty")
    return short_label, CANONICAL_MODEL_NAMES.get(short_label, short_label), Path(raw_path).expanduser()


def metric_block(summary: dict[str, Any]) -> dict[str, Any]:
    adapted = summary.get("ocrbench_v2_adapted", {})
    return {
        "total": summary.get("total", 0),
        "capability": "Relation Extraction + Text Recognition",
        "task": "bankbook_account_extraction",
        "metric_family": {
            "recognition": "account exact match + digit normalized edit similarity",
            "extraction": "key-value F1 over bank/account_holder/account_number",
            "basic_vqa": "false-positive account-number rejection",
            "degeneration": "diagnostic free-running collapse warning, not a core OCRBench score",
        },
        "recognition_score_mean": adapted.get("recognition_score_mean", 0),
        "extraction_f1_mean": adapted.get("extraction_f1_mean", 0),
        "basic_vqa_score_mean": adapted.get("basic_vqa_score_mean", 0),
        "composite_score_mean": adapted.get("composite_score_mean", 0),
        "account_exact_match_rate": summary.get("account_exact_match_rate", 0),
        "candidate_exact_match_rate": summary.get("candidate_exact_match_rate", 0),
        "false_positive_count": summary.get("false_positive_count", 0),
        "free_running_gate_pass_rate": summary.get("free_running_gate_pass_rate", 0),
        "surface_gate_pass_rate": summary.get("surface_gate_pass_rate", 0),
        "error_count": summary.get("error_count", 0),
    }


def run_one_model(
    *,
    short_label: str,
    model_name: str,
    model_path: Path,
    manifest: Path,
    output_root: Path,
    api_url: str,
    timeout: int,
    wait_seconds: int,
    splits: str,
    limit: int,
    prompt: str,
    prompt_preset: str,
    require_account_candidate: bool,
) -> dict[str, Any]:
    if not model_path.exists():
        raise SystemExit(f"Missing model path for {short_label}: {model_path}")
    if is_healthy(api_url):
        raise SystemExit(f"{api_url} already has a healthy server. Stop it first to avoid mixing models.")

    splits_set = {split.strip() for split in splits.split(",") if split.strip()}
    items = read_manifest(manifest, splits_set)
    if limit:
        items = items[:limit]
    if not items:
        raise SystemExit(f"No evaluation items found in {manifest}")

    model_dir = output_root / short_label
    server_config = PracticalEvalConfig(
        model_path=model_path,
        api_url=api_url,
        timeout=timeout,
        wait_seconds=wait_seconds,
        output_root=output_root / "server_logs" / short_label,
    )
    server, server_log = start_server(server_config)
    try:
        wait_for_server(server_config, server, server_log)
        summary = evaluate_items(
            items,
            output_dir=model_dir,
            ocr_fn=lambda image_path: post_ocr(
                image_path,
                api_url,
                timeout,
                prompt,
                max_tokens=128,
                temperature=0.0,
                repetition_penalty=1.15,
                repetition_context_size=64,
                early_stop_account=True,
                prefix_salvage=True,
            ),
            require_account_candidate=require_account_candidate,
            account_only_gold=True,
        )
    finally:
        stop_server(server)

    metadata = {
        "model_short_label": short_label,
        "model_name": model_name,
        "model_path": str(model_path),
        "benchmark": "SettlementOCRBench-mini",
        "benchmark_basis": "OCRBench v2 metric families adapted to Korean bankbook account extraction",
        "manifest": str(manifest),
        "label_warning": (
            "Current manifest labels come from previous high-confidence successes; "
            "only account_number is scored as verified gold until independent manual/Codex review labels are added."
        ),
        "prompt_preset": prompt_preset,
        "metric_block": metric_block(summary),
    }
    (model_dir / "model_eval_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["model_name"] = model_name
    summary["model_short_label"] = short_label
    summary["model_path"] = str(model_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCRBench v2-style mini eval for settlement OCR models.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "deepseek_ocr_high_success_subset_manifest_20260426.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "ocrbench_mini_settlement_20260427",
    )
    parser.add_argument("--model", action="append", type=parse_model, required=True, help="SHORT_LABEL=/path/to/model")
    parser.add_argument("--api-url", default=PracticalEvalConfig.api_url)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--wait-seconds", type=int, default=180)
    parser.add_argument("--splits", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prompt-preset", choices=sorted(PROMPT_PRESETS), default="bank_fields")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--require-account-candidate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompt = args.prompt or PROMPT_PRESETS[args.prompt_preset]
    plan = {
        "benchmark": "SettlementOCRBench-mini",
        "manifest": str(args.manifest),
        "output_root": str(args.output_root),
        "models": [
            {"short_label": short_label, "model_name": model_name, "path": str(path)}
            for short_label, model_name, path in args.model
        ],
        "prompt_preset": args.prompt_preset,
        "splits": args.splits,
        "limit": args.limit,
        "require_account_candidate": args.require_account_candidate,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "eval_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    summaries: dict[str, Any] = {}
    for short_label, model_name, model_path in args.model:
        print(f"\n== {model_name} ({short_label}) ==")
        summaries[short_label] = run_one_model(
            short_label=short_label,
            model_name=model_name,
            model_path=model_path,
            manifest=args.manifest,
            output_root=args.output_root,
            api_url=args.api_url,
            timeout=args.timeout,
            wait_seconds=args.wait_seconds,
            splits=args.splits,
            limit=args.limit,
            prompt=prompt,
            prompt_preset=args.prompt_preset,
            require_account_candidate=args.require_account_candidate,
        )

    report = {
        "benchmark": "SettlementOCRBench-mini",
        "plan": plan,
        "models": {
            label: {
                "model_name": summary["model_name"],
                "metric_block": metric_block(summary),
                "output_dir": summary["output_dir"],
            }
            for label, summary in summaries.items()
        },
        "notes": [
            "This is the first OCRBench v2-style smoke application for our model naming and eval schema.",
            "The current manifest has 3 labeled rows and is not yet a formal benchmark.",
            "Next formal gate should use the 34 independent manual/Codex review rows as gold labels.",
        ],
    }
    (args.output_root / "ocrbench_mini_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
