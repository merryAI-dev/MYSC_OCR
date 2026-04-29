from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .accounts import apply_account_updates, write_account_report
from .core import AccountResult, extract_roster, load_config, parse_overrides, timestamped_output_dir, write_csv, write_text
from .ocr import OcrOptions, extract_account_results
from .organize import build_document_plan, materialize_plan, write_document_reports


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_config(args.config)
    return args.func(args, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="settlement-tool")
    parser.add_argument("--config", default="config.json", help="Path to JSON config")

    subparsers = parser.add_subparsers(required=True)
    analyze = subparsers.add_parser("analyze", help="Create matching and OCR reports")
    add_common_options(analyze)
    analyze.set_defaults(func=cmd_analyze)

    run = subparsers.add_parser("run", help="Create organized documents and updated workbook")
    add_common_options(run)
    run.add_argument("--mode", choices=["copy", "inplace"], default="copy")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cmd_run)
    return parser


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", help="Output directory. Defaults to timestamped folder under config output_root")
    parser.add_argument("--overrides", help="CSV with columns: name,field,value")
    parser.add_argument("--ocr-backend", choices=["tesseract", "hf", "qianfan-local", "chandra", "none"], default="tesseract")
    parser.add_argument("--allow-remote-sensitive", action="store_true")
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--hf-provider", default="auto")
    parser.add_argument("--hf-max-tokens", type=int, default=512)
    parser.add_argument("--hf-timeout-seconds", type=int, default=180)
    parser.add_argument("--hf-image-max-side", type=int, default=1800)
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="Environment variable containing HF token/API key")
    parser.add_argument("--hf-base-url", help="Dedicated Inference Endpoint or OpenAI-compatible base URL")
    parser.add_argument("--hf-bill-to", help="Optional Hugging Face billing account/organization")
    parser.add_argument(
        "--hf-prompt",
        help="Custom prompt for Hugging Face remote VLM OCR. Defaults to Korean bank account extraction.",
    )
    parser.add_argument("--chandra-method", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--chandra-prompt-type", default="ocr_layout")
    parser.add_argument(
        "--chandra-model-checkpoint",
        help="Hugging Face model id or local model directory for Chandra, e.g. /path/to/chandra-ocr-2",
    )
    parser.add_argument("--chandra-torch-device", help="Torch device for Chandra HF mode, e.g. cpu, cuda:0, mps")
    parser.add_argument("--chandra-max-output-tokens", type=int, default=1024)
    parser.add_argument("--chandra-image-max-side", type=int, default=1800)
    parser.add_argument(
        "--chandra-prompt",
        help="Custom prompt for Chandra OCR. Defaults to Korean bank account extraction.",
    )
    parser.add_argument("--no-privacy-check", action="store_true")


def resolve_config(path: str) -> dict[str, object]:
    config_path = Path(path).expanduser()
    config = load_config(config_path)
    config["_config_path"] = str(config_path)
    return config


def cmd_analyze(args: argparse.Namespace, config: dict[str, object]) -> int:
    context = build_context(args, config)
    write_base_reports(context)
    if args.ocr_backend != "none":
        account_results = extract_account_results(context["plan"], build_ocr_options(args))
        apply_account_overrides(account_results, context["overrides"])
        account_report = [
            {
                "name": name,
                "account": result.value or "",
                "confidence": result.confidence,
                "candidates": "; ".join(result.candidates),
                "reason": result.reason,
                "backend": result.backend,
            }
            for name, result in sorted(account_results.items())
        ]
        write_csv(
            context["reports_dir"] / "account_ocr_candidates.csv",
            account_report,
            ["name", "account", "confidence", "candidates", "reason", "backend"],
        )
    write_text(context["output_dir"] / "RUN_SUMMARY.txt", summary_text(context, "analyze"))
    print(context["output_dir"])
    return 0


def cmd_run(args: argparse.Namespace, config: dict[str, object]) -> int:
    context = build_context(args, config)
    documents_dir = context["output_dir"] / "documents"
    write_base_reports(context)
    materialize_plan(context["plan"], documents_dir, dry_run=args.dry_run)

    account_results = {}
    if args.ocr_backend != "none":
        account_results = extract_account_results(context["plan"], build_ocr_options(args))
    apply_account_overrides(account_results, context["overrides"])

    workbook_output = workbook_output_path(context, args)
    if args.dry_run:
        account_report = []
    else:
        if args.mode == "inplace":
            backup = backup_workbook(context["withholding_workbook"])
            write_text(context["reports_dir"] / "workbook_backup.txt", str(backup))
        account_report = apply_account_updates(
            context["withholding_workbook"],
            workbook_output,
            account_results,
        )
    write_account_report(account_report, context["reports_dir"] / "account_updates.csv")
    write_text(context["output_dir"] / "RUN_SUMMARY.txt", summary_text(context, "run", args.dry_run))
    print(context["output_dir"])
    return 0


def build_context(args: argparse.Namespace, config: dict[str, object]) -> dict[str, object]:
    paths = {
        "withholding_workbook": path_from_config(config, "withholding_workbook"),
        "payment_zip": path_from_config(config, "payment_zip"),
        "id_zip": path_from_config(config, "id_zip"),
        "bank_zip": path_from_config(config, "bank_zip"),
    }
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else timestamped_output_dir(config.get("output_root", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    overrides = parse_overrides(args.overrides or config.get("overrides"))
    plan = build_document_plan(
        paths["withholding_workbook"],
        paths["payment_zip"],
        paths["id_zip"],
        paths["bank_zip"],
        overrides=overrides,
    )
    return {
        **paths,
        "output_dir": output_dir,
        "reports_dir": reports_dir,
        "overrides": overrides,
        "plan": plan,
    }


def path_from_config(config: dict[str, object], key: str) -> Path:
    value = config.get(key)
    if not value:
        raise KeyError(f"Missing config key: {key}")
    return Path(str(value)).expanduser()


def write_base_reports(context: dict[str, object]) -> None:
    roster = extract_roster(context["withholding_workbook"])
    write_csv(
        context["reports_dir"] / "roster.csv",
        [
            {"group": person.group, "no": person.no, "name": person.name, "row": person.row}
            for person in roster.people
        ],
        ["group", "no", "name", "row"],
    )
    write_csv(
        context["reports_dir"] / "group_counts.csv",
        [{"group": group, "count": count} for group, count in roster.group_counts().items()],
        ["group", "count"],
    )
    write_document_reports(context["plan"], context["reports_dir"])
    write_text(
        context["reports_dir"] / "inputs.json",
        json.dumps(
            {
                "withholding_workbook": str(context["withholding_workbook"]),
                "payment_zip": str(context["payment_zip"]),
                "id_zip": str(context["id_zip"]),
                "bank_zip": str(context["bank_zip"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def build_ocr_options(args: argparse.Namespace) -> OcrOptions:
    return OcrOptions(
        backend=args.ocr_backend,
        allow_remote_sensitive=args.allow_remote_sensitive,
        hf_model=args.hf_model,
        hf_provider=args.hf_provider or None,
        hf_max_tokens=args.hf_max_tokens,
        hf_timeout_seconds=args.hf_timeout_seconds,
        hf_image_max_side=args.hf_image_max_side,
        hf_prompt=args.hf_prompt,
        hf_token_env=args.hf_token_env,
        hf_base_url=args.hf_base_url,
        hf_bill_to=args.hf_bill_to,
        chandra_method=args.chandra_method,
        chandra_prompt_type=args.chandra_prompt_type,
        chandra_model_checkpoint=args.chandra_model_checkpoint,
        chandra_torch_device=args.chandra_torch_device,
        chandra_max_output_tokens=args.chandra_max_output_tokens,
        chandra_image_max_side=args.chandra_image_max_side,
        chandra_prompt=args.chandra_prompt,
        privacy_check=not args.no_privacy_check,
    )


def apply_account_overrides(
    account_results: dict[str, AccountResult],
    overrides: dict[tuple[str, str], str],
) -> None:
    for (name, field), value in overrides.items():
        if field in {"account_number", "계좌번호"}:
            account_results[name] = AccountResult(value, "high", [value], "override", "manual_override")


def workbook_output_path(context: dict[str, object], args: argparse.Namespace) -> Path:
    source = context["withholding_workbook"]
    if args.mode == "inplace":
        return source
    return context["output_dir"] / f"{source.stem}_계좌번호입력{source.suffix}"


def backup_workbook(path: Path) -> Path:
    backup = path.with_name(f"{path.stem}.backup.{Path(path).stat().st_mtime_ns}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def summary_text(context: dict[str, object], command: str, dry_run: bool = False) -> str:
    plan = context["plan"]
    confirmed = sum(1 for item in plan.items if item.status == "confirmed")
    ambiguous = sum(1 for item in plan.items if item.status == "ambiguous")
    missing = sum(1 for item in plan.items if item.status == "missing")
    return "\n".join(
        [
            f"command: {command}",
            f"dry_run: {dry_run}",
            f"output_dir: {context['output_dir']}",
            f"confirmed_documents: {confirmed}",
            f"ambiguous_documents: {ambiguous}",
            f"missing_documents: {missing}",
            "",
            "See reports/document_summary.csv and reports/document_matches.csv for details.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
