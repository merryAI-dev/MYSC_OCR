import csv
import json
from pathlib import Path

from scripts.build_release_gate import (
    build_manual_autofill_review_queue,
    build_release_gate_outputs,
    scan_release_bundle,
)


def test_build_manual_autofill_review_queue_masks_raw_accounts(tmp_path: Path):
    resolution_csv = tmp_path / "account_resolution_candidates.csv"
    output_dir = tmp_path / "gate"
    resolution_csv.write_text(
        "name,decision,chosen_account,chosen_account_masked,source,candidate_files\n"
        "홍길동,auto_fill_targeted_deepseek,110-123-456789,***-***-**6789,targeted_retry:contrast:visible_numbers,bank.png\n"
        "김철수,no_candidate,,,,\n",
        encoding="utf-8",
    )

    summary = build_manual_autofill_review_queue(resolution_csv=resolution_csv, output_dir=output_dir)
    queue_text = (output_dir / "manual_autofill_review_queue.csv").read_text(encoding="utf-8")

    assert summary["required_count"] == 1
    assert "홍길동" in queue_text
    assert "***-***-**6789" in queue_text
    assert "110-123-456789" not in queue_text
    assert "review_status" in queue_text
    assert "bank_name" in queue_text
    assert "bank_evidence" in queue_text
    assert "bank_confidence" in queue_text


def test_build_manual_autofill_review_queue_includes_openai_reranker_autofill(tmp_path: Path):
    resolution_csv = tmp_path / "account_resolution_candidates.csv"
    output_dir = tmp_path / "gate"
    resolution_csv.write_text(
        "name,decision,chosen_account,chosen_account_masked,source,candidate_files\n"
        "홍길동,auto_fill_openai_reranker,110-123-456789,***-***-**6789,openai_reranker:v2_feature_score,\n",
        encoding="utf-8",
    )

    summary = build_manual_autofill_review_queue(resolution_csv=resolution_csv, output_dir=output_dir)

    assert summary["required_count"] == 1
    assert summary["pending_count"] == 1


def test_scan_release_bundle_flags_raw_artifact_classes_and_pii(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "summary.json").write_text('{"selected_reject_count": 0}\n', encoding="utf-8")
    (bundle / "candidate_features_local.jsonl").write_text('{"candidate_raw":"110-123-456789"}\n', encoding="utf-8")
    (bundle / "candidate_raw_map_local.jsonl").write_text('{"candidate_raw":"222-333-444444"}\n', encoding="utf-8")
    (bundle / "raw.txt").write_text("계좌 110-123-456789", encoding="utf-8")
    (bundle / "Run.command").write_text("/Users/boram/Downloads\n", encoding="utf-8")

    scan = scan_release_bundle(bundle)

    assert scan["blocked_artifact_count"] == 2
    assert scan["pii_match_count"] == 3
    assert scan["local_path_or_token_count"] == 1
    assert any(path.endswith("candidate_features_local.jsonl") for path in scan["blocked_artifact_paths"])
    assert any(path.endswith("candidate_raw_map_local.jsonl") for path in scan["blocked_artifact_paths"])
    assert any(path.endswith("raw.txt") for path in scan["pii_match_paths"])


def test_scan_release_bundle_allows_openai_redacted_payload_and_report(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "candidate_features_redacted.jsonl").write_text(
        '{"source_id":"source-1","candidates":[{"candidate_id":"acct_1","account_shape":"ddd-ddd-dddddd","digit_count":12}]}\n',
        encoding="utf-8",
    )
    (bundle / "openai_reranker_decisions.jsonl").write_text(
        '{"source_id":"source-1","action":"accept","selected_candidate_id":"acct_1","confidence":0.97}\n',
        encoding="utf-8",
    )

    scan = scan_release_bundle(bundle)

    assert scan["blocked_artifact_count"] == 0
    assert scan["pii_match_count"] == 0
    assert scan["local_path_or_token_count"] == 0


def test_scan_release_bundle_allows_kie_redacted_evidence_without_raw_values(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kie_evidence_redacted.jsonl").write_text(
        '{"backend":"paddleocr_kie","field_type":"account_number","confidence_bucket":"high","layout":{"x_bucket":"left"}}\n',
        encoding="utf-8",
    )

    scan = scan_release_bundle(bundle)

    assert scan["blocked_artifact_count"] == 0
    assert scan["pii_match_count"] == 0
    assert scan["local_path_or_token_count"] == 0


def test_scan_release_bundle_flags_forbidden_redacted_keys_without_echoing_values(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "candidate_features_redacted.jsonl").write_text(
        '{"source_name":"sample.png","raw_text_local":"RAW_ACCOUNT_SENTINEL","candidate_raw":"RAW_ACCOUNT_SENTINEL"}\n',
        encoding="utf-8",
    )
    (bundle / "kie_evidence_redacted.jsonl").write_text(
        '{"field_type":"account_number","raw_text_local":"RAW_ACCOUNT_SENTINEL"}\n',
        encoding="utf-8",
    )

    scan = scan_release_bundle(bundle)
    scan_text = json.dumps(scan, ensure_ascii=False)

    assert scan["forbidden_redacted_key_count"] == 2
    assert scan["forbidden_redacted_key_paths"] == [
        "candidate_features_redacted.jsonl",
        "kie_evidence_redacted.jsonl",
    ]
    assert "RAW_ACCOUNT_SENTINEL" not in scan_text


def test_scan_release_bundle_flags_raw_kie_local_artifact(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kie_candidates_local.csv").write_text(
        "candidate_raw,candidate_masked\n110-123-456789,***-***-**6789\n",
        encoding="utf-8",
    )

    scan = scan_release_bundle(bundle)

    assert scan["blocked_artifact_count"] == 1
    assert scan["pii_match_count"] == 1


def test_scan_release_bundle_flags_sensitive_terms_in_openai_payloads_without_echoing_terms(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "candidate_features_redacted.jsonl").write_text(
        '{"source_id":"source-1","person_hint":"홍길동","candidates":[]}\n',
        encoding="utf-8",
    )
    (bundle / "openai_reranker_report.json").write_text(
        '{"source_id":"source-1","action":"review"}\n',
        encoding="utf-8",
    )

    scan = scan_release_bundle(bundle, sensitive_terms={"홍길동"})
    scan_text = json.dumps(scan, ensure_ascii=False)

    assert scan["sensitive_term_match_count"] == 1
    assert scan["sensitive_term_match_paths"] == ["candidate_features_redacted.jsonl"]
    assert "홍길동" not in scan_text


def test_scan_release_bundle_does_not_flag_long_decimal_metrics(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "summary.json").write_text('{"precision": 0.9285714285714286}\n', encoding="utf-8")

    scan = scan_release_bundle(bundle)

    assert scan["pii_match_count"] == 0


def test_build_release_gate_outputs_blocks_until_manual_review_done(tmp_path: Path):
    resolution_csv = tmp_path / "account_resolution_candidates.csv"
    reranker_eval = tmp_path / "reranker_eval.json"
    bundle = tmp_path / "bundle"
    output_dir = tmp_path / "gate"
    bundle.mkdir()
    resolution_csv.write_text(
        "name,decision,chosen_account,chosen_account_masked,source,candidate_files\n"
        "홍길동,auto_fill_targeted_deepseek,110-123-456789,***-***-**6789,targeted_retry:contrast:visible_numbers,bank.png\n",
        encoding="utf-8",
    )
    reranker_eval.write_text(
        json.dumps(
            {
                "best_threshold": 10.0,
                "best_margin": 3.0,
                "best": {
                    "selected_reject_count": 0,
                    "selection_precision": 1.0,
                    "source_accept_recall": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_release_gate_outputs(
        resolution_csv=resolution_csv,
        reranker_eval_path=reranker_eval,
        bundle_path=bundle,
        output_dir=output_dir,
    )
    report_text = (output_dir / "release_gate_report.json").read_text(encoding="utf-8")

    assert report["overall_status"] == "blocked_manual_review"
    assert report["manual_review"]["required_count"] == 1
    assert report["reranker_gate"]["passed"] is True
    assert report["pii_gate"]["passed"] is True
    assert "110-123-456789" not in report_text


def test_build_release_gate_outputs_accepts_human_eval_report_format(tmp_path: Path):
    resolution_csv = tmp_path / "account_resolution_candidates.csv"
    reranker_eval = tmp_path / "human_label_eval.json"
    bundle = tmp_path / "bundle"
    output_dir = tmp_path / "gate"
    bundle.mkdir()
    resolution_csv.write_text(
        "name,decision,chosen_account,chosen_account_masked,source,candidate_files\n",
        encoding="utf-8",
    )
    reranker_eval.write_text(
        json.dumps(
            {
                "summary": {
                    "human_positive_count": 64,
                    "correct_positive": 37,
                    "wrong_positive": 0,
                    "review_false_positive": 0,
                },
                "metrics": {
                    "safe_selection_precision": 1.0,
                    "positive_recall": 0.578125,
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_release_gate_outputs(
        resolution_csv=resolution_csv,
        reranker_eval_path=reranker_eval,
        bundle_path=bundle,
        output_dir=output_dir,
    )

    assert report["overall_status"] == "passed"
    assert report["reranker_gate"]["passed"] is True
    assert report["reranker_gate"]["wrong_positive"] == 0
    assert report["reranker_gate"]["review_false_positive"] == 0
    assert report["reranker_gate"]["positive_recall"] == 0.578125


def test_build_release_gate_report_does_not_echo_absolute_local_paths(tmp_path: Path):
    resolution_csv = tmp_path / "account_resolution_candidates.csv"
    reranker_eval = tmp_path / "human_label_eval.json"
    bundle = tmp_path / "bundle"
    output_dir = tmp_path / "gate"
    bundle.mkdir()
    resolution_csv.write_text(
        "name,decision,chosen_account,chosen_account_masked,source,candidate_files\n",
        encoding="utf-8",
    )
    reranker_eval.write_text(
        json.dumps(
            {
                "summary": {"wrong_positive": 0, "review_false_positive": 0},
                "metrics": {"safe_selection_precision": 1.0},
            }
        ),
        encoding="utf-8",
    )

    build_release_gate_outputs(
        resolution_csv=resolution_csv,
        reranker_eval_path=reranker_eval,
        bundle_path=bundle,
        output_dir=output_dir,
    )
    report_text = (output_dir / "release_gate_report.json").read_text(encoding="utf-8")

    assert str(bundle) not in report_text
    assert str(output_dir) not in report_text
    assert str(reranker_eval) not in report_text


def test_build_release_gate_outputs_preserves_confirmed_manual_review(tmp_path: Path):
    resolution_csv = tmp_path / "account_resolution_candidates.csv"
    reranker_eval = tmp_path / "reranker_eval.json"
    bundle = tmp_path / "bundle"
    output_dir = tmp_path / "gate"
    bundle.mkdir()
    output_dir.mkdir()
    resolution_csv.write_text(
        "name,decision,chosen_account,chosen_account_masked,source,candidate_files\n"
        "홍길동,auto_fill_targeted_deepseek,110-123-456789,***-***-**6789,targeted_retry:contrast:visible_numbers,bank.png\n",
        encoding="utf-8",
    )
    reranker_eval.write_text(
        json.dumps(
            {
                "best_threshold": 10.0,
                "best_margin": 3.0,
                "best": {
                    "selected_reject_count": 0,
                    "selection_precision": 1.0,
                    "source_accept_recall": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "manual_autofill_review_queue.csv").write_text(
        "name,decision,chosen_account_masked,source,candidate_files,bank_name,bank_evidence,bank_confidence,review_status,reviewer_id,review_notes\n"
        "홍길동,auto_fill_targeted_deepseek,***-***-**6789,targeted_retry:contrast:visible_numbers,bank.png,테스트은행,filename,high,confirmed,user,ok\n",
        encoding="utf-8-sig",
    )

    report = build_release_gate_outputs(
        resolution_csv=resolution_csv,
        reranker_eval_path=reranker_eval,
        bundle_path=bundle,
        output_dir=output_dir,
    )
    rows = list(csv.DictReader((output_dir / "manual_autofill_review_queue.csv").open(encoding="utf-8-sig")))

    assert report["overall_status"] == "passed"
    assert report["manual_review"]["pending_count"] == 0
    assert report["manual_review"]["confirmed_count"] == 1
    assert rows[0]["review_status"] == "confirmed"
    assert rows[0]["bank_name"] == "테스트은행"
