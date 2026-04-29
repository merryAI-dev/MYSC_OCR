from scripts.run_downloads_field_ocr_eval import (
    api_config_for_category,
    clean_generated_outputs,
    compare_digit_groups,
    digit_groups,
    limit_manifest_per_purpose,
    mask_sensitive_text,
    summarize_results,
)


def test_mask_sensitive_text_masks_long_digit_groups_but_keeps_shape():
    masked = mask_sensitive_text("계좌 110-123-456789 전화 010-1234-5678 날짜 2026-04-27")

    assert "110-123-456789" not in masked
    assert "010-1234-5678" not in masked
    assert "***-***-**6789" in masked
    assert "***-****-5678" in masked


def test_digit_group_comparison_counts_overlap_without_exposing_values():
    baseline = digit_groups("account 110-123-456789 amount 250000")
    candidate = digit_groups("계좌 110-123-456789")

    comparison = compare_digit_groups(candidate, baseline)

    assert comparison == {
        "candidate_digit_groups": 1,
        "baseline_digit_groups": 2,
        "overlap_digit_groups": 1,
        "digit_group_recall": 0.5,
        "digit_group_precision": 1.0,
    }


def test_summarize_results_splits_depth_and_generality_metrics():
    rows = [
        {
            "model": "six",
            "purpose": "1_depth_bankbook",
            "category": "bankbook_zip_member",
            "error": "",
            "text": "bank account 110-123-456789",
            "account_confidence": "high",
        },
        {
            "model": "six",
            "purpose": "2_generality_ocr",
            "category": "receipt_tax",
            "error": "",
            "text": "세금계산서 2026 50000",
            "account_confidence": "none",
        },
        {
            "model": "eight",
            "purpose": "2_generality_ocr",
            "category": "receipt_tax",
            "error": "Timeout",
            "text": "",
            "account_confidence": "error",
        },
    ]

    summary = summarize_results(rows)

    assert summary["models"]["six"]["all"]["items"] == 2
    assert summary["models"]["six"]["1_depth_bankbook"]["high_accounts"] == 1
    assert summary["models"]["six"]["2_generality_ocr"]["non_empty"] == 1
    assert summary["models"]["eight"]["2_generality_ocr"]["errors"] == 1


def test_limit_manifest_per_purpose_keeps_each_eval_purpose_present():
    rows = [
        {"item_id": "d1", "purpose": "1_depth_bankbook"},
        {"item_id": "d2", "purpose": "1_depth_bankbook"},
        {"item_id": "d3", "purpose": "1_depth_bankbook"},
        {"item_id": "g1", "purpose": "2_generality_ocr"},
        {"item_id": "g2", "purpose": "2_generality_ocr"},
    ]

    limited = limit_manifest_per_purpose(rows, 2)

    assert [row["item_id"] for row in limited] == ["d1", "d2", "g1", "g2"]


def test_clean_generated_outputs_removes_stale_run_artifacts_but_keeps_manifest(tmp_path):
    (tmp_path / "masked_text").mkdir()
    (tmp_path / "masked_text" / "old.txt").write_text("old", encoding="utf-8")
    (tmp_path / "server_logs").mkdir()
    (tmp_path / "server_logs" / "old.log").write_text("old", encoding="utf-8")
    (tmp_path / "field_ocr_results_masked.csv").write_text("old", encoding="utf-8")
    manifest = tmp_path / "downloads_field_ocr_manifest.jsonl"
    manifest.write_text("{}", encoding="utf-8")

    clean_generated_outputs(tmp_path)

    assert not (tmp_path / "masked_text").exists()
    assert not (tmp_path / "server_logs").exists()
    assert not (tmp_path / "field_ocr_results_masked.csv").exists()
    assert manifest.exists()


def test_bankbook_api_config_uses_practical_successful_basic_mode():
    config = api_config_for_category("bankbook_zip_member")

    assert config["mode"] == "basic"
