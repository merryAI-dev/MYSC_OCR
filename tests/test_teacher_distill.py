from settlement_tool.teacher_distill import (
    calibrate_policy_threshold,
    draft_masked_context_teacher_label,
    evaluate_policy_labels,
    evaluate_source_reranker,
    evaluate_source_selection,
    build_candidate_features,
    features_from_kie_csv,
    features_from_ocr_csv,
    mask_digit_context,
    merge_teacher_labels,
    policy_score,
    seed_teacher_policy_label,
    sweep_source_reranker,
    summarize_label_coverage,
    teacher_review_record,
)

from scripts.build_codex_teacher_distill import build_teacher_distill_outputs
from scripts.draft_codex_teacher_labels import build_draft_labels
from scripts.merge_codex_teacher_labels import build_merged_policy_outputs
from scripts.sweep_policy_reranker import build_policy_reranker_sweep_outputs


def test_mask_digit_context_preserves_labels_but_masks_long_numbers():
    text = "계좌번호 110-123-456789 연락처 010-1234-5678"

    masked = mask_digit_context(text)

    assert "계좌번호" in masked
    assert "110-123-456789" not in masked
    assert "***-***-**6789" in masked
    assert "010-1234-5678" not in masked


def test_mask_digit_context_masks_partial_account_prefixes():
    text = "계좌번호 110-2"

    masked = mask_digit_context(text)

    assert "계좌번호" in masked
    assert "110-2" not in masked
    assert "***-*" in masked


def test_build_candidate_features_labels_gold_match_without_persisting_raw_candidate():
    text = """
    예금주 홍길동
    연락처 010-1234-5678
    국민은행 계좌번호 110-123-456789
    """

    features = build_candidate_features(
        text,
        source_id="sample",
        source_name="sample.png",
        gold_account="110-123-456789",
        backend="mlx-deepseek",
        variant="contrast",
        prompt_id="account_structured_ko",
    )

    assert len(features) == 1
    feature = features[0]
    assert feature["candidate_raw"] == "110-123-456789"
    assert feature["candidate_masked"] == "***-***-**6789"
    assert feature["gold_label_available"] is True
    assert feature["gold_exact_match"] is True
    assert feature["has_account_keyword_context"] is True
    assert feature["has_bank_keyword_context"] is True
    assert "110-123-456789" not in feature["teacher_context_masked"]


def test_policy_score_prefers_account_context_over_phone_context():
    [account_feature] = build_candidate_features(
        "신한은행 계좌번호 110-123-456789",
        source_id="account",
        source_name="account.png",
    )
    [phone_feature] = build_candidate_features(
        "고객센터 010-1234-5678",
        source_id="phone",
        source_name="phone.png",
        include_phone_like=True,
    )

    assert policy_score(account_feature) > policy_score(phone_feature)


def test_policy_score_penalizes_prompt_leakage_without_structured_bankbook_context():
    [prompt_leak_feature] = build_candidate_features(
        "KNOWN 계좌번호가 보이면 account_number: 110-123-456789",
        source_id="prompt",
        source_name="prompt.png",
    )
    [structured_feature] = build_candidate_features(
        "<tr><td>계좌번호</td><td>110-123-456789</td></tr><tr><td>예금주</td><td>홍길동</td></tr>",
        source_id="structured",
        source_name="structured.png",
    )

    assert prompt_leak_feature["has_prompt_leakage_context"] is True
    assert structured_feature["has_structured_bankbook_context"] is True
    assert policy_score(structured_feature) > policy_score(prompt_leak_feature)


def test_policy_score_penalizes_candidate_in_wrong_table_field():
    [feature] = build_candidate_features(
        "<tr><td>예금주</td><td>110-123-456789</td></tr><tr><td>계좌번호</td><td>2026.02.28</td></tr>",
        source_id="wrong-field",
        source_name="wrong-field.png",
    )

    assert feature["has_wrong_field_context"] is True
    assert policy_score(feature) < 10.0


def test_features_from_ocr_csv_builds_local_and_review_records(tmp_path):
    text_path = tmp_path / "ocr.txt"
    text_path.write_text("국민은행 계좌번호 110-123-456789", encoding="utf-8")
    csv_path = tmp_path / "ocr.csv"
    csv_path.write_text(
        "source_name,variant,prompt_id,ocr_text_path\n"
        f"sample.png,contrast,account_structured_ko,{text_path}\n",
        encoding="utf-8",
    )

    features = features_from_ocr_csv(
        csv_path,
        gold_by_source_name={"sample.png": "110-123-456789"},
        backend="mlx-deepseek",
    )
    review = teacher_review_record(features[0])

    assert features[0]["candidate_raw"] == "110-123-456789"
    assert features[0]["gold_exact_match"] is True


def test_features_from_kie_csv_preserves_kie_fields_and_local_raw(tmp_path):
    path = tmp_path / "kie_candidates_local.csv"
    path.write_text(
        "source_id,source_name,backend,kie_backend,kie_field_type,kie_label_masked,kie_holder_match_status,kie_holder_field_present,kie_bank_name_present,candidate_raw,candidate_masked,confidence,bbox_json,page_width,page_height,layout_json,error\n"
        'kie:1,sample.png,paddleocr_kie,paddleocr_kie,account_number,계좌번호,match,True,True,RAW_ACCOUNT_SENTINEL,***-***-**6789,0.93,"[10,20,110,50]",1000,1000,"{}",\n',
        encoding="utf-8",
    )

    rows = features_from_kie_csv(path, backend="paddleocr_kie")

    assert len(rows) == 1
    assert rows[0]["candidate_raw"] == "RAW_ACCOUNT_SENTINEL"
    assert rows[0]["candidate_masked"] == "***-***-**6789"
    assert rows[0]["kie_field_type"] == "account_number"
    assert rows[0]["kie_holder_match_status"] == "match"
    assert rows[0]["kie_bank_name_present"] is True
    assert rows[0]["has_direct_account_field_context"] is True
    assert "RAW_ACCOUNT_SENTINEL" not in rows[0]["teacher_context_masked"]


def test_build_candidate_features_marks_gold_unknown_separately_from_negative():
    [feature] = build_candidate_features(
        "국민은행 계좌번호 110-123-456789",
        source_id="sample",
        source_name="sample.png",
    )

    assert feature["gold_label_available"] is False
    assert feature["gold_exact_match"] is None


def test_features_from_ocr_csv_matches_gold_by_extracted_path_basename(tmp_path):
    text_path = tmp_path / "ocr.txt"
    text_path.write_text("국민은행 계좌번호 110-123-456789", encoding="utf-8")
    extracted = tmp_path / "001_sample.png"
    csv_path = tmp_path / "ocr.csv"
    csv_path.write_text(
        "source_name,extracted_path,ocr_text_path\n"
        f"sample.png,{extracted},{text_path}\n",
        encoding="utf-8",
    )

    features = features_from_ocr_csv(
        csv_path,
        gold_by_source_name={"001_sample.png": "110-123-456789"},
    )

    assert features[0]["gold_exact_match"] is True


def test_build_teacher_distill_outputs_writes_local_and_review_jsonl(tmp_path):
    text_path = tmp_path / "ocr.txt"
    text_path.write_text("국민은행 계좌번호 110-123-456789", encoding="utf-8")
    csv_path = tmp_path / "ocr.csv"
    csv_path.write_text(
        "source_name,variant,prompt_id,ocr_text_path\n"
        f"sample.png,contrast,account_structured_ko,{text_path}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "distill"

    summary = build_teacher_distill_outputs(
        input_csvs=[csv_path],
        output_dir=output_dir,
        gold_by_source_name={"sample.png": "110-123-456789"},
        backend="mlx-deepseek",
    )

    assert summary["candidate_count"] == 1
    local_text = (output_dir / "candidate_features_local.jsonl").read_text(encoding="utf-8")
    redacted_text = (output_dir / "candidate_features_redacted.jsonl").read_text(encoding="utf-8")
    raw_map_text = (output_dir / "candidate_raw_map_local.jsonl").read_text(encoding="utf-8")
    review_text = (output_dir / "teacher_review_queue.jsonl").read_text(encoding="utf-8")
    seed_text = (output_dir / "codex_policy_label_seed.jsonl").read_text(encoding="utf-8")
    assert "110-123-456789" in local_text
    assert "110-123-456789" not in redacted_text
    assert "ddd-ddd-dddddd" in redacted_text
    assert "110-123-456789" in raw_map_text
    assert "110-123-456789" not in review_text
    assert "110-123-456789" not in seed_text
    assert "***-***-**6789" in review_text
    assert '"suggested_label": "accept"' in seed_text


def test_build_teacher_distill_outputs_accepts_input_kie_csv(tmp_path):
    kie_csv = tmp_path / "kie_candidates_local.csv"
    kie_csv.write_text(
        "source_id,source_name,backend,kie_backend,kie_field_type,kie_label_masked,candidate_raw,candidate_masked,confidence,bbox_json,page_width,page_height,layout_json,error\n"
        'kie:1,sample.png,paddleocr_kie,paddleocr_kie,account_number,계좌번호,RAW_ACCOUNT_SENTINEL,***-***-**6789,0.93,"[10,20,110,50]",1000,1000,"{}",\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "distill"

    summary = build_teacher_distill_outputs(
        input_csvs=[],
        input_kie_csvs=[kie_csv],
        output_dir=output_dir,
        backend="paddleocr_kie",
    )

    local_text = (output_dir / "candidate_features_local.jsonl").read_text(encoding="utf-8")
    redacted_text = (output_dir / "candidate_features_redacted.jsonl").read_text(encoding="utf-8")

    assert summary["candidate_count"] == 1
    assert summary["kie_candidate_count"] == 1
    assert "RAW_ACCOUNT_SENTINEL" in local_text
    assert "RAW_ACCOUNT_SENTINEL" not in redacted_text


def test_seed_teacher_policy_label_accepts_only_gold_exact_or_rejects_obvious_negative():
    [gold_feature] = build_candidate_features(
        "국민은행 계좌번호 110-123-456789",
        source_id="gold",
        source_name="gold.png",
        gold_account="110-123-456789",
    )
    [unknown_feature] = build_candidate_features(
        "국민은행 계좌번호 110-123-456789",
        source_id="unknown",
        source_name="unknown.png",
    )
    [negative_feature] = build_candidate_features(
        "자동차운전면허증 12-34-567802-31",
        source_id="negative",
        source_name="negative.png",
    )

    assert seed_teacher_policy_label(gold_feature)["suggested_label"] == "accept"
    assert seed_teacher_policy_label(gold_feature)["requires_teacher_review"] is False
    assert seed_teacher_policy_label(unknown_feature)["suggested_label"] == "review_accept_candidate"
    assert seed_teacher_policy_label(unknown_feature)["requires_teacher_review"] is True
    assert seed_teacher_policy_label(negative_feature)["suggested_label"] == "reject"
    assert seed_teacher_policy_label(negative_feature)["requires_teacher_review"] is False


def test_evaluate_policy_labels_scores_accept_reject_threshold():
    rows = [
        {"teacher_policy_score": 14.0, "teacher_label": "accept"},
        {"teacher_policy_score": 2.0, "teacher_label": "reject"},
        {"teacher_policy_score": 2.0, "teacher_label": "accept"},
    ]

    summary = evaluate_policy_labels(rows, threshold=10.0)

    assert summary["labeled_count"] == 3
    assert summary["accuracy"] == 2 / 3
    assert summary["false_negative_count"] == 1
    assert summary["false_positive_count"] == 0


def test_merge_teacher_labels_updates_matching_seed_rows_only():
    seed_rows = [
        {"source_id": "a", "candidate_masked": "***1234", "teacher_policy_score": 14.0, "teacher_label": ""},
        {"source_id": "b", "candidate_masked": "***5678", "teacher_policy_score": 2.0, "teacher_label": ""},
    ]
    label_rows = [
        {"source_id": "a", "candidate_masked": "***1234", "teacher_label": "accept", "teacher_reason": "account label nearby"},
        {"source_id": "x", "candidate_masked": "***9999", "teacher_label": "reject"},
    ]

    merged = merge_teacher_labels(seed_rows, label_rows)

    assert merged[0]["teacher_label"] == "accept"
    assert merged[0]["teacher_reason"] == "account label nearby"
    assert merged[1]["teacher_label"] == ""


def test_summarize_label_coverage_counts_review_rows_only():
    seed_rows = [
        {"source_id": "a", "candidate_masked": "***1234", "requires_teacher_review": True, "teacher_label": ""},
        {"source_id": "b", "candidate_masked": "***5678", "requires_teacher_review": True, "teacher_label": ""},
        {"source_id": "c", "candidate_masked": "***9999", "requires_teacher_review": False, "teacher_label": "accept"},
    ]
    merged_rows = [
        {"source_id": "a", "candidate_masked": "***1234", "teacher_label": "accept", "teacher_id": "manual"},
        {"source_id": "b", "candidate_masked": "***5678", "teacher_label": "", "teacher_id": ""},
        {"source_id": "c", "candidate_masked": "***9999", "teacher_label": "accept", "teacher_id": ""},
    ]

    summary = summarize_label_coverage(seed_rows, merged_rows)

    assert summary["review_row_count"] == 2
    assert summary["labeled_review_row_count"] == 1
    assert summary["unlabeled_review_row_count"] == 1
    assert summary["review_label_counts"] == {"accept": 1}
    assert summary["all_label_counts"] == {"accept": 2}


def test_calibrate_policy_threshold_prefers_no_false_positives_then_recall():
    rows = [
        {"teacher_policy_score": 14.0, "teacher_label": "accept"},
        {"teacher_policy_score": 13.0, "teacher_label": "accept"},
        {"teacher_policy_score": 11.0, "teacher_label": "reject"},
        {"teacher_policy_score": 2.0, "teacher_label": "reject"},
    ]

    calibrated = calibrate_policy_threshold(rows, thresholds=[10.0, 12.0, 14.0])

    assert calibrated["best_threshold"] == 12.0
    assert calibrated["best"]["false_positive_count"] == 0
    assert calibrated["best"]["recall"] == 1.0


def test_evaluate_source_selection_groups_candidates_by_source():
    rows = [
        {"source_name": "a.png", "candidate_masked": "***1111", "teacher_policy_score": 14.0, "teacher_label": "accept"},
        {"source_name": "a.png", "candidate_masked": "***2222", "teacher_policy_score": 10.0, "teacher_label": "reject"},
        {"source_name": "b.png", "candidate_masked": "***3333", "teacher_policy_score": 12.0, "teacher_label": "reject"},
        {"source_name": "c.png", "candidate_masked": "***4444", "teacher_policy_score": 2.0, "teacher_label": "accept"},
    ]

    summary = evaluate_source_selection(rows, threshold=10.0)

    assert summary["source_count"] == 3
    assert summary["selected_count"] == 2
    assert summary["selected_accept_count"] == 1
    assert summary["selected_reject_count"] == 1
    assert summary["missed_accept_source_count"] == 1
    assert summary["selected_reject_examples"][0]["source_id"] == ""
    assert summary["selected_reject_examples"][0]["candidate_masked"] == "***3333"
    assert summary["missed_accept_examples"][0]["candidate_masked"] == "***4444"


def test_evaluate_source_reranker_defers_candidates_inside_margin():
    rows = [
        {"source_name": "a.png", "source_id": "a-1", "candidate_masked": "***1111", "teacher_policy_score": 14.0, "teacher_label": "accept"},
        {"source_name": "a.png", "source_id": "a-2", "candidate_masked": "***2222", "teacher_policy_score": 13.0, "teacher_label": "reject"},
        {"source_name": "b.png", "source_id": "b-1", "candidate_masked": "***3333", "teacher_policy_score": 18.0, "teacher_label": "accept"},
        {"source_name": "b.png", "source_id": "b-2", "candidate_masked": "***4444", "teacher_policy_score": 12.0, "teacher_label": "reject"},
        {"source_name": "c.png", "source_id": "c-1", "candidate_masked": "***5555", "teacher_policy_score": 16.0, "teacher_label": "reject"},
    ]

    summary = evaluate_source_reranker(rows, threshold=10.0, min_margin=2.0)

    assert summary["source_count"] == 3
    assert summary["selected_count"] == 2
    assert summary["selected_accept_count"] == 1
    assert summary["selected_reject_count"] == 1
    assert summary["deferred_conflict_count"] == 1
    assert summary["missed_accept_source_count"] == 1
    assert summary["deferred_conflict_examples"][0]["candidate_masked"] == "***1111"
    assert summary["deferred_conflict_examples"][0]["runner_up_masked"] == "***2222"
    assert summary["selected_reject_examples"][0]["candidate_masked"] == "***5555"


def test_sweep_source_reranker_prefers_zero_false_positive_recall():
    rows = [
        {"source_name": "a.png", "candidate_masked": "***1111", "teacher_policy_score": 14.0, "teacher_label": "accept"},
        {"source_name": "a.png", "candidate_masked": "***2222", "teacher_policy_score": 11.0, "teacher_label": "reject"},
        {"source_name": "b.png", "candidate_masked": "***3333", "teacher_policy_score": 9.0, "teacher_label": "accept"},
        {"source_name": "b.png", "candidate_masked": "***4444", "teacher_policy_score": 11.0, "teacher_label": "reject"},
        {"source_name": "c.png", "candidate_masked": "***5555", "teacher_policy_score": 13.0, "teacher_label": "accept"},
    ]

    summary = sweep_source_reranker(rows, thresholds=[10.0, 12.0, 14.0], margins=[0.0, 2.0])

    assert summary["best_threshold"] == 12.0
    assert summary["best_margin"] == 0.0
    assert summary["best"]["selected_reject_count"] == 0
    assert summary["best"]["selected_accept_count"] == 2
    assert len(summary["grid"]) == 6


def test_build_policy_reranker_sweep_outputs_writes_eval_json(tmp_path):
    merged_path = tmp_path / "codex_policy_labels_merged.jsonl"
    output_dir = tmp_path / "sweep"
    merged_path.write_text(
        '{"source_name":"a.png","candidate_masked":"***1111","teacher_policy_score":14,"teacher_label":"accept"}\n'
        '{"source_name":"a.png","candidate_masked":"***2222","teacher_policy_score":11,"teacher_label":"reject"}\n'
        '{"source_name":"b.png","candidate_masked":"***3333","teacher_policy_score":9,"teacher_label":"accept"}\n'
        '{"source_name":"b.png","candidate_masked":"***4444","teacher_policy_score":11,"teacher_label":"reject"}\n',
        encoding="utf-8",
    )

    summary = build_policy_reranker_sweep_outputs(
        merged_path=merged_path,
        output_dir=output_dir,
        thresholds=[10.0, 12.0],
        margins=[0.0, 2.0],
    )

    assert summary["best_threshold"] == 12.0
    assert (output_dir / "reranker_eval.json").exists()
    assert "***1111" in (output_dir / "reranker_eval.json").read_text(encoding="utf-8")
    assert "candidate_raw" not in (output_dir / "reranker_eval.json").read_text(encoding="utf-8")


def test_build_merged_policy_outputs_writes_merged_rows_and_eval(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output_dir = tmp_path / "merged"
    seed_path.write_text(
        '{"source_id":"a","candidate_masked":"***1234","teacher_policy_score":14,"teacher_label":"","variant":"","prompt_id":""}\n'
        '{"source_id":"b","candidate_masked":"***5678","teacher_policy_score":2,"teacher_label":"reject","variant":"","prompt_id":""}\n',
        encoding="utf-8",
    )
    labels_path.write_text(
        '{"source_id":"a","candidate_masked":"***1234","teacher_label":"accept","teacher_reason":"visible account","variant":"","prompt_id":""}\n',
        encoding="utf-8",
    )

    summary = build_merged_policy_outputs(seed_path=seed_path, label_path=labels_path, output_dir=output_dir)

    assert summary["labeled_count"] == 2
    assert summary["label_coverage"]["review_row_count"] == 0
    assert summary["calibration"]["best"]["false_positive_count"] == 0
    assert "10.0" in summary["source_selection_by_threshold"]
    assert (output_dir / "codex_policy_labels_merged.jsonl").exists()
    assert (output_dir / "policy_eval.json").exists()


def test_draft_masked_context_teacher_label_accepts_strong_masked_account_context():
    row = {
        "source_id": "a",
        "candidate_masked": "***1234",
        "teacher_policy_score": 13.0,
        "has_account_keyword_context": True,
        "has_bank_keyword_context": True,
        "has_negative_keyword_context": False,
        "looks_like_phone": False,
        "teacher_context_masked": "국민은행 계좌번호 ***-***-**1234",
    }

    label = draft_masked_context_teacher_label(row, teacher_id="test")

    assert label["teacher_label"] == "accept"
    assert label["teacher_id"] == "test"
    assert "국민은행" in label["teacher_evidence"]


def test_build_draft_labels_writes_reproducible_label_file(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    output_path = tmp_path / "labels.jsonl"
    seed_path.write_text(
        '{"source_id":"a","candidate_masked":"***1234","teacher_policy_score":13,'
        '"has_account_keyword_context":true,"has_bank_keyword_context":true,'
        '"has_negative_keyword_context":false,"looks_like_phone":false,'
        '"teacher_context_masked":"국민은행 계좌번호 ***1234","variant":"","prompt_id":""}\n',
        encoding="utf-8",
    )

    summary = build_draft_labels(seed_path=seed_path, output_path=output_path, teacher_id="test")

    assert summary["label_counts"]["accept"] == 1
    assert '"teacher_id": "test"' in output_path.read_text(encoding="utf-8")
