from settlement_tool.kie_evidence import bbox_bucket, infer_kie_field_type, normalize_kie_row, redacted_kie_evidence


def test_infer_kie_field_type_identifies_account_label_without_raw_digits():
    assert infer_kie_field_type("계좌번호") == "account_number"
    assert infer_kie_field_type("예금주") == "holder"
    assert infer_kie_field_type("고객번호") == "customer_number"
    assert infer_kie_field_type("연락처") == "phone"


def test_bbox_bucket_returns_only_layout_buckets():
    assert bbox_bucket([10, 20, 110, 50], page_width=1000, page_height=1000) == {
        "x_bucket": "left",
        "y_bucket": "top",
        "width_bucket": "narrow",
        "height_bucket": "short",
    }


def test_normalize_kie_row_keeps_raw_candidate_local_only():
    row = normalize_kie_row(
        source_id="kie:1",
        source_name="/Users/boram/private/source.png",
        backend="paddleocr_kie",
        text="***-***-**6789",
        raw_text_local="RAW_ACCOUNT_SENTINEL",
        label_text="계좌번호",
        bbox=[10, 20, 110, 50],
        page_width=1000,
        page_height=1000,
        confidence=0.93,
    )

    assert row["candidate_raw"] == "RAW_ACCOUNT_SENTINEL"
    assert row["candidate_masked"] == "***-***-**6789"
    assert row["kie_field_type"] == "account_number"
    assert row["kie_backend"] == "paddleocr_kie"
    assert row["kie_label_masked"] == "계좌번호"
    assert row["source_name"] == "/Users/boram/private/source.png"


def test_redacted_kie_evidence_drops_raw_text_name_and_path():
    local_row = normalize_kie_row(
        source_id="kie:1",
        source_name="/Users/boram/private/source.png",
        backend="paddleocr_kie",
        text="***-***-**6789",
        raw_text_local="RAW_ACCOUNT_SENTINEL",
        label_text="계좌번호",
        bbox=[10, 20, 110, 50],
        page_width=1000,
        page_height=1000,
        confidence=0.93,
    )

    evidence = redacted_kie_evidence(local_row)
    serialized = str(evidence)

    assert "RAW_ACCOUNT_SENTINEL" not in serialized
    assert "/Users/boram" not in serialized
    assert evidence["field_type"] == "account_number"
    assert evidence["backend"] == "paddleocr_kie"
    assert evidence["layout"]["x_bucket"] == "left"
