from scripts.autoresearch_ocr_loop import classify_with_labels, deepseek_form_fields


def test_classify_with_labels_does_not_resurrect_prompt_leakage_candidate():
    result = classify_with_labels("KNOWN 계좌번호가 보이면 account_number: 110-123-456789")

    assert result.value is None
    assert result.confidence == "low"


def test_classify_with_labels_keeps_structured_account_candidate():
    result = classify_with_labels(
        "<tr><td>계좌번호</td><td>110-123-456789</td></tr><tr><td>예금주</td><td>홍길동</td></tr>"
    )

    assert result.value == "110-123-456789"
    assert result.confidence == "high"


def test_deepseek_form_fields_uses_recovery_generation_controls():
    fields = deepseek_form_fields(
        "<image>\nread account",
        max_tokens=512,
        early_stop_account=True,
        prefix_salvage=True,
        repetition_penalty=1.05,
        repetition_context_size=64,
    )

    assert fields["max_tokens"] == "512"
    assert fields["early_stop_account"] == "1"
    assert fields["prefix_salvage"] == "1"
    assert fields["repetition_penalty"] == "1.05"
    assert fields["repetition_context_size"] == "64"
