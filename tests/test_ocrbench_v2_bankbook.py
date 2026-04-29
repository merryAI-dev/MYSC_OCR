from settlement_tool.ocrbench_v2_bankbook import (
    BankbookGold,
    BankbookPrediction,
    bankbook_ocrbench_v2_metrics,
    key_value_f1,
    normalized_edit_similarity,
    parse_structured_fields,
)


def test_normalized_edit_similarity_scores_digit_drift():
    assert normalized_edit_similarity("1234567890", "1234567890") == 1.0
    assert normalized_edit_similarity("1234567890", "1234567899") == 0.9


def test_parse_structured_fields_accepts_korean_and_english_labels():
    fields = parse_structured_fields("bank: 국민은행\n예금주: 홍길동\n계좌번호: 123-456")

    assert fields == {
        "bank": "국민은행",
        "account_holder": "홍길동",
        "account_number": "123-456",
    }


def test_bankbook_ocrbench_metrics_separate_recognition_extraction_and_false_positive():
    metrics = bankbook_ocrbench_v2_metrics(
        BankbookGold(account_number="123-456-789012", bank="국민은행", account_holder="홍길동"),
        BankbookPrediction(
            raw_text="bank: 국민은행\n예금주: 홍길동\n계좌번호: 123-456-789999",
            account_number="123-456-789999",
            candidate_accounts=("123-456-789999",),
        ),
    )

    assert metrics["recognition"]["account_exact"] is False
    assert metrics["recognition"]["account_digit_edit_similarity"] < 1.0
    assert metrics["extraction"]["f1"] < 1.0
    assert metrics["basic_vqa"]["score"] == 0.0
    assert metrics["composite_score"] < 1.0


def test_candidate_exact_preserves_surface_rescue_signal():
    metrics = bankbook_ocrbench_v2_metrics(
        BankbookGold(account_number="123-456-789012", bank="국민은행", account_holder="홍길동"),
        BankbookPrediction(
            raw_text="계좌번호 후보: 999-999-999999 / 123-456-789012",
            account_number="999-999-999999",
            candidate_accounts=("999-999-999999", "123-456-789012"),
        ),
    )

    assert metrics["recognition"]["account_exact"] is False
    assert metrics["recognition"]["candidate_account_exact"] is True
    assert metrics["recognition"]["score"] == 1.0


def test_key_value_f1_ignores_unknown_gold_fields():
    metrics = key_value_f1(
        predicted={"account_number": "123-456"},
        expected={"bank": "UNKNOWN", "account_holder": "", "account_number": "123-456"},
    )

    assert metrics["ground_truth_pairs"] == 1
    assert metrics["correct_pairs"] == 1
    assert metrics["f1"] == 1.0
