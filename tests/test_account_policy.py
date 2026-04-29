from settlement_tool.account_policy import (
    mask_account_candidate,
    policy_audit_rows,
    policy_score_candidate,
    rank_account_candidates,
    rerank_account_candidates,
    select_account_candidate,
)


def test_policy_score_candidate_rejects_prompt_leakage_without_document_context():
    decision = policy_score_candidate(
        "KNOWN 계좌번호가 보이면 account_number: 110-123-456789",
        "110-123-456789",
    )

    assert decision.accepted is False
    assert "prompt_leakage_without_structured_context" in decision.reasons


def test_policy_score_candidate_accepts_structured_bankbook_account_row():
    decision = policy_score_candidate(
        "<tr><td>계좌번호</td><td>110-123-456789</td></tr><tr><td>예금주</td><td>홍길동</td></tr>",
        "110-123-456789",
    )

    assert decision.accepted is True
    assert decision.score >= 10.0
    assert "structured_bankbook_context" in decision.reasons


def test_policy_score_candidate_rejects_wrong_field_candidate():
    decision = policy_score_candidate(
        "<tr><td>예금주</td><td>110-123-456789</td></tr><tr><td>계좌번호</td><td>2026.02.28</td></tr>",
        "110-123-456789",
    )

    assert decision.accepted is False
    assert "wrong_field_context" in decision.reasons


def test_policy_score_candidate_penalizes_unstructured_customer_number_metadata():
    decision = policy_score_candidate(
        """
        신한은행 고객님을 가장 소중히 생각하는 은행이 되겠습니다.
        계좌번호 110-123-456789
        예금종류 저축예금
        고객번호 123452169
        신규일 2026-05-11
        발행일 2026-07-14
        """,
        "110-123-456789",
    )

    assert decision.accepted is False
    assert decision.score < 10.0
    assert "customer_number_metadata_context" in decision.reasons


def test_policy_score_candidate_keeps_structured_customer_number_context():
    decision = policy_score_candidate(
        """
        <tr><td>계좌번호</td><td>110-123-456789</td></tr>
        <tr><td>예금주</td><td>홍길동</td></tr>
        고객번호 123452169
        """,
        "110-123-456789",
    )

    assert decision.accepted is True


def test_rank_account_candidates_prefers_structured_evidence_over_prompt_echo():
    text = """
    KNOWN 계좌번호가 보이면 account_number: 999-999-999999
    <tr><td>계좌번호</td><td>110-123-456789</td></tr><tr><td>예금주</td><td>홍길동</td></tr>
    """

    ranked = rank_account_candidates(text, ["999-999-999999", "110-123-456789"])
    selected = select_account_candidate(text, ["999-999-999999", "110-123-456789"])

    assert [item.candidate for item in ranked] == ["110-123-456789", "999-999-999999"]
    assert selected is not None
    assert selected.candidate == "110-123-456789"


def test_rerank_account_candidates_selects_clear_policy_margin():
    text = """
    국민은행 계좌번호 110-123-456789
    후보 메모: 신한은행 333-444-555555
    """

    result = rerank_account_candidates(text, ["333-444-555555", "110-123-456789"], min_margin=2.0)

    assert result.status == "selected"
    assert result.selected is not None
    assert result.selected.candidate == "110-123-456789"
    assert result.reason == "top_policy_margin"


def test_rerank_account_candidates_defers_close_multiple_accepted_candidates():
    text = """
    국민은행 계좌 110-123-456789
    신한은행 계좌 333-444-555555
    """

    result = rerank_account_candidates(text, ["110-123-456789", "333-444-555555"], min_margin=2.0)

    assert result.status == "ambiguous_conflict"
    assert result.selected is None
    assert result.reason == "accepted_candidates_within_margin"


def test_rerank_account_candidates_deduplicates_same_account_digits():
    text = "국민은행 계좌번호 110-123-456789"

    result = rerank_account_candidates(text, ["110-123-456789", "110123456789"])

    assert result.status == "selected"
    assert result.selected is not None
    assert result.selected.candidate == "110-123-456789"
    assert [decision.candidate for decision in result.decisions] == ["110-123-456789"]


def test_policy_audit_rows_are_masked_and_explain_rejection():
    rows = policy_audit_rows(
        "KNOWN 계좌번호가 보이면 account_number: 110-123-456789",
        ["110-123-456789"],
        source_id="row-1",
        source_name="sample.png",
    )

    assert rows == [
        {
            "source_id": "row-1",
            "source_name": "sample.png",
            "candidate_masked": "***-***-**6789",
            "accepted": "0",
            "policy_score": "8.0",
            "policy_reasons": "account_keyword_context;bank_keyword_context;prompt_leakage_without_structured_context",
            "has_prompt_leakage_context": "1",
            "has_wrong_field_context": "0",
            "has_direct_account_field_context": "0",
            "has_structured_bankbook_context": "0",
        }
    ]
    assert "110-123-456789" not in str(rows)


def test_mask_account_candidate_keeps_only_last_four_digits():
    assert mask_account_candidate("110-123-456789") == "***-***-**6789"
