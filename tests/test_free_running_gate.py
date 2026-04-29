from settlement_tool.free_running_gate import free_running_degeneration_metrics, tokenize_free_running_output


def test_gate_accepts_concise_account_candidate_when_required():
    metrics = free_running_degeneration_metrics("계좌번호: 123-456-789012", require_account_candidate=True)

    assert metrics["account_candidate_presence"] is True
    assert metrics["degeneration_pass"] is True
    assert metrics["degeneration_reason"] == "ok"


def test_gate_rejects_repeated_free_running_collapse():
    metrics = free_running_degeneration_metrics("Background " * 20, require_account_candidate=True)

    assert metrics["degeneration_pass"] is False
    assert "top_token_dominance" in metrics["degeneration_reason"]
    assert "long_token_run" in metrics["degeneration_reason"]
    assert "missing_account_candidate" in metrics["degeneration_reason"]


def test_gate_accepts_structured_html_table_with_account_candidate():
    repeated_empty_rows = '<tr><td colspan="1"></td><td colspan="1"></td></tr>\n' * 30
    text = (
        ".\n<table>"
        f"{repeated_empty_rows}"
        "<tr><td>예금주</td><td>권경안님</td></tr>"
        "<tr><td>계좌번호</td><td>3333-19-2297030</td></tr>"
        "</table>"
    )

    metrics = free_running_degeneration_metrics(text, require_account_candidate=True)

    assert metrics["account_candidate_presence"] is True
    assert metrics["degeneration_pass"] is True
    assert metrics["degeneration_reason"] == "ok"


def test_tokenizer_keeps_account_number_as_one_structured_token():
    assert tokenize_free_running_output("계좌번호: 123-456-789012") == ["계좌번호", ":", "123-456-789012"]
