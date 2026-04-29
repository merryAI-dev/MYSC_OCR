from pathlib import Path

from scripts.deepseek_recovery_loop import (
    DegenerationMonitor,
    Evidence,
    build_recovery_manifest,
    choose_recovery_accounts,
    deepseek_form_fields,
    degenerate_output_reason,
)


def test_build_recovery_manifest_excludes_missing_files_from_active_targets():
    base_rows = [
        {"name": "이미완료", "chosen_account": "111-222", "decision": "keep_existing_final_run"},
        {"name": "파일있음", "chosen_account": "", "decision": "no_candidate"},
        {"name": "파일없음", "chosen_account": "", "decision": "no_candidate"},
    ]
    deepseek_rows = [
        {
            "source_name": "5. 통장 사본 업로드 (File responses)/파일있음.jpg",
            "extracted_path": "/tmp/file.jpg",
            "filename_hint": "파일있음",
            "matched_name": "파일있음",
        }
    ]

    manifest = build_recovery_manifest(base_rows, deepseek_rows)

    assert [target.name for target in manifest.active_targets] == ["파일있음"]
    assert manifest.missing_files == ["파일없음"]
    assert manifest.total_unresolved == 2


def test_choose_recovery_accounts_requires_repeated_policy_safe_evidence(tmp_path: Path):
    text_a = tmp_path / "a.txt"
    text_b = tmp_path / "b.txt"
    text_a.write_text("은행 계좌번호 110-123-456789 예금주 홍길동", encoding="utf-8")
    text_b.write_text("account_number: 110-123-456789\naccount_holder: 홍길동", encoding="utf-8")
    single = [
        Evidence("홍길동", "110-123-456789", "account_only", "original", "file.jpg", str(text_a)),
    ]
    repeated = single + [
        Evidence("홍길동", "110-123-456789", "copy_all_text", "contrast", "file.jpg", str(text_b)),
    ]

    assert choose_recovery_accounts(single) == {}

    chosen = choose_recovery_accounts(repeated)

    assert chosen["홍길동"].account == "110-123-456789"
    assert chosen["홍길동"].decision == "auto_fill_recovery_deepseek_consensus"


def test_choose_recovery_accounts_rejects_conflicting_repeated_candidates(tmp_path: Path):
    text_a = tmp_path / "a.txt"
    text_b = tmp_path / "b.txt"
    text_a.write_text("계좌번호 110-123-456789", encoding="utf-8")
    text_b.write_text("계좌번호 220-999-888777", encoding="utf-8")
    evidence = [
        Evidence("홍길동", "110-123-456789", "account_only", "original", "file.jpg", str(text_a)),
        Evidence("홍길동", "220-999-888777", "copy_all_text", "contrast", "file.jpg", str(text_b)),
    ]

    assert choose_recovery_accounts(evidence) == {}


def test_deepseek_form_fields_for_recovery_uses_tiny_without_basic_mode():
    fields = deepseek_form_fields(
        "<image>\nReturn account only.",
        max_tokens=128,
        repetition_penalty=1.05,
        repetition_context_size=64,
    )

    assert fields["content_type"] == "Scene"
    assert fields["subcategory"] == "Verification"
    assert fields["complexity"] == "Tiny"
    assert fields["max_tokens"] == "128"
    assert fields["repetition_penalty"] == "1.05"
    assert fields["repetition_context_size"] == "64"
    assert "mode" not in fields


def test_degenerate_output_reason_flags_prompt_conditioned_repetition():
    assert degenerate_output_reason("and eye, a human eye, the eye, a human eye, a human eye, a human eye")
    assert degenerate_output_reason("是, 是, 是, 是, 是, 是, 是, 是, 是, 是")
    assert degenerate_output_reason(". 000-000-000-000000.000-000-000-000000.000-000-000-000000")

    assert not degenerate_output_reason("은행: 신한은행\n예금주: 홍길동\n계좌번호: 110-123-456789")


def test_degeneration_monitor_stops_after_repeated_degenerate_outputs():
    monitor = DegenerationMonitor(max_degenerate_outputs=3)

    monitor.record("and eye, a human eye, the eye, a human eye, a human eye, a human eye")
    monitor.record("是, 是, 是, 是, 是, 是, 是, 是, 是, 是")
    assert not monitor.should_stop

    monitor.record(". 000-000-000-000000.000-000-000-000000.000-000-000-000000")

    assert monitor.should_stop
    assert monitor.reason.startswith("3 degenerate DeepSeek outputs")
