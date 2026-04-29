import csv
import json
from pathlib import Path

from scripts.evaluate_ocr_server import EvaluationItem, PROMPT_PRESETS, evaluate_items, mask_account


def test_mask_account_preserves_only_last_four_digits():
    assert mask_account("123-456-789012") == "***-***-**9012"


def test_bank_zip_full_success_prompt_preserves_original_teacher_prompt():
    prompt = PROMPT_PRESETS["bank_zip_full_success"]

    assert "OCR this Korean bank account image" in prompt
    assert "Focus on 계좌번호, 예금주, 은행명" in prompt
    assert "account_number" in prompt


def test_evaluate_items_writes_masked_results_without_raw_account_numbers(tmp_path: Path):
    image = tmp_path / "bank.png"
    image.write_bytes(b"fake image bytes")
    output_dir = tmp_path / "out"
    items = [
        EvaluationItem(
            item_id="sample-1",
            split="test",
            name="홍길동",
            image_path=image,
            label_account_number="123-456-789012",
            label_bank="국민은행",
            label_account_holder="홍길동",
        )
    ]

    def fake_ocr(_image_path: Path) -> str:
        return "bank: 국민은행\n계좌번호: 123-456-789012\n예금주: 홍길동"

    summary = evaluate_items(items, output_dir=output_dir, ocr_fn=fake_ocr, require_account_candidate=True)

    assert summary["total"] == 1
    assert summary["account_exact_match"] == 1
    assert summary["false_positive_count"] == 0
    assert summary["free_running_gate_pass"] == 1
    assert summary["surface_gate_pass"] == 1

    result_text = (output_dir / "evaluation_results.csv").read_text(encoding="utf-8-sig")
    assert "123-456-789012" not in result_text
    assert "***-***-**9012" in result_text

    rows = list(csv.DictReader((output_dir / "evaluation_results.csv").open(encoding="utf-8-sig")))
    assert rows[0]["label_account_masked"] == "***-***-**9012"
    assert rows[0]["predicted_account_masked"] == "***-***-**9012"
    assert rows[0]["exact_match"] == "1"
    assert rows[0]["free_running_gate_pass"] == "1"
    assert rows[0]["surface_gate_pass"] == "1"
    assert rows[0]["degeneration_reason"] == "ok"
    assert rows[0]["ocrbench_recognition_score"] == "1.0000"
    assert rows[0]["ocrbench_extraction_f1"] == "1.0000"

    persisted_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert persisted_summary["account_exact_match"] == 1
    assert persisted_summary["free_running_degeneration"]["require_account_candidate"] is True
    assert persisted_summary["ocrbench_v2_adapted"]["composite_score_mean"] == 1.0


def test_evaluate_items_gate_rejects_background_repetition(tmp_path: Path):
    image = tmp_path / "bank.png"
    image.write_bytes(b"fake image bytes")
    output_dir = tmp_path / "out"
    items = [
        EvaluationItem(
            item_id="sample-1",
            split="test",
            name="홍길동",
            image_path=image,
            label_account_number="123-456-789012",
        )
    ]

    def fake_ocr(_image_path: Path) -> str:
        return "Background " * 20

    summary = evaluate_items(items, output_dir=output_dir, ocr_fn=fake_ocr, require_account_candidate=True)

    assert summary["free_running_gate_pass"] == 0
    assert summary["surface_gate_pass"] == 0

    rows = list(csv.DictReader((output_dir / "evaluation_results.csv").open(encoding="utf-8-sig")))
    assert rows[0]["free_running_gate_pass"] == "0"
    assert "missing_account_candidate" in rows[0]["degeneration_reason"]


def test_evaluate_items_can_score_account_only_gold(tmp_path: Path):
    image = tmp_path / "bank.png"
    image.write_bytes(b"fake image bytes")
    items = [
        EvaluationItem(
            item_id="sample-1",
            split="test",
            name="홍길동",
            image_path=image,
            label_account_number="123-456-789012",
            label_bank="국민은행",
            label_account_holder="홍길동",
        )
    ]

    def fake_ocr(_image_path: Path) -> str:
        return "account_number: 123-456-789012"

    summary = evaluate_items(
        items,
        output_dir=tmp_path / "out",
        ocr_fn=fake_ocr,
        account_only_gold=True,
    )

    assert summary["ocrbench_v2_adapted"]["extraction_f1_mean"] == 1.0
