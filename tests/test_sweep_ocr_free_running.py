from pathlib import Path

from PIL import Image

from scripts.evaluate_ocr_server import EvaluationItem
from scripts.sweep_ocr_free_running import (
    build_variants,
    expand_items_with_image_variants,
    reranker_score,
    summarize_variant_harvest,
)


def test_reranker_score_does_not_use_gold_false_positive_field():
    base = {
        "surface_gate_pass": "1",
        "free_running_gate_pass": "1",
        "account_candidate_presence": "1",
        "unique_token_ratio": "0.5",
        "top_token_share": "0.2",
        "max_token_run": "1",
        "false_positive": "0",
    }
    oracle_only_changed = base | {"false_positive": "1"}

    assert reranker_score(base) == reranker_score(oracle_only_changed)


def test_build_variants_includes_decoding_axis():
    variants = build_variants(["candidate_lines"], ["scene_photo_small"], ["baseline", "short64_rep115"])

    assert [variant.variant_id for variant in variants] == [
        "candidate_lines__scene_photo_small__baseline",
        "candidate_lines__scene_photo_small__short64_rep115",
    ]
    assert variants[1].decoding_config["repetition_penalty"] == 1.15


def test_expand_items_with_image_variants_preserves_gold_and_creates_variant_ids(tmp_path: Path):
    source = tmp_path / "bank.png"
    Image.new("RGB", (100, 80), "white").save(source)
    items = [
        EvaluationItem(
            item_id="sample-1",
            split="test",
            name="홍길동",
            image_path=source,
            label_account_number="123-456-789012",
            label_bank="국민은행",
            label_account_holder="홍길동",
        )
    ]

    expanded = expand_items_with_image_variants(items, tmp_path / "out", ["full", "bands_3"])

    assert [item.item_id for item in expanded] == [
        "sample-1::full",
        "sample-1::bands_3_top",
        "sample-1::bands_3_middle",
        "sample-1::bands_3_bottom",
    ]
    assert all(item.label_account_number == "123-456-789012" for item in expanded)
    assert all(item.image_path.exists() for item in expanded)


def test_summarize_variant_harvest_groups_by_original_item():
    rows = [
        {
            "id": "sample-1::full",
            "surface_gate_pass": "0",
            "free_running_gate_pass": "0",
            "account_candidate_presence": "0",
            "unique_token_ratio": "0.1",
            "top_token_share": "0.5",
            "max_token_run": "1",
            "exact_match": "0",
            "candidate_exact_match": "0",
            "false_positive": "0",
        },
        {
            "id": "sample-1::bottom_60",
            "surface_gate_pass": "1",
            "free_running_gate_pass": "1",
            "account_candidate_presence": "1",
            "unique_token_ratio": "0.5",
            "top_token_share": "0.1",
            "max_token_run": "1",
            "exact_match": "0",
            "candidate_exact_match": "1",
            "false_positive": "0",
        },
    ]

    summary = summarize_variant_harvest(rows)

    assert summary["total"] == 1
    assert summary["account_candidate_presence"] == 1
    assert summary["candidate_exact_match"] == 1
    assert summary["selected_image_variant_counts"] == {"bottom_60": 1}
