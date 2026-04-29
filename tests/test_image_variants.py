from pathlib import Path

from PIL import Image

from settlement_tool.image_variants import build_variant_images, parse_variant_ids, render_image_variants


def test_parse_variant_ids_defaults_and_named_default():
    assert parse_variant_ids("") == ["full"]
    assert "grid_2x2" in parse_variant_ids("default")
    assert parse_variant_ids("full,bottom_60") == ["full", "bottom_60"]


def test_build_variant_images_keeps_crops_inside_bounds():
    image = Image.new("RGB", (100, 80), "white")

    variants = build_variant_images(image, "grid_2x2")

    assert len(variants) == 4
    assert all(variant.size[0] > 0 and variant.size[1] > 0 for _, variant in variants)
    assert all(variant.size[0] <= 100 and variant.size[1] <= 80 for _, variant in variants)


def test_render_image_variants_does_not_modify_source(tmp_path: Path):
    source = tmp_path / "bank.png"
    Image.new("RGB", (90, 60), "white").save(source)
    before = source.read_bytes()

    variants = render_image_variants(source, tmp_path / "variants", item_id="sample", variant_ids=["full", "bands_3"])

    assert source.read_bytes() == before
    assert [variant.variant_id for variant in variants] == ["full", "bands_3_top", "bands_3_middle", "bands_3_bottom"]
    assert all(variant.path.exists() for variant in variants)
