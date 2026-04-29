from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from .core import safe_filename_part


DEFAULT_VARIANT_IDS = (
    "full",
    "top_60",
    "middle_60",
    "bottom_60",
    "left_60",
    "right_60",
    "grid_2x2",
    "bands_3",
    "contrast_gray",
    "sharpen_gray",
)


@dataclass(frozen=True)
class ImageVariant:
    item_id: str
    variant_id: str
    path: Path


def parse_variant_ids(value: str | None) -> list[str]:
    if not value:
        return ["full"]
    if value.strip() == "default":
        return list(DEFAULT_VARIANT_IDS)
    variants = [part.strip() for part in value.split(",") if part.strip()]
    return variants or ["full"]


def _crop_fraction(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left = max(0, min(width, round(width * box[0])))
    upper = max(0, min(height, round(height * box[1])))
    right = max(left + 1, min(width, round(width * box[2])))
    lower = max(upper + 1, min(height, round(height * box[3])))
    return image.crop((left, upper, right, lower))


def build_variant_images(image: Image.Image, variant_id: str) -> list[tuple[str, Image.Image]]:
    rgb = image.convert("RGB")
    if variant_id == "full":
        return [("full", rgb.copy())]
    if variant_id == "top_60":
        return [("top_60", _crop_fraction(rgb, (0.0, 0.0, 1.0, 0.60)))]
    if variant_id == "middle_60":
        return [("middle_60", _crop_fraction(rgb, (0.0, 0.20, 1.0, 0.80)))]
    if variant_id == "bottom_60":
        return [("bottom_60", _crop_fraction(rgb, (0.0, 0.40, 1.0, 1.0)))]
    if variant_id == "left_60":
        return [("left_60", _crop_fraction(rgb, (0.0, 0.0, 0.60, 1.0)))]
    if variant_id == "right_60":
        return [("right_60", _crop_fraction(rgb, (0.40, 0.0, 1.0, 1.0)))]
    if variant_id == "grid_2x2":
        return [
            ("grid_2x2_r0c0", _crop_fraction(rgb, (0.0, 0.0, 0.55, 0.55))),
            ("grid_2x2_r0c1", _crop_fraction(rgb, (0.45, 0.0, 1.0, 0.55))),
            ("grid_2x2_r1c0", _crop_fraction(rgb, (0.0, 0.45, 0.55, 1.0))),
            ("grid_2x2_r1c1", _crop_fraction(rgb, (0.45, 0.45, 1.0, 1.0))),
        ]
    if variant_id == "bands_3":
        return [
            ("bands_3_top", _crop_fraction(rgb, (0.0, 0.0, 1.0, 0.40))),
            ("bands_3_middle", _crop_fraction(rgb, (0.0, 0.30, 1.0, 0.70))),
            ("bands_3_bottom", _crop_fraction(rgb, (0.0, 0.60, 1.0, 1.0))),
        ]
    if variant_id == "contrast_gray":
        gray = ImageOps.grayscale(rgb)
        enhanced = ImageOps.autocontrast(gray)
        return [("contrast_gray", enhanced.convert("RGB"))]
    if variant_id == "sharpen_gray":
        gray = ImageOps.grayscale(rgb)
        enhanced = ImageOps.autocontrast(gray).filter(ImageFilter.SHARPEN)
        return [("sharpen_gray", enhanced.convert("RGB"))]
    raise ValueError(f"Unknown image variant id: {variant_id}")


def render_image_variants(
    image_path: Path,
    output_dir: Path,
    *,
    item_id: str,
    variant_ids: list[str],
) -> list[ImageVariant]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[ImageVariant] = []
    with Image.open(image_path) as image:
        for requested_id in variant_ids:
            for variant_id, variant_image in build_variant_images(image, requested_id):
                filename = f"{safe_filename_part(item_id)}__{variant_id}.png"
                path = output_dir / filename
                variant_image.save(path)
                variant_image.close()
                rendered.append(ImageVariant(item_id=item_id, variant_id=variant_id, path=path))
    return rendered
