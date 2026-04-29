from pathlib import Path

import pytest

from settlement_tool.ocr import OcrOptions, extract_text, ocr_backend_label


def test_chandra_backend_reports_missing_dependency_clearly(tmp_path: Path):
    image = tmp_path / "bank.png"
    image.write_bytes(b"not a real image")

    with pytest.raises(RuntimeError, match="chandra-ocr"):
        extract_text(image, OcrOptions(backend="chandra", chandra_method="hf"))


def test_unsupported_ocr_backend_mentions_backend_name(tmp_path: Path):
    image = tmp_path / "bank.png"
    image.write_bytes(b"not a real image")

    with pytest.raises(ValueError, match="unknown"):
        extract_text(image, OcrOptions(backend="unknown"))


def test_chandra_backend_label_includes_local_checkpoint():
    options = OcrOptions(
        backend="chandra",
        chandra_method="hf",
        chandra_model_checkpoint="/models/chandra-ocr-2",
    )

    assert ocr_backend_label(options) == "chandra:hf:/models/chandra-ocr-2"


def test_hf_backend_label_includes_provider_and_model():
    options = OcrOptions(
        backend="hf",
        hf_provider="auto",
        hf_model="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    assert ocr_backend_label(options) == "hf:auto:Qwen/Qwen2.5-VL-7B-Instruct"


def test_hf_backend_label_uses_endpoint_when_base_url_is_set():
    options = OcrOptions(
        backend="hf",
        hf_model="datalab-to/chandra-ocr-2",
        hf_base_url="https://example.endpoints.huggingface.cloud/v1",
    )

    assert (
        ocr_backend_label(options)
        == "hf:endpoint:https://example.endpoints.huggingface.cloud/v1:datalab-to/chandra-ocr-2"
    )
