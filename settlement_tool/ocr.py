from __future__ import annotations

import base64
import io
import mimetypes
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

from .core import AccountResult, classify_account_candidates, extract_zip_member, safe_filename_part
from .organize import DocumentPlan
from .privacy import detect_privacy_spans


@dataclass(frozen=True)
class OcrOptions:
    backend: str = "tesseract"
    allow_remote_sensitive: bool = False
    hf_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    hf_provider: str | None = "auto"
    hf_max_tokens: int = 512
    hf_timeout_seconds: int = 180
    hf_image_max_side: int = 1800
    hf_prompt: str | None = None
    hf_token_env: str | None = "HF_TOKEN"
    hf_base_url: str | None = None
    hf_bill_to: str | None = None
    chandra_method: str = "hf"
    chandra_prompt_type: str = "ocr_layout"
    chandra_model_checkpoint: str | None = None
    chandra_torch_device: str | None = None
    chandra_max_output_tokens: int | None = 1024
    chandra_image_max_side: int = 1800
    chandra_prompt: str | None = None
    tesseract_lang: str = "kor+eng"
    timeout_seconds: int = 120
    privacy_check: bool = True


def extract_account_results(plan: DocumentPlan, options: OcrOptions) -> dict[str, AccountResult]:
    results: dict[str, AccountResult] = {}
    if options.backend == "none":
        return results

    bank_items = [item for item in plan.items if item.doc_type == "통장사본" and item.status == "confirmed"]
    with tempfile.TemporaryDirectory(prefix="settlement_ocr_") as tmp:
        tmp_dir = Path(tmp)
        for index, item in enumerate(bank_items, start=1):
            if not item.zip_path or not item.source_name:
                continue
            print(f"OCR {index}/{len(bank_items)} {item.name} ({ocr_backend_label(options)})", flush=True)
            extracted = tmp_dir / f"{safe_filename_part(item.name)}{Path(item.source_name).suffix}"
            extract_zip_member(item.zip_path, item.source_name, extracted)
            try:
                text = extract_text(extracted, options)
                result = classify_account_candidates(text)
                if options.privacy_check:
                    spans = detect_privacy_spans(text)
                    if spans and result.reason == "no_account_candidate":
                        account_spans = [span.text for span in spans if span.label == "account_number"]
                        if account_spans:
                            result = AccountResult(
                                None,
                                "low",
                                account_spans,
                                "privacy_filter_account_span_but_not_high_confidence",
                                ocr_backend_label(options),
                            )
                result = replace(result, backend=ocr_backend_label(options))
                results[item.name] = result
            except Exception as exc:
                results[item.name] = AccountResult(
                    None,
                    "error",
                    [],
                    f"{type(exc).__name__}: {exc}",
                    ocr_backend_label(options),
                )
    return results


def ocr_backend_label(options: OcrOptions) -> str:
    if options.backend == "chandra":
        checkpoint = f":{options.chandra_model_checkpoint}" if options.chandra_model_checkpoint else ""
        return f"chandra:{options.chandra_method}{checkpoint}"
    if options.backend == "hf":
        target = f"endpoint:{options.hf_base_url}" if options.hf_base_url else options.hf_provider or "default"
        return f"hf:{target}:{options.hf_model}"
    return options.backend


def extract_text(path: Path, options: OcrOptions) -> str:
    if options.backend == "tesseract":
        return extract_text_tesseract(path, options)
    if options.backend == "hf":
        return extract_text_hf(path, options)
    if options.backend == "qianfan-local":
        return extract_text_qianfan_local(path, options)
    if options.backend == "chandra":
        return extract_text_chandra(path, options)
    raise ValueError(f"Unsupported OCR backend: {options.backend}")


def extract_text_tesseract(path: Path, options: OcrOptions) -> str:
    if not shutil.which("tesseract"):
        raise RuntimeError("tesseract is not installed")

    if path.suffix.lower() == ".pdf":
        text = _pdftotext(path, options)
        if text.strip():
            return text
        images = _pdf_to_images(path, options)
        return "\n".join(_tesseract_image(image, options) for image in images)

    return _tesseract_image(path, options)


def extract_text_hf(path: Path, options: OcrOptions) -> str:
    if not options.allow_remote_sensitive:
        raise PermissionError("Remote OCR requires --allow-remote-sensitive")

    try:
        from huggingface_hub import InferenceClient
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for --ocr-backend hf") from exc

    token = _env_value(options.hf_token_env)
    client_kwargs = {"timeout": options.hf_timeout_seconds}
    if options.hf_base_url:
        client_kwargs["base_url"] = options.hf_base_url
        if token:
            client_kwargs["api_key"] = token
    elif options.hf_provider:
        client_kwargs["provider"] = options.hf_provider
        if token:
            client_kwargs["token"] = token
    elif token:
        client_kwargs["token"] = token
    if options.hf_bill_to:
        client_kwargs["bill_to"] = options.hf_bill_to
    client = InferenceClient(**client_kwargs)
    outputs: list[str] = []
    paths = _pages_for_remote(path, options)
    for image_path in paths:
        image_url = _data_url(image_path, options)
        completion = client.chat.completions.create(
            model=options.hf_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {
                            "type": "text",
                            "text": options.hf_prompt or _default_hf_account_prompt(),
                        },
                    ],
                }
            ],
            max_tokens=options.hf_max_tokens,
        )
        outputs.append(completion.choices[0].message.content or "")
    return "\n".join(outputs)


def _default_hf_account_prompt() -> str:
    return (
        "You are reading a Korean bankbook copy for settlement. Extract only visible bank account data. "
        "Return concise plain text with these fields when visible: bank, account_holder, account_number. "
        "Do not guess. If account_number is not visible, return account_number: NOT_FOUND."
    )


def _env_value(name: str | None) -> str | None:
    if not name:
        return None
    value = os.environ.get(name)
    return value if value else None


def extract_text_qianfan_local(path: Path, options: OcrOptions) -> str:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "qianfan-local requires torch and transformers. Use --ocr-backend tesseract or hf, or install model dependencies."
        ) from exc
    raise NotImplementedError(
        "qianfan-local dependency gate is present, but full 5B local inference is intentionally not auto-run on this machine."
    )


_CHANDRA_MANAGER_CACHE: dict[str, object] = {}
_CHANDRA_FAILURE_CACHE: dict[str, RuntimeError] = {}


def extract_text_chandra(path: Path, options: OcrOptions) -> str:
    try:
        from chandra.model import InferenceManager
        from chandra.model.schema import BatchInputItem
        from chandra.settings import settings
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "chandra OCR backend requires chandra-ocr. Install with `pip install chandra-ocr` "
            "or `pip install chandra-ocr[hf]`; local HF mode also requires torch/transformers."
        ) from exc

    if options.chandra_model_checkpoint:
        settings.MODEL_CHECKPOINT = options.chandra_model_checkpoint
    if options.chandra_torch_device:
        settings.TORCH_DEVICE = options.chandra_torch_device

    manager_key = f"{options.chandra_method}:{settings.MODEL_CHECKPOINT}:{settings.TORCH_DEVICE or 'auto'}"
    if manager_key in _CHANDRA_FAILURE_CACHE:
        raise _CHANDRA_FAILURE_CACHE[manager_key]
    manager = _CHANDRA_MANAGER_CACHE.get(manager_key)
    if manager is None:
        try:
            manager = InferenceManager(method=options.chandra_method)
        except Exception as exc:
            error = RuntimeError(
                f"Could not initialize Chandra OCR with method={options.chandra_method!r}. "
                "For method='vllm', start `chandra_vllm` first. For method='hf', ensure model weights are available "
                "from Hugging Face or already cached locally. "
                f"Original error: {type(exc).__name__}: {exc}"
            )
            _CHANDRA_FAILURE_CACHE[manager_key] = error
            raise error from exc
        _CHANDRA_MANAGER_CACHE[manager_key] = manager

    image_paths = _pages_for_chandra(path, options)
    outputs: list[str] = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        batch = [
            BatchInputItem(
                image=image,
                prompt=options.chandra_prompt or _default_chandra_account_prompt(),
                prompt_type=options.chandra_prompt_type,
            )
        ]
        try:
            result = manager.generate(batch, max_output_tokens=options.chandra_max_output_tokens)[0]
        except Exception as exc:
            raise RuntimeError(f"Chandra OCR generation failed for {path.name}: {exc}") from exc
        outputs.append(_chandra_result_text(result))
    return "\n".join(outputs)


def _default_chandra_account_prompt() -> str:
    return (
        "Read this Korean bankbook copy. Extract only visible bank account information. "
        "Return plain text only: bank, account_holder, account_number. Do not describe layout. "
        "Do not guess. If the account number is not visible, return account_number: NOT_FOUND."
    )


def _chandra_result_text(result: object) -> str:
    for attr in ("markdown", "html", "raw", "text"):
        value = getattr(result, attr, None)
        if value:
            return str(value)
    return str(result)


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)


def _pdftotext(path: Path, options: OcrOptions) -> str:
    if not shutil.which("pdftotext"):
        return ""
    proc = _run(["pdftotext", "-layout", str(path), "-"], options.timeout_seconds)
    return proc.stdout if proc.returncode == 0 else ""


def _pdf_to_images(path: Path, options: OcrOptions) -> list[Path]:
    if not shutil.which("pdftoppm"):
        raise RuntimeError("pdftoppm is required to OCR image-only PDFs")
    out_prefix = path.parent / f"{path.stem}_page"
    proc = _run(["pdftoppm", "-png", "-r", "220", str(path), str(out_prefix)], options.timeout_seconds)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "pdftoppm failed")
    return sorted(path.parent.glob(f"{path.stem}_page-*.png"))


def _tesseract_image(path: Path, options: OcrOptions) -> str:
    proc = _run(
        ["tesseract", str(path), "stdout", "-l", options.tesseract_lang, "--psm", "6"],
        options.timeout_seconds,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "tesseract failed")
    return proc.stdout


def _pages_for_remote(path: Path, options: OcrOptions) -> list[Path]:
    if path.suffix.lower() == ".pdf":
        return _pdf_to_images(path, options)
    return [path]


def _pages_for_chandra(path: Path, options: OcrOptions) -> list[Path]:
    if path.suffix.lower() == ".pdf":
        paths = _pdf_to_images(path, options)
    else:
        paths = [path]
    return [_resize_image_for_chandra(image_path, options) for image_path in paths]


def _resize_image_for_chandra(path: Path, options: OcrOptions) -> Path:
    max_side = options.chandra_image_max_side
    if max_side <= 0:
        return path
    try:
        from PIL import Image
    except Exception:
        return path

    with Image.open(path) as image:
        if max(image.size) <= max_side:
            return path
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        resized = path.with_name(f"{path.stem}_chandra_{max_side}.jpg")
        image.save(resized, format="JPEG", quality=90, optimize=True)
        return resized


def _data_url(path: Path, options: OcrOptions) -> str:
    image_bytes, mime = _hf_image_payload(path, options)
    payload = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _hf_image_payload(path: Path, options: OcrOptions) -> tuple[bytes, str]:
    try:
        from PIL import Image
    except Exception:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return path.read_bytes(), mime

    with Image.open(path) as image:
        image = image.convert("RGB")
        max_side = options.hf_image_max_side
        if max_side > 0 and max(image.size) > max_side:
            image.thumbnail((max_side, max_side))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=92, optimize=True)
        return output.getvalue(), "image/jpeg"
