import json
import unicodedata
import zipfile
from pathlib import Path

from scripts.build_downloads_field_ocr_manifest import PROMPTS, build_manifest, classify_download_path


def test_classify_download_path_maps_generality_categories():
    assert classify_download_path(Path("PRE0417/세금계산서.pdf")) == "receipt_tax"
    assert classify_download_path(Path("docs/4. MYSC_사업자등록증.pdf")) == "business_registry"
    assert classify_download_path(Path("foo/이력서_홍길동.pdf")) == "resume_form"
    assert classify_download_path(Path("mermaid-diagram.png")) == "diagram_screenshot"
    assert classify_download_path(Path("2501.00321v2.pdf")) == "paper_pdf"


def test_build_manifest_combines_depth_bankbook_and_generality_samples(tmp_path: Path):
    bank_zip = tmp_path / "5. 통장 사본 업로드 (File responses)-20260424T051727Z-3-001.zip"
    with zipfile.ZipFile(bank_zip, "w") as archive:
        archive.writestr("001_통장사본 - 김테스트.jpg", b"fake")
        archive.writestr("002_계좌개설확인서 - 이테스트.pdf", b"fake")

    (tmp_path / "PRE0417").mkdir()
    (tmp_path / "PRE0417" / "세금계산서.pdf").write_bytes(b"fake")
    (tmp_path / "사업자등록증.pdf").write_bytes(b"fake")
    (tmp_path / "mermaid-diagram.png").write_bytes(b"fake")

    rows = build_manifest(tmp_path, bankbook_limit=2, generality_limit=3)

    purposes = {row["purpose"] for row in rows}
    categories = {row["category"] for row in rows}
    assert purposes == {"1_depth_bankbook", "2_generality_ocr"}
    assert {"bankbook_zip_member", "receipt_tax", "business_registry", "diagram_screenshot"} <= categories
    assert all("prompt" in row and row["prompt"] for row in rows)

    encoded = [json.dumps(row, ensure_ascii=False) for row in rows]
    assert any("001_통장사본" in line for line in encoded)


def test_bankbook_manifest_normalizes_zip_member_names(tmp_path: Path):
    bank_zip = tmp_path / "5. 통장 사본 업로드 (File responses)-20260424T051727Z-3-001.zip"
    decomposed = unicodedata.normalize("NFD", "김민채_통장사본 - 김민채.jpeg")
    with zipfile.ZipFile(bank_zip, "w") as archive:
        archive.writestr(f"5. 통장 사본 업로드 (File responses)/{decomposed}", b"fake")

    [row] = build_manifest(tmp_path, bankbook_limit=1, generality_limit=0)

    assert row["member_name"] == unicodedata.normalize("NFC", row["member_name"])
    assert "김민채_통장사본" in row["member_name"]


def test_bankbook_prompt_matches_practical_successful_bank_account_prompt():
    assert "OCR this Korean bank account image" in PROMPTS["bankbook_account"]
    assert "계좌번호" in PROMPTS["bankbook_account"]
