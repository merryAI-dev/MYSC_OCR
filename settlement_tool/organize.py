from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .core import (
    DOC_BANK,
    DOC_ID,
    DOC_PAYMENT,
    DOC_TYPES,
    FileRef,
    doc_filename,
    extract_roster,
    extract_zip_member,
    match_files_by_name,
    normalize_text,
    safe_filename_part,
    write_csv,
    zip_file_names,
)


@dataclass(frozen=True)
class DocumentItem:
    name: str
    group: str
    no: int
    doc_type: str
    status: str
    source_name: str | None
    zip_path: Path | None
    output_path: Path | None
    reason: str


@dataclass
class DocumentPlan:
    items: list[DocumentItem]

    def report_rows(self) -> list[dict[str, object]]:
        return [
            {
                "group": item.group,
                "no": item.no,
                "name": item.name,
                "doc_type": item.doc_type,
                "status": item.status,
                "source_name": item.source_name or "",
                "output_path": str(item.output_path or ""),
                "reason": item.reason,
            }
            for item in self.items
        ]

    def summary_rows(self) -> list[dict[str, object]]:
        counts: dict[tuple[str, str, str], int] = {}
        for item in self.items:
            key = (item.group, item.doc_type, item.status)
            counts[key] = counts.get(key, 0) + 1
        return [
            {"group": group, "doc_type": doc_type, "status": status, "count": count}
            for (group, doc_type, status), count in sorted(counts.items())
        ]


def _zip_refs(zip_path: Path, names: list[str]) -> list[FileRef]:
    return [FileRef(source_name=name, zip_path=zip_path) for name in names]


def _filter_doc_members(names: list[str], doc_type: str) -> list[str]:
    if doc_type == DOC_PAYMENT:
        return [name for name in names if DOC_PAYMENT in Path(name).name]
    if doc_type == DOC_ID:
        return [name for name in names if DOC_ID in Path(name).name]
    if doc_type == DOC_BANK:
        return [name for name in names if DOC_BANK in Path(name).name or "통장" in Path(name).name]
    return names


def _resolve_matches(names: list[str], refs: list[FileRef], doc_type: str, overrides: dict[tuple[str, str], str]) -> dict[str, tuple[str, FileRef | None, str]]:
    match = match_files_by_name(names, [ref.source_name for ref in refs])
    by_source = {ref.source_name: ref for ref in refs}
    resolved: dict[str, tuple[str, FileRef | None, str]] = {}

    for name in names:
        override = overrides.get((name, doc_type))
        if override:
            ref = by_source.get(override)
            if ref:
                resolved[name] = ("confirmed", ref, "override")
            else:
                resolved[name] = ("missing", None, "override_not_found")
        elif name in match.confirmed:
            ref = by_source[match.confirmed[name].source_name]
            resolved[name] = ("confirmed", ref, "single_name_match")
        elif name in match.ambiguous:
            resolved[name] = ("ambiguous", None, "multiple_name_matches")
        else:
            resolved[name] = ("missing", None, "no_name_match")

    return resolved


def build_document_plan(
    roster_path: Path | str,
    payment_zip: Path | str,
    id_zip: Path | str,
    bank_zip: Path | str,
    overrides: dict[tuple[str, str], str],
) -> DocumentPlan:
    roster = extract_roster(roster_path)
    payment_zip = Path(payment_zip)
    id_zip = Path(id_zip)
    bank_zip = Path(bank_zip)

    payment_names = zip_file_names(payment_zip)
    id_names = zip_file_names(id_zip)
    bank_names = zip_file_names(bank_zip)
    all_cost_refs = _zip_refs(payment_zip, payment_names)

    doc_sources = {
        DOC_PAYMENT: _zip_refs(payment_zip, _filter_doc_members(payment_names, DOC_PAYMENT)),
        DOC_ID: _zip_refs(id_zip, id_names)
        + [ref for ref in all_cost_refs if DOC_ID in Path(ref.source_name).name],
        DOC_BANK: _zip_refs(bank_zip, bank_names)
        + [
            ref
            for ref in all_cost_refs
            if DOC_BANK in Path(ref.source_name).name or "통장" in Path(ref.source_name).name
        ],
    }

    matches = {
        doc_type: _resolve_matches(roster.names, refs, doc_type, overrides)
        for doc_type, refs in doc_sources.items()
    }

    items: list[DocumentItem] = []
    for person in roster.people:
        for doc_type in DOC_TYPES:
            status, ref, reason = matches[doc_type][person.name]
            output_path = None
            source_name = None
            zip_path = None
            if ref:
                output_path = Path(person.group) / doc_filename(person.name, doc_type, ref.suffix)
                source_name = ref.source_name
                zip_path = ref.zip_path
            items.append(
                DocumentItem(
                    name=person.name,
                    group=person.group,
                    no=person.no,
                    doc_type=doc_type,
                    status=status,
                    source_name=source_name,
                    zip_path=zip_path,
                    output_path=output_path,
                    reason=reason,
                )
            )

    return DocumentPlan(items=items)


def materialize_plan(plan: DocumentPlan, output_dir: Path | str, dry_run: bool) -> None:
    output_dir = Path(output_dir)
    for item in plan.items:
        if item.status != "confirmed" or not item.zip_path or not item.source_name or not item.output_path:
            continue
        if dry_run:
            continue
        extract_zip_member(item.zip_path, item.source_name, output_dir / item.output_path)


def write_document_reports(plan: DocumentPlan, report_dir: Path | str) -> None:
    report_dir = Path(report_dir)
    write_csv(
        report_dir / "document_matches.csv",
        plan.report_rows(),
        ["group", "no", "name", "doc_type", "status", "source_name", "output_path", "reason"],
    )
    write_csv(
        report_dir / "document_summary.csv",
        plan.summary_rows(),
        ["group", "doc_type", "status", "count"],
    )


def default_overrides_template(path: Path | str) -> None:
    path = Path(path)
    rows = [
        {
            "name": "김경진",
            "field": DOC_PAYMENT,
            "value": "원본 ZIP 내부 경로를 여기에 입력",
            "note": "예: 기준표 이름과 파일명이 다른 경우",
        }
    ]
    write_csv(path, rows, ["name", "field", "value", "note"])


def make_group_directories(plan: DocumentPlan, output_dir: Path | str) -> None:
    output_dir = Path(output_dir)
    for group in sorted({normalize_text(item.group) for item in plan.items if item.group}):
        (output_dir / safe_filename_part(group)).mkdir(parents=True, exist_ok=True)
