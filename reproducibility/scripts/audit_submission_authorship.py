#!/usr/bin/env python3
"""Audit eight-author consistency across the SAT submission artifacts."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

from pypdf import PdfReader


AUTHORS = [
    "Shuhao Liu",
    "Zhuo Chen",
    "Zhi Ling",
    "Yu Yan",
    "Jie Liu",
    "Qiuxue Wu",
    "Ziyi Kuang",
    "Haiyou Zhang",
]
HAIYOU_ROLES = ["Data curation", "Validation", "Writing"]


def pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    return re.sub(r"<[^>]+>", " ", xml)


def all_names(text: str) -> bool:
    return all(name in text for name in AUTHORS)


def citation_authors(path: Path) -> list[str]:
    names: list[str] = []
    family_name: str | None = None
    in_authors = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "authors:":
            in_authors = True
            continue
        if in_authors and line.startswith("repository-code:"):
            break
        if not in_authors:
            continue
        family_match = re.fullmatch(r"\s*- family-names:\s*(.+)", line)
        if family_match:
            family_name = family_match.group(1).strip()
            continue
        given_match = re.fullmatch(r"\s*given-names:\s*(.+)", line)
        if given_match and family_name:
            names.append(f"{given_match.group(1).strip()} {family_name}")
            family_name = None
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    repository = args.repository.resolve()
    package = args.package.resolve()

    repository_metadata = (
        repository / "manuscript_source" / "author_metadata.tex"
    ).read_text(encoding="utf-8")
    package_metadata = (
        package / "02_OVERLEAF_SOURCE" / "author_metadata.tex"
    ).read_text(encoding="utf-8")
    editorial_metadata = (
        package / "04_EDITORIAL_METADATA" / "SUBMISSION_METADATA.md"
    ).read_text(encoding="utf-8")
    citation_names = citation_authors(repository / "CITATION.cff")

    manuscript = pdf_text(package / "01_UPLOAD_TO_SAT" / "SAT_Manuscript.pdf")
    supplement = pdf_text(
        package / "01_UPLOAD_TO_SAT" / "SAT_Supplementary_Material.pdf"
    )
    declaration = docx_text(
        package / "01_UPLOAD_TO_SAT" / "Declaration_of_Competing_Interests.docx"
    )

    haiyou_credit_context = manuscript[manuscript.find("Haiyou Zhang:") :]
    haiyou_credit_context = haiyou_credit_context.split(
        "Declaration of competing interests", 1
    )[0]

    checks = {
        "repository_author_metadata_has_all_eight": all_names(repository_metadata),
        "overleaf_author_metadata_has_all_eight": all_names(package_metadata),
        "editorial_metadata_has_all_eight": all_names(editorial_metadata),
        "citation_cff_has_exact_author_sequence": citation_names == AUTHORS,
        "manuscript_pdf_has_all_eight": all_names(manuscript),
        "supplement_pdf_has_all_eight": all_names(supplement),
        "declaration_docx_has_all_eight": all_names(declaration),
        "haiyou_email_in_repository_metadata": (
            "haiyou.zhang@kiwiar.com" in repository_metadata
        ),
        "haiyou_email_in_overleaf_metadata": (
            "haiyou.zhang@kiwiar.com" in package_metadata
        ),
        "haiyou_affiliation_c_in_repository_metadata": (
            r"\author[aff-c]{Haiyou Zhang}" in repository_metadata
        ),
        "haiyou_credit_roles_in_manuscript": all(
            role in haiyou_credit_context for role in HAIYOU_ROLES
        ),
        "haiyou_credit_roles_in_editorial_metadata": all(
            role in editorial_metadata[editorial_metadata.find("**Haiyou Zhang:**") :]
            for role in HAIYOU_ROLES
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "status": status,
        "expected_author_count": len(AUTHORS),
        "expected_authors": AUTHORS,
        "citation_author_count": len(citation_names),
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
