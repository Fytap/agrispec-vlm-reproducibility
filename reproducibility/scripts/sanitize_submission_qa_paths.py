#!/usr/bin/env python3
"""Replace workstation paths in copied submission QA reports with package paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    qa = args.package / "05_QA_AND_MANIFEST"

    latex_path = qa / "latex_submission_audit.json"
    latex = json.loads(latex_path.read_text(encoding="utf-8"))
    latex["root"] = "02_OVERLEAF_SOURCE"
    latex_path.write_text(json.dumps(latex, indent=2) + "\n", encoding="utf-8")

    pdf_path = qa / "submission_pdf_audit.json"
    pdf = json.loads(pdf_path.read_text(encoding="utf-8"))
    path_map = {
        "manuscript": "01_UPLOAD_TO_SAT/SAT_Manuscript.pdf",
        "supplement": "01_UPLOAD_TO_SAT/SAT_Supplementary_Material.pdf",
        "cover_letter": "01_UPLOAD_TO_SAT/SAT_Cover_Letter.pdf",
        "graphical_abstract": "01_UPLOAD_TO_SAT/SAT_Graphical_Abstract.pdf",
        "declaration": "05_QA_AND_MANIFEST/Declaration_of_Competing_Interests_preview.pdf",
        "informed_consent": "05_QA_AND_MANIFEST/Ethics_and_Informed_Consent_Statement_preview.pdf",
    }
    for document in pdf["documents"]:
        document["path"] = path_map[document["label"]]
    pdf_path.write_text(json.dumps(pdf, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
