#!/usr/bin/env python3
"""Retain only BibTeX entries cited by one or more LaTeX files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CITE_RE = re.compile(r"\\cite\w*\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}")
KEY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.I)


def cited_keys(tex_paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in tex_paths:
        text = path.read_text(encoding="utf-8")
        for match in CITE_RE.finditer(text):
            keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def split_entries(text: str) -> list[str]:
    entries: list[str] = []
    index = 0
    while True:
        start = text.find("@", index)
        if start < 0:
            break
        brace = text.find("{", start)
        if brace < 0:
            break
        depth = 0
        end = brace
        in_quote = False
        escaped = False
        while end < len(text):
            char = text[end]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_quote = not in_quote
            elif not in_quote and char == "{":
                depth += 1
            elif not in_quote and char == "}":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1
        entries.append(text[start:end].strip())
        index = end
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bib", type=Path, required=True)
    parser.add_argument("--tex", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    keys = cited_keys(args.tex)
    selected = []
    found = set()
    for entry in split_entries(args.bib.read_text(encoding="utf-8")):
        match = KEY_RE.search(entry)
        if match and match.group(1) in keys:
            selected.append(entry)
            found.add(match.group(1))
    missing = sorted(keys - found)
    if missing:
        raise SystemExit(f"Missing BibTeX entries: {', '.join(missing)}")
    args.output.write_text("\n\n".join(selected) + "\n", encoding="utf-8")
    print(f"retained={len(selected)} cited={len(keys)} output={args.output}")


if __name__ == "__main__":
    main()
