#!/usr/bin/env python3
"""v2 wrapper: preserve semantic DDP v1 algorithm, repair source-manifest writing."""
from __future__ import annotations
import csv
from pathlib import Path
import train_p2_sugarbeets_rgb_semantic_specialist_ddp as base

def write_projected(path: Path, fields: list[str], data: object) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for row in data:
            writer.writerow({field: row.get(field, '') for field in fields})

base.write = write_projected
base.main()
