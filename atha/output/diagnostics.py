from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ResidualDiagnosticRecord:
    name: str
    value: float
    normalized: float


def residual_diagnostics_from_mapping(residuals: Mapping[str, float]) -> list[ResidualDiagnosticRecord]:
    return sorted(
        [
            ResidualDiagnosticRecord(name=str(name), value=float(value), normalized=float(value))
            for name, value in residuals.items()
        ],
        key=lambda item: abs(item.normalized),
        reverse=True,
    )


def write_residual_diagnostics_json(path: str | Path, records: list[ResidualDiagnosticRecord]) -> Path:
    out_path = Path(path)
    payload = {
        "format": "atha.residual_diagnostics.v1",
        "residuals": [record.__dict__ for record in records],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def write_residual_diagnostics_csv(path: str | Path, records: list[ResidualDiagnosticRecord]) -> Path:
    out_path = Path(path)
    with out_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "value", "normalized"])
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)
    return out_path
