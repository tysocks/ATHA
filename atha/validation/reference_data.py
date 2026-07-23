"""Reference-dataset schema and ingestion for historical / external correlation.

Workstream 6.4 defines a standard package for bringing hot-fire telemetry,
literature digitizations, map tables, and external-solver exports into ATHA
verification. Each dataset is a folder with:

- ``manifest.yaml`` — provenance, units, alignment, channel mapping
- one or more CSV / HDF5 traces referenced by the manifest
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from atha.output.comparison import load_time_series


@dataclass(frozen=True)
class ChannelMapping:
    """Maps a reference column onto an ATHA telemetry alias or source path."""

    reference_channel: str
    atha_channel: str
    units: str = ""
    scale: float = 1.0
    offset: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class TimeAlignment:
    """Rules for aligning an external time base to ATHA mission time."""

    time_column: str = "TIME"
    time_unit: str = "s"
    time_offset_s: float = 0.0
    time_scale: float = 1.0
    trim_start_s: float | None = None
    trim_end_s: float | None = None


@dataclass(frozen=True)
class ReferenceDataset:
    """In-memory representation of a Workstream 6.4 reference package."""

    id: str
    title: str
    source: str
    provenance: str
    allowed_use: str
    category: str
    path: Path
    data_file: Path
    time_alignment: TimeAlignment
    channels: tuple[ChannelMapping, ...]
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def load_series(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Load the reference trace and apply alignment / channel mapping."""

        time, raw = load_time_series(self.data_file, time_column=self.time_alignment.time_column)
        time = (np.asarray(time, dtype=float) * float(self.time_alignment.time_scale)) + float(
            self.time_alignment.time_offset_s
        )
        mapped: dict[str, np.ndarray] = {}
        for channel in self.channels:
            if channel.reference_channel not in raw:
                raise KeyError(
                    f"reference dataset {self.id!r} missing channel {channel.reference_channel!r} "
                    f"in {self.data_file}"
                )
            values = np.asarray(raw[channel.reference_channel], dtype=float) * float(channel.scale) + float(
                channel.offset
            )
            mapped[channel.atha_channel] = values
        if self.time_alignment.trim_start_s is not None or self.time_alignment.trim_end_s is not None:
            start = float(self.time_alignment.trim_start_s if self.time_alignment.trim_start_s is not None else time[0])
            end = float(self.time_alignment.trim_end_s if self.time_alignment.trim_end_s is not None else time[-1])
            mask = (time >= start - 1.0e-12) & (time <= end + 1.0e-12)
            time = time[mask]
            mapped = {key: values[mask] for key, values in mapped.items()}
        return time, mapped


def load_reference_dataset(path: str | Path) -> ReferenceDataset:
    """Load a reference dataset folder or manifest YAML path."""

    root = Path(path).expanduser().resolve()
    if root.is_dir():
        manifest_path = root / "manifest.yaml"
    else:
        manifest_path = root
        root = root.parent
    if not manifest_path.exists():
        raise FileNotFoundError(f"reference dataset manifest not found: {manifest_path}")
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"reference manifest must be a mapping: {manifest_path}")
    data_rel = raw.get("data_file", raw.get("trace", "trace.csv"))
    data_file = (root / str(data_rel)).resolve()
    if not data_file.exists():
        raise FileNotFoundError(f"reference data file not found: {data_file}")
    alignment_raw = raw.get("time_alignment", {})
    if not isinstance(alignment_raw, Mapping):
        raise ValueError("time_alignment must be a mapping")
    channels_raw = raw.get("channels", [])
    if not isinstance(channels_raw, list) or not channels_raw:
        raise ValueError("reference manifest requires a non-empty channels list")
    channels = tuple(_channel_mapping(item, index) for index, item in enumerate(channels_raw))
    notes_raw = raw.get("notes", [])
    notes = tuple(str(item) for item in notes_raw) if isinstance(notes_raw, list) else (str(notes_raw),)
    return ReferenceDataset(
        id=str(raw.get("id", root.name)),
        title=str(raw.get("title", root.name)),
        source=str(raw.get("source", "unspecified")),
        provenance=str(raw.get("provenance", "")),
        allowed_use=str(raw.get("allowed_use", "verification_only")),
        category=str(raw.get("category", "literature")),
        path=root,
        data_file=data_file,
        time_alignment=TimeAlignment(
            time_column=str(alignment_raw.get("time_column", "TIME")),
            time_unit=str(alignment_raw.get("time_unit", "s")),
            time_offset_s=float(alignment_raw.get("time_offset_s", 0.0)),
            time_scale=float(alignment_raw.get("time_scale", 1.0)),
            trim_start_s=_optional_float(alignment_raw.get("trim_start_s")),
            trim_end_s=_optional_float(alignment_raw.get("trim_end_s")),
        ),
        channels=channels,
        notes=notes,
        metadata={
            key: value
            for key, value in raw.items()
            if key
            not in {
                "id",
                "title",
                "source",
                "provenance",
                "allowed_use",
                "category",
                "data_file",
                "trace",
                "time_alignment",
                "channels",
                "notes",
            }
        },
    )


def discover_reference_datasets(root: str | Path) -> list[ReferenceDataset]:
    """Discover all ``manifest.yaml`` packages under a historical-data root."""

    base = Path(root).expanduser().resolve()
    if not base.exists():
        return []
    datasets: list[ReferenceDataset] = []
    for manifest in sorted(base.rglob("manifest.yaml")):
        datasets.append(load_reference_dataset(manifest))
    return datasets


def _channel_mapping(raw: object, index: int) -> ChannelMapping:
    if not isinstance(raw, Mapping):
        raise ValueError(f"channels[{index}] must be a mapping")
    try:
        reference = str(raw["reference_channel"])
        atha = str(raw.get("atha_channel", raw.get("channel", reference)))
    except KeyError as exc:
        raise ValueError(f"channels[{index}] missing required key {exc.args[0]!r}") from exc
    return ChannelMapping(
        reference_channel=reference,
        atha_channel=atha,
        units=str(raw.get("units", "")),
        scale=float(raw.get("scale", 1.0)),
        offset=float(raw.get("offset", 0.0)),
        description=str(raw.get("description", "")),
    )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
