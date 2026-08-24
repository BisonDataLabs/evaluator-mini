"""Domain models shared by ingestion, analysis and exporters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

SourceKind = Literal["upload", "snowflake"]
JobStatus = Literal["pending", "running", "completed", "failed", "skipped"]


@dataclass(slots=True)
class Lot:
    lot_id: str
    name: str
    geometry: BaseGeometry
    source: SourceKind
    establishment: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_feature(self) -> dict[str, Any]:
        properties = {
            "lot_id": self.lot_id,
            "name": self.name,
            "source": self.source,
            "establishment": self.establishment,
            **self.metadata,
        }
        return {
            "type": "Feature",
            "properties": {k: v for k, v in properties.items() if _json_scalar(v)},
            "geometry": mapping(self.geometry),
        }


def _json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


@dataclass(frozen=True, slots=True)
class ProcessingOptions:
    buffer_m: int = 500
    start_year: int = 1985
    end_year: int = 2025
    stability_seasons: int = 8
    max_cloud_percent: float = 30.0
    zone_counts: tuple[int, ...] = (2, 3, 4)
    export_quadrant_imagery: bool = True
    calculate_productivity: bool = True


@dataclass(slots=True)
class LotResult:
    lot_id: str
    lot_name: str
    output_dir: Path
    status: JobStatus
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    artifacts: list[Path] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def manifest_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["artifacts"] = [str(p) for p in self.artifacts]
        return data
