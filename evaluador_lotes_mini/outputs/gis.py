"""GeoPackage and CSV outputs with explicit CRS and stable schemas."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union

from evaluador_lotes_mini.analysis.stability import (
    STABILITY_LABELS,
    StabilityResult,
    stability_statistics,
)
from evaluador_lotes_mini.analysis.zoning import ZoneAlternative
from evaluador_lotes_mini.geometry import reproject
from evaluador_lotes_mini.models import Lot


def write_geopackage(
    path: Path,
    lot: Lot,
    stability: StabilityResult,
    alternatives: list[ZoneAlternative],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target_crs = stability.grid.crs
    lot_metric = reproject(lot.geometry, "EPSG:4326", target_crs)
    gpd.GeoDataFrame(
        [{"lot_id": lot.lot_id, "name": lot.name, "source": lot.source, "geometry": lot_metric}],
        crs=target_crs,
    ).to_file(path, layer="perimetro", driver="GPKG")

    stability_rows = _stability_polygons(stability)
    if stability_rows:
        gpd.GeoDataFrame(stability_rows, geometry="geometry", crs=target_crs).to_file(
            path, layer="estabilidad_5_clases", driver="GPKG"
        )

    for alternative in alternatives:
        records = [dict(row) for row in alternative.polygons]
        if not records:
            continue
        gpd.GeoDataFrame(records, geometry="geometry", crs=target_crs).to_file(
            path, layer=f"ambientes_k{alternative.k}", driver="GPKG"
        )
    return path


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row if key != "geometry"})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key in fields} for row in rows
        )
    return path


def write_stability_csv(path: Path, stability: StabilityResult) -> Path:
    return write_rows_csv(path, stability_statistics(stability))


def _stability_polygons(stability: StabilityResult) -> list[dict[str, Any]]:
    grouped: dict[int, list] = {class_id: [] for class_id in STABILITY_LABELS}
    for geometry, value in shapes(
        stability.classes,
        mask=stability.classes > 0,
        transform=stability.grid.transform,
    ):
        class_id = int(value)
        polygon = shape(geometry)
        if polygon.area >= 500:
            grouped[class_id].append(polygon)
    pixel_area_ha = stability.grid.resolution**2 / 10_000
    rows: list[dict[str, Any]] = []
    for class_id, polygons in grouped.items():
        if not polygons:
            continue
        mask = stability.classes == class_id
        rows.append(
            {
                "class_id": class_id,
                "label": STABILITY_LABELS[class_id],
                "hectares": round(int(np.count_nonzero(mask)) * pixel_area_ha, 2),
                "z_mean": round(float(np.nanmean(stability.z_mean[mask])), 3),
                "z_std": round(float(np.nanmean(stability.z_std[mask])), 3),
                "geometry": unary_union(polygons).simplify(stability.grid.resolution),
            }
        )
    return rows
