"""GeoPackage and CSV outputs with explicit CRS and stable schemas."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize, shapes
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
    campaign_type: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    target_crs = stability.grid.crs
    lot_metric = reproject(lot.geometry, "EPSG:4326", target_crs)
    gpd.GeoDataFrame(
        [{
            "lot_id": lot.lot_id,
            "name": lot.name,
            "source": lot.source,
            "campaign": campaign_type,
            "geometry": lot_metric,
        }],
        crs=target_crs,
    ).to_file(path, layer="perimetro", driver="GPKG")

    stability_patches = _stability_patches(stability)
    if stability_patches:
        gpd.GeoDataFrame(stability_patches, geometry="geometry", crs=target_crs).to_file(
            path, layer="estabilidad_parches", driver="GPKG"
        )
        dissolved = _dissolve_stability(stability_patches)
        gpd.GeoDataFrame(dissolved, geometry="geometry", crs=target_crs).to_file(
            path, layer="estabilidad_disuelta", driver="GPKG"
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


def _stability_patches(stability: StabilityResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    patch_id = 0
    for geometry, value in shapes(
        stability.classes,
        mask=stability.classes > 0,
        transform=stability.grid.transform,
    ):
        class_id = int(value)
        polygon = shape(geometry)
        if polygon.area > 0:
            patch_id += 1
            mask = rasterize(
                [(polygon, 1)],
                out_shape=stability.grid.shape,
                transform=stability.grid.transform,
                fill=0,
            ).astype(bool)
            mask &= stability.classes == class_id
            rows.append(
                {
                    "patch_id": patch_id,
                    "campaign": stability.campaign_type,
                    "class_id": class_id,
                    "label": STABILITY_LABELS[class_id],
                    "hectares": round(polygon.area / 10_000, 3),
                    "z_mean": round(float(np.nanmean(stability.z_mean[mask])), 3),
                    "z_std": round(float(np.nanmean(stability.z_std[mask])), 3),
                    "geometry": polygon.simplify(stability.grid.resolution),
                }
            )
    return rows


def _dissolve_stability(patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id, label in STABILITY_LABELS.items():
        selected = [row for row in patches if row["class_id"] == class_id]
        if not selected:
            continue
        hectares = sum(float(row["hectares"]) for row in selected)
        rows.append(
            {
                "class_id": class_id,
                "campaign": selected[0]["campaign"],
                "label": label,
                "hectares": round(hectares, 3),
                "z_mean": round(
                    sum(float(row["z_mean"]) * float(row["hectares"]) for row in selected)
                    / hectares,
                    3,
                ),
                "z_std": round(
                    sum(float(row["z_std"]) * float(row["hectares"]) for row in selected)
                    / hectares,
                    3,
                ),
                "geometry": unary_union([row["geometry"] for row in selected]),
            }
        )
    return rows
