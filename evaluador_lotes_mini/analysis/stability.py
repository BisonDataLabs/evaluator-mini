"""Eight-season relative productivity and stability classification."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.features import geometry_mask
from scipy.ndimage import distance_transform_edt, median_filter
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from evaluador_lotes_mini.imagery.grid import RasterGrid, grid_for_geometry, write_raster
from evaluador_lotes_mini.imagery.planetary import PlanetaryImagery, read_ndvi

STABILITY_LABELS = {
    1: "Alto estable",
    2: "Alto variable",
    3: "Medio",
    4: "Bajo variable",
    5: "Bajo persistente",
}


@dataclass(slots=True)
class StabilityResult:
    grid: RasterGrid
    seasonal_ndvi: np.ndarray
    z_mean: np.ndarray
    z_std: np.ndarray
    classes: np.ndarray
    seasons: list[dict[str, Any]]
    artifacts: list[Path]


def calculate_stability(
    imagery: PlanetaryImagery,
    geometry: BaseGeometry,
    output_dir: Path,
    *,
    seasons: int = 8,
    max_cloud_percent: float = 30,
    end_year: int | None = None,
    scenes_per_season: int = 6,
    progress: Callable[[int, int, str], None] | None = None,
) -> StabilityResult:
    grid = grid_for_geometry(geometry, resolution=10)
    end_year = end_year or _last_complete_gruesa_year()
    seasonal_arrays: list[np.ndarray] = []
    season_metadata: list[dict[str, Any]] = []

    for harvest_year in range(end_year - seasons + 1, end_year + 1):
        start = date(harvest_year - 1, 10, 1)
        end = date(harvest_year, 3, 31)
        items = imagery.search_sentinel(
            geometry, start, end, max_cloud_percent=max_cloud_percent, max_items=50
        )
        candidates = sorted(
            items,
            key=lambda item: float(item.properties.get("eo:cloud_cover", 100)),
        )[:scenes_per_season]
        stack: list[np.ndarray] = []
        used: list[str] = []
        for item in candidates:
            try:
                ndvi, valid = read_ndvi(item, grid)
                if valid.mean() < 0.25:
                    continue
                stack.append(ndvi)
                used.append(item.id)
            except Exception:
                continue
        if not stack:
            if progress:
                progress(
                    harvest_year - (end_year - seasons + 1) + 1,
                    seasons,
                    f"{harvest_year - 1}/{str(harvest_year)[-2:]}",
                )
            continue
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            season_max = np.nanmax(np.stack(stack), axis=0).astype("float32")
        seasonal_arrays.append(season_max)
        season_metadata.append(
            {
                "campaign": f"{harvest_year - 1}/{str(harvest_year)[-2:]}",
                "items": used,
                "scene_count": len(used),
            }
        )
        if progress:
            progress(
                harvest_year - (end_year - seasons + 1) + 1,
                seasons,
                f"{harvest_year - 1}/{str(harvest_year)[-2:]}",
            )

    if len(seasonal_arrays) < 3:
        raise RuntimeError(
            f"Solo se pudieron construir {len(seasonal_arrays)} campañas; se requieren al menos 3"
        )

    stack_array = np.stack(seasonal_arrays)
    z_stack = np.full_like(stack_array, np.nan, dtype="float32")
    interior = grid.geometry.buffer(-2 * grid.resolution)
    if interior.is_empty or interior.area < grid.geometry.area * 0.7:
        interior = grid.geometry.buffer(-grid.resolution)
    if interior.is_empty:
        interior = grid.geometry
    inside = geometry_mask(
        [mapping(interior)],
        out_shape=grid.shape,
        transform=grid.transform,
        invert=True,
        all_touched=False,
    )
    for index, season in enumerate(stack_array):
        values = season[inside & np.isfinite(season)]
        if values.size < 20:
            continue
        mean = float(np.mean(values))
        std = max(float(np.std(values)), 0.01)
        z_stack[index] = (season - mean) / std

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        z_mean = np.nanmean(z_stack, axis=0).astype("float32")
        z_std = np.nanstd(z_stack, axis=0).astype("float32")
    valid = inside & np.isfinite(z_mean) & np.isfinite(z_std)
    classes = np.zeros(grid.shape, dtype="uint8")
    classes[valid] = 3
    classes[valid & (z_mean > 0.3) & (z_std < 0.4)] = 1
    classes[valid & (z_mean > 0.3) & (z_std >= 0.4)] = 2
    classes[valid & (z_mean < -0.3) & (z_std >= 0.4)] = 4
    classes[valid & (z_mean < -0.3) & (z_std < 0.4)] = 5
    smoothed = median_filter(classes, size=3, mode="nearest")
    classes[valid] = smoothed[valid]
    full_inside = grid.inside_mask
    perimeter = full_inside & ~valid
    if np.any(valid) and np.any(perimeter):
        nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        classes[perimeter] = classes[tuple(nearest[:, perimeter])]
    classes[~full_inside] = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        write_raster(
            output_dir / "productividad_media_relativa.tif",
            z_mean,
            grid,
            dtype="float32",
            nodata=-9999,
            descriptions=["z_mean"],
        ),
        write_raster(
            output_dir / "variabilidad_temporal.tif",
            z_std,
            grid,
            dtype="float32",
            nodata=-9999,
            descriptions=["z_std"],
        ),
        write_raster(
            output_dir / "estabilidad_5_clases.tif",
            classes,
            grid,
            dtype="uint8",
            nodata=0,
            descriptions=["stability_class"],
        ),
    ]
    return StabilityResult(
        grid=grid,
        seasonal_ndvi=stack_array,
        z_mean=z_mean,
        z_std=z_std,
        classes=classes,
        seasons=season_metadata,
        artifacts=artifacts,
    )


def stability_statistics(result: StabilityResult) -> list[dict[str, Any]]:
    pixel_area_ha = result.grid.resolution**2 / 10_000
    total = int(np.count_nonzero(result.classes))
    rows = []
    for class_id, label in STABILITY_LABELS.items():
        pixels = int(np.count_nonzero(result.classes == class_id))
        rows.append(
            {
                "class_id": class_id,
                "label": label,
                "hectares": round(pixels * pixel_area_ha, 2),
                "percent": round(pixels / total * 100, 1) if total else 0,
            }
        )
    return rows


def _last_complete_gruesa_year() -> int:
    today = date.today()
    return today.year if today.month >= 4 else today.year - 1
