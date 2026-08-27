"""Campaign-specific relative productivity and temporal stability."""

from __future__ import annotations

import json
import re
import warnings
from calendar import monthrange
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
from rasterio.features import geometry_mask
from scipy.ndimage import distance_transform_edt, median_filter
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from evaluador_lotes_mini.imagery.grid import RasterGrid, grid_for_geometry, write_raster
from evaluador_lotes_mini.imagery.planetary import (
    PlanetaryImagery,
    read_ndvi_false_color,
    scene_quality,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CampaignType = Literal["gruesa", "fina"]

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
    analysis_mask: np.ndarray
    campaign_type: CampaignType
    edge_exclusion_m: int
    seasons: list[dict[str, Any]]
    scenes: list[dict[str, Any]]
    artifacts: list[Path]


def calculate_stability(
    imagery: PlanetaryImagery,
    geometry: BaseGeometry,
    output_dir: Path,
    *,
    campaign_type: CampaignType,
    seasons: int = 8,
    max_cloud_percent: float = 30,
    end_year: int | None = None,
    scenes_per_season: int = 6,
    edge_exclusion_m: int = 30,
    cache_arrays: bool = True,
    excluded_item_ids: set[str] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> StabilityResult:
    """Build stability for fine or coarse crops, retaining reviewable scene evidence."""
    grid = grid_for_geometry(geometry, resolution=10)
    end_year = end_year or _last_complete_campaign_year(campaign_type)
    excluded = excluded_item_ids or set()
    seasonal_arrays: list[np.ndarray] = []
    season_metadata: list[dict[str, Any]] = []
    scene_inventory: list[dict[str, Any]] = []
    cache_dir = output_dir / ".cache_escenas"
    preview_dir = output_dir / "revision_escenas"

    for position, campaign_year in enumerate(
        range(end_year - seasons + 1, end_year + 1), start=1
    ):
        start, end = stability_campaign_dates(campaign_type, campaign_year)
        label = campaign_label(campaign_type, campaign_year)
        items = imagery.search_sentinel(
            geometry, start, end, max_cloud_percent=max_cloud_percent, max_items=50
        )
        candidates = sorted(
            items,
            key=lambda item: (
                float(item.properties.get("eo:cloud_cover", 100)),
                item.datetime.timestamp() if item.datetime else 0,
            ),
        )[: max(20, scenes_per_season)]
        stack: list[np.ndarray] = []
        used: list[str] = []
        for item in candidates:
            acquired = item.datetime.date().isoformat() if item.datetime else "sin-fecha"
            cache_path = cache_dir / label.replace("/", "-") / f"{_safe_id(item.id)}.npz"
            preview_path = preview_dir / label.replace("/", "-") / (
                f"{acquired}_{_safe_id(item.id)}_falso_color.png"
            )
            record: dict[str, Any] = {
                "campaign_type": campaign_type,
                "campaign": label,
                "item_id": item.id,
                "date": acquired,
                "cloud_percent": _float_or_none(item.properties.get("eo:cloud_cover")),
                "included": item.id not in excluded,
                "preview": str(preview_path.relative_to(output_dir)),
                "valid_pixel_percent": 0.0,
            }
            try:
                false_color: np.ndarray | None = None
                if cache_arrays and cache_path.exists():
                    cached = np.load(cache_path)
                    ndvi = cached["ndvi"]
                    valid = cached["valid"].astype(bool)
                else:
                    ndvi, valid, false_color = read_ndvi_false_color(item, grid)
                    if cache_arrays:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            cache_path,
                            ndvi=ndvi.astype("float32"),
                            valid=valid.astype("uint8"),
                        )
                valid_fraction = _valid_fraction(valid, grid.inside_mask)
                record["valid_pixel_percent"] = round(valid_fraction * 100, 1)
                quality = scene_quality(valid, grid.inside_mask, grid.resolution)
                record["contaminated_percent"] = quality["contaminated_percent"]
                record["largest_contaminated_patch_m2"] = quality["largest_patch_m2"]
                if not preview_path.exists():
                    if false_color is None:
                        _, _, false_color = read_ndvi_false_color(item, grid)
                    _write_false_color_preview(false_color, preview_path, f"{label} · {acquired}")
                if not quality["passes"]:
                    record["included"] = False
                    record["reason"] = quality["reason"]
                elif item.id in excluded:
                    record["reason"] = "excluida por el usuario"
                else:
                    stack.append(ndvi)
                    used.append(item.id)
            except Exception as exc:
                record["included"] = False
                record["reason"] = f"error de lectura: {exc}"
            scene_inventory.append(record)
            if len(used) >= scenes_per_season:
                break

        if stack:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                season_max = np.nanmax(np.stack(stack), axis=0).astype("float32")
            seasonal_arrays.append(season_max)
            season_metadata.append(
                {
                    "campaign_type": campaign_type,
                    "campaign": label,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "items": used,
                    "scene_count": len(used),
                }
            )
        if progress:
            progress(position, seasons, label)

    if len(seasonal_arrays) < 3:
        raise RuntimeError(
            f"Solo se pudieron construir {len(seasonal_arrays)} campañas de {campaign_type}; "
            "se requieren al menos 3"
        )

    stack_array = np.stack(seasonal_arrays)
    analysis_mask, effective_edge_m = _analysis_mask(grid, edge_exclusion_m)
    z_stack = np.full_like(stack_array, np.nan, dtype="float32")
    for index, season in enumerate(stack_array):
        values = season[analysis_mask & np.isfinite(season)]
        if values.size < 20:
            continue
        mean = float(np.mean(values))
        std = max(float(np.std(values)), 0.01)
        z_stack[index] = (season - mean) / std

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        z_mean = np.nanmean(z_stack, axis=0).astype("float32")
        z_std = np.nanstd(z_stack, axis=0).astype("float32")
    valid = analysis_mask & np.isfinite(z_mean) & np.isfinite(z_std)
    classes = np.zeros(grid.shape, dtype="uint8")
    classes[valid] = 3
    classes[valid & (z_mean > 0.3) & (z_std < 0.4)] = 1
    classes[valid & (z_mean > 0.3) & (z_std >= 0.4)] = 2
    classes[valid & (z_mean < -0.3) & (z_std >= 0.4)] = 4
    classes[valid & (z_mean < -0.3) & (z_std < 0.4)] = 5
    smoothed = median_filter(classes, size=3, mode="nearest")
    classes[valid & (smoothed > 0)] = smoothed[valid & (smoothed > 0)]
    _fill_to_boundary(classes, valid, grid.inside_mask)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        write_raster(
            output_dir / "productividad_media_relativa.tif",
            z_mean,
            grid,
            dtype="float32",
            nodata=-9999,
            descriptions=[f"z_mean_{campaign_type}"],
        ),
        write_raster(
            output_dir / "variabilidad_temporal.tif",
            z_std,
            grid,
            dtype="float32",
            nodata=-9999,
            descriptions=[f"z_std_{campaign_type}"],
        ),
        write_raster(
            output_dir / "estabilidad_5_clases.tif",
            classes,
            grid,
            dtype="uint8",
            nodata=0,
            descriptions=[f"stability_class_{campaign_type}"],
        ),
    ]
    (output_dir / "escenas_utilizadas.json").write_text(
        json.dumps(scene_inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return StabilityResult(
        grid=grid,
        seasonal_ndvi=stack_array,
        z_mean=z_mean,
        z_std=z_std,
        classes=classes,
        analysis_mask=analysis_mask,
        campaign_type=campaign_type,
        edge_exclusion_m=effective_edge_m,
        seasons=season_metadata,
        scenes=scene_inventory,
        artifacts=[*artifacts, output_dir / "escenas_utilizadas.json"],
    )


def stability_statistics(result: StabilityResult) -> list[dict[str, Any]]:
    pixel_area_ha = result.grid.resolution**2 / 10_000
    total = int(np.count_nonzero(result.classes))
    rows = []
    for class_id, label in STABILITY_LABELS.items():
        pixels = int(np.count_nonzero(result.classes == class_id))
        rows.append(
            {
                "campaign_type": result.campaign_type,
                "class_id": class_id,
                "label": label,
                "hectares": round(pixels * pixel_area_ha, 2),
                "percent": round(pixels / total * 100, 1) if total else 0,
            }
        )
    return rows


def stability_campaign_dates(campaign_type: CampaignType, year: int) -> tuple[date, date]:
    if campaign_type == "gruesa":
        return date(year, 1, 1), date(year, 2, monthrange(year, 2)[1])
    return date(year, 9, 1), date(year, 10, 31)


def campaign_label(campaign_type: CampaignType, year: int) -> str:
    if campaign_type == "gruesa":
        return f"{year - 1}/{str(year)[-2:]}"
    return str(year)


def _last_complete_campaign_year(campaign_type: CampaignType) -> int:
    today = date.today()
    if campaign_type == "gruesa":
        return today.year if today.month >= 3 else today.year - 1
    return today.year if today.month >= 11 else today.year - 1


def _analysis_mask(grid: RasterGrid, requested_m: int) -> tuple[np.ndarray, int]:
    attempts = [requested_m, 20, 10, 0]
    for distance_m in dict.fromkeys(max(0, value) for value in attempts):
        interior = grid.geometry.buffer(-distance_m) if distance_m else grid.geometry
        if interior.is_empty:
            continue
        mask = geometry_mask(
            [mapping(interior)],
            out_shape=grid.shape,
            transform=grid.transform,
            invert=True,
            all_touched=False,
        )
        if np.count_nonzero(mask) >= 50:
            return mask, distance_m
    return grid.inside_mask.copy(), 0


def _fill_to_boundary(values: np.ndarray, valid: np.ndarray, full_inside: np.ndarray) -> None:
    perimeter = full_inside & ~valid
    if np.any(valid) and np.any(perimeter):
        nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        values[perimeter] = values[tuple(nearest[:, perimeter])]
    values[~full_inside] = 0


def _write_false_color_preview(data: np.ndarray, path: Path, title: str) -> None:
    moved = np.moveaxis(data, 0, -1)
    image = np.zeros_like(moved, dtype="float32")
    valid = np.all(np.isfinite(moved), axis=2)
    for band in range(3):
        values = moved[:, :, band][valid]
        if values.size:
            low, high = np.percentile(values, [2, 98])
            image[:, :, band] = np.clip((moved[:, :, band] - low) / max(high - low, 1), 0, 1)
    image[~valid] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(image)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def _float_or_none(value: Any) -> float | None:
    return float(value) if value is not None else None


def _valid_fraction(valid: np.ndarray, footprint: np.ndarray) -> float:
    denominator = int(np.count_nonzero(footprint))
    return float(np.count_nonzero(valid & footprint) / denominator) if denominator else 0.0
