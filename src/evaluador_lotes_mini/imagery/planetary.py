"""Search, rank and materialize Sentinel-2 products from Planetary Computer."""

from __future__ import annotations

import json
from calendar import monthrange
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import planetary_computer as pc
from pystac import Item
from pystac_client import Client
from rasterio.features import geometry_mask
from rasterio.warp import Resampling
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from evaluador_lotes_mini.geometry import buffered_wgs84, reproject
from evaluador_lotes_mini.imagery.grid import (
    RasterGrid,
    grid_for_geometry,
    read_asset,
    write_raster,
)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"
CLEAR_SCL = {4, 5, 6, 11}
MAX_CLIP_CONTAMINATION_PERCENT = 0.5
MAX_CONTAMINATED_PATCH_M2 = 1_000
MAX_REPRESENTATIVE_LOT_CONTAMINATION_PERCENT = 0.5
MAX_REPRESENTATIVE_LOT_PATCH_M2 = 5_000
MAX_REPRESENTATIVE_CONTEXT_CONTAMINATION_PERCENT = 2.5
MAX_REPRESENTATIVE_CONTEXT_PATCH_M2 = 15_000


@dataclass(frozen=True, slots=True)
class SelectedScene:
    item_id: str
    acquisition_date: str
    cloud_percent: float | None
    coverage_percent: float
    collection: str
    contaminated_percent: float
    largest_contaminated_patch_m2: float
    requested_buffer_m: int = 0
    effective_buffer_m: int = 0
    lot_contaminated_percent: float = 0.0
    lot_largest_contaminated_patch_m2: float = 0.0


class PlanetaryImagery:
    def __init__(self) -> None:
        self.catalog = Client.open(STAC_URL, modifier=pc.sign_inplace)

    def search_sentinel(
        self,
        geometry: BaseGeometry,
        start: date,
        end: date,
        max_cloud_percent: float = 30,
        max_items: int = 40,
    ) -> list[Item]:
        search = self.catalog.search(
            collections=[S2_COLLECTION],
            intersects=geometry.__geo_interface__,
            datetime=f"{start.isoformat()}/{end.isoformat()}",
            query={"eo:cloud_cover": {"lt": max_cloud_percent}},
            max_items=max_items,
        )
        items = _deduplicate_acquisitions(list(search.items()))
        return sorted(
            items,
            key=lambda item: (
                -_coverage(item, geometry),
                float(item.properties.get("eo:cloud_cover", 100)),
                item.datetime.timestamp() if item.datetime else 0,
            ),
        )

    def select_peak_scene(
        self,
        geometry: BaseGeometry,
        start: date,
        end: date,
        max_cloud_percent: float = 30,
        candidates: int = 6,
        buffer_m: int = 500,
    ) -> tuple[Item, SelectedScene]:
        requested_context = buffered_wgs84(geometry, buffer_m) if buffer_m else geometry
        items = self.search_sentinel(
            requested_context,
            start,
            end,
            max_cloud_percent,
        )
        if not items:
            raise RuntimeError(f"Sin Sentinel-2 entre {start} y {end}")
        ranked = sorted(
            items,
            key=lambda item: (
                float(item.properties.get("eo:cloud_cover", 100)),
                -_coverage(item, requested_context),
            ),
        )
        ndvi_grid = grid_for_geometry(geometry, resolution=30)
        buffer_candidates = _buffer_fallbacks(buffer_m)
        quality_by_item: dict[str, dict[int, dict[str, Any]]] = {}
        ndvi_by_item: dict[str, tuple[float, float]] = {}
        unreadable_items: set[str] = set()
        best_lot_quality: dict[str, Any] | None = None
        best_context_quality: dict[int, dict[str, Any]] = {}
        last_read_error: str | None = None

        selected: Item | None = None
        selected_quality: dict[str, Any] | None = None
        selected_lot_quality: dict[str, Any] | None = None
        effective_buffer_m = buffer_m
        for candidate_buffer in buffer_candidates:
            scored: list[tuple[float, float, Item]] = []
            for item in ranked:
                if item.id in unreadable_items:
                    continue
                try:
                    qualities = quality_by_item.get(item.id)
                    if qualities is None:
                        qualities = read_scene_quality_buffers(
                            item,
                            geometry,
                            buffer_candidates,
                        )
                        quality_by_item[item.id] = qualities
                        lot_quality = qualities[0]
                        best_lot_quality = _better_quality(best_lot_quality, lot_quality)
                        for measured_buffer in buffer_candidates:
                            best_context_quality[measured_buffer] = _better_quality(
                                best_context_quality.get(measured_buffer),
                                qualities[measured_buffer],
                            )
                    lot_quality = qualities[0]
                    if not lot_quality["passes"]:
                        continue

                    quality = qualities[candidate_buffer]
                    if not quality["passes"]:
                        continue

                    ndvi_score = ndvi_by_item.get(item.id)
                    if ndvi_score is None:
                        ndvi, valid = _read_ndvi(item, ndvi_grid)
                        values = ndvi[valid & np.isfinite(ndvi)]
                        valid_fraction = _valid_fraction(valid, ndvi_grid.inside_mask)
                        if not values.size or valid_fraction < 0.25:
                            continue
                        ndvi_score = (float(np.mean(values)), valid_fraction)
                        ndvi_by_item[item.id] = ndvi_score
                    scored.append((ndvi_score[0], ndvi_score[1], item))
                    if len(scored) >= candidates:
                        break
                except Exception as exc:
                    unreadable_items.add(item.id)
                    last_read_error = str(exc)

            if scored:
                selected = max(scored, key=lambda pair: (pair[0], pair[1]))[2]
                selected_quality = quality_by_item[selected.id][candidate_buffer]
                selected_lot_quality = quality_by_item[selected.id][0]
                effective_buffer_m = candidate_buffer
                break

        if selected is None or selected_quality is None or selected_lot_quality is None:
            raise RuntimeError(
                _scene_failure_message(
                    best_lot_quality,
                    best_context_quality,
                    buffer_candidates,
                    last_read_error,
                )
            )
        metadata = SelectedScene(
            item_id=selected.id,
            acquisition_date=(
                selected.datetime.date().isoformat() if selected.datetime else "unknown"
            ),
            cloud_percent=_float_or_none(selected.properties.get("eo:cloud_cover")),
            coverage_percent=round(_coverage(selected, geometry) * 100, 1),
            collection=S2_COLLECTION,
            contaminated_percent=selected_quality["contaminated_percent"],
            largest_contaminated_patch_m2=selected_quality["largest_patch_m2"],
            requested_buffer_m=buffer_m,
            effective_buffer_m=effective_buffer_m,
            lot_contaminated_percent=selected_lot_quality["contaminated_percent"],
            lot_largest_contaminated_patch_m2=selected_lot_quality["largest_patch_m2"],
        )
        return selected, metadata

    def export_scene_products(
        self,
        item: Item,
        lot_geometry: BaseGeometry,
        output_dir: Path,
        buffer_m: int = 500,
        file_prefix: str | None = None,
        file_suffix: str | None = None,
    ) -> tuple[list[Path], dict[str, Any]]:
        context = buffered_wgs84(lot_geometry, buffer_m)
        quality_grid = grid_for_geometry(context, resolution=20)
        quality = read_scene_quality(
            item,
            quality_grid,
            max_contamination_percent=(
                MAX_REPRESENTATIVE_CONTEXT_CONTAMINATION_PERCENT
                if buffer_m
                else MAX_REPRESENTATIVE_LOT_CONTAMINATION_PERCENT
            ),
            max_patch_area_m2=(
                MAX_REPRESENTATIVE_CONTEXT_PATCH_M2
                if buffer_m
                else MAX_REPRESENTATIVE_LOT_PATCH_M2
            ),
        )
        if not quality["passes"]:
            raise RuntimeError(str(quality["reason"]))
        grid = grid_for_geometry(context)
        blue = read_asset(item.assets["B02"].href, grid)
        green = read_asset(item.assets["B03"].href, grid)
        red = read_asset(item.assets["B04"].href, grid)
        nir = read_asset(item.assets["B08"].href, grid)
        scl = read_asset(
            item.assets["SCL"].href,
            grid,
            resampling=Resampling.nearest,
            dtype="uint8",
            nodata=0,
        )
        valid = np.isin(scl, list(CLEAR_SCL)) & grid.inside_mask
        for band in (blue, green, red, nir):
            band[~valid] = 0
        ndvi = np.full(grid.shape, -9999, dtype="float32")
        denominator = nir + red
        good = valid & (denominator != 0)
        ndvi[good] = (nir[good] - red[good]) / denominator[good]

        def product_name(product: str) -> str:
            return _product_filename(product, file_prefix, file_suffix)

        rgb_path = write_raster(
            output_dir / product_name("RGB"),
            [red, green, blue],
            grid,
            dtype="uint16",
            nodata=0,
            descriptions=["Red", "Green", "Blue"],
        )
        ir_path = write_raster(
            output_dir / product_name("IR"),
            [nir, red, green],
            grid,
            dtype="uint16",
            nodata=0,
            descriptions=["NIR", "Red", "Green"],
        )
        ndvi_path = write_raster(
            output_dir / product_name("NDVI"),
            ndvi,
            grid,
            dtype="float32",
            nodata=-9999,
            descriptions=["NDVI"],
        )
        metadata = {
            "item_id": item.id,
            "date": item.datetime.date().isoformat() if item.datetime else None,
            "cloud_percent": _float_or_none(item.properties.get("eo:cloud_cover")),
            "buffer_m": buffer_m,
            "resolution_m": grid.resolution,
            "crs": grid.crs.to_string(),
            "valid_pixel_percent": round(_valid_fraction(valid, grid.inside_mask) * 100, 1),
            "contaminated_percent": quality["contaminated_percent"],
            "largest_contaminated_patch_m2": quality["largest_patch_m2"],
            "quality_status": "aprobada",
            "products": {
                "RGB": rgb_path.name,
                "IR": ir_path.name,
                "NDVI": ndvi_path.name,
            },
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return [rgb_path, ir_path, ndvi_path, output_dir / "metadata.json"], metadata


def campaign_dates(campaign: str, year: int) -> tuple[date, date]:
    if campaign == "gruesa":
        image_year = year + 1
        return date(image_year, 1, 1), date(image_year, 2, monthrange(image_year, 2)[1])
    if campaign == "fina":
        return date(year, 9, 1), date(year, 10, 31)
    raise ValueError(f"Campaña desconocida: {campaign}")


def scene_metadata(item: Item, geometry: BaseGeometry) -> dict[str, Any]:
    return asdict(
        SelectedScene(
            item_id=item.id,
            acquisition_date=item.datetime.date().isoformat() if item.datetime else "unknown",
            cloud_percent=_float_or_none(item.properties.get("eo:cloud_cover")),
            coverage_percent=round(_coverage(item, geometry) * 100, 1),
            collection=S2_COLLECTION,
            contaminated_percent=0.0,
            largest_contaminated_patch_m2=0.0,
        )
    )


def _read_ndvi(item: Item, grid: RasterGrid) -> tuple[np.ndarray, np.ndarray]:
    red = read_asset(item.assets["B04"].href, grid)
    nir = read_asset(item.assets["B08"].href, grid)
    scl = read_asset(
        item.assets["SCL"].href,
        grid,
        resampling=Resampling.nearest,
        dtype="uint8",
        nodata=0,
    )
    valid = np.isin(scl, list(CLEAR_SCL)) & grid.inside_mask & ((nir + red) != 0)
    ndvi = np.full(grid.shape, np.nan, dtype="float32")
    ndvi[valid] = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
    return ndvi, valid


def read_ndvi(item: Item, grid: RasterGrid) -> tuple[np.ndarray, np.ndarray]:
    """Public aligned NDVI reader used by temporal analyses."""
    return _read_ndvi(item, grid)


def read_ndvi_false_color(
    item: Item, grid: RasterGrid
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read aligned NDVI plus NIR/red/green bands for visual scene review."""
    red = read_asset(item.assets["B04"].href, grid)
    green = read_asset(item.assets["B03"].href, grid)
    nir = read_asset(item.assets["B08"].href, grid)
    scl = read_asset(
        item.assets["SCL"].href,
        grid,
        resampling=Resampling.nearest,
        dtype="uint8",
        nodata=0,
    )
    valid = np.isin(scl, list(CLEAR_SCL)) & grid.inside_mask & ((nir + red) != 0)
    ndvi = np.full(grid.shape, np.nan, dtype="float32")
    ndvi[valid] = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
    false_color = np.stack([nir, red, green]).astype("float32")
    false_color[:, ~valid] = np.nan
    return ndvi, valid, false_color


def read_scene_quality(
    item: Item,
    grid: RasterGrid,
    *,
    max_contamination_percent: float = MAX_CLIP_CONTAMINATION_PERCENT,
    max_patch_area_m2: float = MAX_CONTAMINATED_PATCH_M2,
) -> dict[str, Any]:
    """Measure holes/cloud masks inside a clip and apply the APB acceptance rule."""
    scl = read_asset(
        item.assets["SCL"].href,
        grid,
        resampling=Resampling.nearest,
        dtype="uint8",
        nodata=0,
    )
    clear = np.isin(scl, list(CLEAR_SCL)) & grid.inside_mask
    return scene_quality(
        clear,
        grid.inside_mask,
        grid.resolution,
        max_contamination_percent=max_contamination_percent,
        max_patch_area_m2=max_patch_area_m2,
    )


def read_scene_quality_buffers(
    item: Item,
    lot_geometry: BaseGeometry,
    buffers_m: list[int],
) -> dict[int, dict[str, Any]]:
    """Measure the lot and all context buffers from one aligned SCL read."""
    measured_buffers = list(dict.fromkeys([0, *buffers_m]))
    largest_buffer = max(measured_buffers)
    largest_geometry = (
        buffered_wgs84(lot_geometry, largest_buffer) if largest_buffer else lot_geometry
    )
    grid = grid_for_geometry(largest_geometry, resolution=20)
    scl = read_asset(
        item.assets["SCL"].href,
        grid,
        resampling=Resampling.nearest,
        dtype="uint8",
        nodata=0,
    )
    classified_clear = np.isin(scl, list(CLEAR_SCL))
    results: dict[int, dict[str, Any]] = {}
    for buffer_m in measured_buffers:
        geometry_wgs84 = (
            buffered_wgs84(lot_geometry, buffer_m) if buffer_m else lot_geometry
        )
        geometry_metric = reproject(geometry_wgs84, "EPSG:4326", grid.crs)
        footprint = geometry_mask(
            [mapping(geometry_metric)],
            out_shape=grid.shape,
            transform=grid.transform,
            invert=True,
            all_touched=True,
        )
        clear = classified_clear & footprint
        results[buffer_m] = scene_quality(
            clear,
            footprint,
            grid.resolution,
            max_contamination_percent=(
                MAX_REPRESENTATIVE_LOT_CONTAMINATION_PERCENT
                if buffer_m == 0
                else MAX_REPRESENTATIVE_CONTEXT_CONTAMINATION_PERCENT
            ),
            max_patch_area_m2=(
                MAX_REPRESENTATIVE_LOT_PATCH_M2
                if buffer_m == 0
                else MAX_REPRESENTATIVE_CONTEXT_PATCH_M2
            ),
        )
    return results


def scene_quality(
    clear: np.ndarray,
    footprint: np.ndarray,
    resolution_m: float,
    *,
    max_contamination_percent: float = MAX_CLIP_CONTAMINATION_PERCENT,
    max_patch_area_m2: float = MAX_CONTAMINATED_PATCH_M2,
) -> dict[str, Any]:
    """Classify clip contamination, excluding pixels outside the requested footprint."""
    from scipy.ndimage import label

    contaminated = footprint & ~clear
    denominator = int(np.count_nonzero(footprint))
    contaminated_pixels = int(np.count_nonzero(contaminated))
    contaminated_percent = contaminated_pixels / denominator * 100 if denominator else 100.0
    components, count = label(contaminated, structure=np.ones((3, 3), dtype="uint8"))
    largest_pixels = 0
    if count:
        sizes = np.bincount(components.ravel())
        largest_pixels = int(sizes[1:].max(initial=0))
    largest_patch_m2 = largest_pixels * resolution_m**2
    passes = (
        contaminated_percent <= max_contamination_percent
        and largest_patch_m2 <= max_patch_area_m2
    )
    reasons = []
    if contaminated_percent > max_contamination_percent:
        reasons.append(
            f"{contaminated_percent:.2f}% del clip está enmascarado "
            f"(máximo {max_contamination_percent:.2f}%)"
        )
    if largest_patch_m2 > max_patch_area_m2:
        reasons.append(
            f"parche continuo de {largest_patch_m2 / 10_000:.2f} ha "
            f"(máximo {max_patch_area_m2 / 10_000:.2f} ha)"
        )
    return {
        "passes": passes,
        "contaminated_percent": round(contaminated_percent, 3),
        "largest_patch_m2": round(largest_patch_m2, 1),
        "reason": (
            "Escena aprobada"
            if passes
            else "Descartada automáticamente: " + "; ".join(reasons)
        ),
    }


def _buffer_fallbacks(requested_buffer_m: int) -> list[int]:
    """Return the requested context followed by smaller supported contexts."""
    requested = max(0, int(requested_buffer_m))
    values = [requested]
    values.extend(value for value in (300, 100) if value < requested)
    return list(dict.fromkeys(values))


def _better_quality(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Keep the least contaminated quality observation for diagnostics."""
    if current is None:
        return candidate
    current_key = (current["contaminated_percent"], current["largest_patch_m2"])
    candidate_key = (candidate["contaminated_percent"], candidate["largest_patch_m2"])
    return candidate if candidate_key < current_key else current


def _quality_summary(quality: dict[str, Any]) -> str:
    return (
        f"{quality['contaminated_percent']:.2f}% enmascarado, "
        f"parche máximo {quality['largest_patch_m2'] / 10_000:.2f} ha"
    )


def _scene_failure_message(
    best_lot_quality: dict[str, Any] | None,
    best_context_quality: dict[int, dict[str, Any]],
    buffer_candidates: list[int],
    last_read_error: str | None,
) -> str:
    details: list[str] = []
    if best_lot_quality is not None:
        details.append(f"lote: {_quality_summary(best_lot_quality)}")
    for buffer_m in buffer_candidates:
        quality = best_context_quality.get(buffer_m)
        if quality is not None:
            details.append(f"buffer {buffer_m} m: {_quality_summary(quality)}")
    if last_read_error:
        details.append(f"último error de lectura: {last_read_error}")
    suffix = f" Mejor calidad observada: {'; '.join(details)}." if details else ""
    return (
        "No se encontró una escena cuyo lote pase el control y que tenga contexto válido "
        f"en los buffers probados ({', '.join(map(str, buffer_candidates))} m).{suffix}"
    )


def _coverage(item: Item, geometry: BaseGeometry) -> float:
    if not item.geometry:
        return 0.0
    footprint = shape(item.geometry)
    if geometry.area <= 0:
        return 0.0
    return min(1.0, footprint.intersection(geometry).area / geometry.area)


def _float_or_none(value: Any) -> float | None:
    return float(value) if value is not None else None


def _deduplicate_acquisitions(items: list[Item]) -> list[Item]:
    """Keep the newest processing baseline for each sensing time and MGRS tile."""
    selected: dict[tuple[str, str], Item] = {}
    for item in items:
        sensing_time = item.datetime.isoformat() if item.datetime else item.id
        tile = str(item.properties.get("s2:mgrs_tile") or _tile_from_item_id(item.id))
        key = sensing_time, tile
        previous = selected.get(key)
        if previous is None or item.id > previous.id:
            selected[key] = item
    return list(selected.values())


def _tile_from_item_id(item_id: str) -> str:
    parts = item_id.split("_")
    return parts[4] if len(parts) > 4 else item_id


def _valid_fraction(valid: np.ndarray, footprint: np.ndarray) -> float:
    denominator = int(np.count_nonzero(footprint))
    return float(np.count_nonzero(valid & footprint) / denominator) if denominator else 0.0


def _product_filename(product: str, prefix: str | None, suffix: str | None) -> str:
    if prefix and suffix:
        return f"{prefix}_{product}-{suffix}.tif"
    prefix_text = f"{prefix}_" if prefix else ""
    return f"{prefix_text}{product}.tif"
