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
from rasterio.warp import Resampling
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from evaluador_lotes_mini.geometry import buffered_wgs84
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


@dataclass(frozen=True, slots=True)
class SelectedScene:
    item_id: str
    acquisition_date: str
    cloud_percent: float | None
    coverage_percent: float
    collection: str
    contaminated_percent: float
    largest_contaminated_patch_m2: float


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
        items = self.search_sentinel(geometry, start, end, max_cloud_percent)
        if not items:
            raise RuntimeError(f"Sin Sentinel-2 entre {start} y {end}")
        ranked = sorted(
            items,
            key=lambda item: (
                float(item.properties.get("eo:cloud_cover", 100)),
                -_coverage(item, geometry),
            ),
        )
        grid = grid_for_geometry(geometry, resolution=30)
        quality_grid = grid_for_geometry(buffered_wgs84(geometry, buffer_m), resolution=20)
        scored: list[tuple[float, float, Item]] = []
        quality_by_item: dict[str, dict[str, Any]] = {}
        for item in ranked:
            try:
                quality = read_scene_quality(item, quality_grid)
                quality_by_item[item.id] = quality
                if not quality["passes"]:
                    continue
                ndvi, valid = _read_ndvi(item, grid)
                values = ndvi[valid & np.isfinite(ndvi)]
                valid_fraction = _valid_fraction(valid, grid.inside_mask)
                if values.size and valid_fraction >= 0.25:
                    scored.append((float(np.mean(values)), valid_fraction, item))
                    if len(scored) >= candidates:
                        break
            except Exception:
                continue
        if not scored:
            raise RuntimeError(
                "No se encontró una escena sin nubes, sombras o huecos relevantes "
                "dentro del lote y su buffer"
            )
        selected = max(scored, key=lambda pair: (pair[0], pair[1]))[2]
        quality = quality_by_item[selected.id]
        metadata = SelectedScene(
            item_id=selected.id,
            acquisition_date=(
                selected.datetime.date().isoformat() if selected.datetime else "unknown"
            ),
            cloud_percent=_float_or_none(selected.properties.get("eo:cloud_cover")),
            coverage_percent=round(_coverage(selected, geometry) * 100, 1),
            collection=S2_COLLECTION,
            contaminated_percent=quality["contaminated_percent"],
            largest_contaminated_patch_m2=quality["largest_patch_m2"],
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
        quality = read_scene_quality(item, quality_grid)
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


def read_scene_quality(item: Item, grid: RasterGrid) -> dict[str, Any]:
    """Measure holes/cloud masks inside a clip and apply the APB acceptance rule."""
    scl = read_asset(
        item.assets["SCL"].href,
        grid,
        resampling=Resampling.nearest,
        dtype="uint8",
        nodata=0,
    )
    clear = np.isin(scl, list(CLEAR_SCL)) & grid.inside_mask
    return scene_quality(clear, grid.inside_mask, grid.resolution)


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
