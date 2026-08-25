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


@dataclass(frozen=True, slots=True)
class SelectedScene:
    item_id: str
    acquisition_date: str
    cloud_percent: float | None
    coverage_percent: float
    collection: str


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
        )[:candidates]
        grid = grid_for_geometry(geometry, resolution=30)
        scored: list[tuple[float, float, Item]] = []
        for item in ranked:
            try:
                ndvi, valid = _read_ndvi(item, grid)
                values = ndvi[valid & np.isfinite(ndvi)]
                valid_fraction = _valid_fraction(valid, grid.inside_mask)
                if values.size and valid_fraction >= 0.25:
                    scored.append((float(np.mean(values)), valid_fraction, item))
            except Exception:
                continue
        selected = max(scored, key=lambda pair: (pair[0], pair[1]))[2] if scored else ranked[0]
        metadata = SelectedScene(
            item_id=selected.id,
            acquisition_date=(
                selected.datetime.date().isoformat() if selected.datetime else "unknown"
            ),
            cloud_percent=_float_or_none(selected.properties.get("eo:cloud_cover")),
            coverage_percent=round(_coverage(selected, geometry) * 100, 1),
            collection=S2_COLLECTION,
        )
        return selected, metadata

    def export_scene_products(
        self,
        item: Item,
        lot_geometry: BaseGeometry,
        output_dir: Path,
        buffer_m: int = 500,
        file_prefix: str | None = None,
    ) -> tuple[list[Path], dict[str, Any]]:
        context = buffered_wgs84(lot_geometry, buffer_m)
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

        prefix = f"{file_prefix}_" if file_prefix else ""
        rgb_path = write_raster(
            output_dir / f"{prefix}RGB.tif",
            [red, green, blue],
            grid,
            dtype="uint16",
            nodata=0,
            descriptions=["Red", "Green", "Blue"],
        )
        ir_path = write_raster(
            output_dir / f"{prefix}IR.tif",
            [nir, red, green],
            grid,
            dtype="uint16",
            nodata=0,
            descriptions=["NIR", "Red", "Green"],
        )
        ndvi_path = write_raster(
            output_dir / f"{prefix}NDVI.tif",
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
