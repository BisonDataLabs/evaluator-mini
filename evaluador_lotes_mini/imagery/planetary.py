"""Search, rank and materialize Sentinel-2 products from Planetary Computer."""

from __future__ import annotations

import json
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
        items = list(search.items())
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
        scored: list[tuple[float, Item]] = []
        for item in ranked:
            try:
                ndvi, valid = _read_ndvi(item, grid)
                score = float(np.nanmean(np.where(valid, ndvi, np.nan)))
                if np.isfinite(score):
                    scored.append((score, item))
            except Exception:
                continue
        selected = max(scored, key=lambda pair: pair[0])[1] if scored else ranked[0]
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

        rgb_path = write_raster(
            output_dir / "RGB.tif",
            [red, green, blue],
            grid,
            dtype="uint16",
            nodata=0,
            descriptions=["Red", "Green", "Blue"],
        )
        ir_path = write_raster(
            output_dir / "IR.tif",
            [nir, red, green],
            grid,
            dtype="uint16",
            nodata=0,
            descriptions=["NIR", "Red", "Green"],
        )
        ndvi_path = write_raster(
            output_dir / "NDVI.tif",
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
            "valid_pixel_percent": round(valid.mean() * 100, 1),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return [rgb_path, ir_path, ndvi_path, output_dir / "metadata.json"], metadata


def campaign_dates(campaign: str, year: int) -> tuple[date, date]:
    if campaign == "gruesa":
        return date(year, 10, 1), date(year + 1, 3, 31)
    if campaign == "fina":
        return date(year, 4, 1), date(year, 9, 30)
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


def _coverage(item: Item, geometry: BaseGeometry) -> float:
    if not item.geometry:
        return 0.0
    footprint = shape(item.geometry)
    if geometry.area <= 0:
        return 0.0
    return min(1.0, footprint.intersection(geometry).area / geometry.area)


def _float_or_none(value: Any) -> float | None:
    return float(value) if value is not None else None
