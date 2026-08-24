"""Aligned 10 m raster grids and COG window reads."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.features import geometry_mask
from rasterio.transform import Affine, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from evaluador_lotes_mini.geometry import reproject, utm_crs_for


@dataclass(frozen=True, slots=True)
class RasterGrid:
    crs: CRS
    transform: Affine
    width: int
    height: int
    geometry: BaseGeometry
    resolution: float = 10.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def inside_mask(self) -> np.ndarray:
        return geometry_mask(
            [mapping(self.geometry)],
            out_shape=self.shape,
            transform=self.transform,
            invert=True,
            all_touched=True,
        )


def grid_for_geometry(geometry_wgs84: BaseGeometry, resolution: float = 10.0) -> RasterGrid:
    crs = utm_crs_for(geometry_wgs84)
    geometry = reproject(geometry_wgs84, "EPSG:4326", crs)
    min_x, min_y, max_x, max_y = geometry.bounds
    min_x = floor(min_x / resolution) * resolution
    min_y = floor(min_y / resolution) * resolution
    max_x = ceil(max_x / resolution) * resolution
    max_y = ceil(max_y / resolution) * resolution
    width = max(1, int(round((max_x - min_x) / resolution)))
    height = max(1, int(round((max_y - min_y) / resolution)))
    return RasterGrid(
        crs=crs,
        transform=from_origin(min_x, max_y, resolution, resolution),
        width=width,
        height=height,
        geometry=geometry,
        resolution=resolution,
    )


def read_asset(
    url: str,
    grid: RasterGrid,
    *,
    resampling: Resampling = Resampling.bilinear,
    dtype: str = "float32",
    nodata: float = 0,
) -> np.ndarray:
    with (
        rasterio.open(url) as source,
        WarpedVRT(
            source,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.width,
            height=grid.height,
            resampling=resampling,
            nodata=nodata,
        ) as vrt,
    ):
        return vrt.read(1, out_dtype=dtype)


def write_raster(
    path: Path,
    arrays: list[np.ndarray] | np.ndarray,
    grid: RasterGrid,
    *,
    dtype: str,
    nodata: float | int,
    descriptions: list[str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    bands = arrays if isinstance(arrays, list) else [arrays]
    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": len(bands),
        "dtype": dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": min(256, _valid_block(grid.width)),
        "blockysize": min(256, _valid_block(grid.height)),
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as target:
        for index, band in enumerate(bands, start=1):
            prepared = np.nan_to_num(band, nan=nodata, posinf=nodata, neginf=nodata)
            target.write(prepared.astype(dtype), index)
            if descriptions and index <= len(descriptions):
                target.set_band_description(index, descriptions[index - 1])
        factors = [2, 4, 8, 16]
        valid = [
            factor for factor in factors if grid.width // factor >= 1 and grid.height // factor >= 1
        ]
        if valid:
            target.build_overviews(valid, Resampling.nearest)
            target.update_tags(ns="rio_overview", resampling="nearest")
    return path


def _valid_block(size: int) -> int:
    if size >= 16:
        return max(16, (min(size, 256) // 16) * 16)
    return 16
