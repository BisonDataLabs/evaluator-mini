"""Fast in-memory renderers for inspecting GIS outputs in Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape

from evaluador_lotes_mini.geometry import reproject

STABILITY_COLORS = {
    1: (44, 162, 95),
    2: (153, 216, 201),
    3: (255, 255, 191),
    4: (253, 174, 97),
    5: (215, 48, 39),
}
ZONE_COLORS = {
    1: (166, 97, 26),
    2: (223, 194, 125),
    3: (128, 205, 193),
    4: (1, 133, 113),
    5: (0, 68, 27),
}


def render_raster(path: Path, boundary_geojson: Path | None = None) -> np.ndarray:
    with rasterio.open(path) as source:
        data = source.read()
        nodata = source.nodata
        if source.count >= 3:
            image = _stretch_multiband(data[:3], nodata)
        elif "estabilidad_5_clases" in path.stem:
            image = _categorical(data[0], STABILITY_COLORS)
        elif "ambientes_k" in path.stem:
            image = _categorical(data[0], ZONE_COLORS)
        else:
            image = _continuous(data[0], nodata, ndvi="ndvi" in path.stem.lower())
        if boundary_geojson and boundary_geojson.exists():
            _draw_boundary(image, boundary_geojson, source.crs, source.transform, source.res[0])
        return image


def _stretch_multiband(data: np.ndarray, nodata: float | None) -> np.ndarray:
    moved = np.moveaxis(data.astype("float32"), 0, -1)
    valid = np.all(np.isfinite(moved), axis=2)
    if nodata is not None:
        valid &= np.all(moved != nodata, axis=2)
    image = np.zeros_like(moved, dtype="float32")
    for band in range(3):
        values = moved[:, :, band][valid]
        if values.size:
            low, high = np.percentile(values, [2, 98])
            image[:, :, band] = np.clip((moved[:, :, band] - low) / max(high - low, 1), 0, 1)
    return (image * 255).astype("uint8")


def _continuous(data: np.ndarray, nodata: float | None, *, ndvi: bool) -> np.ndarray:
    valid = np.isfinite(data)
    if nodata is not None:
        valid &= data != nodata
    normalized = np.zeros(data.shape, dtype="float32")
    values = data[valid]
    if values.size:
        if ndvi:
            low, high = -0.1, 0.9
        else:
            low, high = np.percentile(values, [2, 98])
        normalized[valid] = np.clip((data[valid] - low) / max(high - low, 1e-6), 0, 1)
    colors = np.array([[165, 42, 42], [255, 230, 128], [0, 110, 55]], dtype="float32")
    position = normalized * 2
    lower = np.floor(position).astype(int).clip(0, 1)
    fraction = (position - lower)[..., None]
    image = colors[lower] * (1 - fraction) + colors[lower + 1] * fraction
    image[~valid] = 0
    return image.astype("uint8")


def _categorical(data: np.ndarray, colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
    image = np.zeros((*data.shape, 3), dtype="uint8")
    for value, color in colors.items():
        image[data == value] = color
    return image


def _draw_boundary(
    image: np.ndarray,
    geojson_path: Path,
    crs,
    transform,
    resolution: float,
) -> None:
    document = json.loads(geojson_path.read_text(encoding="utf-8"))
    geometry = shape(document["features"][0]["geometry"])
    metric = reproject(geometry, "EPSG:4326", crs)
    outline = metric.boundary.buffer(max(resolution * 1.2, 1))
    mask = rasterize([(outline, 1)], out_shape=image.shape[:2], transform=transform, fill=0).astype(
        bool
    )
    image[mask] = (255, 235, 40)
