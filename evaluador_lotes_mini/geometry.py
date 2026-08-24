"""Geometry normalization, validation and metric operations."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable

from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union
from shapely.validation import make_valid

POLYGON_TYPES = {"Polygon", "MultiPolygon"}


def normalize_polygon(geometry: BaseGeometry) -> BaseGeometry:
    """Return a valid non-empty polygonal geometry in WGS84 coordinate order."""
    if geometry.is_empty:
        raise ValueError("La geometría está vacía")
    fixed = make_valid(geometry) if not geometry.is_valid else geometry
    polygons: list[Polygon] = []
    if isinstance(fixed, Polygon):
        polygons = [fixed]
    elif isinstance(fixed, MultiPolygon):
        polygons = list(fixed.geoms)
    elif hasattr(fixed, "geoms"):
        polygons = [g for g in fixed.geoms if isinstance(g, Polygon)]
    if not polygons:
        raise ValueError(f"Se requiere Polygon/MultiPolygon, llegó {geometry.geom_type}")
    merged = unary_union(polygons)
    if merged.is_empty or merged.area <= 0:
        raise ValueError("La geometría no tiene superficie")
    return merged


def utm_crs_for(geometry: BaseGeometry) -> CRS:
    centroid = geometry.centroid
    zone = int((centroid.x + 180) // 6) + 1
    epsg = (32600 if centroid.y >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def reproject(geometry: BaseGeometry, source: CRS | str, target: CRS | str) -> BaseGeometry:
    transformer = Transformer.from_crs(source, target, always_xy=True)
    return transform(transformer.transform, geometry)


def buffered_wgs84(geometry: BaseGeometry, distance_m: float) -> BaseGeometry:
    target = utm_crs_for(geometry)
    metric = reproject(geometry, "EPSG:4326", target)
    buffered = metric.buffer(distance_m)
    return reproject(buffered, target, "EPSG:4326")


def area_hectares(geometry: BaseGeometry) -> float:
    metric = reproject(geometry, "EPSG:4326", utm_crs_for(geometry))
    return metric.area / 10_000.0


def geometry_hash(geometry: BaseGeometry) -> str:
    normalized = normalize_polygon(geometry)
    return hashlib.sha256(normalized.wkb).hexdigest()[:16]


def safe_slug(value: str, fallback: str = "lote") -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text[:80] or fallback


def feature_collection(geometries: Iterable[tuple[BaseGeometry, dict]]) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": props, "geometry": mapping(geom)}
            for geom, props in geometries
        ],
    }
