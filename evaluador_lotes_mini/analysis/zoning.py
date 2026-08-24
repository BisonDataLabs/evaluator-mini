"""Management-zone alternatives derived from temporal stability metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rasterio.features import shapes
from scipy.ndimage import distance_transform_edt, median_filter
from shapely.geometry import shape
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from evaluador_lotes_mini.analysis.stability import StabilityResult
from evaluador_lotes_mini.imagery.grid import write_raster


@dataclass(slots=True)
class ZoneAlternative:
    k: int
    labels: np.ndarray
    silhouette: float
    polygons: list[dict]
    raster_path: Path
    zone_percentages: dict[int, float]


def build_zone_alternatives(
    stability: StabilityResult,
    output_dir: Path,
    zone_counts: tuple[int, ...] = (2, 3, 4),
    min_patch_area_m2: float = 1_000,
) -> list[ZoneAlternative]:
    valid = (
        stability.grid.inside_mask & np.isfinite(stability.z_mean) & np.isfinite(stability.z_std)
    )
    features = np.column_stack([stability.z_mean[valid], stability.z_std[valid]])
    if features.shape[0] < 50:
        raise RuntimeError("Muy pocos píxeles válidos para zonificar")
    scaled = StandardScaler().fit_transform(features)
    alternatives: list[ZoneAlternative] = []
    for k in zone_counts:
        if k >= features.shape[0]:
            continue
        model = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=500)
        raw = model.fit_predict(scaled)
        productivity = [(cluster, features[raw == cluster, 0].mean()) for cluster in range(k)]
        productivity.sort(key=lambda pair: pair[1])
        remap = {cluster: rank + 1 for rank, (cluster, _) in enumerate(productivity)}
        ordered = np.array([remap[value] for value in raw], dtype="uint8")
        grid = np.zeros(stability.grid.shape, dtype="uint8")
        grid[valid] = ordered
        filtered = median_filter(grid, size=5, mode="nearest")
        grid[valid] = filtered[valid]
        full_inside = stability.grid.inside_mask
        perimeter = full_inside & ~valid
        if np.any(perimeter):
            nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
            grid[perimeter] = grid[tuple(nearest[:, perimeter])]
        grid[~full_inside] = 0
        total_pixels = int(np.count_nonzero(grid))
        zone_percentages = {
            zone: round(int(np.count_nonzero(grid == zone)) / total_pixels * 100, 1)
            for zone in range(1, k + 1)
        }
        silhouette = float(
            silhouette_score(
                scaled,
                raw,
                sample_size=min(10_000, len(raw)),
                random_state=42,
            )
        )
        raster_path = write_raster(
            output_dir / f"ambientes_k{k}.tif",
            grid,
            stability.grid,
            dtype="uint8",
            nodata=0,
            descriptions=[f"zones_k{k}"],
        )
        polygons = _polygonize(
            grid,
            stability,
            k=k,
            min_patch_area_m2=min_patch_area_m2,
        )
        alternatives.append(
            ZoneAlternative(k, grid, silhouette, polygons, raster_path, zone_percentages)
        )
    return alternatives


def recommended_zone_count(alternatives: list[ZoneAlternative]) -> int | None:
    eligible = [
        item
        for item in alternatives
        if item.silhouette >= 0.25 and min(item.zone_percentages.values(), default=0) >= 10
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda item: item.silhouette).k


def _polygonize(
    grid: np.ndarray,
    stability: StabilityResult,
    *,
    k: int,
    min_patch_area_m2: float,
) -> list[dict]:
    by_zone: dict[int, list] = {zone: [] for zone in range(1, k + 1)}
    for geometry, value in shapes(grid, mask=grid > 0, transform=stability.grid.transform):
        zone = int(value)
        polygon = shape(geometry)
        if polygon.area >= min_patch_area_m2:
            by_zone[zone].append(polygon)
    rows: list[dict] = []
    pixel_area_ha = stability.grid.resolution**2 / 10_000
    for zone, polygons in by_zone.items():
        if not polygons:
            continue
        merged = unary_union(polygons).simplify(stability.grid.resolution)
        pixels = grid == zone
        productivity_label = "Baja" if zone == 1 else "Alta" if zone == k else "Media"
        rows.append(
            {
                "zone": zone,
                "label": f"Zona {zone} - {productivity_label}",
                "hectares": round(int(np.count_nonzero(pixels)) * pixel_area_ha, 2),
                "z_mean": round(float(np.nanmean(stability.z_mean[pixels])), 3),
                "z_std": round(float(np.nanmean(stability.z_std[pixels])), 3),
                "geometry": merged,
            }
        )
    return rows
