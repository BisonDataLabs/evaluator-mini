from pathlib import Path

import numpy as np

from evaluador_lotes_mini.analysis.zoning import ZoneAlternative, recommended_zone_count


def _alternative(k: int, silhouette: float, percentages: dict[int, float]) -> ZoneAlternative:
    return ZoneAlternative(
        k=k,
        labels=np.zeros((2, 2), dtype="uint8"),
        silhouette=silhouette,
        polygons=[],
        raster_path=Path(f"k{k}.tif"),
        zone_percentages=percentages,
    )


def test_recommendation_rejects_tiny_environment() -> None:
    alternatives = [
        _alternative(2, 0.86, {1: 4.0, 2: 96.0}),
        _alternative(3, 0.70, {1: 20.0, 2: 30.0, 3: 50.0}),
    ]
    assert recommended_zone_count(alternatives) == 3


def test_recommendation_can_return_none() -> None:
    assert recommended_zone_count([_alternative(2, 0.90, {1: 5.0, 2: 95.0})]) is None
