from datetime import UTC, date, datetime
from types import SimpleNamespace

import numpy as np
from pyproj import CRS
from rasterio.transform import from_origin
from shapely.geometry import box

from evaluador_lotes_mini.analysis.stability import (
    _analysis_mask,
    calculate_stability,
    campaign_label,
    stability_campaign_dates,
)
from evaluador_lotes_mini.imagery.grid import RasterGrid
from evaluador_lotes_mini.imagery.planetary import campaign_dates


def test_stability_windows_are_crop_specific() -> None:
    assert stability_campaign_dates("gruesa", 2024) == (date(2024, 1, 1), date(2024, 2, 29))
    assert stability_campaign_dates("fina", 2024) == (date(2024, 9, 1), date(2024, 10, 31))
    assert campaign_label("gruesa", 2024) == "2023/24"
    assert campaign_label("fina", 2024) == "2024"


def test_representative_imagery_uses_requested_crop_windows() -> None:
    assert campaign_dates("gruesa", 2023) == (date(2024, 1, 1), date(2024, 2, 29))
    assert campaign_dates("fina", 2023) == (date(2023, 9, 1), date(2023, 10, 31))


def test_analysis_mask_excludes_thirty_metre_field_edge() -> None:
    geometry = box(0, 0, 200, 200)
    grid = RasterGrid(
        crs=CRS.from_epsg(32721),
        transform=from_origin(0, 200, 10, 10),
        width=20,
        height=20,
        geometry=geometry,
        resolution=10,
    )
    mask, effective_distance = _analysis_mask(grid, 30)
    assert effective_distance == 30
    assert 0 < mask.sum() < grid.inside_mask.sum()


def test_stability_pipeline_writes_review_inventory(monkeypatch, tmp_path) -> None:
    class FakeImagery:
        def search_sentinel(self, geometry, start, end, **kwargs):
            del geometry, end, kwargs
            return [
                SimpleNamespace(
                    id=f"S2-{start.year}",
                    datetime=datetime(start.year, start.month, 15, tzinfo=UTC),
                    properties={"eo:cloud_cover": 4.0},
                )
            ]

    def fake_reader(item, grid):
        del item
        rows, columns = np.indices(grid.shape)
        ndvi = (0.2 + columns / max(grid.width, 1) * 0.5).astype("float32")
        valid = grid.inside_mask
        ndvi[~valid] = np.nan
        false_color = np.stack([ndvi + 0.3, ndvi + 0.1, ndvi]).astype("float32")
        return ndvi, valid, false_color

    monkeypatch.setattr(
        "evaluador_lotes_mini.analysis.stability.read_ndvi_false_color", fake_reader
    )
    result = calculate_stability(
        FakeImagery(),
        box(-60.0, -34.0, -59.99, -33.99),
        tmp_path,
        campaign_type="fina",
        seasons=3,
        end_year=2024,
        cache_arrays=False,
    )
    assert result.campaign_type == "fina"
    assert len(result.seasons) == 3
    assert len(result.scenes) == 3
    assert (tmp_path / "escenas_utilizadas.json").exists()
    assert len(list((tmp_path / "revision_escenas").rglob("*.png"))) == 3
    assert not (tmp_path / ".cache_escenas").exists()
