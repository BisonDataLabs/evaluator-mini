from datetime import UTC, date, datetime
from types import SimpleNamespace

import numpy as np
from shapely.geometry import box, mapping

from evaluador_lotes_mini.imagery import planetary
from evaluador_lotes_mini.imagery.planetary import (
    PlanetaryImagery,
    _buffer_fallbacks,
    _deduplicate_acquisitions,
    _product_filename,
    _valid_fraction,
    scene_quality,
)


def _quality(contaminated_percent: float, largest_patch_m2: float) -> dict:
    return {
        "passes": contaminated_percent <= 0.5 and largest_patch_m2 <= 1_000,
        "contaminated_percent": contaminated_percent,
        "largest_patch_m2": largest_patch_m2,
        "reason": "prueba",
    }


def test_reprocessed_sentinel_acquisitions_are_deduplicated() -> None:
    sensing_time = datetime(2023, 9, 13, 13, 57, 11, tzinfo=UTC)
    original = SimpleNamespace(
        id="S2A_MSIL2A_20230913T135711_R067_T20HPG_20230913T231934",
        datetime=sensing_time,
        properties={"s2:mgrs_tile": "20HPG"},
    )
    reprocessed = SimpleNamespace(
        id="S2A_MSIL2A_20230913T135711_R067_T20HPG_20241025T135823",
        datetime=sensing_time,
        properties={"s2:mgrs_tile": "20HPG"},
    )
    assert _deduplicate_acquisitions([original, reprocessed]) == [reprocessed]


def test_valid_fraction_uses_lot_pixels_as_denominator() -> None:
    footprint = np.array([[True, True, False], [True, True, False]])
    valid = np.array([[True, False, False], [True, True, False]])
    assert _valid_fraction(valid, footprint) == 0.75


def test_cloud_rule_rejects_large_contiguous_patch() -> None:
    footprint = np.ones((100, 100), dtype=bool)
    clear = footprint.copy()
    clear[10:12, 10:16] = False
    quality = scene_quality(clear, footprint, 10)
    assert quality["contaminated_percent"] < 0.5
    assert quality["largest_patch_m2"] == 1_200
    assert quality["passes"] is False


def test_qgis_product_filename_is_short_and_predictable() -> None:
    assert (
        _product_filename("NDVI", "2019-09-17", "fina_Calido_seco")
        == "2019-09-17_NDVI-fina_Calido_seco.tif"
    )


def test_buffer_fallbacks_never_exceed_requested_context() -> None:
    assert _buffer_fallbacks(500) == [500, 300, 100]
    assert _buffer_fallbacks(200) == [200, 100]
    assert _buffer_fallbacks(100) == [100]
    assert _buffer_fallbacks(0) == [0]


def test_scene_selection_reduces_buffer_only_after_lot_passes(monkeypatch) -> None:
    geometry = box(-59.0, -34.0, -58.99, -33.99)
    item = SimpleNamespace(
        id="scene-1",
        datetime=datetime(2024, 1, 11, tzinfo=UTC),
        properties={"eo:cloud_cover": 5.0},
        geometry=mapping(geometry),
    )
    quality_results = {
        0: _quality(0.0, 0.0),
        500: _quality(1.2, 4_000),
        300: _quality(0.8, 2_000),
        100: _quality(0.0, 0.0),
    }
    fake_grid = SimpleNamespace(inside_mask=np.ones((2, 2), dtype=bool))
    monkeypatch.setattr(planetary, "grid_for_geometry", lambda *args, **kwargs: fake_grid)
    monkeypatch.setattr(
        planetary,
        "read_scene_quality_buffers",
        lambda *args, **kwargs: quality_results,
    )
    monkeypatch.setattr(
        planetary,
        "_read_ndvi",
        lambda *args, **kwargs: (
            np.full((2, 2), 0.7, dtype="float32"),
            np.ones((2, 2), dtype=bool),
        ),
    )
    monkeypatch.setattr(
        PlanetaryImagery,
        "search_sentinel",
        lambda *args, **kwargs: [item],
    )
    imagery = object.__new__(PlanetaryImagery)

    _, selection = imagery.select_peak_scene(
        geometry,
        date(2024, 1, 1),
        date(2024, 2, 29),
        buffer_m=500,
    )

    assert selection.requested_buffer_m == 500
    assert selection.effective_buffer_m == 100
    assert selection.lot_contaminated_percent == 0.0
