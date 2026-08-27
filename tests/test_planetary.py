from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np

from evaluador_lotes_mini.imagery.planetary import (
    _deduplicate_acquisitions,
    _product_filename,
    _valid_fraction,
    scene_quality,
)


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
