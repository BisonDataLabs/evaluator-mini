from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np

from evaluador_lotes_mini.imagery.planetary import (
    _deduplicate_acquisitions,
    _valid_fraction,
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
