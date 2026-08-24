from shapely.geometry import Polygon

from evaluador_lotes_mini.geometry import area_hectares, buffered_wgs84, safe_slug


def test_geometry_helpers() -> None:
    lot = Polygon([(-60.0, -34.0), (-59.99, -34.0), (-59.99, -33.99), (-60.0, -33.99)])
    assert 80 < area_hectares(lot) < 120
    assert buffered_wgs84(lot, 500).area > lot.area
    assert safe_slug("Lote Ñandú 12") == "lote-nandu-12"
