import json
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from evaluador_lotes_mini.ingestion.files import read_uploaded_file


def test_geojson_multiple_lots() -> None:
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"LOTE_ID": 7, "LOTE": "Norte"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-60, -34], [-59.99, -34], [-59.99, -33.99], [-60, -34]]],
                },
            }
        ],
    }
    lots = read_uploaded_file("lotes.geojson", json.dumps(document).encode())
    assert len(lots) == 1
    assert lots[0].lot_id == "7"
    assert lots[0].name == "Norte"


def test_zip_with_many_geojson_files() -> None:
    def document(name: str) -> bytes:
        return json.dumps(
            {
                "type": "Feature",
                "properties": {"LOTE": name},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[-60, -34], [-59.99, -34], [-59.99, -33.99], [-60, -34]]
                    ],
                },
            }
        ).encode()

    bundle = BytesIO()
    with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
        archive.writestr("zona-a/norte.geojson", document("Norte"))
        archive.writestr("zona-a/sur.geojson", document("Sur"))

    lots = read_uploaded_file("zona-a.zip", bundle.getvalue())
    assert [lot.name for lot in lots] == ["Norte", "Sur"]
    assert len({lot.lot_id for lot in lots}) == 2


def test_zip_rejects_unsafe_path() -> None:
    bundle = BytesIO()
    with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
        archive.writestr("../lote.geojson", "{}")

    try:
        read_uploaded_file("lotes.zip", bundle.getvalue())
    except ValueError as exc:
        assert "ruta insegura" in str(exc)
    else:
        raise AssertionError("El ZIP inseguro debía rechazarse")
