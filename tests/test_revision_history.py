import zipfile

import numpy as np
from pyproj import CRS
from rasterio.transform import from_origin
from shapely.geometry import box

from evaluador_lotes_mini.imagery.grid import RasterGrid, write_raster
from evaluador_lotes_mini.processor import (
    _changed_class_percent,
    _snapshot_productivity_previews,
    _zip_lot,
)


def _grid() -> RasterGrid:
    return RasterGrid(
        crs=CRS.from_epsg(32720),
        transform=from_origin(0, 200, 10, 10),
        width=20,
        height=20,
        geometry=box(0, 0, 200, 200),
        resolution=10,
    )


def test_revision_snapshot_preserves_lightweight_previews(tmp_path) -> None:
    grid = _grid()
    classes = np.ones(grid.shape, dtype="uint8")
    write_raster(
        tmp_path / "estabilidad/fina/estabilidad_5_clases.tif",
        classes,
        grid,
        dtype="uint8",
        nodata=0,
    )
    write_raster(
        tmp_path / "ambientes/fina/ambientes_k2.tif",
        classes,
        grid,
        dtype="uint8",
        nodata=0,
    )
    previews = _snapshot_productivity_previews(tmp_path, "fina", "revision-1", "antes")
    assert [row["layer"] for row in previews] == ["estabilidad_5_clases", "ambientes_k2"]
    assert all((tmp_path / row["path"]).exists() for row in previews)


def test_changed_class_percentage_is_explicit() -> None:
    before = np.array([[1, 1], [2, 0]], dtype="uint8")
    after = np.array([[1, 2], [2, 0]], dtype="uint8")
    assert _changed_class_percent(before, after) == 33.3


def test_qgis_readme_is_never_packaged(tmp_path) -> None:
    result_dir = tmp_path / "lote"
    qgis_dir = result_dir / "qgis"
    qgis_dir.mkdir(parents=True)
    (qgis_dir / "LEEME.txt").write_text("legacy", encoding="utf-8")
    (qgis_dir / "cargar_resultados.py").write_text("pass", encoding="utf-8")
    package = _zip_lot(result_dir)
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
    assert not any(name.endswith("qgis/LEEME.txt") for name in names)
    assert any(name.endswith("qgis/cargar_resultados.py") for name in names)
