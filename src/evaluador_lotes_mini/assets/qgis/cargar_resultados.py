"""Ejecutar desde la consola Python de QGIS con exec(open(ruta).read())."""

from pathlib import Path

from osgeo import ogr
from qgis.core import (
    QgsCategorizedSymbolRenderer,
    QgsProject,
    QgsRasterLayer,
    QgsRendererCategory,
    QgsSymbol,
    QgsVectorLayer,
)
from qgis.PyQt.QtGui import QColor

BASE = Path(__file__).resolve().parent.parent
PROJECT = QgsProject.instance()
ZONE_PALETTE = ["#a6611a", "#dfc27d", "#80cdc1", "#018571", "#00441b", "#00280f"]
STABILITY_PALETTE = {
    1: ("#2ca25f", "Alto estable"),
    2: ("#99d8c9", "Alto variable"),
    3: ("#ffffbf", "Medio"),
    4: ("#fdae61", "Bajo variable"),
    5: ("#d73027", "Bajo persistente"),
}


def categorized(layer, field, categories):
    renderer_categories = []
    for value, color, label in categories:
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol.setColor(QColor(color))
        renderer_categories.append(QgsRendererCategory(value, symbol, label))
    layer.setRenderer(QgsCategorizedSymbolRenderer(field, renderer_categories))


for geopackage in sorted((BASE / "gis").glob("*.gpkg")):
    dataset = ogr.Open(str(geopackage))
    if dataset is None:
        continue
    for index in range(dataset.GetLayerCount()):
        layer_name = dataset.GetLayerByIndex(index).GetName()
        display_name = f"{geopackage.stem} · {layer_name}"
        layer = QgsVectorLayer(f"{geopackage}|layername={layer_name}", display_name, "ogr")
        if not layer.isValid():
            continue
        if layer_name.startswith("ambientes_k"):
            count = int(layer_name.removeprefix("ambientes_k"))
            categorized(
                layer,
                "zone",
                [
                    (zone, ZONE_PALETTE[zone - 1], f"Zona {zone}")
                    for zone in range(1, count + 1)
                ],
            )
        elif layer_name.startswith("estabilidad_"):
            categorized(
                layer,
                "class_id",
                [
                    (class_id, color, label)
                    for class_id, (color, label) in STABILITY_PALETTE.items()
                ],
            )
        PROJECT.addMapLayer(layer)

for raster in BASE.rglob("*.tif"):
    relative = raster.relative_to(BASE)
    display_name = " · ".join(relative.with_suffix("").parts)
    layer = QgsRasterLayer(str(raster), display_name)
    if layer.isValid():
        PROJECT.addMapLayer(layer)

print("Evaluador de lotes: capas cargadas con campaña, fecha y simbología")
